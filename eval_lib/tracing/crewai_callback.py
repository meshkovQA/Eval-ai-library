# eval_lib/tracing/crewai_callback.py
"""CrewAI trace collector — turns CrewAI event-bus events into eval-lib spans.

Supported CrewAI layouts (both verified against the shipped source):

* ``crewai >= 1.x`` — bus at ``crewai.events.event_bus.crewai_event_bus``;
  event classes lazily exported from ``crewai.events`` and defined under
  ``crewai.events.types.<module>``.
* legacy ``crewai 0.x`` — bus at ``crewai.utilities.events.crewai_event_bus``;
  event classes re-exported from ``crewai.utilities.events``.

On both, ``bus.on(EventClass)`` is a decorator factory keyed by the event
*class* and handlers are invoked as ``handler(source, event)``.

Usage::

    from eval_lib.tracing import tracer
    from eval_lib.tracing.crewai_callback import install_crewai_tracing

    trace_id = tracer.start_trace("crewai_agent")
    collector = install_crewai_tracing()      # binds the active trace
    result = crew.kickoff()
    collector.set_token_usage(result)         # CrewOutput.token_usage -> declared totals

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()

``CrewAITraceCollector()`` is equivalent to ``install_crewai_tracing()`` and
is kept for backward compatibility. Create one collector per trace, *after*
``tracer.start_trace()`` (or pass ``trace_id=``).

Span tree produced::

    crew:<name>            AGENT_STEP  (root)
    └── task:<name>        AGENT_STEP
        └── agent:<role>   AGENT_STEP
            ├── llm:<model>   LLM_CALL   (per-call usage accumulated on the trace)
            └── <tool_name>   TOOL_CALL  (real started_at/finished_at timing)

Why parents are explicit: CrewAI >= 1.x runs sync handlers on a
``ThreadPoolExecutor`` under a *copy* of the emitting context, and events may
be emitted from threads that never had the trace bound (async tasks,
``kickoff_async``). The tracer's context-bound "current span" is therefore
not a usable parent and the trace id may be missing. The collector captures
the trace id once, binds it inside every handler (in a private context copy,
so nothing leaks), and tracks the crew/task/agent hierarchy itself.
"""

from __future__ import annotations

import contextvars
import functools
import importlib
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from .context import get_trace_id, set_trace_id
from .trace_utils import safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

logger = logging.getLogger(__name__)

__all__ = ["CrewAITraceCollector", "install_crewai_tracing"]


# (event class name, module stem under the events package, collector method)
_EVENT_HANDLERS: Tuple[Tuple[str, str, str], ...] = (
    ("CrewKickoffStartedEvent", "crew_events", "_on_crew_start"),
    ("CrewKickoffCompletedEvent", "crew_events", "_on_crew_complete"),
    ("CrewKickoffFailedEvent", "crew_events", "_on_crew_error"),
    ("TaskStartedEvent", "task_events", "_on_task_start"),
    ("TaskCompletedEvent", "task_events", "_on_task_complete"),
    ("TaskFailedEvent", "task_events", "_on_task_error"),
    ("AgentExecutionStartedEvent", "agent_events", "_on_agent_start"),
    ("AgentExecutionCompletedEvent", "agent_events", "_on_agent_complete"),
    ("AgentExecutionErrorEvent", "agent_events", "_on_agent_error"),
    ("LLMCallStartedEvent", "llm_events", "_on_llm_start"),
    ("LLMCallCompletedEvent", "llm_events", "_on_llm_complete"),
    ("LLMCallFailedEvent", "llm_events", "_on_llm_error"),
    ("ToolUsageStartedEvent", "tool_usage_events", "_on_tool_start"),
    ("ToolUsageFinishedEvent", "tool_usage_events", "_on_tool_end"),
    ("ToolUsageErrorEvent", "tool_usage_events", "_on_tool_error"),
)

# Process-global registration state. CrewAI's bus is a process singleton, so
# the handlers are registered exactly once and dispatch to whichever collector
# is currently active (the most recently created / installed one).
_install_lock = threading.Lock()
_installed = False
_active: Optional["CrewAITraceCollector"] = None

_SPAN_TYPE_ERROR = {
    "crew": "CrewError",
    "task": "TaskError",
    "agent": "AgentError",
    "llm": "LLMError",
    "tool": "ToolError",
}


# =========================================================================
# Bus discovery / registration
# =========================================================================


def _load_event_bus() -> Tuple[Any, str, str]:
    """Return ``(bus, events_package, types_package)`` for the installed CrewAI."""
    try:
        from crewai.events.event_bus import crewai_event_bus  # type: ignore[import-not-found]

        return crewai_event_bus, "crewai.events", "crewai.events.types"
    except ImportError as modern_error:
        try:
            from crewai.utilities.events import (  # type: ignore[import-not-found]
                crewai_event_bus,
            )

            return crewai_event_bus, "crewai.utilities.events", "crewai.utilities.events"
        except ImportError:
            raise ImportError(
                "neither crewai.events.event_bus (crewai>=1) nor "
                f"crewai.utilities.events (crewai 0.x) is importable: {modern_error}"
            ) from modern_error


def _resolve_event_class(
    events_package: str, types_package: str, class_name: str, module_stem: str
) -> type:
    """Find an event class: package re-export first, then its defining module."""
    package = importlib.import_module(events_package)
    cls = getattr(package, class_name, None)
    if cls is None:
        module = importlib.import_module(f"{types_package}.{module_stem}")
        cls = getattr(module, class_name)
    return cast(type, cls)


def _make_bus_handler(method_name: str) -> Callable[..., None]:
    """Build the function registered on the bus for one event class.

    Both CrewAI layouts call ``handler(source, event)``. The event is always
    the *last* positional argument, so a ``*args`` signature also tolerates a
    bus that passes only the event. ``inspect.signature`` reports a single
    parameter, which keeps crewai>=1 on its two-argument call path (three
    parameters would mean "wants RuntimeState").
    """

    def _handler(*args: Any) -> None:
        if not args:
            return
        source = args[0] if len(args) > 1 else None
        event = args[-1]
        collector = _active
        if collector is not None:
            getattr(collector, method_name)(source, event)

    _handler.__name__ = _handler.__qualname__ = "evallib_" + method_name.lstrip("_")
    return _handler


def _ensure_installed() -> bool:
    """Register the bus handlers once per process. True when installed."""
    global _installed
    if _installed:
        return True
    try:
        bus, events_package, types_package = _load_event_bus()
    except Exception as e:
        logger.warning(
            "eval_lib.tracing: CrewAI tracing NOT installed — %s: %s. "
            "Crew runs will produce no spans.",
            type(e).__name__,
            e,
        )
        return False

    registered: List[str] = []
    for class_name, module_stem, method_name in _EVENT_HANDLERS:
        try:
            event_cls = _resolve_event_class(events_package, types_package, class_name, module_stem)
            bus.on(event_cls)(_make_bus_handler(method_name))
        except Exception as e:
            logger.debug(
                "eval_lib.tracing: CrewAI event %s not subscribed (%s: %s)",
                class_name,
                type(e).__name__,
                e,
            )
            continue
        registered.append(class_name)

    if not registered:
        logger.warning(
            "eval_lib.tracing: CrewAI tracing NOT installed — no known event classes "
            "could be subscribed via %s. Crew runs will produce no spans.",
            events_package,
        )
        return False

    _installed = True
    logger.info(
        "eval_lib.tracing: CrewAI tracing installed via %s (%d event types)",
        events_package,
        len(registered),
    )
    return True


def _activate(collector: "CrewAITraceCollector") -> bool:
    """Make ``collector`` the dispatch target and make sure handlers exist."""
    global _active
    with _install_lock:
        _active = collector
        return _ensure_installed()


def install_crewai_tracing(trace_id: Optional[str] = None) -> "CrewAITraceCollector":
    """Install CrewAI tracing (idempotent) and bind it to a trace.

    Handlers are registered on the process-global bus only once. Calling this
    again re-binds the active collector to ``trace_id`` (default: the trace
    active in the current context). A collector already bound to a *different*
    trace is left alone and a fresh one becomes active — one collector per
    trace, so spans of two runs never mix.
    """
    resolved = trace_id or get_trace_id()
    current = _active
    if current is not None and (resolved is None or current.trace_id in (None, resolved)):
        if resolved:
            current.trace_id = resolved
        current.registered = _activate(current)
        return current
    return CrewAITraceCollector(trace_id=resolved)


# =========================================================================
# Event field helpers
# =========================================================================


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or dict-key lookup — events are pydantic models, tests use dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(obj: Any, *keys: str) -> Any:
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value
    return None


def _payload(value: Any) -> Any:
    """Shape a value for a span field.

    Strings go through :func:`safe_str` so the configured field cap applies;
    JSON-native containers stay structured (the sender serializes them
    safely); anything else is stringified.
    """
    if value is None or isinstance(value, (bool, int, float, dict, list, tuple)):
        return value
    return safe_str(value)


def _output_text(obj: Any) -> Any:
    """Best representation of a TaskOutput / CrewOutput / plain value.

    ``.raw`` is the human-readable result on both output models; fall back
    to the model's own dump so structure is kept rather than ``str(obj)``.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float, dict, list, tuple)):
        return _payload(obj)
    raw = _get(obj, "raw")
    if isinstance(raw, str) and raw:
        return safe_str(raw)
    mapping = as_mapping(obj)
    if mapping:
        return mapping
    return safe_str(obj)


def _error_fields(event: Any, default_type: str) -> Tuple[str, str]:
    """``(message, error_type)`` for a failure event.

    ``error_type`` comes from the event's own ``error_type`` (a class on
    crewai>=1 ``TaskFailedEvent``), else from the error's class when it is an
    exception, else the per-kind default — never a misleading ``ToolError``
    for a task failure.
    """
    error = _get(event, "error")
    error_type = _get(event, "error_type")
    if isinstance(error_type, type):
        error_type = error_type.__name__
    elif error_type is not None:
        error_type = str(error_type)
    elif isinstance(error, BaseException):
        error_type = type(error).__name__
    else:
        error_type = default_type
    message = safe_str(error) if error is not None else default_type
    return str(message), str(error_type)


def _task_key(event: Any) -> Optional[str]:
    """Stable id of the task an event belongs to (``task_id`` or ``task.id``)."""
    task_id = _get(event, "task_id")
    if task_id:
        return str(task_id)
    task = _get(event, "task")
    if task is None:
        return None
    tid = _get(task, "id")
    return str(tid) if tid is not None else str(id(task))


def _agent_role(event: Any) -> Optional[str]:
    role = _get(event, "agent_role")
    if role:
        return str(role)
    agent = _get(event, "agent")
    if agent is None:
        return None
    if isinstance(agent, str):
        return agent
    role = _get(agent, "role")
    return str(role) if role else None


def _label(text: Any, fallback: str, limit: int = 80) -> str:
    """Short single-line span label; the full text lives in the span input."""
    if text is None:
        return fallback
    lines = str(text).strip().splitlines()
    if not lines:
        return fallback
    line = lines[0].strip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _task_label(event: Any) -> str:
    name = _get(event, "task_name")
    task = _get(event, "task")
    if not name and task is not None:
        name = _get(task, "name") or _get(task, "description")
    if not name:
        name = _get(event, "description")
    return _label(name, "task")


def _epoch(value: Any) -> Optional[float]:
    if isinstance(value, datetime):
        try:
            return value.timestamp()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _token_fields(token_usage: Any) -> Optional[Dict[str, int]]:
    """Trace-level totals from ``UsageMetrics`` (pydantic) or a dict.

    ``UsageMetrics`` fields: ``total_tokens``, ``prompt_tokens``,
    ``cached_prompt_tokens``, ``completion_tokens``, ``reasoning_tokens``,
    ``successful_requests``. ``cached_prompt_tokens`` is CrewAI-specific and
    not known to :func:`usage_from_mapping`, so it is read here.
    """
    mapping = as_mapping(token_usage)
    if not mapping:
        return None
    found = usage_from_mapping(mapping) or {}
    input_tokens = found.get("input_tokens", 0)
    output_tokens = found.get("output_tokens", 0)
    cached = found.get("cached_tokens", 0) or _int(mapping.get("cached_prompt_tokens"))
    reasoning = found.get("reasoning_tokens", 0)
    total = _int(mapping.get("total_tokens")) or (input_tokens + output_tokens)
    if not any((input_tokens, output_tokens, total)):
        return None
    fields = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
    }
    if cached:
        fields["cached_tokens"] = cached
    if reasoning:
        fields["reasoning_tokens"] = reasoning
    return fields


def _declare_token_fields(fields: Dict[str, int]) -> None:
    """Declare token totals as trace facts (explicit kwargs keep the typed API)."""
    tracer.set_trace_metadata(
        input_tokens=fields.get("input_tokens"),
        output_tokens=fields.get("output_tokens"),
        total_tokens=fields.get("total_tokens"),
        cached_tokens=fields.get("cached_tokens"),
        reasoning_tokens=fields.get("reasoning_tokens"),
    )


def _compact(**fields: Any) -> Dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


# =========================================================================
# Collector
# =========================================================================


def _handler(fn: Callable[..., None]) -> Callable[..., None]:
    """Adapt ``fn(self, event)`` so it can be called as ``(event)`` or
    ``(source, event)``, run it with the collector's trace bound in a private
    context copy, and never let an exception escape into the framework."""

    @functools.wraps(fn)
    def wrapper(self: "CrewAITraceCollector", *args: Any) -> None:
        if not args:
            return
        event = args[-1]
        try:
            self._run(fn, self, event)
        except Exception as e:
            logger.warning(
                "eval_lib.tracing: CrewAI %s failed: %s: %s",
                fn.__name__,
                type(e).__name__,
                e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )

    return wrapper


_AgentKey = Tuple[Optional[str], Optional[str]]
_ToolKey = Tuple[Optional[str], Optional[str], str]


class CrewAITraceCollector:
    """Collect CrewAI events into spans of one trace.

    Instantiating it registers the bus handlers (once per process) and makes
    this collector the active dispatch target. Handler methods also accept
    direct calls — ``collector._on_tool_start(event)`` or
    ``(source, event)`` — which is how they are unit-tested.

    Args:
        trace_id: Trace to attach spans to. Defaults to the trace active in
            the current context; if neither is known the first event's
            (copied) context is used.
        install: Register on the CrewAI bus. ``False`` keeps the collector
            passive (direct calls only).
    """

    def __init__(self, trace_id: Optional[str] = None, install: bool = True):
        self.trace_id: Optional[str] = trace_id or get_trace_id()
        self.registered = False
        self._lock = threading.RLock()
        self._crew_stack: List[TraceSpan] = []
        self._task_spans: Dict[Optional[str], TraceSpan] = {}
        self._agent_spans: Dict[_AgentKey, TraceSpan] = {}
        self._llm_by_call: Dict[str, TraceSpan] = {}
        self._llm_stacks: Dict[_AgentKey, List[TraceSpan]] = {}
        self._tool_stacks: Dict[_ToolKey, List[TraceSpan]] = {}
        self._model_reported = False
        if install:
            self.registered = _activate(self)

    # ------------------------------------------------------------ plumbing

    def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn`` with this collector's trace bound, in a context copy.

        The bus may call us on a worker thread whose context has no trace
        id; binding it in a *copy* makes ``tracer.start_span`` /
        ``set_trace_metadata`` work without leaking into the caller.
        """

        def _call() -> Any:
            if self.trace_id is None:
                # Latch from the emitting context (crewai>=1 copies it onto
                # the worker) so later events from bare threads still bind.
                self.trace_id = get_trace_id()
            set_trace_id(self.trace_id)
            with self._lock:
                return fn(*args)

        return contextvars.copy_context().run(_call)

    def _agent_key(self, task_key: Optional[str], role: Optional[str]) -> Optional[_AgentKey]:
        """Key of the open agent span an event belongs to.

        Exact ``(task, role)`` first; then the newest agent on the same task
        (role renamed / unknown) or with the same role (task unknown, as on
        crewai 0.x tool events); with no correlation data at all, the only
        open agent if there is exactly one.
        """
        if (task_key, role) in self._agent_spans:
            return (task_key, role)
        for candidate in reversed(list(self._agent_spans)):
            tk, r = candidate
            if task_key is not None and tk == task_key:
                return candidate
            if task_key is None and role is not None and r == role:
                return candidate
        if task_key is None and role is None and len(self._agent_spans) == 1:
            return next(iter(self._agent_spans))
        return None

    def _parent_id(
        self,
        task_key: Optional[str] = None,
        role: Optional[str] = None,
        *,
        under_agent: bool = True,
    ) -> Optional[str]:
        """Explicit parent: agent > task > crew > root."""
        span: Optional[TraceSpan] = None
        if under_agent:
            key = self._agent_key(task_key, role)
            if key is not None:
                span = self._agent_spans.get(key)
        if span is None:
            span = self._task_spans.get(task_key)
        if span is None and self._crew_stack:
            span = self._crew_stack[-1]
        return span.span_id if span is not None else None

    def _synth(
        self,
        name: str,
        span_type: SpanType,
        parent_id: Optional[str],
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceSpan]:
        """A completion arrived without its start.

        crewai>=1 dispatches handlers on a thread pool, so two adjacent
        events can be processed out of order. Record the completion as its
        own span rather than drop what it carries.
        """
        logger.debug(
            "eval_lib.tracing: CrewAI %s %r completed without a matching start",
            span_type.value,
            name,
        )
        return tracer.start_span(
            name=name,
            span_type=span_type,
            input_data=input_data,
            metadata=metadata or {},
            parent_span_id=parent_id,
            set_current=False,
        )

    def _report_model(self, model: Any) -> None:
        if model and not self._model_reported:
            tracer.set_trace_metadata(model=str(model))
            self._model_reported = True

    def _declare_usage(self, output: Any, total_tokens: Any = None) -> None:
        """Declare trace totals from an output's ``token_usage`` or a bare total."""
        fields = _token_fields(_get(output, "token_usage")) if output is not None else None
        if fields:
            _declare_token_fields(fields)
        elif _int(total_tokens) > 0:
            tracer.set_trace_metadata(total_tokens=_int(total_tokens))

    # ---------------------------------------------------------- crew events

    @_handler
    def _on_crew_start(self, event: Any) -> None:
        name = _get(event, "crew_name") or "crew"
        inputs = _get(event, "inputs")
        parent = self._crew_stack[-1].span_id if self._crew_stack else None
        span = tracer.start_span(
            name=f"crew:{name}",
            span_type=SpanType.AGENT_STEP,
            input_data=_payload(inputs),
            metadata=_compact(crew_name=name),
            parent_span_id=parent,
            set_current=False,
        )
        if span is None:
            return
        self._crew_stack.append(span)
        if parent is None and inputs is not None:
            tracer.set_trace_metadata(input=_payload(inputs))

    @_handler
    def _on_crew_complete(self, event: Any) -> None:
        raw_output = _get(event, "output")
        output = _output_text(raw_output)
        span = self._crew_stack.pop() if self._crew_stack else None
        if span is None:
            name = _get(event, "crew_name") or "crew"
            span = self._synth(f"crew:{name}", SpanType.AGENT_STEP, None)
        tracer.end_span(span, output=output)
        if not self._crew_stack:  # outermost crew: trace-level facts
            if output is not None:
                tracer.set_trace_metadata(output=output)
            self._declare_usage(raw_output, total_tokens=_get(event, "total_tokens"))

    @_handler
    def _on_crew_error(self, event: Any) -> None:
        message, error_type = _error_fields(event, _SPAN_TYPE_ERROR["crew"])
        span = self._crew_stack.pop() if self._crew_stack else None
        if span is None:
            name = _get(event, "crew_name") or "crew"
            span = self._synth(f"crew:{name}", SpanType.AGENT_STEP, None)
        tracer.end_span(span, error=message, error_type=error_type)

    # ---------------------------------------------------------- task events

    @_handler
    def _on_task_start(self, event: Any) -> None:
        key = _task_key(event)
        task = _get(event, "task")
        description = _get(task, "description") if task is not None else _get(event, "description")
        expected = _get(task, "expected_output") if task is not None else None
        input_data = _compact(
            description=safe_str(description),
            expected_output=safe_str(expected),
            context=safe_str(_get(event, "context")),
        )
        span = tracer.start_span(
            name=f"task:{_task_label(event)}",
            span_type=SpanType.AGENT_STEP,
            input_data=input_data or None,
            metadata=_compact(task_id=key),
            parent_span_id=self._parent_id(under_agent=False),
            set_current=False,
        )
        if span is None:
            return
        self._task_spans[key] = span

    @_handler
    def _on_task_complete(self, event: Any) -> None:
        key = _task_key(event)
        output = _output_text(_get(event, "output"))
        span = self._task_spans.pop(key, None)
        if span is None:
            span = self._synth(
                f"task:{_task_label(event)}",
                SpanType.AGENT_STEP,
                self._parent_id(under_agent=False),
                metadata=_compact(task_id=key),
            )
        tracer.end_span(span, output=output)

    @_handler
    def _on_task_error(self, event: Any) -> None:
        key = _task_key(event)
        message, error_type = _error_fields(event, _SPAN_TYPE_ERROR["task"])
        span = self._task_spans.pop(key, None)
        if span is None:
            span = self._synth(
                f"task:{_task_label(event)}",
                SpanType.AGENT_STEP,
                self._parent_id(under_agent=False),
                metadata=_compact(task_id=key),
            )
        tracer.end_span(span, error=message, error_type=error_type)

    # --------------------------------------------------------- agent events

    @_handler
    def _on_agent_start(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        agent = _get(event, "agent")
        agent_id = _get(event, "agent_id") or (
            _get(agent, "id") if agent is not None and not isinstance(agent, str) else None
        )
        span = tracer.start_span(
            name=f"agent:{role or 'agent'}",
            span_type=SpanType.AGENT_STEP,
            input_data=_payload(_get(event, "task_prompt")),
            metadata=_compact(
                role=role,
                task_id=key,
                agent_id=str(agent_id) if agent_id is not None else None,
            ),
            parent_span_id=self._parent_id(key, under_agent=False),
            set_current=False,
        )
        if span is None:
            return
        self._agent_spans[(key, role)] = span

    def _pop_agent(self, task_key: Optional[str], role: Optional[str]) -> Optional[TraceSpan]:
        agent_key = self._agent_key(task_key, role)
        return self._agent_spans.pop(agent_key, None) if agent_key is not None else None

    @_handler
    def _on_agent_complete(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        span = self._pop_agent(key, role)
        if span is None:
            span = self._synth(
                f"agent:{role or 'agent'}",
                SpanType.AGENT_STEP,
                self._parent_id(key, under_agent=False),
                metadata=_compact(role=role, task_id=key),
            )
        tracer.end_span(span, output=_output_text(_get(event, "output")))

    @_handler
    def _on_agent_error(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        message, error_type = _error_fields(event, _SPAN_TYPE_ERROR["agent"])
        span = self._pop_agent(key, role)
        if span is None:
            span = self._synth(
                f"agent:{role or 'agent'}",
                SpanType.AGENT_STEP,
                self._parent_id(key, under_agent=False),
                metadata=_compact(role=role, task_id=key),
            )
        tracer.end_span(span, error=message, error_type=error_type)

    # ----------------------------------------------------------- LLM events

    @_handler
    def _on_llm_start(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        model = _get(event, "model")
        call_id = _get(event, "call_id")
        metadata = _compact(
            model=str(model) if model else None,
            call_id=str(call_id) if call_id else None,
            task_id=key,
            agent_role=role,
            temperature=_get(event, "temperature"),
            top_p=_get(event, "top_p"),
            max_tokens=_get(event, "max_tokens"),
            stream=_get(event, "stream"),
        )
        span = tracer.start_span(
            name=f"llm:{model}" if model else "llm_call",
            span_type=SpanType.LLM_CALL,
            input_data=_payload(_get(event, "messages")),
            metadata=metadata,
            parent_span_id=self._parent_id(key, role),
            set_current=False,
        )
        if span is None:
            return
        if call_id:
            self._llm_by_call[str(call_id)] = span
        self._llm_stacks.setdefault((key, role), []).append(span)
        self._report_model(model)

    def _pop_llm(
        self, call_id: Any, task_key: Optional[str], role: Optional[str]
    ) -> Optional[TraceSpan]:
        """Open LLM span for a completion: by ``call_id``, else LIFO per agent."""
        span = self._llm_by_call.get(str(call_id)) if call_id else None
        if span is None:
            stack = self._llm_stacks.get((task_key, role))
            if stack:
                span = stack[-1]
        if span is None:
            return None
        self._llm_by_call = {k: v for k, v in self._llm_by_call.items() if v is not span}
        for stack_key in list(self._llm_stacks):
            remaining = [s for s in self._llm_stacks[stack_key] if s is not span]
            if remaining:
                self._llm_stacks[stack_key] = remaining
            else:
                del self._llm_stacks[stack_key]
        return span

    @_handler
    def _on_llm_complete(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        model = _get(event, "model")
        call_id = _get(event, "call_id")
        usage = usage_from_mapping(as_mapping(_get(event, "usage")))
        span = self._pop_llm(call_id, key, role)
        if span is None:
            span = self._synth(
                f"llm:{model}" if model else "llm_call",
                SpanType.LLM_CALL,
                self._parent_id(key, role),
                input_data=_payload(_get(event, "messages")),
                metadata=_compact(
                    model=str(model) if model else None,
                    call_id=str(call_id) if call_id else None,
                    task_id=key,
                    agent_role=role,
                ),
            )
        if span is not None:
            call_type = _get(event, "call_type")
            extra = _compact(
                finish_reason=_get(event, "finish_reason"),
                call_type=getattr(call_type, "value", call_type),
            )
            span.metadata.update(extra)
            if usage:
                span.metadata["usage"] = dict(usage)
        tracer.end_span(span, output=_payload(_get(event, "response")))
        # Accumulate per call (never overwrite): the trace's usage block is
        # the running total of every LLM call the crew made.
        tracer.add_trace_usage(
            input_tokens=usage["input_tokens"] if usage else 0,
            output_tokens=usage["output_tokens"] if usage else 0,
            cached_tokens=usage["cached_tokens"] if usage else 0,
            reasoning_tokens=usage["reasoning_tokens"] if usage else 0,
            calls=1,
            trace_id=self.trace_id,
        )
        self._report_model(model)

    @_handler
    def _on_llm_error(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        model = _get(event, "model")
        message, error_type = _error_fields(event, _SPAN_TYPE_ERROR["llm"])
        span = self._pop_llm(_get(event, "call_id"), key, role)
        if span is None:
            span = self._synth(
                f"llm:{model}" if model else "llm_call",
                SpanType.LLM_CALL,
                self._parent_id(key, role),
                metadata=_compact(model=str(model) if model else None),
            )
        tracer.end_span(span, error=message, error_type=error_type)

    # ---------------------------------------------------------- tool events

    @_handler
    def _on_tool_start(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        name = str(_get(event, "tool_name") or "tool")
        span = tracer.start_span(
            name=name,
            span_type=SpanType.TOOL_CALL,
            input_data=_payload(_first(event, "tool_args", "input")),
            metadata=_compact(
                agent_role=role,
                task_id=key,
                tool_class=_get(event, "tool_class"),
                run_attempts=_get(event, "run_attempts"),
            ),
            parent_span_id=self._parent_id(key, role),
            set_current=False,
        )
        if span is None:
            return
        self._tool_stacks.setdefault((key, role, name), []).append(span)

    def _pop_tool(
        self, task_key: Optional[str], role: Optional[str], name: str
    ) -> Optional[TraceSpan]:
        """Open tool span: LIFO per ``(task, agent, tool)`` so concurrent
        same-name calls pair correctly; else the newest open span with that
        name (correlation fields missing on one side, e.g. crewai 0.x)."""
        keys = [(task_key, role, name)] + [
            k for k in reversed(list(self._tool_stacks)) if k[2] == name
        ]
        for stack_key in keys:
            stack = self._tool_stacks.get(stack_key)
            if stack:
                span = stack.pop()
                if not stack:
                    del self._tool_stacks[stack_key]
                return span
        return None

    @_handler
    def _on_tool_end(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        name = str(_get(event, "tool_name") or "tool")
        output = _payload(_get(event, "output"))
        started = _epoch(_get(event, "started_at"))
        finished = _epoch(_get(event, "finished_at"))
        span = self._pop_tool(key, role, name)
        if span is None:
            span = self._synth(
                name,
                SpanType.TOOL_CALL,
                self._parent_id(key, role),
                input_data=_payload(_first(event, "tool_args", "input")),
                metadata=_compact(agent_role=role, task_id=key),
            )
        if span is None:
            return
        # Real timing from the event beats our handler-side clock: handlers
        # run asynchronously on a pool, so "now" is not when the tool ran.
        if started is not None:
            span.start_time = started
        if _get(event, "from_cache"):
            span.metadata["from_cache"] = True
        failure = _get(event, "failure")
        if failure is not None:
            message = _get(failure, "message") or safe_str(failure)
            tracer.end_span(span, output=output, error=message, error_type="ToolFailure")
        else:
            tracer.end_span(span, output=output)
        if finished is not None:
            span.end_time = finished
            span.duration_ms = round(max(0.0, finished - span.start_time) * 1000, 2)

    @_handler
    def _on_tool_error(self, event: Any) -> None:
        key = _task_key(event)
        role = _agent_role(event)
        name = str(_get(event, "tool_name") or "tool")
        message, error_type = _error_fields(event, _SPAN_TYPE_ERROR["tool"])
        span = self._pop_tool(key, role, name)
        if span is None:
            span = self._synth(
                name,
                SpanType.TOOL_CALL,
                self._parent_id(key, role),
                input_data=_payload(_first(event, "tool_args", "input")),
                metadata=_compact(agent_role=role, task_id=key),
            )
        tracer.end_span(span, error=message, error_type=error_type)

    # ------------------------------------------------------------ public API

    def set_token_usage(self, crew_output: Any) -> None:
        """Declare trace-level token totals from ``crew.kickoff()``'s result.

        Accepts a ``CrewOutput`` (its ``token_usage`` is a ``UsageMetrics``
        pydantic model), a bare ``UsageMetrics``, or a dict in either shape.
        Declared totals take precedence over the per-call accumulation in the
        trace payload; on crewai 0.x (no usage on LLM events) they are the
        only token source.
        """
        token_usage = _get(crew_output, "token_usage")
        if token_usage is None and as_mapping(crew_output):
            token_usage = crew_output
        fields = _token_fields(token_usage)
        if not fields:
            return
        self._run(_declare_token_fields, fields)

    def _drain(self) -> List[TraceSpan]:
        spans: List[TraceSpan] = []
        for stack in self._tool_stacks.values():
            spans.extend(reversed(stack))
        for stack in self._llm_stacks.values():
            spans.extend(s for s in reversed(stack) if s not in spans)
        spans.extend(s for s in self._llm_by_call.values() if s not in spans)
        spans.extend(reversed(list(self._agent_spans.values())))
        spans.extend(reversed(list(self._task_spans.values())))
        spans.extend(reversed(self._crew_stack))
        self._tool_stacks.clear()
        self._llm_stacks.clear()
        self._llm_by_call.clear()
        self._agent_spans.clear()
        self._task_spans.clear()
        self._crew_stack.clear()
        return spans

    def close_pending_spans(self) -> int:
        """Close every span still open (interrupted run). Returns the count."""

        def _close() -> int:
            spans = self._drain()
            for span in spans:
                tracer.end_span(span, error="Span never completed", error_type="Incomplete")
            return len(spans)

        return int(self._run(_close))

    @property
    def pending(self) -> int:
        """Number of spans currently open (diagnostics / tests)."""
        with self._lock:
            return (
                sum(len(s) for s in self._tool_stacks.values())
                + len(self._llm_by_call)
                + len(self._agent_spans)
                + len(self._task_spans)
                + len(self._crew_stack)
            )
