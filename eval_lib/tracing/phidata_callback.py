# eval_lib/tracing/phidata_callback.py
"""Phidata / Agno trace collector.

Turns a completed ``RunResponse`` (phidata 2.x) or ``RunOutput`` (agno) into
eval-lib spans. Both shapes are handled by duck-typing — neither SDK is
imported, so the module works with either installed (or with plain dicts).

The collector is exposed under both its historical name
(:class:`PhidataTraceCollector`) and the framework's current name
(:class:`AgnoTraceCollector`). They are the same class.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.phidata_callback import AgnoTraceCollector
    # (or: from eval_lib.tracing.phidata_callback import PhidataTraceCollector)

    collector = AgnoTraceCollector()
    trace_id = tracer.start_trace("agno_agent")

    agent = Agent(model=model, tools=[...])
    response = agent.run("query")
    collector.process_response(response)   # never raises into your code

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()

What one call to :meth:`process_response` records:

* one ``LLM_CALL`` span per assistant message — the text *and* the tool
  calls it issued are both kept — with that message's token usage in the
  span metadata;
* one ``TOOL_CALL`` span per tool invocation, parented under the assistant
  span that issued it, carrying args, result, error flag and duration. When
  the response exposes ``tools`` (phidata: list of dicts; agno: list of
  ``ToolExecution``) that record is preferred because it has everything in
  one place; otherwise the span is assembled from the assistant message's
  ``tool_calls`` and the matching ``role == "tool"`` message (matched by
  ``tool_call_id``, falling back to name in FIFO order);
* token usage through :meth:`AgentTracer.add_trace_usage`, per assistant
  message when the framework reports it there, otherwise once per run from
  the run-level metrics — so several runs in one trace add up instead of
  overwriting each other;
* trace-level facts (``model``, ``input``, ``output``, ``session_id``,
  ``user_id``) through :meth:`AgentTracer.set_trace_metadata`.

Span timestamps come from the frameworks' own timing (phidata
``metrics["time"]``, agno ``duration`` / ``start_time`` / ``end_time``), laid
end-to-end so the trace reflects the run's real durations even though it is
reconstructed after the fact.
"""

import json
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .trace_utils import safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

__all__ = ["PhidataTraceCollector", "AgnoTraceCollector"]

logger = logging.getLogger("eval_lib.tracing")

_INPUT_KEYS = ("input_tokens", "prompt_tokens")
_OUTPUT_KEYS = ("output_tokens", "completion_tokens")
# agno names its cache-hit counter ``cache_read_tokens``; the OpenAI /
# Anthropic spellings are the ones :mod:`.usage` already knows.
_CACHED_KEYS = ("cached_tokens", "cache_read_tokens", "cache_read_input_tokens")
_REASONING_KEYS = ("reasoning_tokens",)


# ---------------------------------------------------------------------------
# Shape-agnostic accessors
# ---------------------------------------------------------------------------


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _role_of(msg: Any) -> str:
    role = _safe_get(msg, "role", "")
    role = getattr(role, "value", role)  # tolerate an Enum role
    return str(role or "").lower()


def _to_number(value: Any) -> Optional[float]:
    """Numeric view of one metric value. Lists are summed; non-numbers give ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        numbers = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return float(sum(numbers)) if numbers else None
    return None


def _metric_raw(metrics: Any, key: str) -> Any:
    if metrics is None:
        return None
    if isinstance(metrics, dict):
        return metrics.get(key)
    return getattr(metrics, key, None)


def _metric_float(metrics: Any, key: str) -> Optional[float]:
    return _to_number(_metric_raw(metrics, key))


def _metric_int(metrics: Any, key: str) -> int:
    """Integer metric that tolerates every shape the two frameworks produce.

    * phidata builds ``RunResponse.metrics`` as ``defaultdict(list)`` — each
      key holds one value *per assistant message* — so a list is summed
      (``int(list)`` is what used to raise ``TypeError`` after every run);
    * agno's ``RunMetrics`` / ``MessageMetrics`` are dataclasses with ``int``
      fields, read by attribute;
    * plain dicts, pydantic models, floats and ``None`` are handled as well.
    """
    value = _to_number(_metric_raw(metrics, key))
    return int(value) if value is not None else 0


def _first_metric_int(metrics: Any, keys: Iterable[str]) -> int:
    for key in keys:
        value = _metric_int(metrics, key)
        if value:
            return value
    return 0


def _usage_from_metrics(metrics: Any) -> Optional[Dict[str, int]]:
    """Token usage from a message- or run-level metrics object.

    Handles phidata's per-message dict (OpenAI-style keys incl. the
    ``prompt_tokens_details`` / ``completion_tokens_details`` sub-dicts),
    phidata's run-level ``defaultdict(list)``, and agno's ``MessageMetrics``
    / ``RunMetrics`` dataclasses. ``None`` when there are no counts.
    """
    if metrics is None:
        return None
    usage = usage_from_mapping(as_mapping(metrics))
    if usage is None:
        # Either not dict-like or the counts are lists (phidata run level).
        input_tokens = _first_metric_int(metrics, _INPUT_KEYS)
        output_tokens = _first_metric_int(metrics, _OUTPUT_KEYS)
        cached = _first_metric_int(metrics, _CACHED_KEYS)
        reasoning = _first_metric_int(metrics, _REASONING_KEYS)
        if not any((input_tokens, output_tokens, cached, reasoning)):
            return None
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached,
            "reasoning_tokens": reasoning,
        }
    if not usage["cached_tokens"]:
        usage["cached_tokens"] = _first_metric_int(metrics, _CACHED_KEYS)
    return usage


def _duration_seconds(metrics: Any) -> Optional[float]:
    """Seconds the framework measured for one message / tool call.

    phidata stores it as ``metrics["time"]``; agno as ``.duration`` or as an
    absolute ``start_time`` / ``end_time`` pair on ``ToolCallMetrics``.
    """
    for key in ("duration", "time"):
        value = _metric_float(metrics, key)
        if value is not None and value >= 0:
            return value
    window = _absolute_window(metrics)
    if window:
        return window[1] - window[0]
    return None


def _absolute_window(metrics: Any) -> Optional[Tuple[float, float]]:
    start = _metric_float(metrics, "start_time")
    end = _metric_float(metrics, "end_time")
    if start is not None and end is not None and 0 < start <= end:
        return start, end
    return None


def _content_to_str(content: Any) -> Optional[str]:
    """Message / run content as text (structured outputs are serialised)."""
    if content is None:
        return None
    if isinstance(content, str):
        return safe_str(content)
    dump = getattr(content, "model_dump_json", None)
    if callable(dump):
        try:
            return safe_str(dump(exclude_none=True))
        except Exception:
            pass
    if isinstance(content, (dict, list)):
        try:
            return safe_str(json.dumps(content, ensure_ascii=False, default=str))
        except Exception:
            pass
    return safe_str(content)


def _parse_args(args: Any) -> Any:
    """Tool arguments as given; a JSON string (OpenAI ``function.arguments``) is decoded."""
    if isinstance(args, str):
        stripped = args.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except ValueError:
                pass
    return args


def _run_input_text(value: Any) -> Optional[str]:
    """Text of agno's ``RunOutput.input`` (a ``RunInput``) or a plain value."""
    if value is None:
        return None
    if isinstance(value, str):
        return safe_str(value)
    as_string = getattr(value, "input_content_string", None)
    if callable(as_string):
        try:
            return safe_str(as_string())
        except Exception:
            pass
    inner = _safe_get(value, "input_content")
    return _content_to_str(inner if inner is not None else value)


# ---------------------------------------------------------------------------
# Tool call bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _ToolRecord:
    """One tool invocation, normalised from any of the shapes we see.

    * ``RunResponse.tools[i]`` (phidata) — dict with ``tool_call_id``,
      ``tool_name``, ``tool_args``, ``content``, ``tool_call_error`` and
      ``metrics={"time": …}``;
    * ``RunOutput.tools[i]`` (agno) — ``ToolExecution`` with ``result``
      instead of ``content`` and ``ToolCallMetrics`` for timing;
    * a ``role == "tool"`` message (both frameworks) — same fields as the
      phidata dict;
    * an OpenAI-style ``tool_calls[i]`` entry (``id`` + ``function``).
    """

    call_id: Optional[str] = None
    name: str = "unknown_tool"
    args: Any = None
    result: Any = None
    error: bool = False
    metrics: Any = None

    @classmethod
    def from_any(cls, obj: Any) -> "_ToolRecord":
        func = _safe_get(obj, "function")
        call_id = _safe_get(obj, "tool_call_id")
        if call_id is None and _safe_get(obj, "role") is None:
            # ``id`` is the call id on a tool_calls entry, but a message id
            # on a Message — only trust it when this is not a message.
            call_id = _safe_get(obj, "id")
        name = _safe_get(obj, "tool_name") or _safe_get(func, "name") or _safe_get(obj, "name")
        args = _safe_get(obj, "tool_args")
        for source, key in ((func, "arguments"), (obj, "arguments"), (obj, "input")):
            if args is not None:
                break
            args = _safe_get(source, key)
        result = _safe_get(obj, "result")
        if result is None:
            result = _safe_get(obj, "content")
        return cls(
            call_id=str(call_id) if call_id is not None else None,
            name=str(name) if name else "unknown_tool",
            args=_parse_args(args),
            result=result,
            error=bool(_safe_get(obj, "tool_call_error") or False),
            metrics=_safe_get(obj, "metrics"),
        )


class _CallQueue:
    """Pair tool results with tool calls: by call id first, else by name in FIFO order."""

    def __init__(self) -> None:
        self._items: List[Tuple[Optional[str], str, Any]] = []
        self._taken: set = set()

    def add(self, call_id: Optional[str], name: str, item: Any) -> None:
        self._items.append((call_id, name, item))

    def take(self, call_id: Optional[str], name: str) -> Any:
        index = None
        if call_id:
            index = self._find(lambda cid, _n: cid == call_id)
        if index is None:
            index = self._find(lambda _cid, n: n == name)
        if index is None:
            return None
        self._taken.add(index)
        return self._items[index][2]

    def remaining(self) -> List[Any]:
        return [item for i, (_, _, item) in enumerate(self._items) if i not in self._taken]

    def _find(self, predicate) -> Optional[int]:
        for i, (cid, name, _) in enumerate(self._items):
            if i not in self._taken and predicate(cid, name):
                return i
        return None


class _Timeline:
    """Sequential layout of spans using the frameworks' own durations.

    :meth:`PhidataTraceCollector.process_response` runs after the agent has
    finished, so the tracer's clock would stamp every span with the same
    instant and a zero duration. Instead the spans are laid end-to-end,
    ending at "now" (when the finished run was handed to us), each as long
    as the framework measured. Absolute timestamps (agno's
    ``ToolCallMetrics.start_time`` / ``end_time``) are used verbatim.
    """

    def __init__(self, total_seconds: float, now: Optional[float] = None) -> None:
        anchor = now if now is not None else time.time()
        self._cursor = anchor - max(total_seconds, 0.0)

    def open(self, span: Optional[TraceSpan]) -> None:
        """Best-effort start stamp before ``end_span`` (streaming mode ships it)."""
        if span is not None:
            span.start_time = self._cursor

    def place(
        self,
        span: Optional[TraceSpan],
        duration: Optional[float] = None,
        window: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Call *after* ``tracer.end_span`` — overrides the tracer's wall clock."""
        if span is None:
            return
        if window:
            start, end = window
        else:
            start = self._cursor
            end = start + max(duration, 0.0) if duration is not None else start
        span.start_time = start
        span.end_time = end
        span.duration_ms = round((end - start) * 1000, 2)
        self._cursor = max(self._cursor, end)


def _total_duration(messages: List[Any], records: List[_ToolRecord], prefer_records: bool) -> float:
    total = 0.0
    for msg in messages:
        role = _role_of(msg)
        if role == "assistant" or (role == "tool" and not prefer_records):
            total += _duration_seconds(_safe_get(msg, "metrics")) or 0.0
    if prefer_records:
        for record in records:
            total += _duration_seconds(record.metrics) or 0.0
    return total


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class PhidataTraceCollector:
    """Collect eval-lib spans from a phidata ``RunResponse`` / agno ``RunOutput``.

    Also available as :data:`AgnoTraceCollector`. Call
    :meth:`process_response` once per completed run, inside an active trace.
    """

    def process_response(self, response: Any) -> None:
        """Record one completed run.

        Args:
            response: A phidata ``RunResponse``, an agno ``RunOutput``, or a
                dict with the same keys.

        Never raises: a malformed response is logged as a warning and
        skipped, so tracing cannot break the agent it observes.
        """
        try:
            self._process(response)
        except Exception as exc:  # tracing must never take the agent down
            logger.warning(
                "eval_lib.tracing: %s could not process the run response: %r",
                type(self).__name__,
                exc,
            )

    # ---------------------------------------------------------------- run

    def _process(self, response: Any) -> None:
        if response is None:
            return

        messages = [
            m
            for m in _as_list(_safe_get(response, "messages"))
            if not _safe_get(m, "from_history", False)  # agno replays memory here
        ]
        records = [_ToolRecord.from_any(t) for t in _as_list(_safe_get(response, "tools"))]
        if not records:
            records = [_ToolRecord.from_any(t) for t in _as_list(_safe_get(response, "tool_calls"))]
        prefer_records = bool(records)

        model = _safe_get(response, "model")
        model = str(model) if model else None

        timeline = _Timeline(_total_duration(messages, records, prefer_records))
        results = _CallQueue()  # records from response.tools, consumed by matching calls
        for record in records:
            results.add(record.call_id, record.name, record)
        pending = _CallQueue()  # spans opened from tool_calls, awaiting a tool message

        llm_spans: List[TraceSpan] = []
        usage_seen = False
        first_user_input: Optional[str] = None

        for msg in messages:
            role = _role_of(msg)
            if role == "assistant":
                span, usage = self._record_assistant(msg, model, timeline)
                if span is not None:
                    llm_spans.append(span)
                usage_seen = usage_seen or usage is not None
                parent_id = span.span_id if span is not None else None
                for raw_call in _as_list(_safe_get(msg, "tool_calls")):
                    call = _ToolRecord.from_any(raw_call)
                    if prefer_records:
                        record = results.take(call.call_id, call.name) or call
                        self._record_tool(record, parent_id, timeline, fallback_args=call.args)
                    else:
                        opened = self._open_tool(call, parent_id, timeline)
                        pending.add(call.call_id, call.name, (call, opened))
            elif role == "tool":
                if prefer_records:
                    continue  # the tools list already carries this result
                result = _ToolRecord.from_any(msg)
                entry = pending.take(result.call_id, result.name)
                if entry is None:
                    self._record_tool(result, None, timeline)
                else:
                    call, opened = entry
                    self._close_tool(opened, result, timeline, fallback_args=call.args)
            elif role == "user" and first_user_input is None:
                first_user_input = _content_to_str(_safe_get(msg, "content"))

        # Tool records no assistant message referenced (or no messages at all).
        for record in results.remaining():
            self._record_tool(record, None, timeline)
        # Calls whose result never arrived (interrupted / paused run).
        for call, opened in pending.remaining():
            self._close_tool(opened, call, timeline, unresolved=True)

        if not usage_seen:
            self._record_run_usage(response, llm_spans)
        self._record_trace_facts(response, model, first_user_input)

    # ---------------------------------------------------------- assistant

    def _record_assistant(
        self, msg: Any, model: Optional[str], timeline: _Timeline
    ) -> Tuple[Optional[TraceSpan], Optional[Dict[str, int]]]:
        metrics = _safe_get(msg, "metrics")
        usage = _usage_from_metrics(metrics)
        duration = _duration_seconds(metrics)
        content = _content_to_str(_safe_get(msg, "content")) or None
        calls = [_ToolRecord.from_any(tc) for tc in _as_list(_safe_get(msg, "tool_calls"))]

        metadata: Dict[str, Any] = {}
        if model:
            metadata["model"] = model
        if usage:
            metadata.update(usage)
        if duration is not None:
            metadata["duration"] = duration
        if calls:
            metadata["tool_calls"] = [
                {"id": c.call_id, "name": c.name, "arguments": c.args} for c in calls
            ]
        reasoning = _safe_get(msg, "reasoning_content")
        if reasoning:
            metadata["reasoning_content"] = safe_str(reasoning)

        span = tracer.start_span(
            name="assistant_response",
            span_type=SpanType.LLM_CALL,
            metadata=metadata,
            set_current=False,
        )
        timeline.open(span)
        # Text and tool calls are both kept: text is the output, the calls
        # live in metadata (and stand in as output when there is no text).
        output: Any = content
        if output is None and calls:
            output = {"tool_calls": metadata["tool_calls"]}
        tracer.end_span(span, output=output)
        timeline.place(span, duration=duration)

        if usage:
            tracer.add_trace_usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                cost_usd=_metric_float(metrics, "cost") or 0.0,
                calls=1,
            )
        return span, usage

    # -------------------------------------------------------------- tools

    def _open_tool(
        self, record: _ToolRecord, parent_id: Optional[str], timeline: _Timeline
    ) -> Optional[TraceSpan]:
        metadata: Dict[str, Any] = {}
        if record.call_id:
            metadata["tool_call_id"] = record.call_id
        # No explicit parent → inherit the caller's context (e.g. a wrapping
        # ``tracer.trace(...)`` block), exactly like a hand-written span.
        kwargs: Dict[str, Any] = {"parent_span_id": parent_id} if parent_id else {}
        span = tracer.start_span(
            name=record.name,
            span_type=SpanType.TOOL_CALL,
            input_data=record.args,
            metadata=metadata,
            set_current=False,
            **kwargs,
        )
        timeline.open(span)
        return span

    def _close_tool(
        self,
        span: Optional[TraceSpan],
        record: _ToolRecord,
        timeline: _Timeline,
        fallback_args: Any = None,
        unresolved: bool = False,
    ) -> None:
        if span is None:
            return
        if span.input is None:
            span.input = record.args if record.args is not None else fallback_args
        if record.call_id and "tool_call_id" not in span.metadata:
            span.metadata["tool_call_id"] = record.call_id
        duration = _duration_seconds(record.metrics)
        window = _absolute_window(record.metrics)
        if duration is not None:
            span.metadata["duration"] = duration
        if unresolved:
            span.metadata["unresolved"] = True

        output = _content_to_str(record.result)
        if record.error:
            tracer.end_span(
                span,
                output=output,
                error=output or "tool_call_error",
                status="error",
                error_type="ToolError",
            )
        else:
            tracer.end_span(span, output=output)
        timeline.place(span, duration=duration, window=window)

    def _record_tool(
        self,
        record: _ToolRecord,
        parent_id: Optional[str],
        timeline: _Timeline,
        fallback_args: Any = None,
    ) -> None:
        if record.args is None and fallback_args is not None:
            record = replace(record, args=fallback_args)
        span = self._open_tool(record, parent_id, timeline)
        self._close_tool(span, record, timeline)

    # ------------------------------------------------------- usage / facts

    def _record_run_usage(self, response: Any, llm_spans: List[TraceSpan]) -> None:
        """Fallback when no assistant message carried its own metrics."""
        metrics = _safe_get(response, "metrics")
        if metrics is None:
            metrics = _safe_get(response, "meta")
        usage = _usage_from_metrics(metrics)
        cost = _metric_float(metrics, "cost")
        if usage is None and not cost:
            return
        usage = usage or {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        tracer.add_trace_usage(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cached_tokens=usage["cached_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            cost_usd=cost or 0.0,
            calls=max(1, len(llm_spans)),
        )
        if llm_spans:
            # Make the run total visible to span-level consumers
            # (``extract_resource_usage``) as well; flagged as run-scoped.
            llm_spans[-1].metadata.update(usage)
            llm_spans[-1].metadata["usage_scope"] = "run"

    def _record_trace_facts(
        self, response: Any, model: Optional[str], first_user_input: Optional[str]
    ) -> None:
        facts: Dict[str, Any] = {}
        if model:
            facts["model"] = model
        session_id = _safe_get(response, "session_id")
        if session_id:
            facts["session_id"] = str(session_id)
        user_id = _safe_get(response, "user_id")
        if user_id:
            facts["user_id"] = str(user_id)
        run_input = _run_input_text(_safe_get(response, "input")) or first_user_input
        if run_input:
            facts["input"] = run_input
        output = _content_to_str(_safe_get(response, "content"))
        if output:
            facts["output"] = output
        if facts:
            tracer.set_trace_metadata(**facts)


#: The framework was renamed phidata → agno; both names refer to the same collector.
AgnoTraceCollector = PhidataTraceCollector
