# eval_lib/tracing/otel_collector.py
"""Universal OpenTelemetry Span Collector for eval-ai-library.

Converts OpenTelemetry spans from ANY agent framework into TraceSpan
objects compatible with eval-lib's reliability metrics.

Works with: CrewAI, AutoGen, Semantic Kernel, Haystack, LlamaIndex,
Smolagents, Phidata/Agno, Mastra, and any OTEL-instrumented framework.

Usage:
    from eval_lib.tracing.otel_collector import EvalLibSpanExporter

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Create exporter and wire into OTEL
    exporter = EvalLibSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Pass provider to your framework:
    # AutoGen: SingleThreadedAgentRuntime(tracer_provider=provider)
    # Mastra: telemetry config
    # Smolagents: SmolagentsInstrumentor().instrument(tracer_provider=provider)
    # etc.

    # After agent runs, extract data for evaluation:
    trace_id = exporter.get_latest_trace_id()
    data = exporter.extract_test_case_data(trace_id)

Design notes
------------
``export()`` is invoked by the OTel SDK — with ``BatchSpanProcessor`` on a
background worker thread that has **no** eval-lib trace context. The
exporter therefore never touches the contextvar-based
``tracer.start_trace`` / ``start_span`` / ``end_span`` API. Every span
carries its own identity:

* the OTel trace id (32 hex chars) **is** the eval-lib ``trace_id``;
* the OTel span id (16 hex chars) **is** the eval-lib ``span_id``, and the
  OTel parent id is the ``parent_span_id`` — so parents resolve regardless
  of export order and interleaved traces never mix.

Spans go straight to ``tracer.sender`` (``add_span`` / ``add_trace_usage``
/ ``set_trace_metadata``) keyed by that explicit id.

A trace is shipped (``sender.flush_trace``) when its **root** span — the
one without a parent — is exported, because in OTel the root ends last.
Pass ``auto_flush=False`` to ship manually via :meth:`flush_trace` /
:meth:`end_trace`. Flushing pops the trace out of the sender buffer, so
the exporter keeps a bounded snapshot of every finished trace: use
:meth:`extract_test_case_data` / :meth:`get_trace` on the exporter and it
works whether or not the trace was already flushed.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .trace_utils import (
    extract_planning_steps,
    extract_reasoning,
    extract_resource_usage,
    extract_tools_called,
    spans_to_trace_steps,
)
from .tracer import tracer
from .types import SpanType, TraceSpan

try:  # The OTel SDK is optional — tests and slim installs use fakes.
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporterBase
    from opentelemetry.sdk.trace.export import SpanExportResult as _SpanExportResult
except ImportError:  # pragma: no cover - exercised when the SDK is absent
    _SpanExporterBase = object  # type: ignore[assignment,misc]
    _SpanExportResult = None  # type: ignore[assignment,misc]

_EXPORT_SUCCESS: Any = _SpanExportResult.SUCCESS if _SpanExportResult is not None else 0

logger = logging.getLogger("eval_lib.tracing")


# ---------------------------------------------------------------------------
# GenAI semantic conventions
# See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
# Both the current names and the pre-1.27 ("legacy") names are read, since
# instrumentations in the wild emit either.
# ---------------------------------------------------------------------------
_GENAI_OPERATION = "gen_ai.operation.name"
_GENAI_SYSTEM_KEYS = ("gen_ai.system", "gen_ai.provider.name")
_GENAI_MODEL_KEYS = ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name")
_GENAI_TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name")
_GENAI_TOOL_CALL_ID = "gen_ai.tool.call.id"
_GENAI_CONVERSATION_ID = "gen_ai.conversation.id"

_INPUT_TOKEN_KEYS = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.token_count.prompt",
)
_OUTPUT_TOKEN_KEYS = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
)
_CACHED_TOKEN_KEYS = ("gen_ai.usage.cache_read_input_tokens", "gen_ai.usage.cached_tokens")
_REASONING_TOKEN_KEYS = ("gen_ai.usage.reasoning_tokens",)

_INPUT_ATTR_KEYS = (
    "gen_ai.input.messages",
    "gen_ai.prompt",
    "gen_ai.tool.call.arguments",
    "input.value",
    "input",
    "tool.parameters",
    "tool.input",
)
_OUTPUT_ATTR_KEYS = (
    "gen_ai.output.messages",
    "gen_ai.completion",
    "gen_ai.tool.call.result",
    "output.value",
    "output",
    "tool.output",
    "tool.result",
)

# Event-based content capture (legacy GenAI convention).
_PROMPT_EVENT = "gen_ai.content.prompt"
_COMPLETION_EVENT = "gen_ai.content.completion"
_MESSAGE_EVENTS = (
    "gen_ai.system.message",
    "gen_ai.user.message",
    "gen_ai.assistant.message",
    "gen_ai.tool.message",
)
_CHOICE_EVENT = "gen_ai.choice"
_EXCEPTION_EVENT = "exception"

_LLM_OPERATIONS = ("chat", "text_completion", "generate_content", "embeddings")
_TOOL_OPERATIONS = ("execute_tool",)
_AGENT_OPERATIONS = ("invoke_agent", "create_agent")
_EXECUTE_TOOL_PREFIX = "execute_tool "

# OpenInference (Arize) — used by the smolagents / LlamaIndex / CrewAI
# instrumentors that the module docstring advertises.
_OPENINFERENCE_KIND = "openinference.span.kind"
_OPENINFERENCE_KIND_MAP = {
    "LLM": SpanType.LLM_CALL,
    "EMBEDDING": SpanType.LLM_CALL,
    "TOOL": SpanType.TOOL_CALL,
    "AGENT": SpanType.AGENT_STEP,
    "CHAIN": SpanType.AGENT_STEP,
    "RETRIEVER": SpanType.RETRIEVAL,
    "RERANKER": SpanType.RETRIEVAL,
}

# Keys consumed into first-class span fields — not repeated in metadata.
_CONSUMED_ATTR_KEYS = frozenset(_INPUT_ATTR_KEYS + _OUTPUT_ATTR_KEYS)

_DEFAULT_MAX_TRACES = 1000


# ---------------------------------------------------------------------------
# Small tolerant accessors — an OTel attribute is whatever the
# instrumentation put there; a malformed one must never abort a batch.
# ---------------------------------------------------------------------------


def _attrs_of(obj: Any) -> Dict[str, Any]:
    """``obj.attributes`` as a plain dict (``{}`` when absent/unusable)."""
    raw = getattr(obj, "attributes", None)
    if not raw:
        return {}
    try:
        return dict(raw)
    except Exception:
        return {}


def _first(attrs: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value is not None and value != "":
            return value
    return None


def _to_int(value: Any) -> Optional[int]:
    """Coerce a token count; ``None`` when it isn't a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _first_int(attrs: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        parsed = _to_int(attrs.get(key))
        if parsed is not None:
            return parsed
    return None


def _maybe_json(value: Any) -> Any:
    """Decode a JSON-encoded string attribute (messages are usually shipped
    that way); anything else passes through untouched."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("[", "{"):
            try:
                return json.loads(stripped)
            except ValueError:
                return value
    return value


def _hex_id(value: Any, width: int) -> Optional[str]:
    """OTel ids are ints; ``0`` is the INVALID id. Return lowercase hex."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"OTel id must be an int, got {type(value).__name__}")
    if value == 0:
        return None
    return format(value, f"0{width}x")


def _status_is_error(status: Any) -> bool:
    """True when ``status.status_code`` is ``StatusCode.ERROR``.

    ``StatusCode`` is an ``enum.Enum`` (not ``IntEnum``), so a bare
    ``== 2`` is always False. Compare by name/value and also accept the
    raw int / string forms a fake or another SDK might use.
    """
    code = getattr(status, "status_code", None)
    if code is None:
        return False
    name = getattr(code, "name", None)
    if isinstance(name, str) and name.upper() == "ERROR":
        return True
    value = getattr(code, "value", None)
    if isinstance(value, int) and not isinstance(value, bool) and value == 2:
        return True
    if isinstance(code, int) and not isinstance(code, bool) and code == 2:
        return True
    if isinstance(code, str) and code.upper() == "ERROR":
        return True
    return False


def _status_description(status: Any) -> Optional[str]:
    description = getattr(status, "description", None)
    return description if isinstance(description, str) and description else None


def _events_of(otel_span: Any) -> List[Any]:
    events = getattr(otel_span, "events", None)
    if not events:
        return []
    try:
        return list(events)
    except Exception:
        return []


def _event_name(event: Any) -> str:
    name = getattr(event, "name", "")
    return name if isinstance(name, str) else ""


# ---------------------------------------------------------------------------
# Span interpretation
# ---------------------------------------------------------------------------


def _classify_span_type(otel_span: Any) -> SpanType:
    """Classify an OTEL span into a SpanType.

    Precedence: ``gen_ai.operation.name`` → model attribute → tool
    attribute → OpenInference kind → name heuristics → ``gen_ai.system``.
    A span that names a model is always an LLM call, whatever it is named.
    """
    attrs = _attrs_of(otel_span)
    name = getattr(otel_span, "name", "") or ""
    name_lower = name.lower() if isinstance(name, str) else ""

    operation = attrs.get(_GENAI_OPERATION)
    if isinstance(operation, str):
        op = operation.strip().lower()
        if op in _LLM_OPERATIONS:
            return SpanType.LLM_CALL
        if op in _TOOL_OPERATIONS:
            return SpanType.TOOL_CALL
        if op in _AGENT_OPERATIONS:
            return SpanType.AGENT_STEP

    if _first(attrs, _GENAI_MODEL_KEYS) is not None:
        return SpanType.LLM_CALL

    if _first(attrs, _GENAI_TOOL_NAME_KEYS) is not None:
        return SpanType.TOOL_CALL

    oi_kind = attrs.get(_OPENINFERENCE_KIND)
    if isinstance(oi_kind, str) and oi_kind.upper() in _OPENINFERENCE_KIND_MAP:
        return _OPENINFERENCE_KIND_MAP[oi_kind.upper()]

    # --- name heuristics -------------------------------------------------
    if name_lower.startswith(_EXECUTE_TOOL_PREFIX):
        return SpanType.TOOL_CALL

    if "tool" in name_lower:
        if "search" in name_lower or "retriev" in name_lower:
            return SpanType.RETRIEVAL
        return SpanType.TOOL_CALL

    if any(kw in name_lower for kw in ("llm", "chat", "generate", "completion", "predict")):
        return SpanType.LLM_CALL

    if any(kw in name_lower for kw in ("reason", "think", "plan", "reflect")):
        return SpanType.REASONING

    if any(kw in name_lower for kw in ("agent", "step", "execute", "run", "invoke")):
        return SpanType.AGENT_STEP

    if any(kw in name_lower for kw in ("retriev", "search", "query", "embed")):
        return SpanType.RETRIEVAL

    if _first(attrs, _GENAI_SYSTEM_KEYS) is not None:
        return SpanType.LLM_CALL

    return SpanType.CUSTOM


def _tool_name(otel_span: Any, attrs: Dict[str, Any]) -> Optional[str]:
    """``gen_ai.tool.name`` → ``tool.name`` → span name after ``execute_tool``."""
    explicit = _first(attrs, _GENAI_TOOL_NAME_KEYS)
    if explicit is not None:
        return str(explicit)
    name = getattr(otel_span, "name", None)
    if isinstance(name, str) and name.startswith(_EXECUTE_TOOL_PREFIX):
        rest = name.removeprefix(_EXECUTE_TOOL_PREFIX).strip()
        if rest:
            return rest
    return None


def _extract_input_output(otel_span: Any) -> Tuple[Any, Any]:
    """Input/output from GenAI attributes, then from GenAI content events."""
    attrs = _attrs_of(otel_span)
    input_data = _maybe_json(_first(attrs, _INPUT_ATTR_KEYS))
    output_data = _maybe_json(_first(attrs, _OUTPUT_ATTR_KEYS))

    if input_data is not None and output_data is not None:
        return input_data, output_data

    messages: List[Dict[str, Any]] = []
    choices: List[Any] = []
    for event in _events_of(otel_span):
        ev_name = _event_name(event)
        ev_attrs = _attrs_of(event)
        if ev_name == _PROMPT_EVENT and input_data is None:
            input_data = _maybe_json(ev_attrs.get("gen_ai.prompt", ev_attrs or None))
        elif ev_name == _COMPLETION_EVENT and output_data is None:
            output_data = _maybe_json(ev_attrs.get("gen_ai.completion", ev_attrs or None))
        elif ev_name in _MESSAGE_EVENTS:
            role = ev_name.removeprefix("gen_ai.").removesuffix(".message")
            message = {"role": ev_attrs.get("role", role)}
            for key, value in ev_attrs.items():
                if key in ("role", "gen_ai.system"):
                    continue
                message[key] = _maybe_json(value)
            messages.append(message)
        elif ev_name == _CHOICE_EVENT:
            choices.append({k: _maybe_json(v) for k, v in ev_attrs.items()})

    if input_data is None and messages:
        input_data = messages
    if output_data is None and choices:
        output_data = choices if len(choices) > 1 else choices[0]

    return input_data, output_data


def _extract_error(otel_span: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """``(is_error, message, error_type)`` from status + ``exception`` events.

    The status is authoritative (an exception that was recorded but
    handled leaves the span OK); the exception event supplies the
    message and class name, the status description is the fallback.
    """
    status = getattr(otel_span, "status", None)
    if not _status_is_error(status):
        return False, None, None

    message: Optional[str] = None
    error_type: Optional[str] = None
    for event in _events_of(otel_span):
        if _event_name(event) != _EXCEPTION_EVENT:
            continue
        ev_attrs = _attrs_of(event)
        # Last recorded exception wins — it is the one that escaped.
        ev_message = ev_attrs.get("exception.message")
        ev_type = ev_attrs.get("exception.type")
        if ev_message is not None:
            message = str(ev_message)
        if ev_type is not None:
            error_type = str(ev_type)

    if not message:
        message = _status_description(status) or "OTEL span error"
    return True, message, error_type


def _extract_usage(attrs: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Flat ``{input_tokens, output_tokens, cached_tokens, reasoning_tokens}``
    — the shape :mod:`.usage` recognises — or ``None`` when no counts."""
    input_tokens = _first_int(attrs, _INPUT_TOKEN_KEYS)
    output_tokens = _first_int(attrs, _OUTPUT_TOKEN_KEYS)
    cached_tokens = _first_int(attrs, _CACHED_TOKEN_KEYS)
    reasoning_tokens = _first_int(attrs, _REASONING_TOKEN_KEYS)
    if all(v is None for v in (input_tokens, output_tokens, cached_tokens, reasoning_tokens)):
        return None
    return {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "cached_tokens": cached_tokens or 0,
        "reasoning_tokens": reasoning_tokens or 0,
    }


def _test_case_data(spans: List[TraceSpan], trace_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as :func:`~eval_lib.tracing.trace_utils.extract_test_case_data`,
    built from an explicit span list so it works after the trace was flushed."""
    if not spans:
        return {}
    result: Dict[str, Any] = {}
    tools = extract_tools_called(spans)
    if tools:
        result["tools_called"] = tools
    steps = spans_to_trace_steps(spans)
    if steps:
        result["execution_trace"] = steps
    resource = extract_resource_usage(spans, trace_meta)
    if any(v is not None for v in resource.values()):
        result["resource_usage"] = resource
    reasoning = extract_reasoning(spans)
    if reasoning:
        result["reasoning"] = reasoning
    planning = extract_planning_steps(spans)
    if planning:
        result["planning_steps"] = planning
    return result


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


@dataclass
class _TraceState:
    """What the exporter remembers about one OTel trace."""

    model_declared: bool = False
    session_declared: bool = False
    flushed: bool = False
    # Snapshot taken right before flush_trace() pops the buffer.
    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvalLibSpanExporter(_SpanExporterBase):  # type: ignore[misc,valid-type]
    """OpenTelemetry SpanExporter that converts OTEL spans to eval-lib TraceSpans.

    Implements the OTEL ``SpanExporter`` interface so it can be plugged into
    any ``TracerProvider`` via ``SimpleSpanProcessor`` or
    ``BatchSpanProcessor``. Safe to drive from the batch worker thread: it
    holds no trace context and keys everything by the OTel trace id.

    Args:
        auto_flush: Ship a trace to the sink as soon as its root span (the
            one without a parent) is exported. Root spans end last in
            OTel, so this is the natural end-of-trace signal. Set to
            ``False`` to call :meth:`flush_trace` yourself.
        max_traces: How many trace ids (and finished-trace snapshots) to
            remember; oldest are evicted first.
    """

    def __init__(self, auto_flush: bool = True, max_traces: int = _DEFAULT_MAX_TRACES):
        self.auto_flush = auto_flush
        self._max_traces = max(1, int(max_traces))
        self._lock = threading.Lock()
        # Insertion-ordered (first-seen) with O(1) membership.
        self._traces: "OrderedDict[str, _TraceState]" = OrderedDict()

    # ------------------------------------------------------------ OTel API

    def export(self, spans: Sequence[Any]) -> Any:
        """Convert a batch of ``ReadableSpan`` objects into eval-lib spans.

        Each span is processed in isolation: one malformed span is logged
        and skipped, the rest of the batch still lands.

        Returns:
            ``SpanExportResult.SUCCESS`` when the OTel SDK is importable,
            otherwise ``0``.
        """
        if not tracer.enabled or tracer.sender is None:
            return _EXPORT_SUCCESS

        for otel_span in spans:
            try:
                self._process_span(otel_span)
            except Exception as e:
                logger.warning(
                    "eval_lib.tracing: skipping OTel span %r: %s: %s",
                    getattr(otel_span, "name", "<unnamed>"),
                    type(e).__name__,
                    e,
                )
        return _EXPORT_SUCCESS

    def shutdown(self) -> None:
        """Ship every trace this exporter has seen that is still buffered."""
        if tracer.sender is None:
            return
        with self._lock:
            trace_ids = list(self._traces)
        for trace_id in trace_ids:
            if tracer.sender.has_trace(trace_id):
                self._finish_trace(trace_id)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is buffered in the exporter itself — every span is handed
        to the sender synchronously inside :meth:`export` — so there is
        nothing to push here. Deliberately does **not** end in-progress
        traces: ``provider.force_flush()`` is routinely called mid-run.
        """
        return True

    # ------------------------------------------------------------ public

    def flush_trace(self, trace_id: Optional[str] = None) -> None:
        """Ship a finished OTel trace explicitly (``auto_flush=False`` flows).

        Args:
            trace_id: OTel trace id hex; defaults to the latest one seen.
        """
        target = trace_id or self.get_latest_trace_id()
        if not target or tracer.sender is None:
            return
        self._finish_trace(target)

    def end_trace(self, trace_id: Optional[str] = None) -> None:
        """Alias for :meth:`flush_trace`, mirroring ``tracer.end_trace``."""
        self.flush_trace(trace_id)

    def get_latest_trace_id(self) -> Optional[str]:
        """Most recently *started* trace id (OTel hex) — pass it to
        :meth:`extract_test_case_data`."""
        with self._lock:
            return next(reversed(self._traces)) if self._traces else None

    def get_all_trace_ids(self) -> List[str]:
        """All trace ids collected by this exporter, oldest first."""
        with self._lock:
            return list(self._traces)

    def get_trace(self, trace_id: Optional[str] = None) -> List[TraceSpan]:
        """Spans of a trace — from the live buffer, or from the snapshot
        kept when the trace was flushed."""
        target = trace_id or self.get_latest_trace_id()
        if not target:
            return []
        if tracer.sender is not None:
            live: List[TraceSpan] = tracer.sender.get_trace(target)
            if live:
                return live
        with self._lock:
            state = self._traces.get(target)
            return list(state.spans) if state else []

    def extract_test_case_data(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """``EvalTestCase`` fields for a trace, before **or after** it was
        flushed (unlike ``trace_utils.extract_test_case_data``, which only
        sees the live buffer)."""
        target = trace_id or self.get_latest_trace_id()
        if not target:
            return {}
        spans: List[TraceSpan] = []
        meta: Dict[str, Any] = {}
        if tracer.sender is not None:
            spans = tracer.sender.get_trace(target)
            meta = tracer.sender.get_trace_metadata(target)
        if not spans:
            with self._lock:
                state = self._traces.get(target)
                if state:
                    spans, meta = list(state.spans), dict(state.metadata)
        return _test_case_data(spans, meta)

    def clear(self) -> None:
        """Forget collected trace ids and snapshots."""
        with self._lock:
            self._traces.clear()

    # ------------------------------------------------------------ internals

    def _state_for(self, trace_id: str) -> _TraceState:
        with self._lock:
            state = self._traces.get(trace_id)
            if state is None:
                state = _TraceState()
                self._traces[trace_id] = state
                while len(self._traces) > self._max_traces:
                    self._traces.popitem(last=False)
            return state

    def _finish_trace(self, trace_id: str) -> None:
        sender = tracer.sender
        if sender is None:
            return
        state = self._state_for(trace_id)
        spans = sender.get_trace(trace_id)
        meta = sender.get_trace_metadata(trace_id)
        with self._lock:
            if spans:
                state.spans = spans
            if meta:
                state.metadata = meta
            state.flushed = True
        sender.flush_trace(trace_id)

    def _process_span(self, otel_span: Any) -> None:
        """Convert one OTel span and hand it to the sender."""
        sender = tracer.sender
        ctx = getattr(otel_span, "context", None)
        if ctx is None:
            return
        trace_id = _hex_id(getattr(ctx, "trace_id", 0), 32)
        span_id = _hex_id(getattr(ctx, "span_id", 0), 16)
        if trace_id is None or span_id is None:
            logger.warning(
                "eval_lib.tracing: OTel span %r has an invalid id — skipped",
                getattr(otel_span, "name", "<unnamed>"),
            )
            return

        parent = getattr(otel_span, "parent", None)
        parent_span_id = _hex_id(getattr(parent, "span_id", 0), 16) if parent is not None else None
        # A span whose parent lives in another process is the local root.
        is_root = parent_span_id is None or bool(getattr(parent, "is_remote", False))

        state = self._state_for(trace_id)

        attrs = _attrs_of(otel_span)
        span_type = _classify_span_type(otel_span)
        input_data, output_data = _extract_input_output(otel_span)
        is_error, error_message, error_type = _extract_error(otel_span)

        otel_name = getattr(otel_span, "name", None)
        name = otel_name if isinstance(otel_name, str) and otel_name else "unknown"
        tool_name = _tool_name(otel_span, attrs)
        if span_type == SpanType.TOOL_CALL and tool_name:
            name = tool_name

        metadata: Dict[str, Any] = {
            "otel_span_id": span_id,
            "otel_trace_id": trace_id,
        }
        kind = getattr(otel_span, "kind", None)
        kind_name = getattr(kind, "name", None)
        if isinstance(kind_name, str):
            metadata["otel_kind"] = kind_name
        metadata.update({k: v for k, v in attrs.items() if k not in _CONSUMED_ATTR_KEYS})

        model = _first(attrs, _GENAI_MODEL_KEYS)
        if model is not None:
            metadata["model"] = str(model)
        system = _first(attrs, _GENAI_SYSTEM_KEYS)
        if system is not None:
            metadata["system"] = str(system)
        if tool_name:
            metadata["tool_name"] = tool_name
        tool_call_id = attrs.get(_GENAI_TOOL_CALL_ID)
        if tool_call_id is not None:
            metadata["tool_call_id"] = str(tool_call_id)
        resource_attrs = _attrs_of(getattr(otel_span, "resource", None))
        service_name = resource_attrs.get("service.name")
        if isinstance(service_name, str) and service_name:
            metadata["service_name"] = service_name

        usage = _extract_usage(attrs)
        if usage:
            metadata.update(usage)

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            span_type=span_type,
            name=name,
            input=input_data,
            metadata=metadata,
        )

        # Timing from OTel (nanoseconds since epoch) — set *before* the
        # span is handed to the sender so a streamed partial_span carries
        # the real numbers, not the exporter's wall clock.
        start_ns = _to_int(getattr(otel_span, "start_time", None))
        end_ns = _to_int(getattr(otel_span, "end_time", None))
        if start_ns:
            span.start_time = start_ns / 1e9
        span.finish(
            output=output_data, error=error_message if is_error else None, error_type=error_type
        )
        if end_ns:
            span.end_time = end_ns / 1e9
        if start_ns and end_ns:
            span.duration_ms = round((end_ns - start_ns) / 1e6, 3)

        sender.add_span(span)
        sender.flush_span(span)

        # ---- trace-level facts -------------------------------------------
        if span_type == SpanType.LLM_CALL and usage:
            sender.add_trace_usage(
                trace_id,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
            )

        trace_meta: Dict[str, Any] = {}
        with self._lock:
            if model is not None and not state.model_declared:
                state.model_declared = True
                trace_meta["model"] = str(model)
            conversation_id = attrs.get(_GENAI_CONVERSATION_ID)
            if conversation_id is not None and not state.session_declared:
                state.session_declared = True
                trace_meta["session_id"] = str(conversation_id)
        if trace_meta:
            sender.set_trace_metadata(trace_id, trace_meta)

        if is_root and self.auto_flush:
            self._finish_trace(trace_id)
