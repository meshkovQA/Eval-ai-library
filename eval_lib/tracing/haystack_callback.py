# eval_lib/tracing/haystack_callback.py
"""Haystack (deepset) trace collector via Tracer interface.

Implements Haystack's ``haystack.tracing.Tracer`` interface so every
pipeline component, agent step, LLM call and tool call becomes an
eval-lib :class:`TraceSpan`.

What Haystack reports (2.x / 3.x):

* Every pipeline component runs under the *constant* operation name
  ``"haystack.component.run"``; the component's identity lives in the
  tags ``haystack.component.name`` (instance name in the pipeline) and
  ``haystack.component.type`` (class name). Inputs/outputs arrive later
  through ``set_content_tag`` as ``haystack.component.input`` /
  ``haystack.component.output``.
* The pipeline root is ``"haystack.pipeline.run"`` with the user input in
  the ``haystack.pipeline.input_data`` tag and the result in the
  ``haystack.pipeline.output_data`` content tag.
* ``Agent`` emits ``haystack.agent.run`` → ``haystack.agent.step`` →
  ``haystack.agent.step.llm`` / ``haystack.agent.step.tool`` (tool name in
  the ``haystack.tool.name`` tag).
* Generator outputs are ``{"replies": [ChatMessage, ...]}``; token usage
  and the model id sit in ``ChatMessage.meta["usage"]`` / ``meta["model"]``.

Content (inputs/outputs) is only captured when content tracing is on —
either explicitly via ``content_tracing=True`` or, by default, through
Haystack's own ``HAYSTACK_CONTENT_TRACING_ENABLED=true`` env var. Names,
types, token usage and the model id are always captured.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.haystack_callback import install_haystack_tracing

    trace_id = tracer.start_trace("haystack")
    install_haystack_tracing()

    # Run your Haystack pipeline
    result = pipeline.run(data={"query": "..."})

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .context import get_parent_span_id, get_trace_id
from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

logger = logging.getLogger("eval_lib.tracing.haystack")

HAYSTACK_CONTENT_TRACING_ENV = "HAYSTACK_CONTENT_TRACING_ENABLED"

# Tags that carry the component/tool identity — arriving at trace() or later.
_NAME_TAGS = ("haystack.component.name", "haystack.tool.name")
_TYPE_TAG = "haystack.component.type"
_IDENTITY_TAGS = _NAME_TAGS + (_TYPE_TAG,)

# Tag keys (or suffixes) whose value is *content* — user data that goes to
# span.input / span.output and is subject to content tracing.
_INPUT_SUFFIXES = (".input", ".input_data")
_OUTPUT_SUFFIXES = (".output", ".output_data")

# Stack of open Haystack spans for the current async task / thread.
# Immutable tuples so a copied context (thread pool, asyncio task) never
# mutates its parent's stack.
_span_stack: ContextVar[Tuple["EvalLibHaystackSpan", ...]] = ContextVar(
    "eval_lib_haystack_span_stack", default=()
)


def _content_tracing_from_env() -> bool:
    return os.getenv(HAYSTACK_CONTENT_TRACING_ENV, "false").lower() == "true"


def _is_input_tag(key: str) -> bool:
    return key == "input" or key.endswith(_INPUT_SUFFIXES)


def _is_output_tag(key: str) -> bool:
    return key == "output" or key.endswith(_OUTPUT_SUFFIXES)


def _is_content_tag(key: str) -> bool:
    return _is_input_tag(key) or _is_output_tag(key)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_OPERATION_TYPES = {
    "haystack.pipeline.run": SpanType.AGENT_STEP,
    "haystack.async_pipeline.run": SpanType.AGENT_STEP,
    "haystack.agent.run": SpanType.AGENT_STEP,
    "haystack.agent.step": SpanType.AGENT_STEP,
    "haystack.agent.step.llm": SpanType.LLM_CALL,
    "haystack.agent.step.tool": SpanType.TOOL_CALL,
    "haystack.chat_generator.run": SpanType.LLM_CALL,
}


def _classify_type_name(type_name: str) -> Optional[SpanType]:
    """Map a Haystack component *class name* to a span type.

    ``None`` when the class name says nothing about the role (e.g.
    ``PromptBuilder``, ``DocumentWriter``) so the caller can fall back.
    """
    if not type_name:
        return None
    lowered = type_name.lower()
    if lowered.endswith("generator") or lowered.endswith("chatgenerator"):
        return SpanType.LLM_CALL
    if lowered.endswith("retriever"):
        return SpanType.RETRIEVAL
    if lowered == "toolinvoker" or "tool" in lowered:
        return SpanType.TOOL_CALL
    if lowered == "agent" or lowered.endswith("agent"):
        return SpanType.AGENT_STEP
    return None


def _classify_operation(name: str) -> SpanType:
    """Fallback classification by operation name (non-Haystack callers).

    Kept for hand-rolled usage of :class:`EvalLibHaystackTracer` where the
    operation name *is* the component (``"OpenAIGenerator"``,
    ``"BM25Retriever"``, ``"Pipeline"``).
    """
    if name in _OPERATION_TYPES:
        return _OPERATION_TYPES[name]

    by_class = _classify_type_name(name)
    if by_class is not None:
        return by_class

    name_lower = name.lower()
    if any(kw in name_lower for kw in ("generator", "llm", "chat", "prompt")):
        return SpanType.LLM_CALL
    if any(kw in name_lower for kw in ("retriever", "search", "embed")):
        return SpanType.RETRIEVAL
    if any(kw in name_lower for kw in ("tool", "function", "converter")):
        return SpanType.TOOL_CALL
    if "pipeline" in name_lower:
        return SpanType.AGENT_STEP
    return SpanType.CUSTOM


def _resolve_identity(operation_name: str, tags: Dict[str, Any]) -> Tuple[str, SpanType]:
    """Resolve ``(span name, span type)`` from the operation name and tags.

    Identity tags win over the (constant) operation name: for pipeline
    components ``operation_name`` is always ``"haystack.component.run"``.
    """
    name = None
    for key in _NAME_TAGS:
        value = tags.get(key)
        if value:
            name = str(value)
            break
    name = name or operation_name

    span_type = _classify_type_name(str(tags.get(_TYPE_TAG) or ""))
    if span_type is None:
        span_type = _classify_operation(operation_name)
    return name, span_type


# ---------------------------------------------------------------------------
# Usage / model extraction from generator outputs
# ---------------------------------------------------------------------------


def _iter_meta_dicts(output: Any) -> Iterator[Dict[str, Any]]:
    """Yield every ``meta`` dict reachable in a generator output.

    Handles ``{"replies": [ChatMessage]}`` (``ChatMessage.meta`` attribute),
    serialized replies (``{"meta": {...}}`` dicts), the legacy
    ``{"replies": [str], "meta": [dict]}`` shape and a bare ``meta`` dict.
    """
    mapping = as_mapping(output)
    if not mapping:
        return

    replies = mapping.get("replies")
    if isinstance(replies, (list, tuple)):
        for reply in replies:
            meta = getattr(reply, "meta", None)
            if meta is None and isinstance(reply, dict):
                meta = reply.get("meta") or reply.get("_meta")
            meta = as_mapping(meta)
            if meta:
                yield meta

    top_meta = mapping.get("meta")
    if isinstance(top_meta, (list, tuple)):
        for item in top_meta:
            item = as_mapping(item)
            if item:
                yield item
    else:
        top_meta = as_mapping(top_meta)
        if top_meta:
            yield top_meta


def _extract_llm_facts(output: Any) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """Return ``(summed usage, model)`` for an LLM component output."""
    totals: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    for meta in _iter_meta_dicts(output):
        usage = usage_from_mapping(meta)
        if usage:
            if totals is None:
                totals = dict(usage)
            else:
                for key, value in usage.items():
                    totals[key] = totals.get(key, 0) + value
        if model is None:
            candidate = meta.get("model")
            if candidate:
                model = str(candidate)
    return totals, model


# ---------------------------------------------------------------------------
# Span wrapper
# ---------------------------------------------------------------------------


class EvalLibHaystackSpan:
    """A span object compatible with Haystack's ``Span`` interface."""

    def __init__(
        self,
        trace_span: Optional[TraceSpan],
        operation_name: str = "",
        content_tracing: bool = True,
    ):
        self._trace_span = trace_span
        self._operation_name = operation_name
        self._content_tracing = content_tracing
        self._tags: Dict[str, Any] = {}

    # -- Haystack Span interface -------------------------------------------

    def set_tag(self, key: str, value: Any) -> None:
        """Set a single tag.

        Identity tags rename/re-type the span; content tags become the
        span's input/output (when content tracing is enabled); everything
        else lands in ``metadata``.
        """
        self._tags[key] = value
        span = self._trace_span
        if span is None:
            return

        if _is_content_tag(key):
            self._apply_content(key, value)
            return

        if span.metadata is None:
            span.metadata = {}
        span.metadata[key] = value

        if key in _IDENTITY_TAGS:
            self._reclassify()

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set multiple tags at once."""
        for key, value in (tags or {}).items():
            self.set_tag(key, value)

    def set_content_tag(self, key: str, value: Any) -> None:
        """Set a content tag (inputs/outputs) — honoured only with content tracing."""
        if not self._content_tracing:
            return
        self._tags[key] = value
        if self._trace_span is None:
            return
        if _is_content_tag(key):
            self._apply_content(key, value)
        else:
            # Unknown content key (e.g. a custom component): keep it, but
            # only because content tracing is on.
            if self._trace_span.metadata is None:
                self._trace_span.metadata = {}
            self._trace_span.metadata[key] = value

    def raw_span(self) -> Optional[TraceSpan]:
        return self._trace_span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        span = self._trace_span
        if span is None:
            return {}
        return {"trace_id": span.trace_id, "span_id": span.span_id}

    # -- helpers -------------------------------------------------------------

    @property
    def span_id(self) -> Optional[str]:
        return self._trace_span.span_id if self._trace_span is not None else None

    def _apply_content(self, key: str, value: Any) -> None:
        if not self._content_tracing:
            return
        span = self._trace_span
        if span is None:
            return
        if _is_input_tag(key):
            if span.input is None:
                span.input = value
        elif _is_output_tag(key):
            span.output = value

    def _reclassify(self) -> None:
        span = self._trace_span
        if span is None:
            return
        name, span_type = _resolve_identity(self._operation_name, self._tags)
        span.name = name
        span.span_type = span_type

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"EvalLibHaystackSpan({self._trace_span!r})"


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class EvalLibHaystackTracer:
    """Haystack Tracer that creates eval-lib TraceSpans.

    Implements the ``haystack.tracing.Tracer`` interface.

    Args:
        content_tracing: Capture component inputs/outputs. ``None`` (the
            default) defers to Haystack's ``HAYSTACK_CONTENT_TRACING_ENABLED``
            env var; pass ``True``/``False`` to force it either way.
    """

    def __init__(self, content_tracing: Optional[bool] = None):
        if content_tracing is None:
            content_tracing = _content_tracing_from_env()
        self.is_content_tracing_enabled = bool(content_tracing)
        # trace_id for which the model has already been declared.
        self._model_declared_for: Optional[str] = None

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span: Any = None,
    ) -> Iterator[EvalLibHaystackSpan]:
        """Create a trace span for a Haystack operation."""
        tags = dict(tags) if tags else {}
        name, span_type = _resolve_identity(operation_name, tags)

        # Split tags into metadata vs. content (input/output).
        metadata: Dict[str, Any] = {
            key: value for key, value in tags.items() if not _is_content_tag(key)
        }

        parent_span_id = self._resolve_parent_id(parent_span)

        span = tracer.start_span(
            name=name,
            span_type=span_type,
            metadata=metadata,
            parent_span_id=parent_span_id,
            # Tool spans become the context span while the tool executes, so
            # a decorated tool function nests here rather than duplicating
            # the call as a sibling.
            set_current=(span_type == SpanType.TOOL_CALL),
        )

        hs_span = EvalLibHaystackSpan(
            span, operation_name=operation_name, content_tracing=self.is_content_tracing_enabled
        )
        # Content-carrying tags from trace() (e.g. haystack.pipeline.input_data).
        for key, value in tags.items():
            hs_span._tags[key] = value
            if _is_content_tag(key):
                hs_span._apply_content(key, value)

        # The outermost Haystack span (pipeline root, or a standalone
        # Agent/component run) defines the trace-level input/output.
        # Nested pipelines must not overwrite it.
        is_outermost = not _span_stack.get()

        token = _span_stack.set(_span_stack.get() + (hs_span,))
        try:
            yield hs_span
        except BaseException as e:
            # BaseException: asyncio.CancelledError / KeyboardInterrupt must
            # close the span too, or it stays open and the stack is stuck.
            self._pop(token)
            if span is not None:
                self._finalize_llm_facts(span)
                if is_outermost:
                    tracer.set_trace_metadata(input=span.input)
                tracer.end_span(span, output=span.output, error=e)
            raise
        else:
            self._pop(token)
            if span is not None:
                self._finalize_llm_facts(span)
                if is_outermost:
                    tracer.set_trace_metadata(input=span.input, output=span.output)
                tracer.end_span(span, output=span.output)

    def current_span(self) -> Optional[EvalLibHaystackSpan]:
        """The innermost span open in this task/thread, or ``None``."""
        stack = _span_stack.get()
        return stack[-1] if stack else None

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _pop(token) -> None:
        try:
            _span_stack.reset(token)
        except (ValueError, RuntimeError):
            # Token created in another context (generator finalized
            # elsewhere) — drop the top entry instead.
            stack = _span_stack.get()
            if stack:
                _span_stack.set(stack[:-1])

    def _resolve_parent_id(self, parent_span: Any) -> Optional[str]:
        """Explicit parent from Haystack, else the innermost open Haystack
        span, else whatever eval-lib span is current (a user's outer
        ``tracer.trace`` / ``@trace_step``)."""
        if parent_span is not None:
            if isinstance(parent_span, EvalLibHaystackSpan):
                return parent_span.span_id
            raw = getattr(parent_span, "raw_span", None)
            raw = raw() if callable(raw) else raw
            if isinstance(raw, TraceSpan):
                return raw.span_id
            span_id = getattr(parent_span, "span_id", None)
            if isinstance(span_id, str):
                return span_id
            # A foreign span object we cannot link to — treat as no parent
            # and fall back to our own stack.
        current = self.current_span()
        if current is not None and current.span_id is not None:
            return current.span_id
        return get_parent_span_id()

    def _finalize_llm_facts(self, span: TraceSpan) -> None:
        """On LLM span exit: pull token usage + model out of the replies."""
        if span.span_type != SpanType.LLM_CALL:
            return
        try:
            usage, model = _extract_llm_facts(span.output)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("eval_lib.tracing: haystack usage extraction failed: %r", e)
            return

        # Some versions also surface these as plain tags.
        if usage is None:
            for key, value in list(span.metadata.items()):
                if key.endswith(".usage") or key == "usage":
                    usage = usage_from_mapping(as_mapping(value))
                    if usage:
                        break
        if model is None:
            for key in ("haystack.llm.model", "model", "haystack.component.model"):
                value = span.metadata.get(key)
                if value:
                    model = str(value)
                    break

        if usage:
            span.metadata["usage"] = dict(usage)
            tracer.add_trace_usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cached_tokens=usage.get("cached_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                calls=1,
            )
        if model:
            span.metadata["model"] = model
            trace_id = get_trace_id()
            if trace_id and self._model_declared_for != trace_id:
                self._model_declared_for = trace_id
                tracer.set_trace_metadata(model=model)


def install_haystack_tracing(content_tracing: Optional[bool] = None) -> EvalLibHaystackTracer:
    """Install eval-lib tracer as Haystack's global tracer.

    Args:
        content_tracing: If True, capture component inputs/outputs. ``None``
            (default) follows ``HAYSTACK_CONTENT_TRACING_ENABLED``.

    Returns:
        The installed :class:`EvalLibHaystackTracer`.
    """
    try:
        import haystack.tracing
    except ImportError:
        raise ImportError("Haystack is required. Install with: pip install haystack-ai")
    haystack_tracer = EvalLibHaystackTracer(content_tracing=content_tracing)
    haystack.tracing.enable_tracing(haystack_tracer)
    return haystack_tracer
