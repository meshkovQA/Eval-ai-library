# eval_lib/tracing/haystack_callback.py
"""Haystack (deepset) trace collector via Tracer interface.

Implements Haystack's Tracer interface to capture pipeline
execution as eval-lib TraceSpans.

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

from typing import Any, Dict, Iterator, Optional
from contextlib import contextmanager
from .types import TraceSpan, SpanType
from .tracer import tracer


class EvalLibHaystackSpan:
    """A span object compatible with Haystack's Span interface."""

    def __init__(self, trace_span: TraceSpan):
        self._trace_span = trace_span
        self._content_tracing = True

    def set_tag(self, key: str, value: Any) -> None:
        if self._trace_span and self._trace_span.metadata is not None:
            self._trace_span.metadata[key] = value

    def set_content_tag(self, key: str, value: Any) -> None:
        """Set content tag (inputs/outputs) — only if content tracing enabled."""
        if not self._content_tracing:
            return

        if self._trace_span:
            if "input" in key.lower() and self._trace_span.input is None:
                self._trace_span.input = value
            elif "output" in key.lower():
                self._trace_span.output = value

    def raw_span(self) -> Optional[TraceSpan]:
        return self._trace_span


class EvalLibHaystackTracer:
    """Haystack Tracer that creates eval-lib TraceSpans.

    Implements the haystack.tracing.Tracer interface.
    """

    def __init__(self, content_tracing: bool = True):
        self.is_content_tracing_enabled = content_tracing

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span: Any = None,
    ) -> Iterator[EvalLibHaystackSpan]:
        """Create a trace span for a Haystack operation."""
        # Classify span type from operation name
        span_type = _classify_operation(operation_name)

        span = tracer.start_span(
            name=operation_name,
            span_type=span_type,
            metadata=dict(tags) if tags else None,
        )

        if span and parent_span and isinstance(parent_span, EvalLibHaystackSpan):
            parent_trace_span = parent_span.raw_span()
            if parent_trace_span:
                span.parent_span_id = parent_trace_span.span_id

        hs_span = EvalLibHaystackSpan(span)
        hs_span._content_tracing = self.is_content_tracing_enabled

        try:
            yield hs_span
            if span:
                tracer.end_span(span, output=span.output)
        except Exception as e:
            if span:
                tracer.end_span(span, error=e)
            raise

    def current_span(self) -> Optional[EvalLibHaystackSpan]:
        return None


def _classify_operation(name: str) -> SpanType:
    """Classify a Haystack operation name into a SpanType."""
    name_lower = name.lower()

    if any(kw in name_lower for kw in ("generator", "llm", "chat", "prompt")):
        return SpanType.LLM_CALL
    elif any(kw in name_lower for kw in ("retriever", "search", "embed")):
        return SpanType.RETRIEVAL
    elif any(kw in name_lower for kw in ("tool", "function", "converter")):
        return SpanType.TOOL_CALL
    elif "pipeline" in name_lower:
        return SpanType.AGENT_STEP

    return SpanType.CUSTOM


def install_haystack_tracing(content_tracing: bool = True):
    """Install eval-lib tracer as Haystack's global tracer.

    Args:
        content_tracing: If True, capture component inputs/outputs.
    """
    try:
        import haystack.tracing
        haystack_tracer = EvalLibHaystackTracer(content_tracing=content_tracing)
        haystack.tracing.enable_tracing(haystack_tracer)
    except ImportError:
        raise ImportError(
            "Haystack is required. Install with: pip install haystack-ai"
        )
