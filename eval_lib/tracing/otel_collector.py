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
    from eval_lib.tracing.trace_utils import extract_test_case_data
    trace_id = exporter.get_latest_trace_id()
    data = extract_test_case_data(trace_id)
"""

from typing import Sequence, Dict, List, Optional, Any
from .types import TraceSpan, SpanType
from .tracer import tracer


# GenAI semantic convention attribute names
# See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
_GENAI_SYSTEM = "gen_ai.system"
_GENAI_MODEL = "gen_ai.request.model"
_GENAI_PROMPT_TOKENS = "gen_ai.usage.input_tokens"
_GENAI_COMPLETION_TOKENS = "gen_ai.usage.output_tokens"
_GENAI_OPERATION = "gen_ai.operation.name"
_TOOL_NAME = "tool.name"
_TOOL_DESCRIPTION = "tool.description"


def _classify_span_type(otel_span: Any) -> SpanType:
    """Classify an OTEL span into a SpanType based on attributes and name."""
    attrs = dict(otel_span.attributes) if otel_span.attributes else {}
    name = otel_span.name or ""
    name_lower = name.lower()

    # Check GenAI semantic conventions
    operation = attrs.get(_GENAI_OPERATION, "")
    if operation in ("chat", "text_completion", "embeddings"):
        return SpanType.LLM_CALL

    # Check for tool-related attributes or name patterns
    if _TOOL_NAME in attrs or "tool" in name_lower:
        if "search" in name_lower or "retriev" in name_lower or "file_search" in name_lower:
            return SpanType.RETRIEVAL
        return SpanType.TOOL_CALL

    # Check for LLM-related name patterns
    if any(kw in name_lower for kw in ("llm", "chat", "generate", "completion", "predict")):
        return SpanType.LLM_CALL

    # Check for reasoning patterns
    if any(kw in name_lower for kw in ("reason", "think", "plan", "reflect")):
        return SpanType.REASONING

    # Check for agent step patterns
    if any(kw in name_lower for kw in ("agent", "step", "execute", "run", "invoke")):
        return SpanType.AGENT_STEP

    # Check for retrieval patterns
    if any(kw in name_lower for kw in ("retriev", "search", "query", "embed")):
        return SpanType.RETRIEVAL

    # Check if it has GenAI model attribute → likely LLM call
    if _GENAI_MODEL in attrs or _GENAI_SYSTEM in attrs:
        return SpanType.LLM_CALL

    return SpanType.CUSTOM


def _extract_input_output(otel_span: Any) -> tuple:
    """Extract input/output data from OTEL span attributes and events."""
    attrs = dict(otel_span.attributes) if otel_span.attributes else {}
    input_data = None
    output_data = None

    # Try standard GenAI attributes
    for key in ("gen_ai.prompt", "input", "tool.parameters", "tool.input"):
        if key in attrs:
            input_data = attrs[key]
            break

    for key in ("gen_ai.completion", "output", "tool.output", "tool.result"):
        if key in attrs:
            output_data = attrs[key]
            break

    # Try events (OTEL events can contain prompt/completion data)
    if otel_span.events:
        for event in otel_span.events:
            event_attrs = dict(event.attributes) if event.attributes else {}
            if "prompt" in event.name.lower() and not input_data:
                input_data = event_attrs.get("gen_ai.prompt", str(event_attrs))
            if "completion" in event.name.lower() and not output_data:
                output_data = event_attrs.get("gen_ai.completion", str(event_attrs))

    return input_data, output_data


class EvalLibSpanExporter:
    """OpenTelemetry SpanExporter that converts OTEL spans to eval-lib TraceSpans.

    Implements the OTEL SpanExporter interface so it can be plugged into
    any TracerProvider via SimpleSpanProcessor or BatchSpanProcessor.
    """

    def __init__(self):
        self._trace_ids: List[str] = []

    def export(self, spans: Sequence[Any]) -> int:
        """Export OTEL spans by converting them to eval-lib TraceSpans.

        Args:
            spans: Sequence of opentelemetry.sdk.trace.ReadableSpan objects.

        Returns:
            SpanExportResult.SUCCESS (0) always.
        """
        for otel_span in spans:
            self._process_span(otel_span)
        return 0  # SUCCESS

    def shutdown(self):
        """Called when the SDK is shutting down."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _process_span(self, otel_span: Any):
        """Convert a single OTEL span to an eval-lib TraceSpan."""
        # Get trace context
        ctx = otel_span.context
        if not ctx:
            return

        trace_id_hex = format(ctx.trace_id, '032x')
        span_id_hex = format(ctx.span_id, '016x')
        parent_id_hex = None
        if otel_span.parent and otel_span.parent.span_id:
            parent_id_hex = format(otel_span.parent.span_id, '016x')

        # Start trace if new trace_id
        if trace_id_hex not in self._trace_ids:
            self._trace_ids.append(trace_id_hex)
            tracer.start_trace(f"otel_{trace_id_hex[:8]}")

        # Classify span type
        span_type = _classify_span_type(otel_span)

        # Extract attributes
        attrs = dict(otel_span.attributes) if otel_span.attributes else {}

        # Extract input/output
        input_data, output_data = _extract_input_output(otel_span)

        # Calculate timing
        start_ns = otel_span.start_time or 0
        end_ns = otel_span.end_time or 0
        start_time = start_ns / 1e9 if start_ns else None
        duration_ms = (end_ns - start_ns) / 1e6 if end_ns and start_ns else None

        # Determine tool name
        name = attrs.get(_TOOL_NAME, otel_span.name or "unknown")

        # Create eval-lib TraceSpan
        span = tracer.start_span(
            name=name,
            span_type=span_type,
            input_data=input_data,
            metadata={
                "otel_span_id": span_id_hex,
                "otel_trace_id": trace_id_hex,
                **{k: v for k, v in attrs.items()
                   if k not in (_TOOL_NAME, "gen_ai.prompt", "gen_ai.completion")},
            },
        )

        if not span:
            return

        # Override IDs to match OTEL hierarchy
        span.span_id = span_id_hex
        span.parent_span_id = parent_id_hex
        if start_time:
            span.start_time = start_time

        # Check for errors
        status = otel_span.status
        if status and hasattr(status, 'status_code'):
            # StatusCode.ERROR = 2
            if status.status_code == 2:
                tracer.end_span(span, error=Exception(
                    status.description or "OTEL span error"
                ))
                return

        tracer.end_span(span, output=output_data)

        # Override timing if OTEL provides it
        if duration_ms:
            span.duration_ms = round(duration_ms, 2)
        if start_time:
            span.start_time = start_time
        if end_ns:
            span.end_time = end_ns / 1e9

        # Set trace-level metadata from GenAI attributes
        model = attrs.get(_GENAI_MODEL)
        input_tokens = attrs.get(_GENAI_PROMPT_TOKENS)
        output_tokens = attrs.get(_GENAI_COMPLETION_TOKENS)

        if model or input_tokens or output_tokens:
            tracer.set_trace_metadata(
                model=model,
                input_tokens=int(input_tokens) if input_tokens else None,
                output_tokens=int(output_tokens) if output_tokens else None,
            )

    def get_latest_trace_id(self) -> Optional[str]:
        """Get the most recent trace ID for extract_test_case_data()."""
        return self._trace_ids[-1] if self._trace_ids else None

    def get_all_trace_ids(self) -> List[str]:
        """Get all trace IDs collected by this exporter."""
        return list(self._trace_ids)

    def clear(self):
        """Clear collected trace IDs."""
        self._trace_ids.clear()
