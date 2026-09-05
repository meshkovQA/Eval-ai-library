# eval_lib/tracing/trace_utils.py
"""Utilities for converting trace data to EvalTestCase fields.

This module bridges the tracing system with the evaluation pipeline,
allowing metrics to consume execution trace data collected from
LangChain, Claude Agent SDK, OpenAI Assistants, or manual tracing.
"""

from typing import List, Dict, Any, Optional
from .types import TraceSpan, SpanType
from .tracer import tracer
from .config import TracingConfig
from .usage import span_token_usage


def safe_str(obj: Any, max_length: Optional[int] = None) -> Optional[str]:
    """Convert a value to a string for storage in a trace field.

    The tracing SDK must not silently drop data, so by default the value is
    preserved in full. A length cap is applied only when one is configured —
    passed explicitly via ``max_length`` or through the
    ``TRACING_MAX_FIELD_LENGTH`` env var (see
    :meth:`TracingConfig.get_max_field_length`). When a cap does apply, the
    truncation is made explicit with a suffix so it is never silent.

    Args:
        obj: Value to stringify (``None`` passes through as ``None``).
        max_length: Optional per-call cap; overrides the global config.

    Returns:
        The full string, a truncated string with a ``…[truncated N chars]``
        marker, or ``None`` when ``obj`` is ``None``.
    """
    if obj is None:
        return None
    s = obj if isinstance(obj, str) else str(obj)
    limit = max_length if max_length is not None else TracingConfig.get_max_field_length()
    if limit and limit > 0 and len(s) > limit:
        return s[:limit] + f"…[truncated {len(s) - limit} chars]"
    return s


def spans_to_trace_steps(spans: List[TraceSpan]) -> List[Dict[str, Any]]:
    """Convert TraceSpan list to TraceStep-compatible dicts.

    Maps the internal TraceSpan format to the TraceStep schema
    used in EvalTestCase.execution_trace.
    """
    steps = []
    for span in sorted(spans, key=lambda s: s.start_time or 0):
        step = {
            "step_id": span.span_id,
            "type": span.span_type.value if span.span_type else "custom",
            "name": span.name,
            "input": span.input,
            "output": span.output,
            "duration_ms": span.duration_ms,
            "timestamp": span.start_time,
            "status": span.status,
            "error": span.error,
            "error_type": span.error_type,
            "parent_step_id": span.parent_span_id,
            "metadata": span.metadata if span.metadata else None,
        }
        steps.append(step)
    return steps


def extract_tools_called(spans: List[TraceSpan]) -> List[str]:
    """Extract tool names from TOOL_CALL spans."""
    tool_span_ids = {
        s.span_id for s in spans
        if s.span_type == SpanType.TOOL_CALL
    }
    tools = []
    for span in spans:
        if span.span_type == SpanType.TOOL_CALL:
            # Only top-level tool calls (parent is not a tool call)
            if span.parent_span_id not in tool_span_ids:
                tools.append(span.name)
    return tools


def extract_reasoning(spans: List[TraceSpan]) -> Optional[str]:
    """Extract reasoning text from REASONING and AGENT_STEP spans."""
    reasoning_parts = []
    for span in sorted(spans, key=lambda s: s.start_time or 0):
        if span.span_type in (SpanType.REASONING, SpanType.AGENT_STEP):
            if span.output and isinstance(span.output, str):
                reasoning_parts.append(span.output)
            elif span.output and isinstance(span.output, dict):
                text = span.output.get("text") or span.output.get("log", "")
                if text:
                    reasoning_parts.append(str(text))
    return "\n".join(reasoning_parts) if reasoning_parts else None


def extract_resource_usage(
    spans: List[TraceSpan],
    trace_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Extract resource usage from LLM_CALL spans and trace metadata.

    Token counts come from every usage shape :mod:`.usage` understands
    (LangChain ``llm_output``, OpenAI ``usage``, Anthropic, LlamaIndex),
    not only LangChain's — the previous version returned ``None`` for an
    OpenAI-shaped span. Declared trace-level totals still take precedence.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    total_duration_ms = 0.0
    model = None

    for span in spans:
        if span.duration_ms:
            total_duration_ms += span.duration_ms

        if span.span_type == SpanType.LLM_CALL:
            usage = span_token_usage(span)
            if usage:
                total_input_tokens += usage["input_tokens"]
                total_output_tokens += usage["output_tokens"]
                total_cached_tokens += usage["cached_tokens"]
                total_reasoning_tokens += usage["reasoning_tokens"]
            if model is None:
                span_model = (span.metadata or {}).get("model")
                if span_model:
                    model = str(span_model)

    # Prefer trace-level metadata if available
    if trace_metadata:
        if trace_metadata.get("input_tokens"):
            total_input_tokens = trace_metadata["input_tokens"]
        if trace_metadata.get("output_tokens"):
            total_output_tokens = trace_metadata["output_tokens"]
        if trace_metadata.get("cached_tokens"):
            total_cached_tokens = trace_metadata["cached_tokens"]
        if trace_metadata.get("reasoning_tokens"):
            total_reasoning_tokens = trace_metadata["reasoning_tokens"]
        if trace_metadata.get("model"):
            model = trace_metadata["model"]

    return {
        "input_tokens": total_input_tokens or None,
        "output_tokens": total_output_tokens or None,
        "total_tokens": (total_input_tokens + total_output_tokens) or None,
        "cached_tokens": total_cached_tokens or None,
        "reasoning_tokens": total_reasoning_tokens or None,
        "duration_ms": round(total_duration_ms, 2) if total_duration_ms else None,
        "model": model,
    }


def extract_planning_steps(spans: List[TraceSpan]) -> Optional[List[str]]:
    """Extract planning steps from REASONING spans."""
    steps = []
    for span in sorted(spans, key=lambda s: s.start_time or 0):
        if span.span_type == SpanType.REASONING and span.output:
            if isinstance(span.output, str):
                steps.append(span.output)
            elif isinstance(span.output, dict):
                text = span.output.get("text") or span.output.get("plan", "")
                if text:
                    steps.append(str(text))
    return steps if steps else None


def extract_test_case_data(trace_id: str) -> Dict[str, Any]:
    """Extract all EvalTestCase-compatible fields from a trace.

    Call this after agent execution but before tracer.end_trace()
    to capture trace data for evaluation.

    Returns a dict with keys matching EvalTestCase optional fields:
    - tools_called
    - execution_trace
    - resource_usage
    - reasoning
    - planning_steps
    """
    if not tracer.enabled or not tracer.sender:
        return {}

    spans = tracer.sender.get_trace(trace_id)
    if not spans:
        return {}

    trace_meta = tracer.sender.get_trace_metadata(trace_id)

    # Accumulated add_trace_usage() totals fill any counter the caller did
    # not declare, so this helper agrees with the shipped payload (declared
    # > accumulated > span roll-up — the same precedence as the sender).
    accumulated = tracer.sender.get_trace_usage(trace_id)
    if accumulated.get("llm_calls"):
        trace_meta = dict(trace_meta)
        for key in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens"):
            if trace_meta.get(key) is None and accumulated.get(key):
                trace_meta[key] = accumulated[key]
        if trace_meta.get("cost_usd") is None and accumulated.get("cost_usd"):
            trace_meta["cost_usd"] = accumulated["cost_usd"]

    result = {}

    tools = extract_tools_called(spans)
    if tools:
        result["tools_called"] = tools

    trace_steps = spans_to_trace_steps(spans)
    if trace_steps:
        result["execution_trace"] = trace_steps

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
