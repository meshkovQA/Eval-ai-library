# eval_lib/tracing/phidata_callback.py
"""Phidata/Agno trace collector.

Extracts trace data from Phidata (now Agno) RunResponse events
and converts them into eval-lib TraceSpans.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.phidata_callback import PhidataTraceCollector

    collector = PhidataTraceCollector()
    trace_id = tracer.start_trace("phidata_agent")

    agent = Agent(model=model, tools=[...])
    response = agent.run("query")

    # Extract trace from response
    collector.process_response(response)

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

from typing import Any, List, Optional
from .types import SpanType
from .tracer import tracer


class PhidataTraceCollector:
    """Collects trace data from Phidata/Agno agent responses.

    Processes RunResponse objects and extracts tool calls,
    messages, and token usage.
    """

    def process_response(self, response: Any):
        """Process a Phidata RunResponse and create TraceSpans.

        Args:
            response: A phidata/agno RunResponse object or dict.
        """
        # Extract messages from response
        messages = _safe_get(response, "messages", None)
        if messages and isinstance(messages, list):
            self._process_messages(messages)

        # Extract tool calls if available directly
        tool_calls = _safe_get(response, "tool_calls", None)
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                self._process_tool_call(tc)

        # Extract token usage
        metrics = _safe_get(response, "metrics", None) or _safe_get(response, "meta", None)
        if metrics:
            input_tokens = _safe_get(metrics, "input_tokens", None) or _safe_get(metrics, "prompt_tokens", None)
            output_tokens = _safe_get(metrics, "output_tokens", None) or _safe_get(metrics, "completion_tokens", None)
            model = _safe_get(response, "model", None)
            if input_tokens or output_tokens:
                tracer.set_trace_metadata(
                    model=model,
                    input_tokens=int(input_tokens) if input_tokens else None,
                    output_tokens=int(output_tokens) if output_tokens else None,
                )

    def _process_messages(self, messages: List[Any]):
        """Process message history to extract tool calls and reasoning."""
        for msg in messages:
            role = _safe_get(msg, "role", "")
            content = _safe_get(msg, "content", "")

            if role == "assistant":
                # Check for tool calls in message
                msg_tool_calls = _safe_get(msg, "tool_calls", None)
                if msg_tool_calls:
                    for tc in msg_tool_calls:
                        self._process_tool_call(tc)
                elif content:
                    span = tracer.start_span(
                        name="assistant_response",
                        span_type=SpanType.LLM_CALL,
                    )
                    if span:
                        tracer.end_span(span, output=str(content)[:1000])

            elif role == "tool":
                # Tool result — try to match with pending tool span
                tool_name = _safe_get(msg, "tool_name", None) or _safe_get(msg, "name", "tool_result")
                span = tracer.start_span(
                    name=f"tool_result:{tool_name}",
                    span_type=SpanType.TOOL_CALL,
                )
                if span:
                    tracer.end_span(span, output=str(content)[:1000] if content else None)

    def _process_tool_call(self, tool_call: Any):
        """Process a single tool call."""
        func = _safe_get(tool_call, "function", tool_call)
        name = _safe_get(func, "name", "unknown_tool")
        arguments = _safe_get(func, "arguments", None)

        span = tracer.start_span(
            name=str(name),
            span_type=SpanType.TOOL_CALL,
            input_data=arguments,
        )
        if span:
            tracer.end_span(span)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
