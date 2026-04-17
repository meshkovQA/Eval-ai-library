# eval_lib/tracing/claude_agent_callback.py
"""Claude Agent SDK trace collector.

Converts Anthropic API response messages into TraceSpan objects,
enabling reliability metrics to analyze tool calls, reasoning steps,
and resource usage from Claude-based agents.

Supports:
- tool_use content blocks → TOOL_CALL spans
- thinking content blocks (extended thinking) → REASONING spans
- text content blocks → LLM_CALL spans
- response.usage → trace-level metadata (tokens)

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.claude_agent_callback import ClaudeAgentTraceCollector

    collector = ClaudeAgentTraceCollector()
    trace_id = tracer.start_trace("claude_agent")

    # Run your Claude agent and collect responses
    response = client.messages.create(...)
    collector.process_response(response)

    # If agent makes tool calls and you get results:
    collector.process_tool_results(tool_results)

    # Extract data for evaluation
    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)

    tracer.end_trace()
"""

from typing import Any, Dict, List, Optional
from .types import TraceSpan, SpanType
from .tracer import tracer


class ClaudeAgentTraceCollector:
    """Collects trace data from Anthropic Claude API responses.

    Processes response content blocks and converts them into TraceSpan
    objects compatible with the eval_lib tracing system.
    """

    def __init__(self):
        self._tool_spans: Dict[str, TraceSpan] = {}  # tool_use_id → span

    def process_response(self, response: Any) -> List[TraceSpan]:
        """Process a Claude API response and create spans.

        Args:
            response: An anthropic.types.Message object or dict with
                      'content', 'usage', 'model', 'stop_reason' fields.

        Returns:
            List of created TraceSpan objects.
        """
        spans = []

        content_blocks = _get_attr_or_key(response, "content", [])
        usage = _get_attr_or_key(response, "usage", None)
        model = _get_attr_or_key(response, "model", None)
        stop_reason = _get_attr_or_key(response, "stop_reason", None)

        for block in content_blocks:
            block_type = _get_attr_or_key(block, "type", "")

            if block_type == "tool_use":
                span = self._process_tool_use(block)
                if span:
                    spans.append(span)

            elif block_type == "thinking":
                span = self._process_thinking(block)
                if span:
                    spans.append(span)

            elif block_type == "text":
                span = self._process_text(block)
                if span:
                    spans.append(span)

        # Set trace-level metadata from usage
        if usage:
            input_tokens = _get_attr_or_key(usage, "input_tokens", None)
            output_tokens = _get_attr_or_key(usage, "output_tokens", None)
            tracer.set_trace_metadata(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens or 0) + (output_tokens or 0),
            )

        return spans

    def process_tool_results(self, tool_results: List[Any]):
        """Process tool results and close the corresponding TOOL_CALL spans.

        Args:
            tool_results: List of tool result content blocks (from the user
                         message following a tool_use response). Each should
                         have 'tool_use_id' and 'content' fields.
        """
        for result in tool_results:
            tool_use_id = _get_attr_or_key(result, "tool_use_id", None)
            if not tool_use_id or tool_use_id not in self._tool_spans:
                continue

            span = self._tool_spans.pop(tool_use_id)
            content = _get_attr_or_key(result, "content", None)
            is_error = _get_attr_or_key(result, "is_error", False)

            if is_error:
                tracer.end_span(
                    span,
                    error=Exception(str(content) if content else "Tool error")
                )
            else:
                tracer.end_span(span, output=content)

    def _process_tool_use(self, block: Any) -> Optional[TraceSpan]:
        """Create a TOOL_CALL span from a tool_use content block."""
        name = _get_attr_or_key(block, "name", "unknown_tool")
        tool_input = _get_attr_or_key(block, "input", {})
        tool_use_id = _get_attr_or_key(block, "id", None)

        span = tracer.start_span(
            name=name,
            span_type=SpanType.TOOL_CALL,
            input_data=tool_input,
            metadata={"tool_use_id": tool_use_id} if tool_use_id else None,
        )

        if span and tool_use_id:
            # Store span to close later when tool result arrives
            self._tool_spans[tool_use_id] = span

        return span

    def _process_thinking(self, block: Any) -> Optional[TraceSpan]:
        """Create a REASONING span from a thinking content block."""
        thinking_text = _get_attr_or_key(block, "thinking", "")

        span = tracer.start_span(
            name="thinking",
            span_type=SpanType.REASONING,
        )
        if span:
            tracer.end_span(span, output=thinking_text)
        return span

    def _process_text(self, block: Any) -> Optional[TraceSpan]:
        """Create an LLM_CALL span from a text content block."""
        text = _get_attr_or_key(block, "text", "")

        span = tracer.start_span(
            name="text_response",
            span_type=SpanType.LLM_CALL,
        )
        if span:
            tracer.end_span(span, output=text)
        return span

    def close_pending_tool_spans(self):
        """Close any tool spans that never received results."""
        for tool_use_id, span in self._tool_spans.items():
            tracer.end_span(
                span,
                error=Exception("Tool result never received")
            )
        self._tool_spans.clear()


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Get value from an object by attribute or dict key."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
