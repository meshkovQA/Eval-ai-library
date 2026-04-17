# eval_lib/tracing/autogen_callback.py
"""AutoGen (Microsoft) trace collector via InterventionHandler.

Intercepts agent-to-agent messages and tool calls in AutoGen v0.4+.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.autogen_callback import AutoGenTraceHandler

    trace_id = tracer.start_trace("autogen")

    handler = AutoGenTraceHandler()
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[handler])

    # ... register agents and run ...

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

from typing import Any, Optional
from .types import SpanType
from .tracer import tracer


class AutoGenTraceHandler:
    """AutoGen InterventionHandler that creates eval-lib TraceSpans.

    Implements the autogen_core.InterventionHandler interface for
    intercepting messages between agents.
    """

    async def on_send(self, message: Any, *, sender: Any = None, recipient: Any = None) -> Any:
        """Called when a message is sent between agents."""
        msg_type = type(message).__name__
        sender_name = _agent_name(sender)
        recipient_name = _agent_name(recipient)

        # Classify message type
        if "ToolCall" in msg_type or "FunctionCall" in msg_type:
            span = tracer.start_span(
                name=f"tool_call:{sender_name}→{recipient_name}",
                span_type=SpanType.TOOL_CALL,
                input_data=_safe_str(message),
                metadata={"sender": sender_name, "recipient": recipient_name},
            )
            if span:
                tracer.end_span(span, output=_safe_str(message))
        else:
            span = tracer.start_span(
                name=f"message:{sender_name}→{recipient_name}",
                span_type=SpanType.AGENT_STEP,
                input_data=_safe_str(message),
                metadata={"sender": sender_name, "recipient": recipient_name, "msg_type": msg_type},
            )
            if span:
                tracer.end_span(span, output=_safe_str(message))

        return message  # Pass through — do not modify

    async def on_publish(self, message: Any, *, sender: Any = None) -> Any:
        """Called when a message is published (broadcast)."""
        sender_name = _agent_name(sender)
        span = tracer.start_span(
            name=f"publish:{sender_name}",
            span_type=SpanType.AGENT_STEP,
            input_data=_safe_str(message),
            metadata={"sender": sender_name, "event": "publish"},
        )
        if span:
            tracer.end_span(span, output=_safe_str(message))
        return message

    async def on_response(self, message: Any, *, sender: Any = None, recipient: Any = None) -> Any:
        """Called when a response is received."""
        sender_name = _agent_name(sender)
        recipient_name = _agent_name(recipient)
        span = tracer.start_span(
            name=f"response:{sender_name}→{recipient_name}",
            span_type=SpanType.LLM_CALL,
            input_data=_safe_str(message),
            metadata={"sender": sender_name, "recipient": recipient_name},
        )
        if span:
            tracer.end_span(span, output=_safe_str(message))
        return message


def _agent_name(agent: Any) -> str:
    """Extract a readable name from an agent reference."""
    if agent is None:
        return "unknown"
    if isinstance(agent, str):
        return agent
    name = getattr(agent, "name", None) or getattr(agent, "type", None)
    if name:
        return str(name)
    return type(agent).__name__


def _safe_str(obj: Any) -> Optional[str]:
    """Safely convert to string, truncating if too long."""
    if obj is None:
        return None
    s = str(obj)
    return s[:2000] if len(s) > 2000 else s
