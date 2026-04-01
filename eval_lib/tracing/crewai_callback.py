# eval_lib/tracing/crewai_callback.py
"""CrewAI trace collector using Event Bus.

Listens to CrewAI execution events and converts them into TraceSpan
objects for reliability evaluation.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.crewai_callback import CrewAITraceCollector

    collector = CrewAITraceCollector()
    trace_id = tracer.start_trace("crewai_agent")

    # Just import the collector BEFORE crew.kickoff()
    # It auto-registers with crewai_event_bus
    result = crew.kickoff()

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()

    # Token usage from CrewAI output:
    # result.token_usage → {"total_tokens": ..., "prompt_tokens": ..., ...}
"""

from typing import Any, Dict, Optional
from .types import TraceSpan, SpanType
from .tracer import tracer


class CrewAITraceCollector:
    """Collects trace data from CrewAI's event bus.

    Automatically registers as a listener when instantiated.
    Works with CrewAI's BaseEventListener pattern.
    """

    def __init__(self):
        self._agent_spans: Dict[str, TraceSpan] = {}
        self._task_spans: Dict[str, TraceSpan] = {}
        self._registered = False
        self._try_register()

    def _try_register(self):
        """Try to register with CrewAI event bus if available."""
        try:
            from crewai.utilities.events import crewai_event_bus
            crewai_event_bus.on("agent_execution_started", self._on_agent_start)
            crewai_event_bus.on("agent_execution_completed", self._on_agent_complete)
            crewai_event_bus.on("agent_execution_error", self._on_agent_error)
            crewai_event_bus.on("task_started", self._on_task_start)
            crewai_event_bus.on("task_completed", self._on_task_complete)
            crewai_event_bus.on("task_failed", self._on_task_error)
            crewai_event_bus.on("tool_usage_started", self._on_tool_start)
            crewai_event_bus.on("tool_usage_finished", self._on_tool_end)
            crewai_event_bus.on("tool_usage_error", self._on_tool_error)
            self._registered = True
        except (ImportError, AttributeError):
            # CrewAI not installed or incompatible version
            pass

    # ---- Agent events ----

    def _on_agent_start(self, event: Any):
        agent_name = _safe_get(event, "agent", "unknown_agent")
        if isinstance(agent_name, str):
            name = agent_name
        else:
            name = _safe_get(agent_name, "role", str(agent_name))

        span = tracer.start_span(
            name=f"agent:{name}",
            span_type=SpanType.AGENT_STEP,
            metadata={"event": "agent_execution_started"},
        )
        if span:
            self._agent_spans[name] = span

    def _on_agent_complete(self, event: Any):
        agent_name = _safe_get(event, "agent", "unknown_agent")
        if isinstance(agent_name, str):
            name = agent_name
        else:
            name = _safe_get(agent_name, "role", str(agent_name))

        span = self._agent_spans.pop(name, None)
        if span:
            output = _safe_get(event, "output", None)
            tracer.end_span(span, output=str(output) if output else None)

    def _on_agent_error(self, event: Any):
        agent_name = _safe_get(event, "agent", "unknown_agent")
        if isinstance(agent_name, str):
            name = agent_name
        else:
            name = _safe_get(agent_name, "role", str(agent_name))

        span = self._agent_spans.pop(name, None)
        if span:
            error = _safe_get(event, "error", "Agent execution error")
            tracer.end_span(span, error=Exception(str(error)))

    # ---- Task events ----

    def _on_task_start(self, event: Any):
        task_desc = _safe_get(event, "description", "unknown_task")
        task_id = _safe_get(event, "task_id", str(id(event)))
        span = tracer.start_span(
            name=f"task:{str(task_desc)[:50]}",
            span_type=SpanType.AGENT_STEP,
            input_data=str(task_desc),
            metadata={"event": "task_started", "task_id": task_id},
        )
        if span:
            self._task_spans[str(task_id)] = span

    def _on_task_complete(self, event: Any):
        task_id = _safe_get(event, "task_id", str(id(event)))
        span = self._task_spans.pop(str(task_id), None)
        if span:
            output = _safe_get(event, "output", None)
            tracer.end_span(span, output=str(output) if output else None)

    def _on_task_error(self, event: Any):
        task_id = _safe_get(event, "task_id", str(id(event)))
        span = self._task_spans.pop(str(task_id), None)
        if span:
            error = _safe_get(event, "error", "Task failed")
            tracer.end_span(span, error=Exception(str(error)))

    # ---- Tool events ----

    def _on_tool_start(self, event: Any):
        tool_name = _safe_get(event, "tool_name", "unknown_tool")
        tool_input = _safe_get(event, "input", None)
        span = tracer.start_span(
            name=str(tool_name),
            span_type=SpanType.TOOL_CALL,
            input_data=tool_input,
        )
        if span:
            self._agent_spans[f"tool:{tool_name}:{id(event)}"] = span

    def _on_tool_end(self, event: Any):
        tool_name = _safe_get(event, "tool_name", "unknown_tool")
        # Find matching span
        key = None
        for k in list(self._agent_spans.keys()):
            if k.startswith(f"tool:{tool_name}:"):
                key = k
                break
        if key:
            span = self._agent_spans.pop(key)
            output = _safe_get(event, "output", None)
            tracer.end_span(span, output=str(output) if output else None)

    def _on_tool_error(self, event: Any):
        tool_name = _safe_get(event, "tool_name", "unknown_tool")
        key = None
        for k in list(self._agent_spans.keys()):
            if k.startswith(f"tool:{tool_name}:"):
                key = k
                break
        if key:
            span = self._agent_spans.pop(key)
            error = _safe_get(event, "error", "Tool error")
            tracer.end_span(span, error=Exception(str(error)))

    # ---- Utility ----

    def set_token_usage(self, crew_output: Any):
        """Extract token usage from CrewAI CrewOutput and set as trace metadata.

        Call this after crew.kickoff():
            result = crew.kickoff()
            collector.set_token_usage(result)
        """
        token_usage = _safe_get(crew_output, "token_usage", None)
        if token_usage and isinstance(token_usage, dict):
            tracer.set_trace_metadata(
                input_tokens=token_usage.get("prompt_tokens"),
                output_tokens=token_usage.get("completion_tokens")
                              or token_usage.get("completion_tokens"),
                total_tokens=token_usage.get("total_tokens"),
            )

    def close_pending_spans(self):
        """Close any spans that were not properly closed."""
        for key, span in list(self._agent_spans.items()):
            tracer.end_span(span, error=Exception("Span never completed"))
        self._agent_spans.clear()
        for key, span in list(self._task_spans.items()):
            tracer.end_span(span, error=Exception("Task never completed"))
        self._task_spans.clear()


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key safely."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
