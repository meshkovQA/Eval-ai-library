# eval_lib/tracing/smolagents_callback.py
"""Smolagents (Hugging Face) trace collector via step_callbacks.

Converts Smolagents MemoryStep objects into TraceSpan objects
for reliability evaluation.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.smolagents_callback import smolagents_step_callback

    trace_id = tracer.start_trace("smolagents")

    agent = CodeAgent(
        tools=[...],
        model=model,
        step_callbacks=[smolagents_step_callback],
    )
    result = agent.run("do something")

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

from typing import Any
from .types import SpanType
from .tracer import tracer


def smolagents_step_callback(step: Any, agent: Any = None):
    """Callback function for Smolagents agent.step_callbacks.

    Processes each MemoryStep and creates corresponding TraceSpans.

    Args:
        step: A smolagents MemoryStep object containing:
            - step.task (for TaskStep)
            - step.tool_calls (for ActionStep)
            - step.observations (tool output)
            - step.model_output (LLM reasoning)
            - step.error (if any)
        agent: The agent instance (optional, for metadata).
    """
    step_type = type(step).__name__

    if step_type == "TaskStep":
        _process_task_step(step)
    elif step_type == "ActionStep":
        _process_action_step(step)
    elif step_type == "PlanningStep":
        _process_planning_step(step)
    elif step_type == "SystemPromptStep":
        pass  # Skip system prompt steps
    else:
        _process_generic_step(step, step_type)


def _process_task_step(step: Any):
    """Process a TaskStep — the initial task assignment."""
    task = getattr(step, "task", None)
    span = tracer.start_span(
        name="task_assignment",
        span_type=SpanType.AGENT_STEP,
        input_data=str(task) if task else None,
    )
    if span:
        tracer.end_span(span, output=str(task))


def _process_action_step(step: Any):
    """Process an ActionStep — LLM reasoning + tool calls."""
    # 1. LLM reasoning (model_output)
    model_output = getattr(step, "model_output", None)
    if model_output:
        reasoning_span = tracer.start_span(
            name="reasoning",
            span_type=SpanType.REASONING,
        )
        if reasoning_span:
            tracer.end_span(reasoning_span, output=str(model_output))

    # 2. Tool calls
    tool_calls = getattr(step, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            tool_name = getattr(tc, "name", None) or getattr(tc, "tool_name", "unknown_tool")
            tool_args = getattr(tc, "arguments", None) or getattr(tc, "tool_input", {})

            span = tracer.start_span(
                name=str(tool_name),
                span_type=SpanType.TOOL_CALL,
                input_data=tool_args,
            )
            if span:
                # Tool output is in observations
                observations = getattr(step, "observations", None)
                error = getattr(step, "error", None)
                if error:
                    tracer.end_span(span, error=Exception(str(error)))
                else:
                    tracer.end_span(span, output=str(observations) if observations else None)

    # 3. If no tool calls but has observations (code execution)
    elif getattr(step, "observations", None):
        span = tracer.start_span(
            name="code_execution",
            span_type=SpanType.TOOL_CALL,
            input_data=str(model_output)[:500] if model_output else None,
        )
        if span:
            error = getattr(step, "error", None)
            if error:
                tracer.end_span(span, error=Exception(str(error)))
            else:
                tracer.end_span(span, output=str(getattr(step, "observations", "")))

    # 4. Duration if available
    duration = getattr(step, "duration", None)
    if duration:
        # Attach to last span created
        pass  # Duration is per-step, not per-span


def _process_planning_step(step: Any):
    """Process a PlanningStep — agent's plan/strategy."""
    plan = getattr(step, "plan", None) or getattr(step, "model_output", None)
    span = tracer.start_span(
        name="planning",
        span_type=SpanType.REASONING,
    )
    if span:
        tracer.end_span(span, output=str(plan) if plan else None)


def _process_generic_step(step: Any, step_type: str):
    """Process any other step type."""
    span = tracer.start_span(
        name=step_type.lower(),
        span_type=SpanType.CUSTOM,
    )
    if span:
        output = getattr(step, "model_output", None) or getattr(step, "output", None)
        tracer.end_span(span, output=str(output) if output else None)
