# eval_lib/tracing/smolagents_callback.py
"""Smolagents (Hugging Face) trace collector via step_callbacks.

Converts Smolagents ``MemoryStep`` objects into TraceSpans for
reliability evaluation.

Span tree per ``ActionStep``::

    step_{n}                 AGENT_STEP  (real timing from step.timing,
    │                                     output = step.action_output,
    │                                     error = step.error)
    ├── llm_call             LLM_CALL    (input = model_input_messages,
    │                                     output = model output text,
    │                                     model + usage from step.token_usage)
    └── <tool name>          TOOL_CALL   (one per tool call, observations
                                          as output; "code_execution" for
                                          a CodeAgent step without tool calls)

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.smolagents_callback import install_smolagents_tracing

    trace_id = tracer.start_trace("smolagents")

    agent = CodeAgent(tools=[...], model=model)
    install_smolagents_tracing(agent)      # every step class, incl. planning/final
    result = agent.run("do something")

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()

``smolagents_step_callback`` also works as a plain callable in
``step_callbacks=[...]`` — but a *list* registration only fires for
``ActionStep`` (smolagents wires lists to ``ActionStep`` for backward
compatibility); use :func:`install_smolagents_tracing` or a dict keyed by
step class to see planning and final-answer steps too.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .context import get_trace_id
from .trace_utils import safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan

logger = logging.getLogger("eval_lib.tracing.smolagents")

_PRIMITIVES = (str, int, float, bool)

# Per-agent bookkeeping: id(agent) -> (trace_id, task) already declared.
_declared_tasks: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
# trace_id for which the model has already been declared.
_model_declared_for: Optional[str] = None


# ---------------------------------------------------------------------------
# Public callback
# ---------------------------------------------------------------------------


def smolagents_step_callback(step: Any, agent: Any = None) -> None:
    """Callback function for Smolagents ``agent.step_callbacks``.

    Processes each ``MemoryStep`` and creates the corresponding TraceSpans.
    Never raises into the agent loop — failures are logged as warnings.

    Args:
        step: A smolagents MemoryStep (``ActionStep``, ``PlanningStep``,
            ``TaskStep``, ``FinalAnswerStep``, ...).
        agent: The agent instance (smolagents passes ``agent=self`` to
            callbacks that accept it). Used for the model id and the task.
    """
    try:
        _dispatch(step, agent)
    except Exception as e:
        logger.warning(
            "eval_lib.tracing: smolagents callback failed for %s: %r",
            type(step).__name__,
            e,
        )


def _dispatch(step: Any, agent: Any) -> None:
    step_type = type(step).__name__

    if step_type == "TaskStep":
        _process_task_step(step)
    elif step_type == "ActionStep":
        _declare_task(agent)
        _process_action_step(step, agent)
    elif step_type == "PlanningStep":
        _declare_task(agent)
        _process_planning_step(step, agent)
    elif step_type == "FinalAnswerStep":
        _process_final_answer_step(step)
    elif step_type == "SystemPromptStep":
        pass  # Skip system prompt steps
    else:
        _process_generic_step(step, step_type)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def _step_classes() -> List[type]:
    try:
        from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, TaskStep
    except ImportError:
        return []
    return [ActionStep, PlanningStep, FinalAnswerStep, TaskStep]


def install_smolagents_tracing(agent: Any, callback: Any = None) -> Any:
    """Register :func:`smolagents_step_callback` for every step class.

    Supports the three registration shapes smolagents has used:

    * ``CallbackRegistry`` (``agent.step_callbacks.register(cls, cb)``) —
      the runtime form on a constructed agent.
    * ``dict`` ``{StepClass: [callbacks]}``.
    * plain ``list`` — appended (fires for ``ActionStep`` only; that is a
      smolagents limitation of list registration).

    Args:
        agent: A constructed smolagents agent (``CodeAgent``,
            ``ToolCallingAgent``, ...). A bare registry / dict / list is
            accepted too.
        callback: Override the callback to register (default
            :func:`smolagents_step_callback`).

    Returns:
        ``agent``.
    """
    cb = callback or smolagents_step_callback
    registry = getattr(agent, "step_callbacks", agent)
    if registry is None:
        registry = []
        try:
            agent.step_callbacks = registry
        except Exception:
            pass

    classes = _step_classes()

    register = getattr(registry, "register", None)
    if callable(register):
        if not classes:
            existing = getattr(registry, "_callbacks", None)
            classes = list(existing.keys()) if isinstance(existing, dict) else []
        if not classes:
            raise ImportError(
                "smolagents is required to resolve step classes. "
                "Install with: pip install smolagents"
            )
        for cls in classes:
            if not _registered_in(registry, cls, cb):
                register(cls, cb)
        return agent

    if isinstance(registry, dict):
        targets = list(classes) or list(registry.keys())
        if not targets:
            raise ImportError(
                "smolagents is required to resolve step classes. "
                "Install with: pip install smolagents"
            )
        for cls in targets:
            current = registry.get(cls)
            if current is None:
                registry[cls] = [cb]
            elif isinstance(current, list):
                if cb not in current:
                    current.append(cb)
            elif current is not cb:
                registry[cls] = [current, cb]
        return agent

    if isinstance(registry, list):
        if cb not in registry:
            registry.append(cb)
        return agent

    raise TypeError(f"Unsupported step_callbacks container: {type(registry).__name__}")


def _registered_in(registry: Any, cls: type, cb: Any) -> bool:
    existing = getattr(registry, "_callbacks", None)
    if isinstance(existing, dict):
        return cb in (existing.get(cls) or [])
    return False


# ---------------------------------------------------------------------------
# Trace-level facts
# ---------------------------------------------------------------------------


def _agent_task(agent: Any) -> Optional[str]:
    if agent is None:
        return None
    task = getattr(agent, "task", None)
    if isinstance(task, str) and task:
        return task
    memory = getattr(agent, "memory", None)
    steps = getattr(memory, "steps", None)
    if isinstance(steps, list):
        for step in steps:
            if type(step).__name__ == "TaskStep":
                found = getattr(step, "task", None)
                if isinstance(found, str) and found:
                    return found
                break
    return None


def _declare_task(agent: Any) -> None:
    """Record the agent's task as the trace input — once per agent+trace."""
    if agent is None:
        return
    task = _agent_task(agent)
    if not task:
        return
    key = id(agent)
    trace_id = get_trace_id()
    if _declared_tasks.get(key) == (trace_id, task):
        return
    _declared_tasks[key] = (trace_id, task)
    tracer.set_trace_metadata(input=task)


def _agent_model_id(agent: Any) -> Optional[str]:
    model = getattr(agent, "model", None) if agent is not None else None
    if model is None:
        return None
    model_id = getattr(model, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    return None


def _declare_model(span: Optional[TraceSpan], model: Optional[str]) -> None:
    global _model_declared_for
    if not model:
        return
    if span is not None:
        span.metadata["model"] = model
    trace_id = get_trace_id()
    if trace_id and _model_declared_for != trace_id:
        _model_declared_for = trace_id
        tracer.set_trace_metadata(model=model)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _step_timing(step: Any) -> Tuple[Optional[float], Optional[float]]:
    timing = getattr(step, "timing", None)
    if timing is None:
        return None, None
    start = getattr(timing, "start_time", None)
    end = getattr(timing, "end_time", None)
    return (start if _is_number(start) else None), (end if _is_number(end) else None)


def _apply_timing(span: Optional[TraceSpan], start: Optional[float], end: Optional[float]) -> None:
    """Overwrite the span's clock with the framework's real timestamps.

    Must run *after* ``end_span`` (``finish()`` stamps ``end_time=now``).
    """
    if span is None or start is None:
        return
    span.start_time = float(start)
    if end is not None:
        span.end_time = float(end)
        span.duration_ms = round((float(end) - float(start)) * 1000, 2)


def _apply_child_timing(span: Optional[TraceSpan], start: Optional[float]) -> None:
    """Children have no timing of their own: pin them to the step start so
    the timeline nests, with zero duration and an explicit flag."""
    if span is None or start is None:
        return
    # 1 µs after the step start so the step span sorts first on a timeline.
    pinned = float(start) + 1e-6
    span.start_time = pinned
    span.end_time = pinned
    span.duration_ms = 0.0
    span.metadata["duration_unknown"] = True


def _token_usage(step: Any) -> Optional[Dict[str, int]]:
    usage = getattr(step, "token_usage", None)
    if usage is None:
        message = getattr(step, "model_output_message", None)
        usage = getattr(message, "token_usage", None) if message is not None else None
    if usage is None:
        return None
    if isinstance(usage, dict):
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    else:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
    if not (_is_number(input_tokens) or _is_number(output_tokens)):
        return None
    return {
        "input_tokens": int(input_tokens) if _is_number(input_tokens) else 0,
        "output_tokens": int(output_tokens) if _is_number(output_tokens) else 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }


def _record_usage(span: Optional[TraceSpan], usage: Optional[Dict[str, int]]) -> None:
    if not usage:
        return
    if span is not None:
        span.metadata["usage"] = dict(usage)
    tracer.add_trace_usage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        calls=1,
    )


def _message_to_plain(message: Any) -> Any:
    if isinstance(message, dict):
        return message
    role = getattr(message, "role", None)
    role = getattr(role, "value", role)
    content = getattr(message, "content", None)
    if not isinstance(content, _PRIMITIVES + (list, dict)):
        content = safe_str(content)
    plain: Dict[str, Any] = {"role": str(role) if role is not None else None, "content": content}
    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        plain["tool_calls"] = [safe_str(tc) for tc in tool_calls]
    return plain


def _messages_to_plain(messages: Any) -> Any:
    if messages is None:
        return None
    if isinstance(messages, list):
        return [_message_to_plain(m) for m in messages]
    return safe_str(messages)


def _model_output_text(step: Any) -> Any:
    message = getattr(step, "model_output_message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, (str, list)):
            return content
    model_output = getattr(step, "model_output", None)
    if isinstance(model_output, (str, list)):
        return model_output
    if model_output is not None:
        return safe_str(model_output)
    return None


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, _PRIMITIVES + (dict, list)):
        return value
    return safe_str(value)


def _error_parts(error: Any) -> Tuple[Any, Optional[str]]:
    """``(error, error_type)`` with the framework's real class name."""
    if error is None:
        return None, None
    if isinstance(error, BaseException):
        return error, type(error).__name__
    return str(error), type(error).__name__


def _reasoning_text(step: Any, output_text: Any) -> Optional[str]:
    """Explicit reasoning distinct from the model output, if the step has any."""
    for attr in ("reasoning", "reasoning_content", "thought"):
        value = getattr(step, attr, None)
        if isinstance(value, str) and value.strip() and value != output_text:
            return value
    message = getattr(step, "model_output_message", None)
    if message is not None:
        for attr in ("reasoning", "reasoning_content"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip() and value != output_text:
                return value
    return None


# ---------------------------------------------------------------------------
# Step processors
# ---------------------------------------------------------------------------


def _process_task_step(step: Any) -> None:
    """Process a TaskStep — the initial task assignment."""
    task = getattr(step, "task", None)
    task_text = task if isinstance(task, str) else (safe_str(task) if task is not None else None)
    span = tracer.start_span(
        name="task_assignment",
        span_type=SpanType.AGENT_STEP,
        input_data=task_text,
    )
    if task_text:
        tracer.set_trace_metadata(input=task_text)
    tracer.end_span(span, output=task_text)


def _process_action_step(step: Any, agent: Any) -> None:
    """Process an ActionStep — one LLM call + tool calls under a step span."""
    step_number = getattr(step, "step_number", None)
    step_name = f"step_{step_number}" if _is_number(step_number) else "step"
    start, end = _step_timing(step)

    error, error_type = _error_parts(getattr(step, "error", None))
    observations = getattr(step, "observations", None)
    observations_text = (
        observations
        if isinstance(observations, str)
        else (safe_str(observations) if observations is not None else None)
    )
    output_text = _model_output_text(step)

    step_metadata: Dict[str, Any] = {}
    if _is_number(step_number):
        step_metadata["step_number"] = int(step_number)
    is_final = getattr(step, "is_final_answer", None)
    if isinstance(is_final, bool):
        step_metadata["is_final_answer"] = is_final

    step_span = tracer.start_span(
        name=step_name,
        span_type=SpanType.AGENT_STEP,
        metadata=step_metadata,
        set_current=False,
    )
    parent_id = step_span.span_id if step_span is not None else None

    # 1. LLM call
    model_input = getattr(step, "model_input_messages", None)
    if (
        model_input is not None
        or output_text is not None
        or getattr(step, "token_usage", None) is not None
    ):
        llm_span = tracer.start_span(
            name="llm_call",
            span_type=SpanType.LLM_CALL,
            input_data=_messages_to_plain(model_input),
            parent_span_id=parent_id,
            set_current=False,
        )
        _declare_model(llm_span, _agent_model_id(agent))
        _record_usage(llm_span, _token_usage(step))
        tracer.end_span(llm_span, output=output_text)
        _apply_child_timing(llm_span, start)

    # 2. Explicit reasoning, only when distinct from the model output
    reasoning = _reasoning_text(step, output_text)
    if reasoning:
        reasoning_span = tracer.start_span(
            name="reasoning",
            span_type=SpanType.REASONING,
            parent_span_id=parent_id,
            set_current=False,
        )
        tracer.end_span(reasoning_span, output=reasoning)
        _apply_child_timing(reasoning_span, start)

    # 3. Tool calls
    tool_calls = getattr(step, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        shared = len(tool_calls) > 1
        for tc in tool_calls:
            tool_name = getattr(tc, "name", None)
            if not tool_name:
                tool_name = getattr(tc, "tool_name", None) or "unknown_tool"
            tool_args = getattr(tc, "arguments", None)
            if tool_args is None:
                tool_args = getattr(tc, "tool_input", None)
            tool_metadata: Dict[str, Any] = {}
            call_id = getattr(tc, "id", None)
            if isinstance(call_id, str) and call_id:
                tool_metadata["tool_call_id"] = call_id
            if shared:
                tool_metadata["shared_observation"] = True

            tool_span = tracer.start_span(
                name=str(tool_name),
                span_type=SpanType.TOOL_CALL,
                input_data=_plain_value(tool_args),
                metadata=tool_metadata,
                parent_span_id=parent_id,
                set_current=False,
            )
            if error is not None and not shared:
                tracer.end_span(
                    tool_span, output=observations_text, error=error, error_type=error_type
                )
            else:
                tracer.end_span(tool_span, output=observations_text)
            _apply_child_timing(tool_span, start)

    # 4. No tool calls but observations (code execution)
    elif observations_text is not None:
        code = getattr(step, "code_action", None)
        code_span = tracer.start_span(
            name="code_execution",
            span_type=SpanType.TOOL_CALL,
            input_data=(
                code if isinstance(code, str) else (safe_str(output_text) if output_text else None)
            ),
            parent_span_id=parent_id,
            set_current=False,
        )
        if error is not None:
            tracer.end_span(code_span, output=observations_text, error=error, error_type=error_type)
        else:
            tracer.end_span(code_span, output=observations_text)
        _apply_child_timing(code_span, start)

    # 5. Close the step
    action_output = getattr(step, "action_output", None)
    step_output = _plain_value(action_output)
    if step_output is None and observations_text is not None:
        step_output = observations_text
    tracer.end_span(step_span, output=step_output, error=error, error_type=error_type)
    _apply_timing(step_span, start, end)


def _process_planning_step(step: Any, agent: Any) -> None:
    """Process a PlanningStep — the agent's plan (an LLM call in its own right)."""
    plan = getattr(step, "plan", None)
    if plan is None:
        plan = getattr(step, "model_output", None)
    plan_text = plan if isinstance(plan, str) else (safe_str(plan) if plan is not None else None)
    start, end = _step_timing(step)

    span = tracer.start_span(
        name="planning",
        span_type=SpanType.REASONING,
        set_current=False,
    )
    parent_id = span.span_id if span is not None else None

    usage = _token_usage(step)
    model_input = getattr(step, "model_input_messages", None)
    if usage or isinstance(model_input, list):
        llm_span = tracer.start_span(
            name="planning_llm_call",
            span_type=SpanType.LLM_CALL,
            input_data=_messages_to_plain(model_input),
            parent_span_id=parent_id,
            set_current=False,
        )
        _declare_model(llm_span, _agent_model_id(agent))
        _record_usage(llm_span, usage)
        tracer.end_span(llm_span, output=plan_text)
        _apply_child_timing(llm_span, start)

    tracer.end_span(span, output=plan_text)
    _apply_timing(span, start, end)


def _process_final_answer_step(step: Any) -> None:
    """Process a FinalAnswerStep — the run's final output."""
    output = _plain_value(getattr(step, "output", None))
    span = tracer.start_span(
        name="final_answer",
        span_type=SpanType.AGENT_STEP,
        set_current=False,
    )
    if output is not None:
        tracer.set_trace_metadata(output=output)
    tracer.end_span(span, output=output)


def _process_generic_step(step: Any, step_type: str) -> None:
    """Process any other step type."""
    span = tracer.start_span(
        name=step_type.lower(),
        span_type=SpanType.CUSTOM,
        set_current=False,
    )
    output = getattr(step, "model_output", None)
    if output is None:
        output = getattr(step, "output", None)
    tracer.end_span(span, output=_plain_value(output))
