# eval_lib/tracing/openai_assistants_callback.py
"""OpenAI Assistants API trace collector.

Converts OpenAI Assistants Run and RunStep objects into TraceSpan objects,
enabling reliability metrics to analyze tool calls, execution traces,
and resource usage from OpenAI Assistant-based agents.

Supports:
- RunStep type=tool_calls → TOOL_CALL spans (function, code_interpreter, file_search)
- RunStep type=message_creation → LLM_CALL spans
- Run.usage → trace-level metadata (tokens)
- Per-step timing (created_at, completed_at → duration_ms)

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.openai_assistants_callback import OpenAIAssistantsTraceCollector

    collector = OpenAIAssistantsTraceCollector()
    trace_id = tracer.start_trace("openai_assistant")

    # Run your assistant
    run = client.beta.threads.runs.create_and_poll(thread_id=thread_id, ...)

    # Collect trace from completed run
    steps = client.beta.threads.runs.steps.list(thread_id, run.id)
    collector.process_run(run, steps.data)

    # Extract data for evaluation
    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)

    tracer.end_trace()
"""

from typing import Any, Dict, List, Optional
from .types import TraceSpan, SpanType
from .tracer import tracer


class OpenAIAssistantsTraceCollector:
    """Collects trace data from OpenAI Assistants API runs.

    Processes Run and RunStep objects and converts them into TraceSpan
    objects compatible with the eval_lib tracing system.
    """

    def process_run(
        self,
        run: Any,
        run_steps: List[Any],
        messages: Optional[List[Any]] = None,
    ) -> List[TraceSpan]:
        """Process a completed Run and its steps into trace spans.

        Args:
            run: An openai.types.beta.threads.Run object or dict.
            run_steps: List of RunStep objects from
                       client.beta.threads.runs.steps.list().
            messages: Optional list of thread messages for extracting
                      assistant output text.

        Returns:
            List of created TraceSpan objects.
        """
        spans = []

        # Create root AGENT_STEP span for the entire run
        run_id = _get_attr_or_key(run, "id", "unknown_run")
        model = _get_attr_or_key(run, "model", None)
        status = _get_attr_or_key(run, "status", None)
        instructions = _get_attr_or_key(run, "instructions", None)

        run_span = tracer.start_span(
            name=f"assistant_run",
            span_type=SpanType.AGENT_STEP,
            input_data={"instructions": instructions} if instructions else None,
            metadata={"run_id": run_id, "model": model, "status": status},
        )

        # Process each run step
        # Sort by created_at to maintain chronological order
        sorted_steps = sorted(
            run_steps,
            key=lambda s: _get_attr_or_key(s, "created_at", 0) or 0
        )

        for step in sorted_steps:
            step_spans = self._process_run_step(step)
            spans.extend(step_spans)

        # Close run span
        if run_span:
            run_error = _get_attr_or_key(run, "last_error", None)
            if run_error:
                error_msg = _get_attr_or_key(run_error, "message", str(run_error))
                tracer.end_span(run_span, error=Exception(error_msg))
            else:
                output = self._extract_output_from_messages(messages)
                tracer.end_span(run_span, output=output)
            spans.append(run_span)

        # Set trace-level metadata
        usage = _get_attr_or_key(run, "usage", None)
        if usage:
            prompt_tokens = _get_attr_or_key(usage, "prompt_tokens", None)
            completion_tokens = _get_attr_or_key(usage, "completion_tokens", None)
            tracer.set_trace_metadata(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
            )

        return spans

    def _process_run_step(self, step: Any) -> List[TraceSpan]:
        """Process a single RunStep into one or more spans."""
        spans = []
        step_type = _get_attr_or_key(step, "type", "")
        step_id = _get_attr_or_key(step, "id", None)
        step_details = _get_attr_or_key(step, "step_details", None)
        step_status = _get_attr_or_key(step, "status", None)

        # Calculate duration from timestamps
        created_at = _get_attr_or_key(step, "created_at", None)
        completed_at = _get_attr_or_key(step, "completed_at", None)

        if step_type == "message_creation":
            span = tracer.start_span(
                name="message_creation",
                span_type=SpanType.LLM_CALL,
                metadata={"step_id": step_id},
            )
            if span:
                # Extract message id from step_details
                msg_id = None
                if step_details:
                    msg_creation = _get_attr_or_key(step_details, "message_creation", None)
                    if msg_creation:
                        msg_id = _get_attr_or_key(msg_creation, "message_id", None)

                span.metadata["message_id"] = msg_id
                self._set_span_timing(span, created_at, completed_at)

                if step_status == "failed":
                    last_error = _get_attr_or_key(step, "last_error", None)
                    tracer.end_span(span, error=Exception(
                        str(last_error) if last_error else "Step failed"
                    ))
                else:
                    tracer.end_span(span)
                spans.append(span)

        elif step_type == "tool_calls" and step_details:
            tool_calls = _get_attr_or_key(step_details, "tool_calls", [])
            for tool_call in tool_calls:
                tool_span = self._process_tool_call(
                    tool_call, step_id, created_at, completed_at, step_status
                )
                if tool_span:
                    spans.append(tool_span)

        return spans

    def _process_tool_call(
        self,
        tool_call: Any,
        step_id: Optional[str],
        created_at: Any,
        completed_at: Any,
        step_status: Optional[str],
    ) -> Optional[TraceSpan]:
        """Process a single tool call within a RunStep."""
        tc_type = _get_attr_or_key(tool_call, "type", "")
        tc_id = _get_attr_or_key(tool_call, "id", None)

        if tc_type == "function":
            function = _get_attr_or_key(tool_call, "function", {})
            name = _get_attr_or_key(function, "name", "unknown_function")
            arguments = _get_attr_or_key(function, "arguments", None)
            output = _get_attr_or_key(function, "output", None)

            span = tracer.start_span(
                name=name,
                span_type=SpanType.TOOL_CALL,
                input_data=arguments,
                metadata={"tool_call_id": tc_id, "step_id": step_id},
            )

        elif tc_type == "code_interpreter":
            code_input = _get_attr_or_key(
                _get_attr_or_key(tool_call, "code_interpreter", {}),
                "input", None
            )
            code_outputs = _get_attr_or_key(
                _get_attr_or_key(tool_call, "code_interpreter", {}),
                "outputs", []
            )
            output = [_get_attr_or_key(o, "logs", None) or "image" for o in code_outputs]

            span = tracer.start_span(
                name="code_interpreter",
                span_type=SpanType.TOOL_CALL,
                input_data=code_input,
                metadata={"tool_call_id": tc_id, "step_id": step_id},
            )

        elif tc_type == "file_search":
            span = tracer.start_span(
                name="file_search",
                span_type=SpanType.RETRIEVAL,
                metadata={"tool_call_id": tc_id, "step_id": step_id},
            )
            output = None

        else:
            return None

        if span:
            self._set_span_timing(span, created_at, completed_at)
            if step_status == "failed":
                tracer.end_span(span, error=Exception("Tool call step failed"))
            else:
                tracer.end_span(span, output=output)

        return span

    def _set_span_timing(
        self,
        span: TraceSpan,
        created_at: Any,
        completed_at: Any,
    ):
        """Set span timing from OpenAI timestamps (unix seconds)."""
        if created_at is not None:
            span.start_time = float(created_at)
        if completed_at is not None:
            span.end_time = float(completed_at)
            if span.start_time:
                span.duration_ms = round(
                    (span.end_time - span.start_time) * 1000, 2
                )

    def _extract_output_from_messages(
        self, messages: Optional[List[Any]]
    ) -> Optional[str]:
        """Extract the last assistant message text from thread messages."""
        if not messages:
            return None

        for msg in reversed(messages):
            role = _get_attr_or_key(msg, "role", "")
            if role == "assistant":
                content = _get_attr_or_key(msg, "content", [])
                text_parts = []
                for block in content:
                    if _get_attr_or_key(block, "type", "") == "text":
                        text = _get_attr_or_key(
                            _get_attr_or_key(block, "text", {}),
                            "value", ""
                        )
                        if text:
                            text_parts.append(text)
                if text_parts:
                    return "\n".join(text_parts)
        return None


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Get value from an object by attribute or dict key."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
