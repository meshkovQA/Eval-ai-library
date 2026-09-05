# eval_lib/tracing/openai_assistants_callback.py
"""OpenAI Assistants API trace collector.

Converts OpenAI Assistants ``Run`` / ``RunStep`` / ``Message`` objects into
TraceSpan objects, enabling reliability metrics to analyze tool calls,
execution traces, and resource usage from OpenAI Assistant-based agents.

Span tree produced for one run::

    assistant_run (AGENT_STEP)          timed run.created_at → completed_at
    ├── message_creation (LLM_CALL)     output = the created message's text
    ├── <function name> (TOOL_CALL)     input = parsed arguments, output = result
    ├── code_interpreter (TOOL_CALL)    output = logs and image file ids
    ├── file_search (RETRIEVAL)         output = file_search.results
    └── <other type> (TOOL_CALL)        unknown tool types keep their raw type

Timing comes from the API's unix timestamps — the run has usually long
finished by the time it is collected, so the wall clock at collection
time says nothing about how long a step took.

Token usage is **accumulated** on the trace (``tracer.add_trace_usage``)
so several runs of one thread add up instead of overwriting each other:
per ``RunStep.usage`` when the steps carry it, otherwise once from
``Run.usage``.

Outcome: a run in ``cancelled`` / ``expired`` / ``failed`` / ``incomplete``
is recorded as an error (``error_type`` = ``last_error.code`` when the API
gives one, else the status); a run that is still ``queued`` /
``in_progress`` / ``requires_action`` is recorded with status ``running``.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.openai_assistants_callback import OpenAIAssistantsTraceCollector

    collector = OpenAIAssistantsTraceCollector()
    trace_id = tracer.start_trace("openai_assistant")

    # Run your assistant
    run = client.beta.threads.runs.create_and_poll(thread_id=thread_id, ...)

    # Collect trace from completed run
    steps = client.beta.threads.runs.steps.list(thread_id, run.id)
    messages = client.beta.threads.messages.list(thread_id)
    collector.process_run(run, steps.data, messages.data)

    # Extract data for evaluation
    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)

    tracer.end_trace()
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

logger = logging.getLogger("eval_lib.tracing")

_RUN_ERROR_STATUSES = frozenset({"cancelled", "expired", "failed", "incomplete"})
_RUN_PENDING_STATUSES = frozenset({"queued", "in_progress", "requires_action", "cancelling"})
_STEP_ERROR_STATUSES = frozenset({"cancelled", "expired", "failed"})
_STEP_PENDING_STATUSES = frozenset({"in_progress"})
# Whichever terminal timestamp the API filled in is the end of the object.
_END_TIMESTAMP_FIELDS = ("completed_at", "failed_at", "cancelled_at", "expired_at")


class OpenAIAssistantsTraceCollector:
    """Collects trace data from OpenAI Assistants API runs.

    Processes Run and RunStep objects and converts them into TraceSpan
    objects compatible with the eval_lib tracing system. Accepts the SDK's
    pydantic models, plain dicts, or cursor pages (``.data``).
    """

    def process_run(
        self,
        run: Any,
        run_steps: Any,
        messages: Any = None,
    ) -> List[TraceSpan]:
        """Process a completed Run and its steps into trace spans.

        Args:
            run: An openai.types.beta.threads.Run object or dict.
            run_steps: RunStep objects from ``client.beta.threads.runs.steps.list()``
                (the page itself or its ``.data``).
            messages: Optional thread messages (``messages.list()`` page or
                its ``.data``) — used for the assistant's final answer, the
                user prompt that started the run and per-step message text.

        Returns:
            List of created TraceSpan objects. Never raises: a failure is
            logged as a warning and an empty list is returned.
        """
        try:
            return self._process_run(run, _as_list(run_steps), _as_list(messages))
        except Exception as exc:
            logger.warning(
                "eval_lib.tracing: OpenAIAssistantsTraceCollector.process_run failed: %r",
                exc,
                exc_info=True,
            )
            return []

    def _process_run(self, run: Any, run_steps: List[Any], messages: List[Any]) -> List[TraceSpan]:
        spans: List[TraceSpan] = []

        run_id = _get(run, "id", None) or "unknown_run"
        model = _get(run, "model", None)
        status = _get(run, "status", None)
        instructions = _get(run, "instructions", None) or None
        thread_id = _get(run, "thread_id", None)
        run_created = _ts(_get(run, "created_at", None))
        run_started = _ts(_get(run, "started_at", None))
        run_ended = _first_timestamp(run, _END_TIMESTAMP_FIELDS)

        user_input = _latest_user_message(messages, run_created)
        input_data: Dict[str, Any] = {}
        if instructions:
            input_data["instructions"] = instructions
        if user_input:
            input_data["user_message"] = user_input

        metadata: Dict[str, Any] = {"run_id": run_id, "model": model, "status": status}
        if thread_id:
            metadata["thread_id"] = thread_id
        assistant_id = _get(run, "assistant_id", None)
        if assistant_id:
            metadata["assistant_id"] = assistant_id
        if run_started is not None:
            metadata["started_at"] = run_started
            if run_created is not None:
                metadata["queue_ms"] = round((run_started - run_created) * 1000, 2)

        # Root span for the whole run. It nests under whatever the caller
        # has open (an outer ``tracer.trace(...)``) but never becomes the
        # context parent itself — every child gets an explicit parent.
        run_span = tracer.start_span(
            name="assistant_run",
            span_type=SpanType.AGENT_STEP,
            input_data=input_data or None,
            metadata=metadata,
            set_current=False,
        )
        parent_id = run_span.span_id if run_span is not None else None

        message_index = _index_message_text(messages)

        # Chronological order — the steps page is newest-first by default.
        sorted_steps = sorted(run_steps, key=lambda s: _ts(_get(s, "created_at", None)) or 0)
        steps_had_usage = False
        for step in sorted_steps:
            step_spans, had_usage = self._process_run_step(step, parent_id, model, message_index)
            spans.extend(step_spans)
            steps_had_usage = steps_had_usage or had_usage

        # Per-step usage is preferred (it is what add_trace_usage was made
        # for); fall back to the run total exactly once when steps carry none.
        run_usage = usage_from_mapping(as_mapping(_get(run, "usage", None)))
        if run_usage and not steps_had_usage:
            tracer.add_trace_usage(
                input_tokens=run_usage["input_tokens"],
                output_tokens=run_usage["output_tokens"],
                cached_tokens=run_usage["cached_tokens"],
                reasoning_tokens=run_usage["reasoning_tokens"],
            )

        output = _extract_output_from_messages(messages, run_id)
        error_message, error_type = _run_error(run, status)

        if run_span is not None:
            end_kwargs = _end_kwargs(error_message, error_type, status in _RUN_PENDING_STATUSES)
            _end_span_timed(run_span, run_created, run_ended, output=output, **end_kwargs)
            spans.append(run_span)

        # Trace-level facts. Tokens are deliberately NOT declared here —
        # declaring would overwrite the accumulated totals of earlier runs.
        trace_meta: Dict[str, Any] = {}
        if model:
            trace_meta["model"] = model
        if thread_id:
            trace_meta["session_id"] = thread_id
        trace_input = user_input or instructions
        if trace_input:
            trace_meta["input"] = trace_input
        if output is not None:
            trace_meta["output"] = output
        if error_message:
            trace_meta["status"] = "error"
            trace_meta["error"] = error_message
            trace_meta["error_type"] = error_type
        if trace_meta:
            tracer.set_trace_metadata(**trace_meta)

        return spans

    def _process_run_step(
        self,
        step: Any,
        parent_id: Optional[str],
        model: Optional[str],
        message_index: Dict[str, str],
    ) -> Tuple[List[TraceSpan], bool]:
        """Process a single RunStep into one or more spans.

        Returns the spans and whether the step carried its own usage.
        """
        spans: List[TraceSpan] = []
        step_type = _get(step, "type", "") or ""
        step_id = _get(step, "id", None)
        step_details = _get(step, "step_details", None)
        step_status = _get(step, "status", None)

        created_at = _ts(_get(step, "created_at", None))
        ended_at = _first_timestamp(step, _END_TIMESTAMP_FIELDS)

        usage = usage_from_mapping(as_mapping(_get(step, "usage", None)))
        if usage:
            tracer.add_trace_usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
            )

        error_message, error_type = _step_error(step, step_status)
        end_kwargs = _end_kwargs(error_message, error_type, step_status in _STEP_PENDING_STATUSES)

        if step_type == "message_creation":
            message_id = _get(_get(step_details, "message_creation", None), "message_id", None)
            metadata: Dict[str, Any] = {"step_id": step_id, "message_id": message_id}
            if model:
                metadata["model"] = model
            if usage:
                metadata["usage"] = dict(usage)
            span = tracer.start_span(
                name="message_creation",
                span_type=SpanType.LLM_CALL,
                metadata=metadata,
                parent_span_id=parent_id,
                set_current=False,
            )
            if span is not None:
                output = message_index.get(message_id) if message_id else None
                _end_span_timed(span, created_at, ended_at, output=output, **end_kwargs)
                spans.append(span)

        elif step_type == "tool_calls":
            for tool_call in _get(step_details, "tool_calls", None) or []:
                span = self._process_tool_call(
                    tool_call, step_id, parent_id, created_at, ended_at, end_kwargs
                )
                if span is not None:
                    spans.append(span)

        return spans, bool(usage)

    def _process_tool_call(
        self,
        tool_call: Any,
        step_id: Optional[str],
        parent_id: Optional[str],
        created_at: Optional[float],
        ended_at: Optional[float],
        end_kwargs: Dict[str, Any],
    ) -> Optional[TraceSpan]:
        """Process a single tool call within a RunStep."""
        tc_type = _get(tool_call, "type", "") or ""
        tc_id = _get(tool_call, "id", None)
        metadata: Dict[str, Any] = {"tool_call_id": tc_id, "step_id": step_id, "tool_type": tc_type}
        output: Any = None

        if tc_type == "function":
            function = _get(tool_call, "function", None)
            name = _get(function, "name", None) or "unknown_function"
            arguments = _parse_json_maybe(_get(function, "arguments", None))
            output = _get(function, "output", None)
            span = tracer.start_span(
                name=name,
                span_type=SpanType.TOOL_CALL,
                input_data=arguments,
                metadata=metadata,
                parent_span_id=parent_id,
                set_current=False,
            )

        elif tc_type == "code_interpreter":
            code_interpreter = _get(tool_call, "code_interpreter", None)
            output = _code_interpreter_output(code_interpreter)
            span = tracer.start_span(
                name="code_interpreter",
                span_type=SpanType.TOOL_CALL,
                input_data=_get(code_interpreter, "input", None),
                metadata=metadata,
                parent_span_id=parent_id,
                set_current=False,
            )

        elif tc_type == "file_search":
            file_search = _get(tool_call, "file_search", None)
            output = _file_search_output(file_search)
            ranking = as_mapping(_get(file_search, "ranking_options", None))
            if ranking:
                metadata["ranking_options"] = ranking
            span = tracer.start_span(
                name="file_search",
                span_type=SpanType.RETRIEVAL,
                metadata=metadata,
                parent_span_id=parent_id,
                set_current=False,
            )

        else:
            # A tool type this collector does not know yet — still a tool
            # call; keep the raw payload so nothing is lost.
            span = tracer.start_span(
                name=tc_type or "unknown_tool",
                span_type=SpanType.TOOL_CALL,
                input_data=as_mapping(tool_call) or tool_call,
                metadata=metadata,
                parent_span_id=parent_id,
                set_current=False,
            )

        if span is not None:
            _end_span_timed(span, created_at, ended_at, output=output, **end_kwargs)
        return span


# ------------------------------------------------------------- module helpers


def _end_span_timed(
    span: TraceSpan, start: Optional[float], end: Optional[float], **end_kwargs: Any
) -> None:
    """Finish ``span`` and stamp it with the API's own timestamps.

    ``tracer.end_span`` → ``span.finish()`` sets ``end_time`` to *now* and
    recomputes ``duration_ms`` from ``start_time``, so the API timestamps
    have to be applied around it: ``start_time`` before, ``end_time`` /
    ``duration_ms`` after. Without the second step a step created an hour
    ago is reported as an hour long. When the API gives a start but no end
    (the object is still in progress) the duration is left unknown rather
    than measured against the collection-time clock.
    """
    if start is not None:
        span.start_time = start
    tracer.end_span(span, **end_kwargs)
    if start is None:
        return
    if end is not None:
        span.end_time = end
        span.duration_ms = round((end - start) * 1000, 2)
    else:
        span.end_time = None
        span.duration_ms = None


def _end_kwargs(
    error_message: Optional[str], error_type: Optional[str], pending: bool
) -> Dict[str, Any]:
    if error_message:
        return {"error": error_message, "error_type": error_type}
    if pending:
        return {"status": "running"}
    return {}


def _run_error(run: Any, status: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(message, error_type) for a run that did not complete, else (None, None)."""
    last_error = _get(run, "last_error", None)
    code = _get(last_error, "code", None)
    message = _get(last_error, "message", None)
    if status not in _RUN_ERROR_STATUSES and not last_error:
        return None, None
    if not message:
        if status == "incomplete":
            reason = _get(_get(run, "incomplete_details", None), "reason", None)
            message = f"run incomplete: {reason}" if reason else "run incomplete"
        else:
            message = f"run {status}" if status else "run failed"
    return str(message), str(code or status or "error")


def _step_error(step: Any, status: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(message, error_type) for a step that did not complete, else (None, None)."""
    last_error = _get(step, "last_error", None)
    code = _get(last_error, "code", None)
    message = _get(last_error, "message", None)
    if status not in _STEP_ERROR_STATUSES and not last_error:
        return None, None
    if not message:
        message = f"step {status}" if status else "step failed"
    return str(message), str(code or status or "error")


def _code_interpreter_output(code_interpreter: Any) -> Optional[List[Any]]:
    """Logs as text, images as ``{"type": "image", "file_id": …}``."""
    result: List[Any] = []
    for item in _get(code_interpreter, "outputs", None) or []:
        item_type = _get(item, "type", None)
        image = _get(item, "image", None)
        if item_type == "image" or image is not None:
            result.append({"type": "image", "file_id": _get(image, "file_id", None)})
        else:
            logs = _get(item, "logs", None)
            result.append(logs if logs is not None else {"type": item_type})
    return result or None


def _file_search_output(file_search: Any) -> Optional[List[Dict[str, Any]]]:
    """``file_search.results`` (present when requested via ``include``)."""
    results = _get(file_search, "results", None)
    if not results:
        return None
    output: List[Dict[str, Any]] = []
    for result in results:
        item: Dict[str, Any] = {
            "file_id": _get(result, "file_id", None),
            "file_name": _get(result, "file_name", None),
            "score": _get(result, "score", None),
        }
        texts = [
            text
            for text in (_get(chunk, "text", None) for chunk in _get(result, "content", None) or [])
            if text
        ]
        if texts:
            item["content"] = texts
        output.append(item)
    return output


def _parse_json_maybe(value: Any) -> Any:
    """Function arguments arrive as a JSON string — decode when possible."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except ValueError:
                return value
    return value


def _message_text(message: Any) -> Optional[str]:
    """Concatenated text of a thread message (``content[].text.value``)."""
    content = _get(message, "content", None)
    if isinstance(content, str):
        return content or None
    parts: List[str] = []
    for block in content or []:
        if _get(block, "type", None) not in (None, "text"):
            continue
        text = _get(block, "text", None)
        value = text if isinstance(text, str) else _get(text, "value", None)
        if value:
            parts.append(str(value))
    return "\n".join(parts) if parts else None


def _index_message_text(messages: List[Any]) -> Dict[str, str]:
    """message id → text, so a message_creation step can show what it produced."""
    index: Dict[str, str] = {}
    for message in messages:
        message_id = _get(message, "id", None)
        if not message_id:
            continue
        text = _message_text(message)
        if text:
            index[str(message_id)] = text
    return index


def _newest_first(messages: List[Any]) -> List[Any]:
    """Sort by ``created_at`` descending; ties keep list order (the API's
    default listing is already newest-first)."""
    return sorted(messages, key=lambda m: _ts(_get(m, "created_at", None)) or 0, reverse=True)


def _latest_user_message(messages: List[Any], run_created: Optional[float]) -> Optional[str]:
    """Text of the user message that started the run."""
    users = [m for m in messages if _get(m, "role", None) == "user"]
    if not users:
        return None
    if run_created is not None:
        before_run = [m for m in users if (_ts(_get(m, "created_at", None)) or 0) <= run_created]
        if before_run:
            users = before_run
    for message in _newest_first(users):
        text = _message_text(message)
        if text:
            return text
    return None


def _extract_output_from_messages(messages: List[Any], run_id: Optional[str]) -> Optional[str]:
    """The newest assistant reply produced by *this* run.

    Messages are filtered by ``run_id`` when they carry the field, then the
    newest by ``created_at`` wins. ``messages.list()`` returns newest-first,
    so scanning by list position from the end picks the *oldest* reply —
    hence the explicit ordering. Without a ``run_id`` match the newest
    assistant message overall is used.
    """
    assistant = [m for m in messages if _get(m, "role", None) == "assistant"]
    if not assistant:
        return None
    if run_id:
        matched = [m for m in assistant if _get(m, "run_id", None) == run_id]
        if matched:
            assistant = matched
    for message in _newest_first(assistant):
        text = _message_text(message)
        if text:
            return text
    return None


def _first_timestamp(obj: Any, fields: Tuple[str, ...]) -> Optional[float]:
    for field in fields:
        value = _ts(_get(obj, field, None))
        if value is not None:
            return value
    return None


def _ts(value: Any) -> Optional[float]:
    """Unix seconds as float from an int/float/datetime, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    return None


def _as_list(value: Any) -> List[Any]:
    """Accept a list, a cursor page (``.data``), any iterable, or ``None``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    data = getattr(value, "data", None)
    if isinstance(data, (list, tuple)):
        return list(data)
    try:
        return list(value)
    except TypeError:
        return []


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get value from an object by attribute or dict key (``None``-safe)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
