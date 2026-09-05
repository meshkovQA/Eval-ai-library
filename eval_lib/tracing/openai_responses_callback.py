# eval_lib/tracing/openai_responses_callback.py
"""OpenAI Responses API and Chat Completions tracing.

The Responses API is the successor of the Assistants API: one
``client.responses.create()`` call returns a ``Response`` whose ``output``
list mixes the assistant message with every tool call the model made
(function calls, web / file search, code interpreter, MCP, image
generation, …) and its reasoning items. Each response becomes one span
tree::

    response (LLM_CALL)                 input = request input, output = output_text
    ├── <function name> (TOOL_CALL)     stays OPEN until its function_call_output arrives
    ├── reasoning (REASONING)           output = reasoning summary
    ├── web_search (TOOL_CALL)          input = the search action
    ├── file_search (RETRIEVAL)         input = queries, output = results
    ├── code_interpreter (TOOL_CALL)    input = code, output = logs / image urls
    ├── <mcp tool> (TOOL_CALL)          server_label in metadata
    └── image_generation (TOOL_CALL)    output = size of the result, not the bytes

Function calls are executed by *your* code between two ``responses.create``
calls; the result comes back as a ``function_call_output`` input item of
the next call. The collector keeps the ``TOOL_CALL`` span open in between
and closes it when that item is seen, so the span's duration is the real
tool execution time.

Usage — instrument an ``OpenAI`` / ``AsyncOpenAI`` client in place
(``responses.create`` / ``.parse`` / ``.stream`` and
``chat.completions.create`` / ``.parse`` are wrapped)::

    from openai import OpenAI
    from eval_lib.tracing import tracer
    from eval_lib.tracing.openai_responses_callback import trace_openai_client

    trace_id = tracer.start_trace("agent")
    client = trace_openai_client(OpenAI())

    r = client.responses.create(model="gpt-4.1", input="Weather in Paris?", tools=[...])
    call = next(item for item in r.output if item.type == "function_call")
    result = run_tool(call.name, call.arguments)          # your code
    r2 = client.responses.create(
        model="gpt-4.1",
        previous_response_id=r.id,
        input=[{"type": "function_call_output", "call_id": call.call_id, "output": result}],
    )
    print(r2.output_text)

    # Chat Completions go through the same client
    c = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()

Or feed the objects to the collector yourself (when the client is created
somewhere you cannot reach)::

    from eval_lib.tracing.openai_responses_callback import OpenAIResponsesTraceCollector

    collector = OpenAIResponsesTraceCollector()
    collector.process_input_items(request_input)      # closes pending tool spans
    response = client.responses.create(model=..., input=request_input)
    collector.process_response(response, request_input=request_input)
    ...
    collector.close_pending()                          # before tracer.end_trace()

Streaming through the wrapper: ``responses.create(stream=True)`` and
``responses.stream()`` record the response from the terminal
``response.completed`` / ``response.failed`` / ``response.incomplete``
event; ``chat.completions.create(stream=True)`` accumulates the deltas and
takes the usage from the final chunk (send
``stream_options={"include_usage": True}`` to get one).

Token usage is **accumulated** on the trace (``tracer.add_trace_usage``)
per response — including cached and reasoning tokens — and the cost is
estimated from ``eval_lib.model_catalog`` when the model is known.

Nothing here imports ``openai``: every object is duck-typed by attribute
(pydantic models and plain dicts both work), so the module loads on a slim
install and tolerates version drift in the SDK types.
"""

import functools
import inspect
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from .context import get_trace_id, set_trace_id
from .trace_utils import safe_str as _safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

logger = logging.getLogger("eval_lib.tracing")

__all__ = ["OpenAIResponsesTraceCollector", "trace_openai_client"]

_TRACED_ATTR = "__eval_lib_traced__"

# Set while an SDK stream helper (``responses.stream()``) issues its request
# through the instance's ``create`` — which is our wrapper. The inner call
# must not record a second span for the same response.
_SUPPRESS_INNER: ContextVar[bool] = ContextVar("eval_lib_openai_suppress_inner", default=False)

# ``Response.status`` values that mean the response is still being produced
# (``background=True`` requests come back like this).
_RESPONSE_PENDING = frozenset({"in_progress", "queued"})
# Output-item statuses that mean "still going".
_ITEM_PENDING = frozenset(
    {"in_progress", "searching", "generating", "interpreting", "calling", "queued"}
)
# Stream events that carry the finished ``Response``.
_TERMINAL_EVENTS = frozenset({"response.completed", "response.failed", "response.incomplete"})
# A tool result that announces itself as a failure.
_ERROR_PREFIX = re.compile(r"^\s*error\b", re.IGNORECASE)
# Output items whose result arrives as a ``<type>_output`` input item of the
# next request — the span is kept open until then.
_OPEN_TOOL_ITEMS = {
    "function_call": None,  # named after the function
    "custom_tool_call": None,
    "local_shell_call": "local_shell",
    "computer_call": "computer",
}


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class OpenAIResponsesTraceCollector:
    """Turns Responses API / Chat Completions objects into trace spans.

    Accepts the SDK's pydantic models or plain dicts. Every public method
    is a no-op-on-failure: a tracing error is logged as a warning and never
    reaches the caller.

    Args:
        trace_id: Trace to record into. Defaults to the trace active when
            the collector is constructed; it is re-bound inside each method
            when the running context has none (a thread pool, a callback
            fired from another task).
    """

    _MAX_PENDING = 256

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self._trace_id = trace_id or get_trace_id()
        # call_id → open TOOL_CALL span, insertion-ordered so the oldest is
        # evicted first when a loop never reports results.
        self._pending: "OrderedDict[str, TraceSpan]" = OrderedDict()
        self._lock = threading.Lock()
        self._model_set_for: Optional[str] = None
        self._input_set_for: Optional[str] = None

    # -- public API ---------------------------------------------------------

    def process_response(
        self,
        response: Any,
        request_input: Any = None,
        *,
        instructions: Any = None,
        started_at: Optional[float] = None,
        ended_at: Optional[float] = None,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceSpan]:
        """Record one ``Response`` as an ``LLM_CALL`` span plus its children.

        Args:
            response: ``openai.types.responses.Response`` (or ``ParsedResponse``)
                or an equivalent dict.
            request_input: The ``input`` argument of the request. Becomes the
                span input; without it ``response.instructions`` is used.
            instructions: The ``instructions`` argument of the request
                (recorded in the span metadata; defaults to
                ``response.instructions``).
            started_at: Unix time the request was sent. Defaults to
                ``response.created_at``.
            ended_at: Unix time the response was received. Defaults to
                ``response.completed_at``, else now.
            parent_span_id: Explicit parent span. ``None`` nests the span
                under whatever span is open in the current context.
            metadata: Extra span metadata.

        Returns:
            The ``LLM_CALL`` span, or ``None`` when there is no active trace
            or tracing failed.
        """
        try:
            self._bind_trace()
            return self._process_response(
                response,
                request_input,
                instructions,
                started_at,
                ended_at,
                parent_span_id,
                metadata,
            )
        except Exception as exc:
            logger.warning(
                "eval_lib.tracing: OpenAIResponsesTraceCollector.process_response failed: %r",
                exc,
                exc_info=True,
            )
            return None

    def process_input_items(self, items: Any) -> int:
        """Inspect the ``input`` of a request *before* it is sent.

        ``function_call_output`` items (and the other ``*_call_output``
        kinds) close the matching open ``TOOL_CALL`` span with the tool's
        result — an output that looks like an error (a dict with a truthy
        ``"error"`` key, or a string starting with ``Error``) marks the span
        failed with ``error_type="ToolError"``. Unknown ``call_id`` values
        are ignored. The first user message seen for a trace is recorded as
        the trace input.

        Returns the number of tool spans closed.
        """
        try:
            self._bind_trace()
            self._record_trace_input(items)
            return self._close_tool_outputs(items)
        except Exception as exc:
            logger.warning(
                "eval_lib.tracing: OpenAIResponsesTraceCollector.process_input_items failed: %r",
                exc,
                exc_info=True,
            )
            return 0

    def process_chat_completion(
        self,
        completion: Any,
        messages: Any = None,
        *,
        started_at: Optional[float] = None,
        ended_at: Optional[float] = None,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceSpan]:
        """Record one ``ChatCompletion`` as an ``LLM_CALL`` span.

        Chat Completions never execute tools, so ``tool_calls`` are recorded
        on the span (as the output when there is no text, always in the
        metadata) instead of as child spans.
        """
        try:
            self._bind_trace()
            return self._process_chat_completion(
                completion, messages, started_at, ended_at, parent_span_id, metadata
            )
        except Exception as exc:
            logger.warning(
                "eval_lib.tracing: OpenAIResponsesTraceCollector.process_chat_completion failed: %r",
                exc,
                exc_info=True,
            )
            return None

    def process_exception(
        self,
        error: BaseException,
        *,
        api: str = "responses",
        request_input: Any = None,
        model: Optional[str] = None,
        started_at: Optional[float] = None,
        ended_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceSpan]:
        """Record a request that raised (``RateLimitError``, timeout, …).

        Produces a failed ``LLM_CALL`` span whose ``error_type`` is the
        exception class name.
        """
        try:
            self._bind_trace()
            meta: Dict[str, Any] = {"model": model}
            if metadata:
                meta.update(metadata)
            span = self._start(
                "chat_completion" if api == "chat" else "response",
                SpanType.LLM_CALL,
                input_data=_jsonable(request_input),
                metadata=meta,
            )
            end = _coerce_ts(ended_at) or time.time()
            _end_span_timed(span, _coerce_ts(started_at) or end, end, error=error)
            return span
        except Exception as exc:
            logger.warning(
                "eval_lib.tracing: OpenAIResponsesTraceCollector.process_exception failed: %r",
                exc,
                exc_info=True,
            )
            return None

    def close_pending(self, error: str = "tool result never observed") -> int:
        """Close tool spans whose result never came back.

        Call it before ``tracer.end_trace()`` when a loop is aborted.
        Returns the number of spans closed.
        """
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for span in pending:
            try:
                tracer.end_span(span, error=error)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("eval_lib.tracing: closing pending tool span failed: %r", exc)
        return len(pending)

    @property
    def pending_call_ids(self) -> List[str]:
        """``call_id`` values of tool calls still waiting for their output."""
        with self._lock:
            return list(self._pending.keys())

    # -- internals: responses -----------------------------------------------

    def _process_response(
        self,
        response: Any,
        request_input: Any,
        instructions: Any,
        started_at: Optional[float],
        ended_at: Optional[float],
        parent_span_id: Optional[str],
        extra_metadata: Optional[Dict[str, Any]],
    ) -> Optional[TraceSpan]:
        if response is None:
            return None

        response_id = _get(response, "id")
        model = _get(response, "model")
        status = _get(response, "status")

        end = _coerce_ts(ended_at) or _ts(_get(response, "completed_at")) or time.time()
        start = _coerce_ts(started_at) or _ts(_get(response, "created_at")) or end
        if end < start:
            end = start

        response_instructions = (
            instructions if instructions is not None else _get(response, "instructions")
        )
        if request_input is not None:
            input_data = _jsonable(request_input)
        else:
            input_data = _jsonable(response_instructions)

        meta: Dict[str, Any] = {"response_id": response_id, "model": model, "status": status}
        previous = _get(response, "previous_response_id")
        if previous:
            meta["previous_response_id"] = previous
        temperature = _get(response, "temperature")
        if temperature is not None:
            meta["temperature"] = temperature
        if request_input is not None and response_instructions:
            meta["instructions"] = _jsonable(response_instructions)
        conversation_id = _conversation_id(response)
        if conversation_id:
            meta["conversation_id"] = conversation_id
        if extra_metadata:
            meta.update(extra_metadata)

        usage = _usage_record(_get(response, "usage"))
        cost: Optional[float] = None
        if usage:
            meta["usage"] = dict(usage)
            cost = _estimate_cost(model, usage)
            if cost is not None:
                meta["estimated_cost_usd"] = cost

        text = _response_text(response)
        refusal = _response_refusal(response)
        if refusal:
            meta["refusal"] = refusal

        span = self._start(
            "response",
            SpanType.LLM_CALL,
            input_data=input_data,
            metadata=meta,
            parent_span_id=parent_span_id,
        )
        # The parent is finished first so it precedes its children in the
        # trace (they share its start time; the sort is stable).
        _end_span_timed(span, start, end, output=text, **_response_end_kwargs(response, status))

        parent_id = span.span_id if span is not None else None
        for item in _as_list(_get(response, "output")):
            self._process_output_item(item, parent_id, start, end)

        if usage:
            tracer.add_trace_usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                cost_usd=cost or 0.0,
                calls=1,
            )
        self._record_trace_input(request_input)
        self._record_trace_facts(model=model, session_id=conversation_id, output=text)
        return span

    def _process_output_item(
        self, item: Any, parent_id: Optional[str], start: float, end: float
    ) -> None:
        item_type = _get(item, "type") or ""
        item_status = _get(item, "status")
        base_meta: Dict[str, Any] = {"item_id": _get(item, "id"), "item_type": item_type}

        if item_type == "message":
            # Its text is the response output; a refusal is already in the
            # response span metadata.
            return

        if item_type in _OPEN_TOOL_ITEMS:
            self._open_tool_item(item, item_type, parent_id, base_meta, end)
            return

        if item_type == "reasoning":
            summary = _join_texts(_get(item, "summary"))
            content = _join_texts(_get(item, "content"))
            meta = {**base_meta, "encrypted": bool(_get(item, "encrypted_content"))}
            span = self._start(
                "reasoning", SpanType.REASONING, metadata=meta, parent_span_id=parent_id
            )
            _end_span_timed(
                span,
                start,
                end,
                output=summary or content or None,
                **_item_end_kwargs(item_status, "reasoning"),
            )
            return

        if item_type == "web_search_call":
            span = self._start(
                "web_search",
                SpanType.TOOL_CALL,
                input_data=_jsonable(_get(item, "action")),
                metadata=base_meta,
                parent_span_id=parent_id,
            )
            _end_span_timed(span, start, end, **_item_end_kwargs(item_status, "web_search"))
            return

        if item_type == "file_search_call":
            span = self._start(
                "file_search",
                SpanType.RETRIEVAL,
                input_data=_jsonable(_get(item, "queries")),
                metadata=base_meta,
                parent_span_id=parent_id,
            )
            _end_span_timed(
                span,
                start,
                end,
                output=_file_search_results(_get(item, "results")),
                **_item_end_kwargs(item_status, "file_search"),
            )
            return

        if item_type == "code_interpreter_call":
            meta = dict(base_meta)
            container = _get(item, "container_id")
            if container:
                meta["container_id"] = container
            span = self._start(
                "code_interpreter",
                SpanType.TOOL_CALL,
                input_data=_get(item, "code"),
                metadata=meta,
                parent_span_id=parent_id,
            )
            _end_span_timed(
                span,
                start,
                end,
                output=_code_interpreter_outputs(_get(item, "outputs")),
                **_item_end_kwargs(item_status, "code_interpreter"),
            )
            return

        if item_type == "mcp_call":
            name = _get(item, "name") or "mcp_call"
            meta = {**base_meta, "server_label": _get(item, "server_label")}
            approval = _get(item, "approval_request_id")
            if approval:
                meta["approval_request_id"] = approval
            span = self._start(
                name,
                SpanType.TOOL_CALL,
                input_data=_parse_json_maybe(_get(item, "arguments")),
                metadata=meta,
                parent_span_id=parent_id,
            )
            error = _get(item, "error")
            end_kwargs = (
                {"error": _safe_str(error)} if error else _item_end_kwargs(item_status, name)
            )
            _end_span_timed(
                span, start, end, output=_parse_json_maybe(_get(item, "output")), **end_kwargs
            )
            return

        if item_type == "mcp_list_tools":
            meta = {**base_meta, "server_label": _get(item, "server_label")}
            tools = [_get(t, "name") for t in _as_list(_get(item, "tools")) if _get(t, "name")]
            span = self._start(
                "mcp_list_tools", SpanType.TOOL_CALL, metadata=meta, parent_span_id=parent_id
            )
            error = _get(item, "error")
            _end_span_timed(
                span,
                start,
                end,
                output=tools or None,
                **({"error": _safe_str(error)} if error else {}),
            )
            return

        if item_type == "mcp_approval_request":
            name = _get(item, "name") or "mcp_approval_request"
            meta = {
                **base_meta,
                "server_label": _get(item, "server_label"),
                "approval_required": True,
            }
            span = self._start(
                name,
                SpanType.TOOL_CALL,
                input_data=_parse_json_maybe(_get(item, "arguments")),
                metadata=meta,
                parent_span_id=parent_id,
            )
            _end_span_timed(span, start, end, status="running")
            return

        if item_type == "image_generation_call":
            result = _get(item, "result")
            output = {"result_bytes": len(result)} if isinstance(result, (str, bytes)) else None
            span = self._start(
                "image_generation", SpanType.TOOL_CALL, metadata=base_meta, parent_span_id=parent_id
            )
            _end_span_timed(
                span, start, end, output=output, **_item_end_kwargs(item_status, "image_generation")
            )
            return

        # A type this collector does not know yet — still a tool call; keep
        # the raw payload so nothing is lost.
        span = self._start(
            item_type or "unknown",
            SpanType.TOOL_CALL,
            input_data=_safe_str(item),
            metadata=base_meta,
            parent_span_id=parent_id,
        )
        _end_span_timed(span, start, end, **_item_end_kwargs(item_status, item_type or "unknown"))

    def _open_tool_item(
        self,
        item: Any,
        item_type: str,
        parent_id: Optional[str],
        base_meta: Dict[str, Any],
        started: float,
    ) -> None:
        """Open a TOOL_CALL span for a call whose result arrives in the next request."""
        call_id = _get(item, "call_id")
        meta = {**base_meta, "call_id": call_id}
        if item_type == "function_call":
            name = _get(item, "name") or "function"
            input_data = _parse_json_maybe(_get(item, "arguments"))
            namespace = _get(item, "namespace")
            if namespace:
                meta["namespace"] = namespace
        elif item_type == "custom_tool_call":
            name = _get(item, "name") or "custom_tool"
            input_data = _get(item, "input")
        else:
            name = _OPEN_TOOL_ITEMS[item_type] or item_type
            action = _get(item, "action")
            if action is None and item_type == "computer_call":
                action = _get(item, "actions")
            input_data = _jsonable(action)
            checks = _as_list(_get(item, "pending_safety_checks"))
            if checks:
                meta["pending_safety_checks"] = [_jsonable(c) for c in checks]

        span = self._start(
            name, SpanType.TOOL_CALL, input_data=input_data, metadata=meta, parent_span_id=parent_id
        )
        if span is None:
            return
        # The tool runs after the response came back.
        span.start_time = started

        if not isinstance(call_id, str) or not call_id:
            tracer.end_span(span)  # nothing to match it against later
            return

        evicted: List[TraceSpan] = []
        with self._lock:
            self._pending[call_id] = span
            while len(self._pending) > self._MAX_PENDING:
                evicted.append(self._pending.popitem(last=False)[1])
        for stale in evicted:
            tracer.end_span(stale, error="tool result never observed")

    def _close_tool_outputs(self, items: Any) -> int:
        if items is None or isinstance(items, (str, bytes)):
            return 0
        closed = 0
        for item in _as_list(items):
            item_type = _get(item, "type")
            if not isinstance(item_type, str) or not item_type.endswith("_call_output"):
                continue
            call_id = _get(item, "call_id")
            with self._lock:
                span = self._pending.pop(call_id, None) if isinstance(call_id, str) else None
            if span is None:
                continue
            output = _tool_output_repr(_get(item, "output"))
            error = _output_error(output)
            if error:
                tracer.end_span(span, output=output, error=error, error_type="ToolError")
            else:
                tracer.end_span(span, output=output)
            closed += 1
        return closed

    # -- internals: chat completions ---------------------------------------

    def _process_chat_completion(
        self,
        completion: Any,
        messages: Any,
        started_at: Optional[float],
        ended_at: Optional[float],
        parent_span_id: Optional[str],
        extra_metadata: Optional[Dict[str, Any]],
    ) -> Optional[TraceSpan]:
        if completion is None:
            return None

        model = _get(completion, "model")
        end = _coerce_ts(ended_at) or time.time()
        start = _coerce_ts(started_at) or _ts(_get(completion, "created")) or end
        if end < start:
            end = start

        choices = _as_list(_get(completion, "choices"))
        choice = choices[0] if choices else None
        message = _get(choice, "message")
        content = _get(message, "content")
        refusal = _get(message, "refusal")
        tool_calls = _chat_tool_calls(_get(message, "tool_calls"))

        meta: Dict[str, Any] = {
            "completion_id": _get(completion, "id"),
            "model": model,
            "finish_reason": _get(choice, "finish_reason"),
        }
        if len(choices) > 1:
            meta["choices"] = len(choices)
        if refusal:
            meta["refusal"] = refusal
        if tool_calls:
            meta["tool_calls"] = tool_calls
        if extra_metadata:
            meta.update(extra_metadata)

        usage = _usage_record(_get(completion, "usage"))
        cost: Optional[float] = None
        if usage:
            meta["usage"] = dict(usage)
            cost = _estimate_cost(model, usage)
            if cost is not None:
                meta["estimated_cost_usd"] = cost

        if content:
            output: Any = content if isinstance(content, str) else _jsonable(content)
        else:
            output = tool_calls or content

        span = self._start(
            "chat_completion",
            SpanType.LLM_CALL,
            input_data=_jsonable(messages),
            metadata=meta,
            parent_span_id=parent_span_id,
        )
        _end_span_timed(span, start, end, output=output)

        if usage:
            tracer.add_trace_usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                cost_usd=cost or 0.0,
                calls=1,
            )
        self._record_trace_input(messages)
        self._record_trace_facts(
            model=model, session_id=None, output=content if isinstance(content, str) else None
        )
        return span

    # -- internals: shared --------------------------------------------------

    def _bind_trace(self) -> None:
        if self._trace_id and not get_trace_id():
            set_trace_id(self._trace_id)

    @staticmethod
    def _start(
        name: str,
        span_type: SpanType,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> Optional[TraceSpan]:
        """``tracer.start_span`` that never moves the context pointer.

        ``parent_span_id=None`` inherits the context parent (so a response
        nests under an outer ``tracer.trace(...)`` block); a string is an
        explicit parent.
        """
        kwargs: Dict[str, Any] = {}
        if parent_span_id is not None:
            kwargs["parent_span_id"] = parent_span_id
        return tracer.start_span(
            name=name,
            span_type=span_type,
            input_data=input_data,
            metadata=metadata,
            set_current=False,
            **kwargs,
        )

    def _record_trace_input(self, request_input: Any) -> None:
        """The first user message of the conversation becomes the trace input."""
        trace_id = get_trace_id()
        if not trace_id or self._input_set_for == trace_id:
            return
        text = _first_user_text(request_input)
        if text is None:
            return
        self._input_set_for = trace_id
        tracer.set_trace_metadata(input=text)

    def _record_trace_facts(
        self, model: Optional[str], session_id: Optional[str], output: Optional[str]
    ) -> None:
        trace_id = get_trace_id()
        if not trace_id:
            return
        facts: Dict[str, Any] = {}
        if isinstance(model, str) and model and self._model_set_for != trace_id:
            facts["model"] = model
            self._model_set_for = trace_id
        if session_id:
            facts["session_id"] = session_id
        if output:
            facts["output"] = output
        if facts:
            tracer.set_trace_metadata(**facts)


# ---------------------------------------------------------------------------
# Client instrumentation
# ---------------------------------------------------------------------------


def trace_openai_client(
    client: Any,
    *,
    collector: Optional[OpenAIResponsesTraceCollector] = None,
    trace_id: Optional[str] = None,
) -> Any:
    """Instrument an ``OpenAI`` / ``AsyncOpenAI`` client in place.

    Wraps, when present, ``client.responses.create`` / ``.parse`` /
    ``.stream`` and ``client.chat.completions.create`` / ``.parse``. The
    wrappers time the call, close pending tool spans from the request's
    ``function_call_output`` items, record the result (or the exception,
    which is re-raised) and hand back the real return value untouched —
    tracing failures are logged, never raised. Streaming returns are
    wrapped in a pass-through proxy that records the response when the
    stream finishes.

    Idempotent: wrapping the same client twice is a no-op.

    Args:
        client: Any object with that attribute layout (duck-typed; the
            ``openai`` package is never imported).
        collector: Collector to record into (one is created when omitted).
        trace_id: Trace to bind when the calling context has none.

    Returns:
        The same ``client``.
    """
    collector = collector or OpenAIResponsesTraceCollector(trace_id=trace_id)
    try:
        responses = getattr(client, "responses", None)
        if responses is not None:
            _wrap_method(responses, "create", collector, api="responses")
            _wrap_method(responses, "parse", collector, api="responses")
            _wrap_stream_manager(responses, "stream", collector)
        completions = getattr(getattr(client, "chat", None), "completions", None)
        if completions is not None:
            _wrap_method(completions, "create", collector, api="chat")
            _wrap_method(completions, "parse", collector, api="chat")
    except Exception as exc:
        logger.warning("eval_lib.tracing: trace_openai_client failed: %r", exc, exc_info=True)
    return client


def _is_coroutine_function(fn: Any) -> bool:
    """``inspect.iscoroutinefunction`` that sees through decorator wrappers.

    ``AsyncCompletions.create`` is wrapped by the SDK's ``required_args``
    decorator, which hides the coroutine flag from the bound method.
    """
    if inspect.iscoroutinefunction(fn):
        return True
    try:
        return inspect.iscoroutinefunction(inspect.unwrap(fn))
    except Exception:
        return False


def _mark_wrapper(wrapper: Any, original: Any) -> Any:
    try:
        functools.update_wrapper(wrapper, original)
    except Exception:
        pass
    setattr(wrapper, _TRACED_ATTR, True)
    setattr(wrapper, "__eval_lib_original__", original)
    return wrapper


def _wrap_method(
    resource: Any, name: str, collector: OpenAIResponsesTraceCollector, *, api: str
) -> None:
    original = getattr(resource, name, None)
    if original is None or not callable(original) or getattr(original, _TRACED_ATTR, False):
        return

    if _is_coroutine_function(original):

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _SUPPRESS_INNER.get():
                return await original(*args, **kwargs)
            call = _TracedCall(collector, api, kwargs, method=name)
            call.before()
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                call.failed(exc)
                raise
            return call.finish(result)

        setattr(resource, name, _mark_wrapper(async_wrapper, original))
        return

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        if _SUPPRESS_INNER.get():
            return original(*args, **kwargs)
        call = _TracedCall(collector, api, kwargs, method=name)
        call.before()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            call.failed(exc)
            raise
        if inspect.isawaitable(result):
            # An async method that hid its coroutine flag: finish on await.
            return call.finish_awaitable(result)
        return call.finish(result)

    setattr(resource, name, _mark_wrapper(sync_wrapper, original))


def _wrap_stream_manager(
    resource: Any, name: str, collector: OpenAIResponsesTraceCollector
) -> None:
    """``responses.stream()`` returns a context manager that performs the
    request on ``__enter__`` / ``__aenter__`` and yields an event stream."""
    original = getattr(resource, name, None)
    if original is None or not callable(original) or getattr(original, _TRACED_ATTR, False):
        return

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call = _TracedCall(collector, "responses", kwargs, method=name, stream=True)
        try:
            manager = original(*args, **kwargs)
        except BaseException as exc:
            call.before()
            call.failed(exc)
            raise
        return _StreamManagerProxy(manager, call)

    setattr(resource, name, _mark_wrapper(wrapper, original))


class _TracedCall:
    """State of one wrapped API call, from request to recorded span."""

    def __init__(
        self,
        collector: OpenAIResponsesTraceCollector,
        api: str,
        kwargs: Dict[str, Any],
        *,
        method: str,
        stream: bool = False,
    ) -> None:
        self.collector = collector
        self.api = api
        self.method = method
        self.stream = bool(stream or kwargs.get("stream"))
        self.model = kwargs.get("model")
        self.instructions = kwargs.get("instructions")
        self.request_input = kwargs.get("messages") if api == "chat" else kwargs.get("input")
        self.started_at = time.time()
        self._done = False
        # streaming state
        self._terminal_seen = False
        self._stream_error: Optional[str] = None
        self._stream_error_code: Optional[str] = None
        self._chat_stream: Optional[_ChatStreamAccumulator] = None

    def _metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"api_method": f"{self.api}.{self.method}"}
        if self.stream:
            meta["stream"] = True
        return meta

    # -- request ------------------------------------------------------------

    def before(self) -> None:
        self.started_at = time.time()
        try:
            self.collector.process_input_items(self.request_input)
        except Exception as exc:  # pragma: no cover - process_input_items never raises
            logger.warning("eval_lib.tracing: OpenAI request tracing failed: %r", exc)

    def failed(self, error: BaseException) -> None:
        if self._done:
            return
        self._done = True
        try:
            self.collector.process_exception(
                error,
                api=self.api,
                request_input=self.request_input,
                model=self.model,
                started_at=self.started_at,
                ended_at=time.time(),
                metadata=self._metadata(),
            )
        except Exception as exc:  # pragma: no cover - process_exception never raises
            logger.warning("eval_lib.tracing: OpenAI error tracing failed: %r", exc)

    # -- non-streaming result ------------------------------------------------

    def finish(self, result: Any) -> Any:
        try:
            if self.stream and _is_stream(result):
                return _StreamProxy(result, self)
            self.record(result)
        except Exception as exc:
            logger.warning("eval_lib.tracing: OpenAI response tracing failed: %r", exc)
        return result

    async def finish_awaitable(self, awaitable: Any) -> Any:
        try:
            result = await awaitable
        except BaseException as exc:
            self.failed(exc)
            raise
        return self.finish(result)

    def record(self, result: Any) -> None:
        if self._done:
            return
        self._done = True
        ended_at = time.time()
        if self.api == "chat":
            self.collector.process_chat_completion(
                result,
                messages=self.request_input,
                started_at=self.started_at,
                ended_at=ended_at,
                metadata=self._metadata(),
            )
        else:
            self.collector.process_response(
                result,
                request_input=self.request_input,
                instructions=self.instructions,
                started_at=self.started_at,
                ended_at=ended_at,
                metadata=self._metadata(),
            )

    # -- streaming ----------------------------------------------------------

    def on_event(self, event: Any) -> None:
        try:
            if self.api == "chat":
                if self._chat_stream is None:
                    self._chat_stream = _ChatStreamAccumulator()
                self._chat_stream.add(event)
                return
            event_type = _get(event, "type")
            if event_type in _TERMINAL_EVENTS:
                self._terminal_seen = True
                self.record(_get(event, "response"))
            elif event_type == "error":
                self._stream_error = _get(event, "message") or "stream error"
                self._stream_error_code = _get(event, "code")
        except Exception as exc:
            logger.warning("eval_lib.tracing: OpenAI stream event tracing failed: %r", exc)

    def stream_ended(self) -> None:
        """Iteration finished, or the stream was closed."""
        if self._done:
            return
        try:
            if self.api == "chat":
                accumulator = self._chat_stream or _ChatStreamAccumulator()
                self.record(accumulator.completion())
                return
            # Responses: no terminal event came through.
            self._done = True
            span = self.collector._start(
                "response",
                SpanType.LLM_CALL,
                input_data=_jsonable(self.request_input),
                metadata={"model": self.model, **self._metadata()},
            )
            if self._stream_error:
                _end_span_timed(
                    span,
                    self.started_at,
                    time.time(),
                    error=self._stream_error,
                    error_type=self._stream_error_code or "ResponseError",
                )
            else:
                _end_span_timed(
                    span,
                    self.started_at,
                    time.time(),
                    error="stream ended without a terminal event",
                    error_type="StreamInterrupted",
                )
        except Exception as exc:
            logger.warning("eval_lib.tracing: OpenAI stream end tracing failed: %r", exc)

    def stream_failed(self, error: BaseException) -> None:
        self.failed(error)


def _is_stream(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, dict)):
        return False
    # pydantic models are iterable (over their fields) — a Response /
    # ChatCompletion is never a stream.
    if hasattr(value, "output") or hasattr(value, "choices"):
        return False
    return hasattr(value, "__iter__") or hasattr(value, "__aiter__")


class _StreamProxy:
    """Pass-through for a sync or async event stream that reports each
    event to the call and finalises when iteration ends."""

    def __init__(self, stream: Any, call: _TracedCall) -> None:
        self._stream = stream
        self._call = call
        self._iterator: Any = None
        self._aiterator: Any = None
        self._is_async = not hasattr(stream, "__iter__") and hasattr(stream, "__aiter__")

    # sync iteration
    def __iter__(self) -> "_StreamProxy":
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self._stream)
        try:
            event = next(self._iterator)
        except StopIteration:
            self._call.stream_ended()
            raise
        except BaseException as exc:
            self._call.stream_failed(exc)
            raise
        self._call.on_event(event)
        return event

    # async iteration
    def __aiter__(self) -> "_StreamProxy":
        return self

    async def __anext__(self) -> Any:
        if self._aiterator is None:
            self._aiterator = self._stream.__aiter__()
        try:
            event = await self._aiterator.__anext__()
        except StopAsyncIteration:
            self._call.stream_ended()
            raise
        except BaseException as exc:
            self._call.stream_failed(exc)
            raise
        self._call.on_event(event)
        return event

    # context management
    def __enter__(self) -> "_StreamProxy":
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        self._call.stream_ended()
        exit_ = getattr(self._stream, "__exit__", None)
        return exit_(exc_type, exc, tb) if callable(exit_) else None

    async def __aenter__(self) -> "_StreamProxy":
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        self._call.stream_ended()
        exit_ = getattr(self._stream, "__aexit__", None)
        return (await exit_(exc_type, exc, tb)) if callable(exit_) else None

    def close(self) -> Any:
        self._call.stream_ended()
        close = getattr(self._stream, "close", None)
        return close() if callable(close) else None

    # SDK helpers on ResponseStream that iterate the stream internally — they
    # have to go through the proxy or the events are never seen.
    def until_done(self) -> Any:
        if self._is_async:
            return self._until_done_async()
        for _ in self:
            pass
        return self

    async def _until_done_async(self) -> "_StreamProxy":
        async for _ in self:
            pass
        return self

    def get_final_response(self) -> Any:
        if self._is_async:
            return self._get_final_response_async()
        self.until_done()
        return self._stream.get_final_response()

    async def _get_final_response_async(self) -> Any:
        await self._until_done_async()
        return await self._stream.get_final_response()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)

    def __repr__(self) -> str:
        return f"_StreamProxy({self._stream!r})"


class _StreamManagerProxy:
    """Wraps the ``ResponseStreamManager`` returned by ``responses.stream()``."""

    def __init__(self, manager: Any, call: _TracedCall) -> None:
        self._manager = manager
        self._call = call

    def __enter__(self) -> Any:
        self._call.before()
        token = _SUPPRESS_INNER.set(True)
        try:
            stream = self._manager.__enter__()
        except BaseException as exc:
            self._call.failed(exc)
            raise
        finally:
            _SUPPRESS_INNER.reset(token)
        return _StreamProxy(stream, self._call)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc is not None:
            self._call.stream_failed(exc)
        else:
            self._call.stream_ended()
        return self._manager.__exit__(exc_type, exc, tb)

    async def __aenter__(self) -> Any:
        self._call.before()
        token = _SUPPRESS_INNER.set(True)
        try:
            stream = await self._manager.__aenter__()
        except BaseException as exc:
            self._call.failed(exc)
            raise
        finally:
            _SUPPRESS_INNER.reset(token)
        return _StreamProxy(stream, self._call)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc is not None:
            self._call.stream_failed(exc)
        else:
            self._call.stream_ended()
        return await self._manager.__aexit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._manager, name)


class _ChatStreamAccumulator:
    """Rebuilds a ``ChatCompletion``-shaped dict from streamed chunks."""

    def __init__(self) -> None:
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.created: Optional[float] = None
        self.finish_reason: Optional[str] = None
        self.usage: Any = None
        self.content: List[str] = []
        self.refusal: List[str] = []
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.chunks = 0

    def add(self, chunk: Any) -> None:
        self.chunks += 1
        for attr in ("id", "model", "created"):
            value = _get(chunk, attr)
            if value is not None:
                setattr(self, attr, value)
        usage = _get(chunk, "usage")
        if usage is not None:
            self.usage = usage
        choices = _as_list(_get(chunk, "choices"))
        if not choices:
            return
        choice = choices[0]
        delta = _get(choice, "delta")
        content = _get(delta, "content")
        if isinstance(content, str):
            self.content.append(content)
        refusal = _get(delta, "refusal")
        if isinstance(refusal, str):
            self.refusal.append(refusal)
        for tool_call in _as_list(_get(delta, "tool_calls")):
            index = _get(tool_call, "index")
            if not isinstance(index, int):
                index = len(self.tool_calls)
            entry = self.tool_calls.setdefault(
                index, {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
            )
            call_id = _get(tool_call, "id")
            if call_id:
                entry["id"] = call_id
            function = _get(tool_call, "function")
            name = _get(function, "name")
            if name:
                entry["function"]["name"] = name
            arguments = _get(function, "arguments")
            if isinstance(arguments, str):
                entry["function"]["arguments"] += arguments
        finish_reason = _get(choice, "finish_reason")
        if finish_reason:
            self.finish_reason = finish_reason

    def completion(self) -> Dict[str, Any]:
        tool_calls = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self.content) or None,
            "refusal": "".join(self.refusal) or None,
            "tool_calls": tool_calls or None,
        }
        return {
            "id": self.id,
            "object": "chat.completion",
            "model": self.model,
            "created": self.created,
            "choices": [{"index": 0, "finish_reason": self.finish_reason, "message": message}],
            "usage": self.usage,
        }


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _end_span_timed(
    span: Optional[TraceSpan], start: Optional[float], end: Optional[float], **end_kwargs: Any
) -> None:
    """Finish ``span`` stamped with the given timestamps.

    ``tracer.end_span`` → ``span.finish()`` sets ``end_time`` to *now* and
    recomputes ``duration_ms`` from ``start_time``, so the timestamps are
    applied around it: ``start_time`` before, ``end_time`` / ``duration_ms``
    after.
    """
    if span is None:
        return
    if start is not None:
        span.start_time = start
    tracer.end_span(span, **end_kwargs)
    if start is not None and end is not None:
        span.end_time = end
        span.duration_ms = round(max(end - start, 0.0) * 1000, 2)


def _response_end_kwargs(response: Any, status: Optional[str]) -> Dict[str, Any]:
    """``end_span`` keywords for a response's outcome."""
    error = _get(response, "error")
    message = _get(error, "message") if error is not None else None
    if status == "failed" or message:
        return {
            "error": str(message or "response failed"),
            "error_type": str(_get(error, "code") or "ResponseError"),
        }
    if status == "incomplete":
        reason = _get(_get(response, "incomplete_details"), "reason")
        return {"status": "error", "error": str(reason or "incomplete"), "error_type": "Incomplete"}
    if status == "cancelled":
        return {"error": "response cancelled", "error_type": "Cancelled"}
    if status in _RESPONSE_PENDING:
        return {"status": "running"}
    return {}


def _item_end_kwargs(status: Optional[str], name: str) -> Dict[str, Any]:
    """``end_span`` keywords for an output item's ``status``."""
    if status == "failed":
        return {"error": f"{name} failed"}
    if status == "incomplete":
        return {"status": "error", "error": f"{name} incomplete", "error_type": "Incomplete"}
    if status in _ITEM_PENDING:
        return {"status": "running"}
    return {}


def _usage_record(usage: Any) -> Optional[Dict[str, int]]:
    """Token counts from a Responses ``ResponseUsage`` or a Chat ``CompletionUsage``.

    :func:`usage_from_mapping` reads ``input_tokens`` / ``output_tokens``
    and the Chat ``prompt_tokens_details`` / ``completion_tokens_details``
    blocks, but not the Responses spelling ``input_tokens_details.cached_tokens``
    / ``output_tokens_details.reasoning_tokens`` — lifted here.
    """
    mapping = as_mapping(usage)
    if not mapping:
        return None
    found = usage_from_mapping(mapping)
    if found is None:
        return None
    if not found["cached_tokens"]:
        details = as_mapping(mapping.get("input_tokens_details"))
        if details:
            found["cached_tokens"] = _int(details.get("cached_tokens"))
    if not found["reasoning_tokens"]:
        details = as_mapping(mapping.get("output_tokens_details"))
        if details:
            found["reasoning_tokens"] = _int(details.get("reasoning_tokens"))
    total = _int(mapping.get("total_tokens"))
    if total:
        found["total_tokens"] = total
    return found


def _estimate_cost(model: Any, usage: Dict[str, int]) -> Optional[float]:
    """USD for one call from the model catalog; ``None`` when the model is unknown.

    ``input_tokens`` includes the cached ones (OpenAI reports the cache hit
    as a subset), so the cached share is billed at the cache-read rate and
    the rest at the input rate; reasoning tokens are already inside
    ``output_tokens``.
    """
    if not isinstance(model, str) or not model:
        return None
    try:
        from ..model_catalog import get_cost_per_million

        pricing = get_cost_per_million(model)
    except Exception:
        return None
    if not pricing:
        return None
    input_rate = pricing.get("input") or 0.0
    output_rate = pricing.get("output") or 0.0
    cache_rate = pricing.get("cache_read")
    if cache_rate is None:
        cache_rate = input_rate
    cached = min(usage.get("cached_tokens", 0), usage.get("input_tokens", 0))
    fresh = usage.get("input_tokens", 0) - cached
    per_million = 1_000_000.0
    return round(
        (fresh / per_million) * input_rate
        + (cached / per_million) * cache_rate
        + (usage.get("output_tokens", 0) / per_million) * output_rate,
        6,
    )


def _response_text(response: Any) -> Optional[str]:
    """``Response.output_text`` (an SDK property) or the same aggregation by hand."""
    text = _get(response, "output_text")
    if isinstance(text, str) and text:
        return text
    parts: List[str] = []
    for item in _as_list(_get(response, "output")):
        if _get(item, "type") != "message":
            continue
        for part in _as_list(_get(item, "content")):
            if _get(part, "type") == "output_text":
                value = _get(part, "text")
                if isinstance(value, str):
                    parts.append(value)
    return "".join(parts) or None


def _response_refusal(response: Any) -> Optional[str]:
    parts: List[str] = []
    for item in _as_list(_get(response, "output")):
        if _get(item, "type") != "message":
            continue
        for part in _as_list(_get(item, "content")):
            if _get(part, "type") == "refusal":
                value = _get(part, "refusal")
                if isinstance(value, str) and value:
                    parts.append(value)
    return "\n".join(parts) or None


def _conversation_id(response: Any) -> Optional[str]:
    conversation = _get(response, "conversation")
    if isinstance(conversation, str):
        return conversation or None
    value = _get(conversation, "id")
    return value if isinstance(value, str) and value else None


def _join_texts(parts: Any) -> str:
    """Concatenate the ``text`` of reasoning summary / content parts."""
    texts: List[str] = []
    for part in _as_list(parts):
        value = part if isinstance(part, str) else _get(part, "text")
        if isinstance(value, str) and value:
            texts.append(value)
    return "\n".join(texts)


def _file_search_results(results: Any) -> Optional[List[Dict[str, Any]]]:
    output: List[Dict[str, Any]] = []
    for result in _as_list(results):
        entry: Dict[str, Any] = {
            "file_id": _get(result, "file_id"),
            "filename": _get(result, "filename"),
            "score": _get(result, "score"),
        }
        text = _get(result, "text")
        if isinstance(text, str) and text:
            entry["text"] = text
        attributes = _get(result, "attributes")
        if attributes:
            entry["attributes"] = _jsonable(attributes)
        output.append(entry)
    return output or None


def _code_interpreter_outputs(outputs: Any) -> Optional[List[Any]]:
    """Logs as text, images as ``{"type": "image", "url": …}``."""
    result: List[Any] = []
    for item in _as_list(outputs):
        item_type = _get(item, "type")
        if item_type == "image" or _get(item, "url") is not None:
            entry: Dict[str, Any] = {"type": "image", "url": _get(item, "url")}
            file_id = _get(item, "file_id")
            if file_id:
                entry["file_id"] = file_id
            result.append(entry)
        else:
            logs = _get(item, "logs")
            result.append(logs if logs is not None else _jsonable(item))
    return result or None


def _chat_tool_calls(tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
    """``[{id, name, arguments}]`` for Chat Completions ``message.tool_calls``."""
    output: List[Dict[str, Any]] = []
    for call in _as_list(tool_calls):
        function = _get(call, "function")
        if function is not None:
            name = _get(function, "name")
            arguments = _parse_json_maybe(_get(function, "arguments"))
        else:
            custom = _get(call, "custom")
            name = _get(custom, "name")
            arguments = _get(custom, "input")
        output.append({"id": _get(call, "id"), "name": name, "arguments": arguments})
    return output or None


def _tool_output_repr(output: Any) -> Any:
    """A tool result as recorded on the span (JSON decoded when possible;
    screenshots are reduced to their size)."""
    if isinstance(output, str):
        return _parse_json_maybe(output)
    if isinstance(output, (list, tuple)):
        return [_tool_output_repr(part) for part in output]
    if _get(output, "type") == "computer_screenshot":
        entry: Dict[str, Any] = {"type": "computer_screenshot"}
        file_id = _get(output, "file_id")
        if file_id:
            entry["file_id"] = file_id
        image_url = _get(output, "image_url")
        if isinstance(image_url, str):
            entry["image_url"] = (
                f"<data url, {len(image_url)} chars>"
                if image_url.startswith("data:")
                else image_url
            )
        return entry
    return _jsonable(output)


def _output_error(output: Any) -> Optional[str]:
    """The error message when a tool result announces a failure."""
    if isinstance(output, dict):
        error = output.get("error")
        if error:
            return _safe_str(error)
    elif isinstance(output, str) and _ERROR_PREFIX.match(output):
        return output
    return None


def _first_user_text(request_input: Any) -> Optional[str]:
    """Text of the first user message in a Responses ``input`` or a Chat
    ``messages`` list (a plain string *is* the user message)."""
    if isinstance(request_input, str):
        return request_input or None
    for item in _as_list(request_input):
        if isinstance(item, str):
            return item or None
        if _get(item, "role") != "user":
            continue
        if _get(item, "type") not in (None, "message"):
            continue
        text = _message_content_text(_get(item, "content"))
        if text:
            return text
    return None


def _message_content_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content or None
    parts: List[str] = []
    for part in _as_list(content):
        if isinstance(part, str):
            parts.append(part)
            continue
        if _get(part, "type") in (None, "input_text", "text", "output_text"):
            value = _get(part, "text")
            if isinstance(value, str) and value:
                parts.append(value)
    return "\n".join(parts) or None


def _parse_json_maybe(value: Any) -> Any:
    """Function arguments / outputs arrive as JSON strings — decode when possible."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except ValueError:
                return value
    return value


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """A JSON-friendly rendering of request / response fragments."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if _depth > 12:
        return _safe_str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in value]
    mapping = as_mapping(value)
    if mapping is not None:
        return _jsonable(mapping, _depth + 1)
    return _safe_str(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int(value: Any) -> int:
    if _is_int(value):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return 0


def _ts(value: Any) -> Optional[float]:
    """Unix seconds as float from an int/float/datetime, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        try:
            return float(timestamp())
        except Exception:
            return None
    return None


def _coerce_ts(value: Optional[float]) -> Optional[float]:
    return _ts(value)


def _as_list(value: Any) -> List[Any]:
    """Accept a list, a page (``.data``), any iterable, or ``None``."""
    if value is None or isinstance(value, (str, bytes, dict)):
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
    """Attribute or dict key, ``None``-safe."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
