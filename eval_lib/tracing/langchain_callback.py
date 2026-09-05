# eval_lib/tracing/langchain_callback.py
"""LangChain / LangGraph callback handler for automatic tracing.

Usage::

    from eval_lib.tracing import tracer
    from eval_lib.tracing.langchain_callback import EvalLibCallbackHandler

    handler = EvalLibCallbackHandler()
    trace_id = tracer.start_trace("my_agent")
    result = await graph.ainvoke(inputs, config={"callbacks": [handler]})
    tracer.end_trace()

Design notes
------------
* **Parentage comes from LangChain, not from the context pointer.** Every
  callback carries ``run_id``/``parent_run_id``. The handler maps
  ``run_id -> TraceSpan`` and passes ``parent_span_id`` explicitly (with
  ``set_current=False``) on every ``start_span``. Relying on the tracer's
  contextvar produced flat traces: for ``ainvoke``/``astream``/LangGraph,
  LangChain runs a non-inline sync handler via
  ``run_in_executor(copy_context().run, ...)`` and every contextvar write is
  discarded with the copied context.

* **``run_inline = True``.** With this flag LangChain calls the (sync)
  handler directly in the event-loop thread instead of a worker thread, so
  the active trace id is visible and events arrive in order. The handler
  only does dictionary bookkeeping and in-memory buffering, so it is safe to
  run inline. There is deliberately no ``AsyncCallbackHandler`` variant: a
  class cannot define both a sync and an async ``on_llm_end``, and an inline
  sync handler already behaves identically for async callers.

* **Tokens are counted once per call.** ``langchain-openai`` reports the same
  usage in ``llm_output.token_usage``, ``message.response_metadata`` *and*
  ``message.usage_metadata``. One source is chosen per call with precedence
  ``usage_metadata > response_metadata > llm_output`` and forwarded to
  :meth:`AgentTracer.add_trace_usage`, which accumulates. The same numbers
  are attached to the LLM span's metadata so span roll-ups agree.

* **Trace-level input/output come from the root run** (``parent_run_id is
  None``): input on its start, output on its end. ``on_agent_finish`` (legacy
  ``AgentExecutor``) still sets the output, but the root chain end is
  authoritative and always wins — a nested agent can no longer freeze the
  outer result.

* **No cross-run state.** The only instance state is ``run_id`` keyed
  bookkeeping, cleaned up when each run ends. Sharing one handler (including
  the module-level ``callback_handler``) between concurrent runs and traces is
  safe.

* **Never breaks the chain.** Every callback is wrapped: an internal error is
  logged at WARNING on ``eval_lib.tracing`` and swallowed.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .context import get_trace_id
from .trace_utils import safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

logger = logging.getLogger("eval_lib.tracing")

#: LangGraph / LangChain tag on internal plumbing runs (``ChannelWrite``,
#: ``RunnableCallable``, ...). Those get no span; their children are attached
#: to the nearest visible ancestor instead.
HIDDEN_TAG = "langsmith:hidden"

# Keys tried, in order, when reducing a chain payload to the user's text.
_INPUT_KEYS = ("input", "question", "query", "prompt", "human_input", "user_input", "messages")
_OUTPUT_KEYS = ("output", "answer", "result", "response", "text", "messages")

_MAX_PARENT_DEPTH = 256
_MAX_TEXT_DEPTH = 8

_ROLE_ALIASES = {"user": "human", "assistant": "ai"}


def _guarded(method: Callable[..., Any]) -> Callable[..., Any]:
    """Log-and-swallow wrapper: a tracing bug must never break the user's chain."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            logger.warning(
                "eval_lib.tracing: LangChain callback %s failed and was ignored: %s: %s",
                method.__name__,
                type(exc).__name__,
                exc,
            )
            return None

    return wrapper


class EvalLibCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that creates spans for LangChain operations.

    Chains/graph nodes become ``AGENT_STEP`` spans, (chat) model calls
    ``LLM_CALL``, tools ``TOOL_CALL`` and retrievers ``RETRIEVAL``. Token
    usage, model name and the trace's input/output are collected without
    any change to the agent code. See the module docstring for the design.
    """

    # Run in the caller's thread/context — required for correct parentage
    # under ainvoke/astream/LangGraph (see module docstring).
    run_inline = True
    raise_error = False

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        # Open spans keyed by LangChain run id.
        self._run_spans: Dict[UUID, TraceSpan] = {}
        # Hidden runs (no span) -> their parent, so descendants re-parent.
        self._hidden_parents: Dict[UUID, Optional[UUID]] = {}
        # Runs started with parent_run_id=None; they own the trace metadata.
        self._root_runs: Set[UUID] = set()

    # ------------------------------------------------------------ bookkeeping

    def _resolve_parent_locked(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        """Span id for ``parent_run_id``, walking through hidden runs. Lock held."""
        run_id = parent_run_id
        for _ in range(_MAX_PARENT_DEPTH):
            if run_id is None:
                return None
            span = self._run_spans.get(run_id)
            if span is not None:
                return span.span_id
            if run_id not in self._hidden_parents:
                return None
            run_id = self._hidden_parents[run_id]
        return None

    def _begin_run(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        tags: Optional[Sequence[str]],
        *,
        name: str,
        span_type: SpanType,
        input_data: Any,
        metadata: Dict[str, Any],
    ) -> Optional[TraceSpan]:
        """Open a span for ``run_id`` under LangChain's reported parent."""
        hidden = _is_hidden(tags)
        with self._lock:
            if parent_run_id is None:
                self._root_runs.add(run_id)
            if hidden:
                self._hidden_parents[run_id] = parent_run_id
                return None
            parent_span_id = self._resolve_parent_locked(parent_run_id)

        span = tracer.start_span(
            name=name,
            span_type=span_type,
            input_data=input_data,
            metadata=metadata,
            parent_span_id=parent_span_id,
            set_current=False,
        )
        if span is not None:
            with self._lock:
                self._run_spans[run_id] = span
        return span

    def _finish_run(self, run_id: UUID) -> Tuple[Optional[TraceSpan], bool]:
        """Forget ``run_id``. Returns ``(span or None, was_root)``."""
        with self._lock:
            span = self._run_spans.pop(run_id, None)
            self._hidden_parents.pop(run_id, None)
            is_root = run_id in self._root_runs
            self._root_runs.discard(run_id)
        return span, is_root

    def _end_with_error(self, run_id: UUID, error: BaseException) -> None:
        span, _ = self._finish_run(run_id)
        if span is None:
            return
        if isinstance(error, Exception):
            tracer.end_span(span, error=error)
        else:
            # CancelledError / KeyboardInterrupt / SystemExit: LangChain reports
            # them too. Keep the real class name as the error type.
            tracer.end_span(
                span,
                error=str(error) or type(error).__name__,
                error_type=type(error).__name__,
            )

    # ============ Chain callbacks ============

    @_guarded
    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Chain / LCEL runnable / LangGraph node started."""
        span_meta = _span_metadata(run_id, tags, metadata)
        run_type = kwargs.get("run_type")
        if isinstance(run_type, str) and run_type:
            span_meta["run_type"] = run_type

        span = self._begin_run(
            run_id,
            parent_run_id,
            tags,
            name=_run_name(kwargs, serialized, default="chain"),
            span_type=SpanType.AGENT_STEP,
            input_data=inputs,
            metadata=span_meta,
        )
        if span is not None and parent_run_id is None:
            text = _text_from_payload(inputs, _INPUT_KEYS, role="human")
            if text:
                _declare_trace_metadata(span.trace_id, input=text)

    @_guarded
    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Chain finished. The root chain's output is the trace output."""
        span, is_root = self._finish_run(run_id)
        if span is None:
            return
        try:
            if is_root:
                text = _text_from_payload(outputs, _OUTPUT_KEYS, role="ai")
                if text:
                    _declare_trace_metadata(span.trace_id, output=text)
        finally:
            tracer.end_span(span, output=outputs)

    @_guarded
    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Chain raised."""
        self._end_with_error(run_id, error)

    # ============ LLM callbacks ============

    def _start_llm_run(
        self,
        serialized: Optional[Dict[str, Any]],
        *,
        input_data: Any,
        input_text_source: Any,
        default_name: str,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        span_meta = _span_metadata(run_id, tags, metadata)
        model = _model_from_start(serialized, metadata, kwargs)
        if model:
            span_meta["model"] = model

        span = self._begin_run(
            run_id,
            parent_run_id,
            tags,
            name=_run_name(kwargs, serialized, default=default_name),
            span_type=SpanType.LLM_CALL,
            input_data=input_data,
            metadata=span_meta,
        )
        if span is None:
            return
        if model:
            _declare_model_once(span.trace_id, model)
        if parent_run_id is None:
            # A bare ``llm.invoke(...)`` — the model call *is* the run.
            text = _text_from_payload(input_text_source, _INPUT_KEYS, role="human")
            if text:
                _declare_trace_metadata(span.trace_id, input=text)

    @_guarded
    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Completion-style LLM started."""
        prompt_list = list(prompts) if prompts is not None else []
        self._start_llm_run(
            serialized,
            input_data={"prompts": prompt_list},
            input_text_source=prompt_list,
            default_name="llm",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    @_guarded
    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List["BaseMessage"]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Chat model started. LangChain emits one event per message list."""
        flat = _flatten_lists(messages if messages is not None else [])
        self._start_llm_run(
            serialized,
            input_data={"messages": [_message_to_dict(m) for m in flat]},
            input_text_source=flat,
            default_name="chat_model",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    @_guarded
    def on_llm_end(self, response: "LLMResult", *, run_id: UUID, **kwargs: Any) -> None:
        """LLM finished: record output, model and (once) this call's usage.

        The span is closed in a ``finally`` so a malformed provider payload
        cannot lose it.
        """
        span, is_root = self._finish_run(run_id)
        if span is None:
            return

        output: Optional[Dict[str, Any]] = None
        try:
            output = _llm_output_payload(response)

            if span.metadata is None:
                span.metadata = {}
            model = _llm_result_model(response)
            if model:
                span.metadata["model"] = model
                _declare_model_once(span.trace_id, model)

            usage = _llm_result_usage(response)
            if usage:
                span.metadata.update(usage)
            # One accumulate per call — even without counts, so ``llm_calls``
            # stays an honest tally.
            tracer.add_trace_usage(trace_id=span.trace_id or None, **(usage or {}))

            if is_root:
                text = _last_generation_text(response)
                if text:
                    _declare_trace_metadata(span.trace_id, output=text)
        finally:
            tracer.end_span(span, output=output)

    @_guarded
    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """LLM raised."""
        self._end_with_error(run_id, error)

    # ============ Tool callbacks ============

    @_guarded
    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Tool started."""
        span_meta = _span_metadata(run_id, tags, metadata)
        tool_call_id = kwargs.get("tool_call_id")
        if tool_call_id:
            span_meta["tool_call_id"] = str(tool_call_id)
        if isinstance(serialized, dict):
            description = serialized.get("description")
            if isinstance(description, str) and description:
                span_meta["description"] = description

        self._begin_run(
            run_id,
            parent_run_id,
            tags,
            name=_run_name(kwargs, serialized, default="tool"),
            span_type=SpanType.TOOL_CALL,
            input_data=inputs if isinstance(inputs, dict) else {"input": input_str},
            metadata=span_meta,
        )

    @_guarded
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Tool finished. A ``ToolMessage`` with ``status="error"`` is recorded as failed."""
        span, _ = self._finish_run(run_id)
        if span is None:
            return
        payload: Any = None
        error: Optional[str] = None
        try:
            payload, error = _tool_output(output)
        finally:
            tracer.end_span(span, output=payload, error=error)

    @_guarded
    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Tool raised."""
        self._end_with_error(run_id, error)

    # ============ Agent callbacks ============

    @_guarded
    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Agent chose a tool. The tool call itself arrives via ``on_tool_start``."""
        return None

    @_guarded
    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Legacy ``AgentExecutor`` finished.

        ``run_id`` is the executor's own chain run. The output is declared
        now, and overwritten by the root chain end if that is a different,
        outer run.
        """
        text = _agent_finish_text(finish)
        if not text:
            return
        with self._lock:
            span = self._run_spans.get(run_id)
        trace_id = span.trace_id if span is not None else get_trace_id()
        _declare_trace_metadata(trace_id, output=text)

    # ============ Retriever callbacks ============

    @_guarded
    def on_retriever_start(
        self,
        serialized: Optional[Dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Retriever started."""
        self._begin_run(
            run_id,
            parent_run_id,
            tags,
            name=_run_name(kwargs, serialized, default="retriever"),
            span_type=SpanType.RETRIEVAL,
            input_data={"query": query},
            metadata=_span_metadata(run_id, tags, metadata),
        )

    @_guarded
    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        """Retriever finished."""
        span, _ = self._finish_run(run_id)
        if span is None:
            return
        payload: Optional[Dict[str, Any]] = None
        try:
            payload = {"documents": [_document_text(doc) for doc in (documents or [])]}
        finally:
            tracer.end_span(span, output=payload)

    @_guarded
    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Retriever raised."""
        self._end_with_error(run_id, error)


# ---------------------------------------------------------------------------
# Trace-level metadata
# ---------------------------------------------------------------------------


def _declare_trace_metadata(trace_id: Optional[str], **fields: Any) -> None:
    """Declare trace facts on the span's own trace, whatever the current context.

    Goes through :meth:`AgentTracer.set_trace_metadata` when the target is the
    active trace; otherwise (callback delivered in another context) straight
    to the sender by id.
    """
    fields = {key: value for key, value in fields.items() if value is not None}
    if not fields or not tracer.enabled or tracer.sender is None:
        return
    if trace_id and trace_id != get_trace_id():
        tracer.sender.set_trace_metadata(trace_id, fields)
    else:
        tracer.set_trace_metadata(**fields)


def _declare_model_once(trace_id: Optional[str], model: str) -> None:
    """Record the trace's model the first time one is seen (never overwrite)."""
    if not trace_id or not tracer.enabled or tracer.sender is None:
        return
    if tracer.sender.get_trace_metadata(trace_id).get("model"):
        return
    _declare_trace_metadata(trace_id, model=model)


# ---------------------------------------------------------------------------
# Run naming / metadata
# ---------------------------------------------------------------------------


def _is_hidden(tags: Optional[Sequence[str]]) -> bool:
    if not tags:
        return False
    try:
        return HIDDEN_TAG in tags
    except TypeError:
        return False


def _run_name(kwargs: Dict[str, Any], serialized: Optional[Dict[str, Any]], *, default: str) -> str:
    """``kwargs["name"]`` (LCEL/LangGraph pass ``serialized=None``) > serialized name > class id."""
    name = kwargs.get("name")
    if isinstance(name, str) and name:
        return name
    if isinstance(serialized, dict):
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        ident = serialized.get("id")
        if isinstance(ident, (list, tuple)) and ident and isinstance(ident[-1], str):
            return ident[-1]
    return default


def _span_metadata(
    run_id: UUID, tags: Optional[Sequence[str]], metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        meta.update(metadata)
    if tags:
        meta["tags"] = list(tags)
    meta["run_id"] = str(run_id)
    return meta


def _model_from_start(
    serialized: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> Optional[str]:
    """Model name at call start: LangSmith params > invocation params > serialized kwargs."""
    candidates: List[Any] = []
    if isinstance(metadata, dict):
        candidates.append(metadata.get("ls_model_name"))
    params = as_mapping(kwargs.get("invocation_params"))
    if params:
        candidates.extend(params.get(key) for key in ("model_name", "model", "model_id"))
    if isinstance(serialized, dict):
        serialized_kwargs = as_mapping(serialized.get("kwargs"))
        if serialized_kwargs:
            candidates.extend(
                serialized_kwargs.get(key) for key in ("model_name", "model", "model_id")
            )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _is_message(obj: Any) -> bool:
    """Duck-typed LangChain message: BaseMessage, ``{"role"/"type", "content"}`` or ``(role, text)``."""
    if obj is None or isinstance(obj, (str, bytes, bool, int, float, list)):
        return False
    if isinstance(obj, tuple):
        return len(obj) == 2 and isinstance(obj[0], str)
    if isinstance(obj, dict):
        return "content" in obj and ("role" in obj or "type" in obj)
    return hasattr(obj, "content") and (hasattr(obj, "type") or hasattr(obj, "role"))


def _message_role(msg: Any) -> Optional[str]:
    if isinstance(msg, tuple):
        role: Any = msg[0]
    else:
        role = _attr(msg, "type") or _attr(msg, "role")
    if not isinstance(role, str):
        return None
    normalised: str = role.lower()
    if normalised.endswith("messagechunk"):  # AIMessageChunk -> ai
        normalised = normalised[: -len("messagechunk")]
    return _ROLE_ALIASES.get(normalised, normalised)


def _message_content(msg: Any) -> Any:
    if isinstance(msg, tuple):
        return msg[1]
    return _attr(msg, "content")


def _content_text(content: Any) -> Optional[str]:
    """Plain text of a message content — a string or a list of content blocks."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and block.get("type") in (None, "text"):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) if parts else safe_str(content)
    return safe_str(content)


def _message_tool_calls(msg: Any) -> List[Any]:
    calls = _attr(msg, "tool_calls")
    if not calls:
        extra = as_mapping(_attr(msg, "additional_kwargs"))
        calls = extra.get("tool_calls") if extra else None
    if not calls or isinstance(calls, (str, bytes)):
        return []
    result: List[Any] = []
    for call in calls:
        mapping = as_mapping(call)
        result.append(dict(mapping) if mapping else safe_str(call))
    return result


def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """Serializable view of a message: role, content, and tool-call linkage."""
    if not _is_message(msg):
        return {"type": type(msg).__name__, "content": safe_str(msg)}
    data: Dict[str, Any] = {"type": _message_role(msg), "content": _message_content(msg)}
    name = _attr(msg, "name")
    if isinstance(name, str) and name:
        data["name"] = name
    tool_calls = _message_tool_calls(msg)
    if tool_calls:
        data["tool_calls"] = tool_calls
    tool_call_id = _attr(msg, "tool_call_id")
    if tool_call_id:
        data["tool_call_id"] = str(tool_call_id)
    return data


def _flatten_lists(items: Any) -> List[Any]:
    """Flatten nested *lists* (``List[List[BaseMessage]]``). Tuples stay — they may be messages."""
    if not isinstance(items, list):
        return [items]
    flat: List[Any] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(_flatten_lists(item))
        else:
            flat.append(item)
    return flat


def _text_from_payload(
    value: Any, keys: Sequence[str], *, role: str, depth: int = 0
) -> Optional[str]:
    """Reduce a chain input/output to the user-facing text.

    Strings pass through; messages give their content; dicts are searched for
    ``keys`` in order; message lists yield the last message with ``role``
    (``"human"`` for inputs, ``"ai"`` for outputs). Anything else is
    stringified.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if depth > _MAX_TEXT_DEPTH:
        return safe_str(value)
    if _is_message(value):
        return _content_text(_message_content(value))
    if isinstance(value, dict):
        for key in keys:
            if value.get(key) is not None:
                text = _text_from_payload(value[key], keys, role=role, depth=depth + 1)
                if text is not None:
                    return text
        return safe_str(value)
    if isinstance(value, (list, tuple)):
        items = _flatten_lists(list(value))
        messages = [item for item in items if _is_message(item)]
        if messages:
            for msg in reversed(messages):
                if _message_role(msg) == role:
                    return _content_text(_message_content(msg))
            return _content_text(_message_content(messages[-1]))
        if len(items) == 1:
            return _text_from_payload(items[0], keys, role=role, depth=depth + 1)
        return safe_str(value)
    return safe_str(value)


# ---------------------------------------------------------------------------
# LLMResult digestion
# ---------------------------------------------------------------------------


def _flatten_generations(response: Any) -> List[Any]:
    generations = getattr(response, "generations", None) or []
    flat: List[Any] = []
    for group in generations:
        if isinstance(group, (list, tuple)):
            flat.extend(group)
        elif group is not None:
            flat.append(group)
    return flat


def _generation_text(gen: Any) -> Optional[str]:
    text = getattr(gen, "text", None)
    if isinstance(text, str) and text:
        return text
    message = getattr(gen, "message", None)
    if message is not None:
        return _content_text(_message_content(message))
    return text if isinstance(text, str) else None


def _last_generation_text(response: Any) -> Optional[str]:
    for gen in reversed(_flatten_generations(response)):
        text = _generation_text(gen)
        if text:
            return text
    return None


def _generation_to_dict(gen: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {"text": _generation_text(gen)}
    message = getattr(gen, "message", None)
    if message is not None:
        data["message"] = _message_to_dict(message)
    info = as_mapping(getattr(gen, "generation_info", None))
    if info and info.get("finish_reason") is not None:
        data["finish_reason"] = info["finish_reason"]
    return data


def _llm_output_payload(response: Any) -> Dict[str, Any]:
    """Span output for an LLM call. Token counts are kept out of it — they
    live in the span metadata, so the roll-up has exactly one source."""
    generations: List[List[Dict[str, Any]]] = []
    for group in getattr(response, "generations", None) or []:
        items = group if isinstance(group, (list, tuple)) else [group]
        generations.append([_generation_to_dict(gen) for gen in items])
    payload: Dict[str, Any] = {"generations": generations}

    llm_output = as_mapping(getattr(response, "llm_output", None))
    if llm_output:
        trimmed = {k: v for k, v in llm_output.items() if k not in ("token_usage", "usage")}
        if trimmed:
            payload["llm_output"] = trimmed
    return payload


def _normalise_usage(candidate: Any) -> Optional[Dict[str, int]]:
    """Token record from one usage-shaped payload, or ``None``.

    LangChain's ``usage_metadata`` nests cache/reasoning counts under
    ``input_token_details.cache_read`` / ``output_token_details.reasoning``;
    they are lifted to the keys :func:`usage_from_mapping` understands.
    ``None`` values are ignored by the helper, never compared.
    """
    mapping = as_mapping(candidate)
    if not mapping:
        return None
    mapping = dict(mapping)
    input_details = as_mapping(mapping.get("input_token_details"))
    if input_details and mapping.get("cached_tokens") is None:
        mapping["cached_tokens"] = _as_int(input_details.get("cache_read"))
    output_details = as_mapping(mapping.get("output_token_details"))
    if output_details and mapping.get("reasoning_tokens") is None:
        mapping["reasoning_tokens"] = _as_int(output_details.get("reasoning"))
    return usage_from_mapping(mapping)


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _sum_usage(records: Sequence[Dict[str, int]]) -> Dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    for record in records:
        for key in total:
            total[key] += int(record.get(key) or 0)
    return total


def _llm_result_usage(response: Any) -> Optional[Dict[str, int]]:
    """This call's usage from exactly one source.

    Precedence: ``message.usage_metadata`` > ``message.response_metadata``
    (``token_usage``/``usage``) > ``llm_output.token_usage``.

    With ``n > 1`` candidates, per-generation ``usage_metadata`` is summed
    only when each generation carries its *own distinct* record. Providers
    such as OpenAI and Gemini stamp the call's total on every candidate;
    identical records are therefore counted once.
    """
    generations = _flatten_generations(response)

    per_generation = [
        _normalise_usage(getattr(getattr(gen, "message", None), "usage_metadata", None))
        for gen in generations
    ]
    found = [usage for usage in per_generation if usage]
    if found:
        if len(generations) > 1 and len(found) == len(generations):
            distinct = {tuple(sorted(usage.items())) for usage in found}
            if len(distinct) > 1:
                return _sum_usage(found)
        return found[0]

    for gen in generations:
        usage = _normalise_usage(getattr(getattr(gen, "message", None), "response_metadata", None))
        if usage:
            return usage

    return _normalise_usage(getattr(response, "llm_output", None))


def _llm_result_model(response: Any) -> Optional[str]:
    """Model reported by the provider: ``llm_output`` > ``response_metadata``."""
    llm_output = as_mapping(getattr(response, "llm_output", None)) or {}
    for key in ("model_name", "model"):
        value = llm_output.get(key)
        if isinstance(value, str) and value:
            return value
    for gen in _flatten_generations(response):
        response_metadata = as_mapping(
            getattr(getattr(gen, "message", None), "response_metadata", None)
        )
        if not response_metadata:
            continue
        for key in ("model_name", "model"):
            value = response_metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return None


# ---------------------------------------------------------------------------
# Tools / agents / retrievers
# ---------------------------------------------------------------------------


def _tool_output(output: Any) -> Tuple[Any, Optional[str]]:
    """``(span output, in-band error)`` for a tool result."""
    if _is_message(output):
        text = _content_text(_message_content(output))
        if _attr(output, "status") == "error":
            return text, text or "tool reported status=error"
        return text, None
    return output, None


def _agent_finish_text(finish: Any) -> Optional[str]:
    return_values = getattr(finish, "return_values", None)
    if return_values is not None:
        return _text_from_payload(return_values, _OUTPUT_KEYS, role="ai")
    log = getattr(finish, "log", None)
    return log if isinstance(log, str) and log else None


def _document_text(doc: Any) -> Optional[str]:
    content = _attr(doc, "page_content")
    if isinstance(content, str):
        return content
    return safe_str(doc)


# Shared instance for convenience. Safe to reuse across runs and traces: the
# handler keeps only per-run bookkeeping that is dropped when each run ends.
callback_handler = EvalLibCallbackHandler()
