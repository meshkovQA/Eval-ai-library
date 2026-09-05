# eval_lib/tracing/llamaindex_callback.py
"""LlamaIndex trace collector using the Instrumentation module.

Converts LlamaIndex events and spans into eval-lib TraceSpans
for reliability evaluation.

Covers both generations of the LlamaIndex agent API:

* **Workflow API** (0.12+, ``AgentWorkflow`` / ``FunctionAgent``) — LLM calls
  go through the *chat* path (``LLMChatStartEvent``/``LLMChatEndEvent``), and
  tool calls are never announced as ``AgentToolCallEvent``. Tool coverage
  therefore comes from the span handler, which sees every ``@dispatcher.span``
  decorated method — ``FunctionTool.acall`` included.
* **Legacy API** (``ReActAgent``/``OpenAIAgent`` and friends) — keeps working
  through ``AgentToolCallEvent`` and the completion events.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.llamaindex_callback import install_llamaindex_tracing

    trace_id = tracer.start_trace("llamaindex_agent")

    # Install handlers on LlamaIndex's root dispatcher
    install_llamaindex_tracing()

    # Run your LlamaIndex agent (workflow or legacy)
    agent = FunctionAgent(tools=tools, llm=llm)
    response = await agent.run("query")

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from .types import TraceSpan, SpanType
from .tracer import tracer
from .trace_utils import safe_str
from .usage import usage_from_response

# ---------------------------------------------------------------------------
# Event names. Matched by class name so a llama-index upgrade that moves a
# class between modules doesn't silently disable tracing.
# ---------------------------------------------------------------------------

# The chat path — what FunctionAgent / AgentWorkflow actually use.
_LLM_START_EVENTS = {
    "LLMChatStartEvent",
    "LLMCompletionStartEvent",
    "LLMPredictStartEvent",
    "LLMStructuredPredictStartEvent",
}
_LLM_END_EVENTS = {
    "LLMChatEndEvent",
    "LLMCompletionEndEvent",
    "LLMPredictEndEvent",
    "LLMStructuredPredictEndEvent",
}

# Qualname fragments used to classify a dispatcher span. LlamaIndex builds
# span ids as f"{func.__qualname__}-{uuid}", so the prefix identifies the
# instrumented method even when no event is emitted for it.
_TOOL_QUALNAMES = ("functiontool.", "queryenginetool.", "basetool.", "toolspec.")
_LLM_QUALNAMES = (".chat", ".achat", ".complete", ".acomplete", ".predict", ".apredict",
                  ".stream_chat", ".astream_chat")
_RETRIEVAL_QUALNAMES = ("retriever.retrieve", "retriever.aretrieve", ".retrieve", ".aretrieve")
_AGENT_QUALNAMES = ("agentworkflow.", "functionagent.", "reactagent.", "agentrunner.",
                    "workflow.run", ".take_step", ".run_step")


# Dispatcher spans currently open, keyed by LlamaIndex span id. The event
# handler consults this to *enrich* the span the span handler already opened
# instead of creating a second one: LlamaIndex decorates `OpenAI.achat` with
# @dispatcher.span **and** emits LLMChatStartEvent from inside it, and an
# event's `span_id` is exactly the enclosing dispatcher span's id.
_ACTIVE_SPANS: "OrderedDict[str, TraceSpan]" = OrderedDict()
_ACTIVE_LOCK = threading.Lock()
# Spans normally leave on span_exit/span_drop; the cap only matters when a
# run is torn down without them and would otherwise pin entries for the
# life of the process.
_ACTIVE_MAX = 4096


def _register_active(id_: str, span: TraceSpan) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_SPANS[id_] = span
        _ACTIVE_SPANS.move_to_end(id_)
        while len(_ACTIVE_SPANS) > _ACTIVE_MAX:
            _ACTIVE_SPANS.popitem(last=False)


def _unregister_active(id_: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_SPANS.pop(id_, None)


def _active_span(id_: Optional[str]) -> Optional[TraceSpan]:
    if not id_:
        return None
    with _ACTIVE_LOCK:
        return _ACTIVE_SPANS.get(id_)


def _qualname_from_span_id(id_: str) -> str:
    """Extract the instrumented method's qualname from a LlamaIndex span id.

    Ids look like ``"FunctionTool.acall-3f2a…"``; everything before the last
    ``-`` that starts the uuid is the qualname.
    """
    if not id_:
        return ""
    head = id_.rsplit("-", 5)[0] if id_.count("-") >= 5 else id_.split("-", 1)[0]
    return head


class EvalLibEventHandler:
    """LlamaIndex EventHandler that logs events to eval-lib tracer.

    Implements the llama_index.core.instrumentation.event_handlers.BaseEventHandler
    interface.

    In-flight spans are keyed by the event's ``span_id`` rather than held in a
    single attribute: one handler instance is shared by every concurrent
    request, so a single ``_current_llm_span`` field would be overwritten by
    whichever call started most recently and close the wrong span.
    """

    @classmethod
    def class_name(cls) -> str:
        return "EvalLibEventHandler"

    def __init__(self) -> None:
        self._llm_spans: Dict[str, TraceSpan] = {}
        self._retrieval_spans: Dict[str, TraceSpan] = {}
        self._lock = threading.Lock()

    def handle(self, event: Any, **kwargs):
        """Process a LlamaIndex event."""
        event_type = type(event).__name__

        if event_type in _LLM_START_EVENTS:
            self._handle_llm_start(event)
        elif event_type in _LLM_END_EVENTS:
            self._handle_llm_end(event)
        elif event_type == "AgentToolCallEvent":
            self._handle_tool_call(event)
        elif event_type == "RetrievalStartEvent":
            self._handle_retrieval_start(event)
        elif event_type == "RetrievalEndEvent":
            self._handle_retrieval_end(event)

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _key(event: Any) -> str:
        """Correlate start/end events. ``span_id`` is per-call; fall back to
        the event's own id so a missing span_id doesn't collapse every
        concurrent call onto one key."""
        return str(getattr(event, "span_id", None) or getattr(event, "id_", "") or "default")

    # --------------------------------------------------------------- LLM

    def _begin(
        self,
        event: Any,
        registry: Dict[str, TraceSpan],
        name: str,
        span_type: SpanType,
        input_data: Optional[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Enrich the enclosing dispatcher span, or open one if there is none.

        Creating a span unconditionally would double-count every LLM call
        whenever the span handler is installed, and leave the duplicate
        unparented.
        """
        existing = _active_span(getattr(event, "span_id", None))
        if existing is not None:
            existing.name = name
            existing.span_type = span_type
            if input_data is not None:
                existing.input = input_data
            if metadata:
                existing.metadata = {**(existing.metadata or {}), **metadata}
            return

        span = tracer.start_span(
            name=name,
            span_type=span_type,
            input_data=input_data,
            metadata=metadata or None,
        )
        if span:
            with self._lock:
                registry[self._key(event)] = span

    def _complete(
        self,
        event: Any,
        registry: Dict[str, TraceSpan],
        output: Optional[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach the result, closing the span only if we opened it."""
        existing = _active_span(getattr(event, "span_id", None))
        if existing is not None:
            if output is not None:
                existing.output = output
            if metadata:
                existing.metadata = {**(existing.metadata or {}), **metadata}
            return  # the span handler owns the lifecycle

        with self._lock:
            span = registry.pop(self._key(event), None)
        if not span:
            return
        if metadata:
            span.metadata = {**(span.metadata or {}), **metadata}
        tracer.end_span(span, output=output)

    def _handle_llm_start(self, event: Any):
        model = (
            getattr(event, "model_dict", None)
            or getattr(event, "model_name", None)
            or getattr(event, "model", None)
        )
        # Chat path carries `messages`; completion path carries `prompt`.
        payload = getattr(event, "messages", None)
        if payload is None:
            payload = getattr(event, "prompt", None)

        metadata: Dict[str, Any] = {}
        if isinstance(model, dict):
            name = model.get("model") or model.get("model_name")
            if name:
                metadata["model"] = str(name)
        elif model:
            metadata["model"] = str(model)

        self._begin(
            event,
            self._llm_spans,
            name="llm_call",
            span_type=SpanType.LLM_CALL,
            input_data=safe_str(payload) if payload is not None else None,
            metadata=metadata,
        )

    def _handle_llm_end(self, event: Any):
        response = getattr(event, "response", None) or getattr(event, "completion", None)

        # ChatResponse.additional_kwargs is already populated by
        # llama_index's _get_response_token_counts() — no need to re-parse raw.
        usage = usage_from_response(response)

        self._complete(
            event,
            self._llm_spans,
            output=safe_str(response) if response is not None else None,
            metadata=usage,
        )

        if usage:
            # Accumulate — one LLMChatEndEvent per API call. set_trace_metadata
            # would overwrite, leaving only the last call's counts.
            tracer.add_trace_usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
            )

    # -------------------------------------------------------------- tools

    def _handle_tool_call(self, event: Any):
        """Legacy-API tool call. The workflow API never emits this event —
        those tool calls are captured by :class:`EvalLibSpanHandler`."""
        tool_name = getattr(event, "tool_name", None) or getattr(event, "name", "unknown_tool")
        tool_args = getattr(event, "tool_kwargs", None) or getattr(event, "arguments", {})
        tool_output = getattr(event, "tool_output", None)

        existing = _active_span(getattr(event, "span_id", None))
        if existing is not None:
            existing.name = str(tool_name)
            existing.span_type = SpanType.TOOL_CALL
            if tool_args:
                existing.input = tool_args
            if tool_output is not None:
                existing.output = safe_str(tool_output)
            return

        span = tracer.start_span(
            name=str(tool_name),
            span_type=SpanType.TOOL_CALL,
            input_data=tool_args,
        )
        if span:
            tracer.end_span(span, output=safe_str(tool_output) if tool_output else None)

    # ---------------------------------------------------------- retrieval

    def _handle_retrieval_start(self, event: Any):
        query = getattr(event, "query", None) or getattr(event, "str_or_query_bundle", None)
        self._begin(
            event,
            self._retrieval_spans,
            name="retrieval",
            span_type=SpanType.RETRIEVAL,
            input_data=safe_str(query) if query else None,
            metadata=None,
        )

    def _handle_retrieval_end(self, event: Any):
        nodes = getattr(event, "nodes", None)
        output = None
        if nodes:
            output = [safe_str(getattr(n, "text", n)) for n in nodes]
        self._complete(event, self._retrieval_spans, output=output)


class EvalLibSpanHandler:
    """LlamaIndex SpanHandler that maps LlamaIndex spans to eval-lib spans.

    Implements the llama_index.core.instrumentation.span_handlers.BaseSpanHandler
    interface.

    This is the component that makes the workflow API observable: every method
    decorated with ``@dispatcher.span`` opens one here, including
    ``FunctionTool.acall``, which the workflow agents use for tool execution
    and which emits no event of its own.

    Parentage comes exclusively from LlamaIndex's own ``parent_span_id``. The
    handler does **not** move the tracer's context pointer — letting both
    mechanisms run at once (start_span moving the contextvar, then overwriting
    ``parent_span_id`` by hand) produced contradictory trees under concurrency.
    """

    def __init__(self):
        self._spans: Dict[str, TraceSpan] = {}
        self._lock = threading.Lock()
        # BaseSpanHandler exposes these; keep them so code that introspects a
        # span handler (or a version that reads them) still works.
        self.open_spans: Dict[str, TraceSpan] = self._spans

    @classmethod
    def class_name(cls) -> str:
        return "EvalLibSpanHandler"

    # -- Dispatcher entry points -------------------------------------------
    # The dispatcher calls span_enter/span_exit/span_drop, NOT new_span
    # directly — BaseSpanHandler normally provides them. Implementing them
    # here keeps the handler working as a duck type, without which every
    # dispatcher span raised AttributeError and no span was ever recorded.

    def span_enter(
        self,
        *,
        id_: str,
        bound_args: Any = None,
        instance: Any = None,
        parent_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if id_ in self._spans:
            return
        self.new_span(
            id_=id_,
            bound_args=bound_args,
            instance=instance,
            parent_span_id=parent_id,
            tags=tags,
            **kwargs,
        )

    def span_exit(
        self,
        *,
        id_: str,
        bound_args: Any = None,
        instance: Any = None,
        result: Any = None,
        **kwargs,
    ) -> None:
        self.prepare_to_exit_span(
            id_=id_, bound_args=bound_args, instance=instance, result=result, **kwargs
        )

    def span_drop(
        self,
        *,
        id_: str,
        bound_args: Any = None,
        instance: Any = None,
        err: Optional[BaseException] = None,
        **kwargs,
    ) -> None:
        self.prepare_to_drop_span(
            id_=id_, bound_args=bound_args, instance=instance, err=err, **kwargs
        )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _classify(id_: str, instance: Any, tags: Optional[Dict[str, Any]]) -> tuple:
        """Return ``(SpanType, name)`` for a LlamaIndex span."""
        qualname = _qualname_from_span_id(id_)
        lowered = qualname.lower()

        if any(frag in lowered for frag in _TOOL_QUALNAMES):
            return SpanType.TOOL_CALL, _tool_name(instance, qualname)
        if any(frag in lowered for frag in _RETRIEVAL_QUALNAMES):
            return SpanType.RETRIEVAL, "retrieval"
        if any(frag in lowered for frag in _AGENT_QUALNAMES):
            return SpanType.AGENT_STEP, qualname or "agent_step"
        if any(frag in lowered for frag in _LLM_QUALNAMES):
            return SpanType.LLM_CALL, "llm_call"

        # Fall back to the older tag/instance heuristics.
        if tags:
            tag_str = str(tags).lower()
            if "tool" in tag_str:
                return SpanType.TOOL_CALL, str(tags.get("tool_name", "tool_call"))
            if "retriev" in tag_str:
                return SpanType.RETRIEVAL, "retrieval"
            if "llm" in tag_str:
                return SpanType.LLM_CALL, "llm_call"
            if "agent" in tag_str:
                return SpanType.AGENT_STEP, "agent_step"

        if instance is not None:
            instance_type = type(instance).__name__
            lowered_instance = instance_type.lower()
            if "tool" in lowered_instance:
                return SpanType.TOOL_CALL, _tool_name(instance, instance_type)
            if "agent" in lowered_instance or "workflow" in lowered_instance:
                return SpanType.AGENT_STEP, f"agent:{instance_type}"
            if "retriev" in lowered_instance:
                return SpanType.RETRIEVAL, "retrieval"

        return SpanType.CUSTOM, qualname or "llamaindex_span"

    def new_span(
        self,
        id_: str,
        bound_args: Any = None,
        instance: Any = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Called when LlamaIndex creates a new span."""
        span_type, name = self._classify(id_, instance, tags)

        # Resolve the parent through LlamaIndex's own id mapping. `None` here
        # is meaningful (a root span), so it is passed explicitly rather than
        # letting the tracer fall back to the ambient context.
        with self._lock:
            parent = self._spans.get(parent_span_id) if parent_span_id else None
        resolved_parent = parent.span_id if parent else None

        span = tracer.start_span(
            name=name,
            span_type=span_type,
            input_data=_bound_args_input(bound_args, span_type),
            metadata=dict(tags) if tags else None,
            parent_span_id=resolved_parent,
            set_current=False,
        )
        if span:
            with self._lock:
                self._spans[id_] = span
            # Publish it so the event handler enriches this span instead of
            # opening a duplicate for the same call.
            _register_active(id_, span)

        return id_

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any = None,
        instance: Any = None,
        result: Any = None,
        **kwargs,
    ):
        """Called when a LlamaIndex span exits successfully."""
        with self._lock:
            span = self._spans.pop(id_, None)
        _unregister_active(id_)
        if not span:
            return

        # A tool that reports failure in-band is an error, not a success.
        error = _tool_error(result)
        # Keep whatever the event handler already attached (e.g. the parsed
        # ChatResponse) when the raw dispatcher result adds nothing.
        output = safe_str(result) if result is not None else span.output
        tracer.end_span(span, output=output, error=error)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any = None,
        instance: Any = None,
        err: Optional[BaseException] = None,
        **kwargs,
    ):
        """Called when a LlamaIndex span drops due to error."""
        with self._lock:
            span = self._spans.pop(id_, None)
        _unregister_active(id_)
        if not span:
            return
        tracer.end_span(span, error=err or "Span dropped")


def _tool_name(instance: Any, fallback: str) -> str:
    """Best-effort tool name from a FunctionTool/BaseTool instance."""
    metadata = getattr(instance, "metadata", None)
    for source in (metadata, instance):
        if source is None:
            continue
        name = getattr(source, "name", None)
        if name:
            return str(name)
    return fallback or "tool_call"


def _bound_args_input(bound_args: Any, span_type: SpanType) -> Optional[Any]:
    """Extract call arguments as span input.

    ``bound_args`` is an ``inspect.BoundArguments``; ``self`` is dropped since
    it is the instance, not an argument.
    """
    arguments = getattr(bound_args, "arguments", None)
    if not arguments:
        return None
    try:
        payload = {k: v for k, v in arguments.items() if k != "self"}
    except Exception:
        return None
    if not payload:
        return None
    return safe_str(payload)


def _tool_error(result: Any) -> Optional[str]:
    """Return an error message when a tool output signals failure in-band."""
    if result is None:
        return None
    if getattr(result, "is_error", False):
        content = getattr(result, "content", None) or getattr(result, "raw_output", None)
        return safe_str(content) or "tool reported is_error=True"
    return None


def install_llamaindex_tracing():
    """Install eval-lib tracing handlers on LlamaIndex's root dispatcher.

    Call this once before running any LlamaIndex agent. Installing twice is
    a no-op — repeated calls would otherwise duplicate every span.
    """
    try:
        import llama_index.core.instrumentation as instrument
    except ImportError:
        raise ImportError(
            "LlamaIndex is required for this integration. "
            "Install with: pip install llama-index-core"
        )

    dispatcher = instrument.get_dispatcher()

    if not any(
        type(h).__name__ == "EvalLibEventHandler"
        for h in getattr(dispatcher, "event_handlers", [])
    ):
        dispatcher.add_event_handler(EvalLibEventHandler())

    if not any(
        type(h).__name__ == "EvalLibSpanHandler"
        for h in getattr(dispatcher, "span_handlers", [])
    ):
        dispatcher.add_span_handler(EvalLibSpanHandler())
