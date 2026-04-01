# eval_lib/tracing/llamaindex_callback.py
"""LlamaIndex trace collector using the Instrumentation module.

Converts LlamaIndex events and spans into eval-lib TraceSpans
for reliability evaluation.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.llamaindex_callback import install_llamaindex_tracing

    trace_id = tracer.start_trace("llamaindex_agent")

    # Install handlers on LlamaIndex's root dispatcher
    install_llamaindex_tracing()

    # Run your LlamaIndex agent
    agent = ReActAgent.from_tools(tools, llm=llm)
    response = agent.chat("query")

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

from typing import Any, Dict, Optional
from .types import TraceSpan, SpanType
from .tracer import tracer


class EvalLibEventHandler:
    """LlamaIndex EventHandler that logs events to eval-lib tracer.

    Implements the llama_index.core.instrumentation.event_handlers.BaseEventHandler
    interface.
    """

    @classmethod
    def class_name(cls) -> str:
        return "EvalLibEventHandler"

    def handle(self, event: Any, **kwargs):
        """Process a LlamaIndex event."""
        event_type = type(event).__name__

        if event_type == "AgentToolCallEvent":
            self._handle_tool_call(event)
        elif event_type == "LLMCompletionStartEvent":
            self._handle_llm_start(event)
        elif event_type == "LLMCompletionEndEvent":
            self._handle_llm_end(event)
        elif event_type == "RetrievalStartEvent":
            self._handle_retrieval_start(event)
        elif event_type == "RetrievalEndEvent":
            self._handle_retrieval_end(event)

    def _handle_tool_call(self, event: Any):
        tool_name = getattr(event, "tool_name", None) or getattr(event, "name", "unknown_tool")
        tool_args = getattr(event, "tool_kwargs", None) or getattr(event, "arguments", {})
        tool_output = getattr(event, "tool_output", None)

        span = tracer.start_span(
            name=str(tool_name),
            span_type=SpanType.TOOL_CALL,
            input_data=tool_args,
        )
        if span:
            tracer.end_span(span, output=str(tool_output) if tool_output else None)

    def _handle_llm_start(self, event: Any):
        model = getattr(event, "model_name", None) or getattr(event, "model", None)
        prompt = getattr(event, "prompt", None) or getattr(event, "messages", None)

        span = tracer.start_span(
            name="llm_call",
            span_type=SpanType.LLM_CALL,
            input_data=str(prompt)[:500] if prompt else None,
            metadata={"model": str(model)} if model else None,
        )
        if span:
            # Store span for matching with end event
            self._current_llm_span = span

    def _handle_llm_end(self, event: Any):
        span = getattr(self, "_current_llm_span", None)
        if span:
            response = getattr(event, "response", None) or getattr(event, "completion", None)
            token_info = getattr(event, "token_usage", None)

            tracer.end_span(span, output=str(response)[:500] if response else None)

            if token_info:
                tracer.set_trace_metadata(
                    input_tokens=getattr(token_info, "prompt_tokens", None),
                    output_tokens=getattr(token_info, "completion_tokens", None),
                    total_tokens=getattr(token_info, "total_tokens", None),
                )
            self._current_llm_span = None

    def _handle_retrieval_start(self, event: Any):
        query = getattr(event, "query", None) or getattr(event, "str_or_query_bundle", None)
        span = tracer.start_span(
            name="retrieval",
            span_type=SpanType.RETRIEVAL,
            input_data=str(query) if query else None,
        )
        if span:
            self._current_retrieval_span = span

    def _handle_retrieval_end(self, event: Any):
        span = getattr(self, "_current_retrieval_span", None)
        if span:
            nodes = getattr(event, "nodes", None)
            output = None
            if nodes:
                output = [getattr(n, "text", str(n))[:200] for n in nodes[:5]]
            tracer.end_span(span, output=output)
            self._current_retrieval_span = None


class EvalLibSpanHandler:
    """LlamaIndex SpanHandler that maps LlamaIndex spans to eval-lib spans.

    Implements the llama_index.core.instrumentation.span_handlers.BaseSpanHandler
    interface.
    """

    def __init__(self):
        self._spans: Dict[str, TraceSpan] = {}

    @classmethod
    def class_name(cls) -> str:
        return "EvalLibSpanHandler"

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
        # Determine span type from tags or instance type
        span_type = SpanType.CUSTOM
        name = "llamaindex_span"

        if tags:
            tag_str = str(tags).lower()
            if "llm" in tag_str:
                span_type = SpanType.LLM_CALL
                name = "llm_call"
            elif "tool" in tag_str:
                span_type = SpanType.TOOL_CALL
                name = tags.get("tool_name", "tool_call")
            elif "retriev" in tag_str:
                span_type = SpanType.RETRIEVAL
                name = "retrieval"
            elif "agent" in tag_str:
                span_type = SpanType.AGENT_STEP
                name = "agent_step"

        if instance:
            instance_type = type(instance).__name__.lower()
            if "agent" in instance_type:
                span_type = SpanType.AGENT_STEP
                name = f"agent:{type(instance).__name__}"

        span = tracer.start_span(
            name=name,
            span_type=span_type,
            metadata=tags,
        )
        if span:
            if parent_span_id and parent_span_id in self._spans:
                span.parent_span_id = self._spans[parent_span_id].span_id
            self._spans[id_] = span

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
        span = self._spans.pop(id_, None)
        if span:
            output = str(result)[:500] if result else None
            tracer.end_span(span, output=output)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any = None,
        instance: Any = None,
        err: Optional[Exception] = None,
        **kwargs,
    ):
        """Called when a LlamaIndex span drops due to error."""
        span = self._spans.pop(id_, None)
        if span:
            tracer.end_span(span, error=err or Exception("Span dropped"))


def install_llamaindex_tracing():
    """Install eval-lib tracing handlers on LlamaIndex's root dispatcher.

    Call this once before running any LlamaIndex agent.
    """
    try:
        import llama_index.core.instrumentation as instrument

        dispatcher = instrument.get_dispatcher()
        dispatcher.add_event_handler(EvalLibEventHandler())
        dispatcher.add_span_handler(EvalLibSpanHandler())
    except ImportError:
        raise ImportError(
            "LlamaIndex is required for this integration. "
            "Install with: pip install llama-index-core"
        )
