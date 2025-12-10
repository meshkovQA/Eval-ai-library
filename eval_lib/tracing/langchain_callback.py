# eval_lib/tracing/langchain_callback.py
"""LangChain callback handler for automatic tracing"""

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from .tracer import tracer
from .types import SpanType

logger = logging.getLogger(__name__)


class EvalLibCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that creates spans for all LangChain operations.

    Automatically collects and sets trace-level metadata (model, tokens, output)
    without requiring any changes to the agent code.
    """

    def __init__(self):
        super().__init__()
        # Map run_id -> span for tracking nested spans
        self._run_spans: Dict[UUID, Any] = {}
        # Aggregated token usage for the trace
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._model_name: Optional[str] = None
        self._user_input: Optional[str] = None
        self._final_output: Optional[str] = None
        # Track if metadata was already set for this trace
        self._metadata_set: bool = False

    def _reset_state(self):
        """Reset state for a new trace"""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._model_name = None
        self._user_input = None
        self._final_output = None
        self._metadata_set = False

    def _set_trace_metadata_once(self, output: Optional[str] = None):
        """Set trace metadata (only once per trace)"""
        if self._metadata_set:
            return

        if output is not None:
            self._final_output = output

        # Set trace-level metadata
        tracer.set_trace_metadata(
            model=self._model_name,
            input=self._user_input,
            output=self._final_output,
            input_tokens=self._total_input_tokens if self._total_input_tokens > 0 else None,
            output_tokens=self._total_output_tokens if self._total_output_tokens > 0 else None,
            total_tokens=(self._total_input_tokens + self._total_output_tokens) if (self._total_input_tokens + self._total_output_tokens) > 0 else None
        )
        self._metadata_set = True

    # ============ Chain callbacks ============

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts running"""
        # Reset state at the start of a new top-level chain
        if parent_run_id is None:
            self._reset_state()
            # Capture user input from the first chain
            if inputs and self._user_input is None:
                # Try common input field names
                user_input = (
                    inputs.get('input') or
                    inputs.get('question') or
                    inputs.get('query') or
                    inputs.get('prompt') or
                    inputs.get('human_input') or
                    inputs.get('user_input')
                )
                if user_input and isinstance(user_input, str):
                    self._user_input = user_input

        if serialized is None:
            name = "Chain"
        else:
            name = serialized.get("name") or serialized.get("id", ["Unknown"])[-1]

        span = tracer.start_span(
            name=name,
            span_type=SpanType.AGENT_STEP,
            input_data=inputs,
            metadata={"tags": tags, **(metadata or {})}
        )
        if span:
            self._run_spans[run_id] = span

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a chain finishes"""
        span = self._run_spans.pop(run_id, None)
        if span:
            tracer.end_span(span, output=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors"""
        span = self._run_spans.pop(run_id, None)
        if span:
            tracer.end_span(span, error=error)

    # ============ LLM callbacks ============

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts"""
        if serialized is None:
            name = "LLM"
        else:
            name = serialized.get("name") or serialized.get("id", ["LLM"])[-1]

        span = tracer.start_span(
            name=name,
            span_type=SpanType.LLM_CALL,
            input_data={"prompts": prompts},
            metadata={"tags": tags, **(metadata or {})}
        )
        if span:
            self._run_spans[run_id] = span

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts"""
        if serialized is None:
            name = "ChatModel"
        else:
            name = serialized.get("name") or serialized.get("id", ["ChatModel"])[-1]

        # Convert messages to serializable format and capture user input
        messages_data = []
        for msg_list in messages:
            for msg in msg_list:
                messages_data.append({"type": msg.type, "content": msg.content})
                # Capture user input from human/user messages if not yet captured
                if self._user_input is None and msg.type in ('human', 'user'):
                    self._user_input = msg.content

        span = tracer.start_span(
            name=name,
            span_type=SpanType.LLM_CALL,
            input_data={"messages": messages_data},
            metadata={"tags": tags, **(metadata or {})}
        )
        if span:
            self._run_spans[run_id] = span

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends"""
        span = self._run_spans.pop(run_id, None)
        if span:
            # Extract output text
            output_data = {
                "generations": [
                    [{"text": gen.text, "message": getattr(gen, "message", None)}
                     for gen in gens]
                    for gens in response.generations
                ]
            }

            # Try to get token usage and model from llm_output (older format)
            if response.llm_output:
                output_data["llm_output"] = response.llm_output

                # Aggregate token usage from llm_output
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage:
                    self._total_input_tokens += token_usage.get("prompt_tokens", 0)
                    self._total_output_tokens += token_usage.get("completion_tokens", 0)

                # Capture model name from llm_output
                if "model_name" in response.llm_output:
                    self._model_name = response.llm_output["model_name"]

            # Try to get token usage and model from response_metadata (newer format)
            # This is used by newer versions of langchain-openai
            if response.generations and response.generations[0]:
                for gen in response.generations[0]:
                    # Try to get message with response_metadata
                    message = getattr(gen, "message", None)
                    if message:
                        response_metadata = getattr(message, "response_metadata", None)
                        if response_metadata:
                            # Get token usage from response_metadata
                            token_usage = response_metadata.get("token_usage", {})
                            if token_usage:
                                prompt_tokens = token_usage.get("prompt_tokens", 0)
                                completion_tokens = token_usage.get("completion_tokens", 0)
                                if prompt_tokens > 0:
                                    self._total_input_tokens += prompt_tokens
                                if completion_tokens > 0:
                                    self._total_output_tokens += completion_tokens

                            # Get model name from response_metadata
                            if "model_name" in response_metadata and not self._model_name:
                                self._model_name = response_metadata["model_name"]

                        # Also try usage_metadata (another format used by some providers)
                        usage_metadata = getattr(message, "usage_metadata", None)
                        if usage_metadata:
                            input_tokens = usage_metadata.get("input_tokens", 0)
                            output_tokens = usage_metadata.get("output_tokens", 0)
                            if input_tokens > 0:
                                self._total_input_tokens += input_tokens
                            if output_tokens > 0:
                                self._total_output_tokens += output_tokens

                # Capture final output text from last generation
                last_gen = response.generations[0][-1] if response.generations[0] else None
                if last_gen:
                    self._final_output = getattr(last_gen, "text", None)

            tracer.end_span(span, output=output_data)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors"""
        span = self._run_spans.pop(run_id, None)
        if span:
            tracer.end_span(span, error=error)

    # ============ Tool callbacks ============

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts"""
        if serialized is None:
            name = "Tool"
        else:
            name = serialized.get("name", "Tool")

        span = tracer.start_span(
            name=name,
            span_type=SpanType.TOOL_CALL,
            input_data=inputs or {"input": input_str},
            metadata={"tags": tags, **(metadata or {})}
        )
        if span:
            self._run_spans[run_id] = span

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool ends"""
        span = self._run_spans.pop(run_id, None)
        if span:
            tracer.end_span(span, output=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors"""
        span = self._run_spans.pop(run_id, None)
        if span:
            tracer.end_span(span, error=error)

    # ============ Agent callbacks ============

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action"""
        # Agent actions are typically followed by tool calls
        # which we track separately
        pass

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes - set trace metadata with final output"""
        # Extract final output from AgentFinish
        final_output = None
        if hasattr(finish, 'return_values'):
            return_values = finish.return_values
            if isinstance(return_values, dict):
                final_output = return_values.get('output', str(return_values))
            else:
                final_output = str(return_values)
        elif hasattr(finish, 'log'):
            final_output = finish.log

        self._set_trace_metadata_once(output=final_output)

    # ============ Retriever callbacks ============

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts"""
        if serialized is None:
            name = "Retriever"
        else:
            name = serialized.get("name", "Retriever")

        span = tracer.start_span(
            name=name,
            span_type=SpanType.RETRIEVAL,
            input_data={"query": query},
            metadata={"tags": tags, **(metadata or {})}
        )
        if span:
            self._run_spans[run_id] = span

    def on_retriever_end(
        self,
        documents: List[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when retriever ends"""
        span = self._run_spans.pop(run_id, None)
        if span:
            tracer.end_span(span, output={"documents": [doc.page_content for doc in documents]})

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors"""
        span = self._run_spans.pop(run_id, None)
        if span:
            tracer.end_span(span, error=error)


# Singleton instance for easy import
callback_handler = EvalLibCallbackHandler()
