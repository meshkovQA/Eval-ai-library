# eval_lib/tracing/claude_agent_callback.py
"""Claude trace collector — raw Anthropic Messages API *and* claude_agent_sdk.

Converts response / stream messages into TraceSpan objects so that reliability
metrics can analyze tool calls, reasoning steps and resource usage from
Claude-based agents.

Two entry points:

* :meth:`ClaudeAgentTraceCollector.process_response` — raw
  ``anthropic.types.Message`` (or dict with ``content`` / ``usage`` /
  ``model`` / ``stop_reason``). Content blocks carry a ``type`` field
  (``"text"`` / ``"thinking"`` / ``"tool_use"``).

* :meth:`ClaudeAgentTraceCollector.process_sdk_message` — a message from
  ``claude_agent_sdk``'s stream (``AssistantMessage`` / ``UserMessage``
  / ``ResultMessage``). SDK blocks are typed classes (``TextBlock``,
  ``ThinkingBlock``, ``ToolUseBlock``, ``ToolResultBlock``) without a
  ``type`` attribute — we duck-type them by class name and attribute
  presence.

We intentionally do NOT import anything from ``claude_agent_sdk`` — that
keeps ``eval_lib`` free of a hard dependency on the SDK.

Basic usage:

.. code-block:: python

    from eval_lib.tracing import tracer
    from eval_lib.tracing.claude_agent_callback import ClaudeAgentTraceCollector

    collector = ClaudeAgentTraceCollector()
    trace_id = tracer.start_trace("claude_agent")

    # raw Anthropic API:
    response = client.messages.create(...)
    collector.process_response(response)

    # OR claude_agent_sdk stream:
    async for msg in client.receive_messages():
        collector.process_sdk_message(msg)

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

from typing import Any, Dict, List, Optional

from .tracer import tracer
from .types import SpanType, TraceSpan


class ClaudeAgentTraceCollector:
    """Collects trace data from Anthropic Claude responses.

    Handles both the raw Messages API shape and the ``claude_agent_sdk``
    streaming shape. Safe to reuse across turns of a conversation — the
    collector holds only ``tool_use_id → span`` bookkeeping.
    """

    def __init__(self):
        self._tool_spans: Dict[str, TraceSpan] = {}  # tool_use_id → span

    # ---------------------------------------------------------------- SDK

    def process_sdk_message(self, msg: Any) -> List[TraceSpan]:
        """Dispatch a ``claude_agent_sdk`` stream message to the right handler.

        Recognised message classes (matched by ``type(msg).__name__``):

        * ``AssistantMessage`` → same treatment as ``process_response``,
          but with duck-typed ``TextBlock`` / ``ThinkingBlock`` /
          ``ToolUseBlock`` inside ``.content``.
        * ``UserMessage`` → forward any ``ToolResultBlock`` entries in
          ``.content`` to :meth:`process_tool_results`.
        * ``ResultMessage`` → extract trace-level metadata
          (``usage`` / ``model`` / ``total_cost_usd`` / ``duration_ms`` /
          ``num_turns`` / ``session_id``).

        Falls back to ``process_response`` for anything else that quacks
        like an Anthropic response (``.content`` + ``.usage``).

        Returns the list of spans created for this message (empty for
        ``ResultMessage`` / ``UserMessage``).
        """
        cls_name = type(msg).__name__

        if cls_name == "AssistantMessage" or (
            cls_name not in ("UserMessage", "ResultMessage")
            and _has_attr(msg, "content")
            and _has_attr(msg, "usage")
        ):
            return self._process_sdk_assistant(msg)

        if cls_name == "UserMessage":
            self._process_sdk_user(msg)
            return []

        if cls_name == "ResultMessage" or _has_attr(msg, "total_cost_usd"):
            self._process_sdk_result(msg)
            return []

        # Unknown shape but has content — try to salvage.
        if _has_attr(msg, "content"):
            return self.process_response(msg)

        return []

    def _process_sdk_assistant(self, msg: Any) -> List[TraceSpan]:
        """Process an ``AssistantMessage`` from claude_agent_sdk."""
        spans: List[TraceSpan] = []
        content_blocks = getattr(msg, "content", None) or []
        model = getattr(msg, "model", None)

        for block in content_blocks:
            span = self._process_sdk_block(block)
            if span is not None:
                spans.append(span)

        # AssistantMessage on the SDK sometimes lacks .usage — usage lives
        # on ResultMessage. Still record whatever model we saw so trace
        # metadata is populated as soon as possible.
        if model is not None:
            self._extract_and_set_metadata(msg, model_hint=model, usage_hint=None)
        return spans

    def _process_sdk_block(self, block: Any) -> Optional[TraceSpan]:
        """Duck-type a single SDK content block into the right span kind."""
        cls_name = type(block).__name__

        # Cheap class-name match first (works for SDK block classes).
        if cls_name == "ToolUseBlock":
            return self._process_tool_use(block)
        if cls_name == "ThinkingBlock":
            return self._process_thinking(block)
        if cls_name == "TextBlock":
            return self._process_text(block)

        # Fallback: attribute-based duck typing. Order matters — ToolUseBlock
        # and ThinkingBlock look enough like TextBlock that we must check
        # their distinctive attributes first.
        if _has_attr(block, "name") and _has_attr(block, "input"):
            return self._process_tool_use(block)
        if _has_attr(block, "thinking"):
            return self._process_thinking(block)
        if _has_attr(block, "text"):
            return self._process_text(block)

        # Old-style dict blocks with `.type` — delegate to legacy path.
        block_type = _get_attr_or_key(block, "type", "")
        if block_type == "tool_use":
            return self._process_tool_use(block)
        if block_type == "thinking":
            return self._process_thinking(block)
        if block_type == "text":
            return self._process_text(block)

        return None

    def _process_sdk_user(self, msg: Any) -> None:
        """Extract ToolResultBlock entries from a UserMessage."""
        content_blocks = getattr(msg, "content", None) or []
        tool_results = []
        for block in content_blocks:
            cls_name = type(block).__name__
            if cls_name == "ToolResultBlock" or _has_attr(block, "tool_use_id"):
                tool_results.append(block)
            elif _get_attr_or_key(block, "type", "") == "tool_result":
                tool_results.append(block)
        if tool_results:
            self.process_tool_results(tool_results)

    def _process_sdk_result(self, msg: Any) -> None:
        """Pull trace-level metadata off a ResultMessage."""
        self._extract_and_set_metadata(
            msg,
            model_hint=getattr(msg, "model", None),
            usage_hint=getattr(msg, "usage", None),
        )

    # -------------------------------------------------------------- legacy

    def process_response(self, response: Any) -> List[TraceSpan]:
        """Process a raw Anthropic Messages API response.

        Args:
            response: An ``anthropic.types.Message`` or dict with
                ``content`` / ``usage`` / ``model`` / ``stop_reason``.

        Returns:
            List of created ``TraceSpan`` objects.
        """
        spans = []

        content_blocks = _get_attr_or_key(response, "content", []) or []
        usage = _get_attr_or_key(response, "usage", None)
        model = _get_attr_or_key(response, "model", None)

        for block in content_blocks:
            block_type = _get_attr_or_key(block, "type", "")

            if block_type == "tool_use":
                span = self._process_tool_use(block)
            elif block_type == "thinking":
                span = self._process_thinking(block)
            elif block_type == "text":
                span = self._process_text(block)
            else:
                # No `.type` field — fall back to duck typing (SDK-style
                # blocks that landed here by accident still work).
                span = self._process_sdk_block(block)

            if span is not None:
                spans.append(span)

        # Set trace-level metadata from usage. Cost isn't available on the
        # raw API surface, so we leave that to `_extract_and_set_metadata`
        # which will estimate from tokens+model as a last resort.
        if usage is not None or model is not None:
            self._extract_and_set_metadata(response, model_hint=model, usage_hint=usage)

        return spans

    def process_tool_results(self, tool_results: List[Any]):
        """Close the TOOL_CALL span matching each incoming ToolResultBlock."""
        for result in tool_results:
            tool_use_id = _get_attr_or_key(result, "tool_use_id", None)
            if not tool_use_id or tool_use_id not in self._tool_spans:
                continue

            span = self._tool_spans.pop(tool_use_id)
            content = _get_attr_or_key(result, "content", None)
            is_error = _get_attr_or_key(result, "is_error", False)

            if is_error:
                tracer.end_span(
                    span, error=Exception(str(content) if content else "Tool error")
                )
            else:
                tracer.end_span(span, output=content)

    # ------------------------------------------------------------- helpers

    def _process_tool_use(self, block: Any) -> Optional[TraceSpan]:
        """Create a TOOL_CALL span from a tool_use block."""
        name = _get_attr_or_key(block, "name", "unknown_tool")
        tool_input = _get_attr_or_key(block, "input", {})
        tool_use_id = _get_attr_or_key(block, "id", None)

        span = tracer.start_span(
            name=name,
            span_type=SpanType.TOOL_CALL,
            input_data=tool_input,
            metadata={"tool_use_id": tool_use_id} if tool_use_id else None,
        )

        if span and tool_use_id:
            # Store the span so process_tool_results can close it later.
            self._tool_spans[tool_use_id] = span

        return span

    def _process_thinking(self, block: Any) -> Optional[TraceSpan]:
        """Create a REASONING span from a thinking block."""
        thinking_text = _get_attr_or_key(block, "thinking", "")
        span = tracer.start_span(name="thinking", span_type=SpanType.REASONING)
        if span:
            tracer.end_span(span, output=thinking_text)
        return span

    def _process_text(self, block: Any) -> Optional[TraceSpan]:
        """Create an LLM_CALL span from a text block."""
        text = _get_attr_or_key(block, "text", "")
        span = tracer.start_span(name="text_response", span_type=SpanType.LLM_CALL)
        if span:
            tracer.end_span(span, output=text)
        return span

    def _extract_and_set_metadata(
        self,
        msg: Any,
        model_hint: Optional[str] = None,
        usage_hint: Optional[Any] = None,
    ) -> None:
        """Push whatever trace-level metadata we can find into the tracer.

        Accepts every shape we've seen in the wild: raw API ``response``
        (``.usage`` + ``.model``) *and* SDK ``ResultMessage``
        (``.usage`` + ``.model`` + ``.total_cost_usd`` + ``.duration_ms``
        + ``.num_turns`` + ``.session_id``).

        When ``total_cost_usd`` isn't available we try to estimate it
        from tokens + model via :mod:`eval_lib.model_catalog`; the trace
        payload marks that with ``cost_source="estimated"``.
        """
        model = model_hint if model_hint is not None else getattr(msg, "model", None)
        usage = usage_hint if usage_hint is not None else getattr(msg, "usage", None)

        input_tokens = _get_attr_or_key(usage, "input_tokens", None) if usage else None
        output_tokens = _get_attr_or_key(usage, "output_tokens", None) if usage else None
        total_tokens = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        total_cost_usd = getattr(msg, "total_cost_usd", None)
        duration_ms = getattr(msg, "duration_ms", None)
        num_turns = getattr(msg, "num_turns", None)
        session_id = getattr(msg, "session_id", None)

        cost_source: Optional[str] = None
        cost_usd: Optional[float] = None
        if total_cost_usd is not None:
            cost_usd = float(total_cost_usd)
            cost_source = "reported"
        elif model and (input_tokens or output_tokens):
            estimated = _estimate_cost(model, input_tokens or 0, output_tokens or 0)
            if estimated is not None:
                cost_usd = estimated
                cost_source = "estimated"

        payload: Dict[str, Any] = {}
        if model is not None:
            payload["model"] = model
        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens
        if total_tokens is not None:
            payload["total_tokens"] = total_tokens
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd
        if cost_source is not None:
            payload["cost_source"] = cost_source
        if duration_ms is not None:
            payload["response_time"] = round(duration_ms / 1000, 3)
        if num_turns is not None:
            payload["num_turns"] = num_turns
        if session_id is not None:
            payload["session_id"] = session_id

        if payload:
            tracer.set_trace_metadata(**payload)

    def close_pending_tool_spans(self):
        """Close any tool spans that never received results."""
        for _tool_use_id, span in list(self._tool_spans.items()):
            tracer.end_span(span, error=Exception("Tool result never received"))
        self._tool_spans.clear()


def _has_attr(obj: Any, name: str) -> bool:
    """Return True if ``obj`` has attribute ``name`` (dict key also counts)."""
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Get value from an object by attribute or dict key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _estimate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> Optional[float]:
    """Estimate USD cost from token counts using eval_lib's model catalog.

    Returns ``None`` when the model is unknown to both LiteLLM and the
    local override table — we don't want to guess.
    """
    try:
        from ..model_catalog import get_cost_per_million
    except Exception:
        # Any import failure — the catalog transitively pulls litellm,
        # which is optional under the [tracing] extra.
        return None

    try:
        pricing = get_cost_per_million(model)
    except Exception:
        return None
    if not pricing:
        return None

    per_million = 1_000_000.0
    return round(
        (input_tokens / per_million) * pricing.get("input", 0.0)
        + (output_tokens / per_million) * pricing.get("output", 0.0),
        6,
    )
