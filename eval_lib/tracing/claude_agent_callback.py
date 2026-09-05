# eval_lib/tracing/claude_agent_callback.py
"""Claude trace collector — raw Anthropic Messages API *and* claude_agent_sdk.

Converts response / stream messages into TraceSpan objects so that reliability
metrics can analyze tool calls, reasoning steps and resource usage from
Claude-based agents.

Span tree produced for one assistant turn::

    assistant_turn (AGENT_STEP)             one per API message
    ├── thinking (REASONING)
    ├── text_response (LLM_CALL)
    ├── <tool name> (TOOL_CALL)             open until its ToolResultBlock arrives
    │   └── assistant_turn (AGENT_STEP)     sub-agent turns nest under the Task
    │       └── ...                         tool that spawned them
    └── web_search (TOOL_CALL, server_tool) closed at the end of the same message

Every span is created with an explicit parent and ``set_current=False`` —
the collector never moves the shared context pointer. Parallel tool calls
therefore become siblings, and an outer ``tracer.trace(...)`` block opened
by the caller stays the root of the whole tree.

Two entry points:

* :meth:`ClaudeAgentTraceCollector.process_response` — raw
  ``anthropic.types.Message`` (or dict with ``content`` / ``usage`` /
  ``model`` / ``stop_reason``). Content blocks carry a ``type`` field
  (``"text"`` / ``"thinking"`` / ``"tool_use"`` / ``"server_tool_use"`` …).

* :meth:`ClaudeAgentTraceCollector.process_sdk_message` — a message from
  ``claude_agent_sdk``'s stream (``AssistantMessage`` / ``UserMessage``
  / ``ResultMessage`` / ``SystemMessage``). SDK blocks are typed classes
  (``TextBlock``, ``ThinkingBlock``, ``ToolUseBlock``, ``ToolResultBlock``,
  ``ServerToolUseBlock``, ``ServerToolResultBlock``) without a ``type``
  attribute — we duck-type them by class name and attribute presence.
  Never raises into the caller's ``async for`` loop: a failure is logged
  as a warning and the message is skipped.

Token accounting: every ``AssistantMessage`` (one API call) is **added** to
the trace via ``tracer.add_trace_usage`` — including cache-read tokens —
and the final ``ResultMessage`` **declares** the authoritative totals and
``total_cost_usd``. Declared totals take precedence in the payload, so the
two stay self-consistent. The CLI may split one API message into several
``AssistantMessage`` objects (one per content block) that repeat the same
``message_id`` and usage; those are de-duplicated by ``message_id``.

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

import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .trace_utils import safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

logger = logging.getLogger("eval_lib.tracing")

# The collector may live for a whole long-running session, so remembered
# ids are capped instead of growing without bound.
_MAX_REMEMBERED_TOOLS = 512
_MAX_REMEMBERED_MESSAGES = 64

_CACHE_CREATION_KEYS = ("cache_creation_input_tokens", "cache_creation_tokens")

_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}


class ClaudeAgentTraceCollector:
    """Collects trace data from Anthropic Claude responses.

    Handles both the raw Messages API shape and the ``claude_agent_sdk``
    streaming shape. Safe to reuse across turns of a conversation — the
    collector holds only bookkeeping (open tool spans, the last model seen,
    whether the trace input has been recorded).
    """

    def __init__(self) -> None:
        self._tool_spans: Dict[str, TraceSpan] = {}  # tool_use_id → open span
        self._server_tool_ids: Set[str] = set()  # subset of _tool_spans run server-side
        # Recently closed tool spans: late server-tool results attach to
        # them, and sub-agent turns can still find their parent Task.
        self._closed_tool_spans: "OrderedDict[str, TraceSpan]" = OrderedDict()
        # API message id → its assistant_turn span (dedupes split messages).
        self._turns_by_message_id: "OrderedDict[str, TraceSpan]" = OrderedDict()
        self._model: Optional[str] = None
        self._input_set = False
        self._last_text: Optional[str] = None
        self._last_main_turn: Optional[TraceSpan] = None
        self._cache_creation_total = 0
        #: The ``assistant_turn`` span of the most recently processed message.
        self.last_turn_span: Optional[TraceSpan] = None

    # ---------------------------------------------------------------- SDK

    def process_sdk_message(self, msg: Any) -> List[TraceSpan]:
        """Dispatch a ``claude_agent_sdk`` stream message to the right handler.

        Recognised message classes (matched by ``type(msg).__name__``):

        * ``AssistantMessage`` → one ``assistant_turn`` span with a child
          span per content block; the message's ``usage`` is accumulated.
        * ``UserMessage`` → ``str`` content becomes the trace input (first
          main-thread message only); ``ToolResultBlock`` entries close the
          matching tool spans.
        * ``ResultMessage`` → closes anything still pending, then declares
          trace-level totals (tokens / cost / timing / outcome).
        * ``SystemMessage`` (``init``) → session id and configured model.

        Falls back to ``process_response`` for anything else that quacks
        like an Anthropic response (``.content`` + ``.usage``).

        Returns the block-level spans created for this message (empty for
        ``UserMessage`` / ``ResultMessage`` / ``SystemMessage``); the
        enclosing turn span is available as :attr:`last_turn_span`.
        Never raises — failures are logged and yield an empty list.
        """
        try:
            return self._dispatch_sdk_message(msg)
        except Exception as exc:
            logger.warning(
                "eval_lib.tracing: ClaudeAgentTraceCollector failed to process %s: %r",
                type(msg).__name__,
                exc,
                exc_info=True,
            )
            return []

    def _dispatch_sdk_message(self, msg: Any) -> List[TraceSpan]:
        cls_name = type(msg).__name__

        if cls_name == "AssistantMessage" or (
            cls_name not in ("UserMessage", "ResultMessage", "SystemMessage")
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

        if cls_name == "SystemMessage" or (_has_attr(msg, "subtype") and _has_attr(msg, "data")):
            self._process_sdk_system(msg)
            return []

        # Unknown shape but has content — try to salvage.
        if _has_attr(msg, "content"):
            return self.process_response(msg)

        return []

    def _process_sdk_assistant(self, msg: Any) -> List[TraceSpan]:
        """Process an ``AssistantMessage`` from claude_agent_sdk."""
        return self._process_assistant_turn(
            content=_get_attr_or_key(msg, "content", None) or [],
            model=_get_attr_or_key(msg, "model", None),
            usage=as_mapping(_get_attr_or_key(msg, "usage", None)),
            message_id=_get_attr_or_key(msg, "message_id", None),
            stop_reason=_get_attr_or_key(msg, "stop_reason", None),
            parent_tool_use_id=_get_attr_or_key(msg, "parent_tool_use_id", None),
            error=_get_attr_or_key(msg, "error", None),
        )

    def _process_sdk_user(self, msg: Any) -> None:
        """Record the trace input and close tool spans from a ``UserMessage``."""
        content = _get_attr_or_key(msg, "content", None)
        parent_tool_use_id = _get_attr_or_key(msg, "parent_tool_use_id", None)
        is_main_thread = parent_tool_use_id is None

        if isinstance(content, str):
            # The user's prompt — one value, never iterated character by
            # character. Sub-agent prompts (parent_tool_use_id set) are not
            # the trace input.
            if content and is_main_thread and not self._input_set:
                tracer.set_trace_metadata(input=content)
                self._input_set = True
            return

        texts: List[str] = []
        for block in content or []:
            kind = _classify_block(block)
            if kind == "tool_result":
                self._close_tool_result(block)
            elif kind == "text":
                text = _get_attr_or_key(block, "text", None)
                if text:
                    texts.append(str(text))

        if texts and is_main_thread and not self._input_set:
            tracer.set_trace_metadata(input="\n".join(texts))
            self._input_set = True

    def _process_sdk_system(self, msg: Any) -> None:
        """Pull session id / configured model off the ``init`` system message."""
        if _get_attr_or_key(msg, "subtype", None) != "init":
            return
        data = as_mapping(_get_attr_or_key(msg, "data", None)) or {}
        payload: Dict[str, Any] = {}
        session_id = data.get("session_id")
        if session_id:
            payload["session_id"] = session_id
        model = data.get("model")
        if model and self._model is None:
            self._model = str(model)
            payload["model"] = self._model
        if payload:
            tracer.set_trace_metadata(**payload)

    def _process_sdk_result(self, msg: Any) -> None:
        """Close pending spans and declare trace-level totals from a ``ResultMessage``.

        ``ResultMessage`` carries no ``model`` field; the model comes from
        the assistant turns seen earlier or from ``model_usage`` keys.
        """
        # Tools whose result never arrived are closed now — a ResultMessage
        # is the end of the turn, nothing can still answer them.
        self.close_pending_tool_spans()

        usage = as_mapping(_get_attr_or_key(msg, "usage", None))
        norm = usage_from_mapping(usage) if usage else None
        cache_creation = _int_from(usage, _CACHE_CREATION_KEYS)
        model_usage = as_mapping(_get_attr_or_key(msg, "model_usage", None))
        model = self._model or _model_from_model_usage(model_usage)

        cost_usd: Optional[float] = None
        cost_source: Optional[str] = None
        total_cost_usd = _get_attr_or_key(msg, "total_cost_usd", None)
        if total_cost_usd is not None:
            cost_usd = float(total_cost_usd)
            cost_source = "reported"
        else:
            reported = _reported_cost_from_model_usage(model_usage)
            if reported is not None:
                cost_usd, cost_source = reported, "reported"
            else:
                estimated = _estimated_cost_from_model_usage(model_usage)
                if estimated is None and model and norm:
                    estimated = _estimate_cost(
                        model,
                        norm["input_tokens"],
                        norm["output_tokens"],
                        norm["cached_tokens"],
                        cache_creation,
                    )
                if estimated is not None:
                    cost_usd, cost_source = estimated, "estimated"

        payload: Dict[str, Any] = {}
        if model:
            payload["model"] = model
        if norm:
            payload["input_tokens"] = norm["input_tokens"]
            payload["output_tokens"] = norm["output_tokens"]
            payload["total_tokens"] = norm["input_tokens"] + norm["output_tokens"]
            payload["cached_tokens"] = norm["cached_tokens"]
            if norm["reasoning_tokens"]:
                payload["reasoning_tokens"] = norm["reasoning_tokens"]
        if cache_creation:
            payload["cache_creation_tokens"] = cache_creation
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd
            payload["cost_source"] = cost_source

        duration_ms = _get_attr_or_key(msg, "duration_ms", None)
        if duration_ms is not None:
            payload["response_time"] = round(float(duration_ms) / 1000, 3)
        for key in (
            "duration_api_ms",
            "num_turns",
            "session_id",
            "stop_reason",
            "subtype",
            "terminal_reason",
            "api_error_status",
        ):
            value = _get_attr_or_key(msg, key, None)
            if value is not None:
                payload[key] = value

        result = _get_attr_or_key(msg, "result", None)
        structured_output = _get_attr_or_key(msg, "structured_output", None)
        if structured_output is not None:
            payload["structured_output"] = structured_output
        output = result if result is not None else self._last_text
        if output is None and structured_output is not None:
            output = structured_output
        if output is not None:
            payload["output"] = output

        subtype = _get_attr_or_key(msg, "subtype", None)
        is_error = bool(_get_attr_or_key(msg, "is_error", False))
        if is_error or (isinstance(subtype, str) and subtype.startswith("error")):
            errors = _get_attr_or_key(msg, "errors", None)
            if errors:
                message = "; ".join(str(e) for e in errors)
            elif isinstance(subtype, str) and subtype:
                message = subtype
            else:
                message = "agent run reported an error"
            error_type = subtype if isinstance(subtype, str) and subtype else "error"
            payload["status"] = "error"
            payload["error"] = message
            payload["error_type"] = error_type
            turn = self._last_main_turn
            if turn is not None and turn.status != "error":
                _mark_span_error(turn, message, error_type)

        if payload:
            tracer.set_trace_metadata(**payload)

    # -------------------------------------------------------------- legacy

    def process_response(self, response: Any) -> List[TraceSpan]:
        """Process a raw Anthropic Messages API response.

        Args:
            response: An ``anthropic.types.Message`` or dict with
                ``content`` / ``usage`` / ``model`` / ``stop_reason``.

        Returns:
            List of block-level ``TraceSpan`` objects created (the enclosing
            ``assistant_turn`` span is available as :attr:`last_turn_span`).
        """
        return self._process_assistant_turn(
            content=_get_attr_or_key(response, "content", None) or [],
            model=_get_attr_or_key(response, "model", None),
            usage=as_mapping(_get_attr_or_key(response, "usage", None)),
            message_id=_get_attr_or_key(response, "id", None),
            stop_reason=_get_attr_or_key(response, "stop_reason", None),
            parent_tool_use_id=None,
            error=None,
        )

    def process_tool_results(self, tool_results: List[Any]):
        """Close the TOOL_CALL span matching each incoming ToolResultBlock.

        A result with ``is_error=True`` keeps its content as the span output
        (the error body is usually the most useful thing in the span) and
        is recorded with ``error_type="ToolError"``.
        """
        for result in tool_results:
            self._close_tool_result(result)

    def close_pending_tool_spans(self):
        """Close any tool spans that never received results.

        Called automatically on ``ResultMessage``. Client tools are marked
        as errors (their result genuinely never came back); server-side
        tools are closed cleanly — the API ran them, the SDK just never
        surfaced a result block.
        """
        for tool_use_id, span in list(self._tool_spans.items()):
            if tool_use_id in self._server_tool_ids:
                tracer.end_span(span)
            else:
                tracer.end_span(
                    span, error="Tool result never received", error_type="ToolResultMissing"
                )
            self._remember(self._closed_tool_spans, tool_use_id, span, _MAX_REMEMBERED_TOOLS)
        self._tool_spans.clear()
        self._server_tool_ids.clear()

    # ------------------------------------------------------------- turn

    def _process_assistant_turn(
        self,
        *,
        content: Any,
        model: Optional[str],
        usage: Optional[Dict[str, Any]],
        message_id: Optional[str],
        stop_reason: Optional[str],
        parent_tool_use_id: Optional[str],
        error: Optional[str],
    ) -> List[TraceSpan]:
        """One API message → one ``assistant_turn`` span with a child per block."""
        if model:
            self._set_model(str(model))

        turn = self._turns_by_message_id.get(message_id) if message_id else None
        reused = turn is not None
        if not reused:
            metadata = {
                key: value
                for key, value in (
                    ("model", model),
                    ("message_id", message_id),
                    ("stop_reason", stop_reason),
                    ("parent_tool_use_id", parent_tool_use_id),
                )
                if value is not None
            }
            turn = self._start_span(
                "assistant_turn",
                SpanType.AGENT_STEP,
                metadata=metadata,
                parent=self._resolve_parent(parent_tool_use_id),
                inherit_context=True,
            )
            if turn is not None and message_id:
                self._remember(
                    self._turns_by_message_id, message_id, turn, _MAX_REMEMBERED_MESSAGES
                )
        turn_id = turn.span_id if turn is not None else None

        spans: List[TraceSpan] = []
        server_tool_ids: List[str] = []
        texts: List[str] = []
        for block in content:
            kind = _classify_block(block)
            span: Optional[TraceSpan] = None
            if kind == "tool_use":
                span = self._open_tool_span(block, turn_id, server=False)
            elif kind == "server_tool_use":
                span = self._open_tool_span(block, turn_id, server=True)
                tool_use_id = _get_attr_or_key(block, "id", None)
                if tool_use_id:
                    server_tool_ids.append(tool_use_id)
            elif kind == "tool_result":
                self._close_tool_result(block)
            elif kind == "thinking":
                span = self._process_thinking(block, turn_id)
            elif kind == "text":
                span = self._process_text(block, turn_id)
                text = _get_attr_or_key(block, "text", None)
                if text:
                    texts.append(str(text))
            if span is not None:
                spans.append(span)

        # Server-side tools run inside the API call, so whatever result the
        # SDK surfaces arrives in this same message. Anything still open is
        # finished — close it so it neither leaks nor stays "running".
        for tool_use_id in server_tool_ids:
            if tool_use_id in self._tool_spans:
                self._finish_tool(tool_use_id, content=None, is_error=False)

        if usage and not reused:
            self._record_call_usage(usage, model or self._model, turn)

        if turn is not None:
            error_message = f"assistant message error: {error}" if error else None
            if reused:
                _extend_span(turn)
                if error_message:
                    _mark_span_error(turn, error_message, str(error))
            elif error_message:
                tracer.end_span(turn, error=error_message, error_type=str(error))
            else:
                tracer.end_span(turn)

        if parent_tool_use_id is None:
            if texts:
                self._last_text = "\n".join(texts)
            if turn is not None:
                self._last_main_turn = turn
        self.last_turn_span = turn
        return spans

    def _open_tool_span(
        self, block: Any, parent_id: Optional[str], *, server: bool
    ) -> Optional[TraceSpan]:
        """Create a TOOL_CALL span from a ``tool_use`` / ``server_tool_use`` block."""
        name = _get_attr_or_key(block, "name", None) or "unknown_tool"
        tool_input = _get_attr_or_key(block, "input", None)
        if tool_input is None:
            tool_input = {}
        tool_use_id = _get_attr_or_key(block, "id", None)

        metadata: Dict[str, Any] = {}
        if tool_use_id:
            metadata["tool_use_id"] = tool_use_id
        if server:
            metadata["server_tool"] = True

        span = self._start_span(
            name,
            SpanType.TOOL_CALL,
            input_data=tool_input,
            metadata=metadata or None,
            parent=parent_id,
        )
        if span is not None and tool_use_id:
            # Store the span so the matching result block can close it later.
            self._tool_spans[tool_use_id] = span
            if server:
                self._server_tool_ids.add(tool_use_id)
        return span

    def _close_tool_result(self, block: Any) -> None:
        tool_use_id = _get_attr_or_key(block, "tool_use_id", None)
        if not tool_use_id:
            return
        self._finish_tool(
            tool_use_id,
            content=_get_attr_or_key(block, "content", None),
            is_error=bool(_get_attr_or_key(block, "is_error", False)),
        )

    def _finish_tool(self, tool_use_id: str, *, content: Any, is_error: bool) -> None:
        span = self._tool_spans.pop(tool_use_id, None)
        self._server_tool_ids.discard(tool_use_id)
        error_message = (safe_str(content) or "tool reported is_error") if is_error else None

        if span is None:
            # A result for a span already closed (server tool result that
            # arrived in a later split message) — attach it in place; the
            # span object is what gets serialized at flush.
            late = self._closed_tool_spans.get(tool_use_id)
            if late is not None:
                if content is not None and late.output is None:
                    late.output = content
                if error_message and late.status != "error":
                    _mark_span_error(late, error_message, "ToolError")
            return

        if error_message:
            tracer.end_span(span, output=content, error=error_message, error_type="ToolError")
        else:
            tracer.end_span(span, output=content)
        self._remember(self._closed_tool_spans, tool_use_id, span, _MAX_REMEMBERED_TOOLS)

    def _process_thinking(self, block: Any, parent_id: Optional[str]) -> Optional[TraceSpan]:
        """Create a REASONING span from a thinking block."""
        thinking_text = _get_attr_or_key(block, "thinking", "")
        span = self._start_span("thinking", SpanType.REASONING, parent=parent_id)
        if span is not None:
            tracer.end_span(span, output=thinking_text)
        return span

    def _process_text(self, block: Any, parent_id: Optional[str]) -> Optional[TraceSpan]:
        """Create an LLM_CALL span from a text block."""
        text = _get_attr_or_key(block, "text", "")
        span = self._start_span("text_response", SpanType.LLM_CALL, parent=parent_id)
        if span is not None:
            tracer.end_span(span, output=text)
        return span

    def _record_call_usage(
        self, usage: Dict[str, Any], model: Optional[str], turn: Optional[TraceSpan]
    ) -> None:
        """Accumulate one API call's tokens (and an estimated cost) on the trace."""
        norm = usage_from_mapping(usage)
        cache_creation = _int_from(usage, _CACHE_CREATION_KEYS)
        if norm is None and not cache_creation:
            return
        norm = norm or dict(_EMPTY_USAGE)

        cost = None
        if model:
            cost = _estimate_cost(
                model,
                norm["input_tokens"],
                norm["output_tokens"],
                norm["cached_tokens"],
                cache_creation,
            )
        tracer.add_trace_usage(
            input_tokens=norm["input_tokens"],
            output_tokens=norm["output_tokens"],
            cached_tokens=norm["cached_tokens"],
            reasoning_tokens=norm["reasoning_tokens"],
            cost_usd=cost or 0.0,
            calls=1,
        )
        if cache_creation:
            # add_trace_usage has no slot for cache writes; keep a running
            # total as a trace fact so it is not lost.
            self._cache_creation_total += cache_creation
            tracer.set_trace_metadata(cache_creation_tokens=self._cache_creation_total)

        if turn is not None:
            turn.metadata["usage"] = {
                "input_tokens": norm["input_tokens"],
                "output_tokens": norm["output_tokens"],
                "cached_tokens": norm["cached_tokens"],
                "cache_creation_tokens": cache_creation,
            }
            if cost is not None:
                turn.metadata["estimated_cost_usd"] = cost

    # ------------------------------------------------------------- helpers

    def _set_model(self, model: str) -> None:
        if model != self._model:
            self._model = model
            tracer.set_trace_metadata(model=model)

    def _resolve_parent(self, parent_tool_use_id: Optional[str]) -> Optional[str]:
        """Span id of the tool that spawned a sub-agent, if we know it."""
        if not parent_tool_use_id:
            return None
        span = self._tool_spans.get(parent_tool_use_id) or self._closed_tool_spans.get(
            parent_tool_use_id
        )
        return span.span_id if span is not None else None

    @staticmethod
    def _start_span(
        name: str,
        span_type: SpanType,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional[str] = None,
        inherit_context: bool = False,
    ) -> Optional[TraceSpan]:
        """``tracer.start_span`` with an explicit parent and ``set_current=False``.

        ``parent=None`` means "root" unless ``inherit_context`` is set, in
        which case the span nests under whatever the caller has open (an
        outer ``tracer.trace(...)`` block).
        """
        kwargs: Dict[str, Any] = {}
        if parent is not None:
            kwargs["parent_span_id"] = parent
        elif not inherit_context:
            kwargs["parent_span_id"] = None
        return tracer.start_span(
            name=name,
            span_type=span_type,
            input_data=input_data,
            metadata=metadata,
            set_current=False,
            **kwargs,
        )

    @staticmethod
    def _remember(store: "OrderedDict[str, TraceSpan]", key: str, span: TraceSpan, cap: int):
        store[key] = span
        store.move_to_end(key)
        while len(store) > cap:
            store.popitem(last=False)


# ------------------------------------------------------------- module helpers


def _classify_block(block: Any) -> Optional[str]:
    """Map any block shape to one of: tool_use / server_tool_use / tool_result /
    thinking / text — or ``None`` for something we do not trace."""
    cls_name = type(block).__name__

    # SDK dataclasses (claude_agent_sdk) and anthropic.types pydantic models
    # share these class names.
    if cls_name == "ToolUseBlock":
        return "tool_use"
    if cls_name == "ServerToolUseBlock":
        return "server_tool_use"
    if cls_name.endswith("ToolResultBlock"):
        # ToolResultBlock, ServerToolResultBlock, WebSearchToolResultBlock,
        # WebFetchToolResultBlock, CodeExecutionToolResultBlock …
        return "tool_result"
    if cls_name in ("ThinkingBlock", "RedactedThinkingBlock"):
        return "thinking"
    if cls_name == "TextBlock":
        return "text"

    # Dict blocks and typed API blocks carry a `type` discriminator.
    block_type = _get_attr_or_key(block, "type", None)
    if isinstance(block_type, str) and block_type:
        if block_type == "tool_use":
            return "tool_use"
        if block_type == "server_tool_use":
            return "server_tool_use"
        if block_type == "tool_result" or block_type.endswith("_tool_result"):
            return "tool_result"
        if block_type in ("thinking", "redacted_thinking"):
            return "thinking"
        if block_type == "text":
            return "text"

    # Last resort: attribute duck typing. Order matters — a result block
    # and a tool_use block both look enough like a text block that their
    # distinctive attributes must be checked first.
    if _has_attr(block, "tool_use_id"):
        return "tool_result"
    if _has_attr(block, "name") and _has_attr(block, "input"):
        return "tool_use"
    if _has_attr(block, "thinking"):
        return "thinking"
    if _has_attr(block, "text"):
        return "text"
    return None


def _mark_span_error(span: TraceSpan, message: str, error_type: str) -> None:
    """Flip an already-finished span to ``error`` in place."""
    span.status = "error"
    span.error = message
    span.error_type = error_type


def _extend_span(span: TraceSpan) -> None:
    """Push a finished span's end time to now (late blocks joined the turn)."""
    now = datetime.now().timestamp()
    if span.end_time is None or now > span.end_time:
        span.end_time = now
        span.duration_ms = round((now - span.start_time) * 1000, 2)


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


def _int_from(mapping: Optional[Dict[str, Any]], keys: Tuple[str, ...]) -> int:
    """First integer-valued key of ``mapping`` among ``keys`` (0 when absent)."""
    if not mapping:
        return 0
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _model_usage_entries(model_usage: Optional[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """``ResultMessage.model_usage`` as ``[(model, entry_dict), …]``."""
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for model, entry in (model_usage or {}).items():
        mapping = as_mapping(entry)
        if mapping is not None and model:
            entries.append((str(model), mapping))
    return entries


def _model_usage_tokens(entry: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """(input, output, cache_read, cache_creation) from one ModelUsage entry."""
    return (
        _int_from(entry, ("inputTokens", "input_tokens")),
        _int_from(entry, ("outputTokens", "output_tokens")),
        _int_from(entry, ("cacheReadInputTokens", "cache_read_input_tokens")),
        _int_from(entry, ("cacheCreationInputTokens", "cache_creation_input_tokens")),
    )


def _model_from_model_usage(model_usage: Optional[Dict[str, Any]]) -> Optional[str]:
    """The model that did the most work according to ``model_usage``."""
    best: Optional[str] = None
    best_tokens = -1
    for model, entry in _model_usage_entries(model_usage):
        tokens = sum(_model_usage_tokens(entry))
        if tokens > best_tokens:
            best, best_tokens = model, tokens
    return best


def _reported_cost_from_model_usage(model_usage: Optional[Dict[str, Any]]) -> Optional[float]:
    """Sum of per-model ``costUSD`` — only when every entry reports one."""
    entries = _model_usage_entries(model_usage)
    if not entries:
        return None
    total = 0.0
    for _model, entry in entries:
        cost = entry.get("costUSD", entry.get("cost_usd"))
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            return None
        total += float(cost)
    return round(total, 6)


def _estimated_cost_from_model_usage(model_usage: Optional[Dict[str, Any]]) -> Optional[float]:
    """Per-model price estimate summed over ``model_usage``; ``None`` if any model is unknown."""
    entries = _model_usage_entries(model_usage)
    if not entries:
        return None
    total = 0.0
    for model, entry in entries:
        estimate = _estimate_cost(model, *_model_usage_tokens(entry))
        if estimate is None:
            return None
        total += estimate
    return round(total, 6)


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> Optional[float]:
    """Estimate USD cost from token counts using eval_lib's model catalog.

    ``input_tokens`` are the *uncached* prompt tokens (Anthropic reports
    cache reads and writes separately). Cache reads are billed at the
    catalog's ``cache_read`` rate and cache writes at ``cache_write``; when
    the catalog does not know a cache rate the plain input rate is used.

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

    input_rate = pricing.get("input") or 0.0
    output_rate = pricing.get("output") or 0.0
    cache_read_rate = pricing.get("cache_read")
    cache_write_rate = pricing.get("cache_write")
    if cache_read_rate is None:
        cache_read_rate = input_rate
    if cache_write_rate is None:
        cache_write_rate = input_rate

    per_million = 1_000_000.0
    return round(
        (input_tokens / per_million) * input_rate
        + (output_tokens / per_million) * output_rate
        + (cache_read_tokens / per_million) * cache_read_rate
        + (cache_creation_tokens / per_million) * cache_write_rate,
        6,
    )
