# eval_lib/tracing/autogen_callback.py
"""AutoGen (Microsoft, v0.4+) tracing.

Two independent pieces — use both for a complete picture:

* :class:`AutoGenTraceHandler` — an ``autogen_core.InterventionHandler``
  plugged into ``SingleThreadedAgentRuntime``. It sees every inter-agent
  message and turns agentchat events into spans: ``TextMessage`` and
  friends → ``AGENT_STEP``, ``ToolCallRequestEvent`` → one open
  ``TOOL_CALL`` span per ``FunctionCall`` that is closed by the matching
  ``ToolCallExecutionEvent`` (paired by ``call_id``), ``ThoughtEvent`` →
  ``REASONING``. Token counts that agentchat attaches to messages
  (``models_usage``) are accumulated into the trace.

* :class:`TracedChatCompletionClient` — a duck-typed proxy around any
  ``autogen_core.models.ChatCompletionClient``. ``AssistantAgent`` calls
  ``model_client.create()`` directly, which the runtime never sees, so
  this is the only way to get an ``LLM_CALL`` span per model call with
  its prompt, completion, model name and ``CreateResult.usage``.

Usage::

    from autogen_agentchat.agents import AssistantAgent
    from autogen_core import SingleThreadedAgentRuntime
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from eval_lib.tracing import tracer
    from eval_lib.tracing.autogen_callback import (
        AutoGenTraceHandler, TracedChatCompletionClient,
    )

    trace_id = tracer.start_trace("autogen")

    model_client = TracedChatCompletionClient(OpenAIChatCompletionClient(model="gpt-4o"))
    agent = AssistantAgent("assistant", model_client=model_client, tools=[...])

    handler = AutoGenTraceHandler()          # captures the active trace id
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[handler])
    # ... register agents / run a team on this runtime ...

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()

Token accounting: when a ``TracedChatCompletionClient`` has reported usage
for a trace, the handler stops accumulating ``models_usage`` from messages
for that trace so a model call is never counted twice. Wrap *all* model
clients of a team, or none — a mix undercounts the unwrapped agents (or
pass ``usage_from_messages=True`` to force message-level accounting and
leave the clients unwrapped).

Nothing in this module imports ``autogen``; every object is duck-typed
by class name and attributes, so it loads on a slim install and tolerates
version drift in the agentchat message classes.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .context import get_trace_id, set_trace_id
from .trace_utils import safe_str as _safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan

logger = logging.getLogger("eval_lib.tracing")

__all__ = ["AutoGenTraceHandler", "TracedChatCompletionClient"]


# ---------------------------------------------------------------------------
# Trace-level registry: traces where a TracedChatCompletionClient reported
# usage. The handler consults it to avoid double counting ``models_usage``.
# ---------------------------------------------------------------------------

_CLIENT_TRACED: "OrderedDict[str, None]" = OrderedDict()
_CLIENT_TRACED_MAX = 512


def _mark_client_usage(trace_id: Optional[str]) -> None:
    if not trace_id:
        return
    _CLIENT_TRACED[trace_id] = None
    _CLIENT_TRACED.move_to_end(trace_id)
    while len(_CLIENT_TRACED) > _CLIENT_TRACED_MAX:
        _CLIENT_TRACED.popitem(last=False)


def _client_reported_usage(trace_id: Optional[str]) -> bool:
    return bool(trace_id) and trace_id in _CLIENT_TRACED


# ---------------------------------------------------------------------------
# Small duck-typing helpers
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _token_pair(usage: Any) -> Optional[Tuple[int, int]]:
    """``RequestUsage``-like (``prompt_tokens``/``completion_tokens``) → ints.

    Strict on types so a ``Mock`` or an unrelated object never yields
    phantom counts. Returns ``None`` when nothing usable is present.
    """
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    else:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
    if not (_is_int(prompt) or _is_int(completion)):
        return None
    return (int(prompt) if _is_int(prompt) else 0, int(completion) if _is_int(completion) else 0)


def _is_function_call(obj: Any) -> bool:
    return isinstance(getattr(obj, "name", None), str) and hasattr(obj, "arguments")


def _is_execution_result(obj: Any) -> bool:
    return isinstance(getattr(obj, "call_id", None), str) and hasattr(obj, "content")


def _content_repr(content: Any) -> Any:
    """JSON-friendly rendering of a message / result ``content`` field."""
    if content is None or isinstance(content, (str, int, float, bool)):
        return content
    if isinstance(content, (list, tuple)):
        return [_content_repr(c) for c in content]
    if _is_function_call(content):
        return {
            "id": getattr(content, "id", None),
            "name": content.name,
            "arguments": _content_repr(content.arguments),
        }
    if _is_execution_result(content):
        return {
            "call_id": content.call_id,
            "name": getattr(content, "name", None),
            "content": _content_repr(content.content),
            "is_error": getattr(content, "is_error", None),
        }
    return _safe_str(content)


def _agent_name(agent: Any) -> str:
    """Readable name for an ``AgentId`` / agent / plain string."""
    if agent is None:
        return "unknown"
    if isinstance(agent, str):
        return agent
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name:
        return name
    agent_type = getattr(agent, "type", None)
    if isinstance(agent_type, str) and agent_type:
        key = getattr(agent, "key", None)
        if isinstance(key, str) and key and key != "default":
            return f"{agent_type}/{key}"
        return agent_type
    return type(agent).__name__


def _is_chat_message(message: Any) -> bool:
    """agentchat ``BaseChatMessage`` / ``BaseAgentEvent`` look-alike."""
    return isinstance(getattr(message, "source", None), str) and hasattr(message, "content")


# Pure control signals of a group chat — no payload worth a span.
_CONTROL_EVENTS = frozenset(
    {
        "GroupChatRequestPublish",
        "GroupChatReset",
        "GroupChatPause",
        "GroupChatResume",
    }
)
# Per-token streaming deltas: the full message follows anyway.
_NOISE_EVENTS = frozenset({"ModelClientStreamingChunkEvent"})


# ---------------------------------------------------------------------------
# Intervention handler
# ---------------------------------------------------------------------------


class AutoGenTraceHandler:
    """``autogen_core.InterventionHandler`` that records eval-lib spans.

    Hook signatures match ``autogen_core`` 0.4+ exactly (the runtime calls
    them with keyword-only ``message_context=`` / ``recipient=`` /
    ``sender=``); extra keywords are accepted for forward compatibility,
    and a legacy ``sender=`` on ``on_send``/``on_publish`` still works.

    Every hook returns the message **unchanged** — even if tracing itself
    fails (the error is logged), because returning anything else alters or
    drops the user's message.

    Args:
        trace_id: Trace to record into. Defaults to the trace active when
            the handler is constructed; the runtime may run hooks from a
            task that never inherited the trace context, so the id is
            re-bound inside each hook when the context lacks one.
        usage_from_messages: Accumulate ``models_usage`` from agentchat
            messages. ``None`` (default) = yes, unless a
            :class:`TracedChatCompletionClient` already reports usage for
            this trace; ``True`` = always; ``False`` = never.
    """

    _MAX_PENDING_TOOLS = 256
    _MAX_SEEN_IDS = 4096

    def __init__(
        self,
        *,
        trace_id: Optional[str] = None,
        usage_from_messages: Optional[bool] = None,
    ) -> None:
        self._trace_id = trace_id or get_trace_id()
        self._usage_from_messages = usage_from_messages
        # Open TOOL_CALL spans waiting for their ToolCallExecutionEvent:
        # (call_id, function name, span) in request order.
        self._pending_tools: List[Tuple[str, str, TraceSpan]] = []
        # agentchat message ids already traced — the same message reaches
        # the runtime twice (GroupChatMessage log + GroupChatAgentResponse).
        self._seen_ids: "OrderedDict[str, None]" = OrderedDict()

    # -- InterventionHandler protocol ---------------------------------------

    async def on_send(
        self,
        message: Any,
        *,
        message_context: Any = None,
        recipient: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Called for ``runtime.send_message()`` (direct / RPC messages)."""
        try:
            self._bind_trace()
            sender = self._sender_from(message_context, kwargs)
            self._record(
                message,
                meta={
                    "hook": "send",
                    "sender": _agent_name(sender),
                    "recipient": _agent_name(recipient),
                },
                direction="input",
            )
        except Exception as e:  # never let tracing break the runtime
            logger.warning("eval_lib.tracing: AutoGen on_send tracing failed: %r", e)
        return message

    async def on_publish(
        self,
        message: Any,
        *,
        message_context: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Called for ``runtime.publish_message()`` (topic broadcasts)."""
        try:
            self._bind_trace()
            sender = self._sender_from(message_context, kwargs)
            meta: Dict[str, Any] = {"hook": "publish", "sender": _agent_name(sender)}
            topic = getattr(message_context, "topic_id", None)
            if topic is not None:
                meta["topic"] = _safe_str(topic)
            self._record(message, meta=meta, direction="input")
        except Exception as e:
            logger.warning("eval_lib.tracing: AutoGen on_publish tracing failed: %r", e)
        return message

    async def on_response(
        self,
        message: Any,
        *,
        sender: Any = None,
        recipient: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Called with the value an agent's RPC handler returned."""
        try:
            self._bind_trace()
            self._record(
                message,
                meta={
                    "hook": "response",
                    "sender": _agent_name(sender),
                    "recipient": _agent_name(recipient),
                },
                direction="output",
            )
        except Exception as e:
            logger.warning("eval_lib.tracing: AutoGen on_response tracing failed: %r", e)
        return message

    # -- public helpers -----------------------------------------------------

    def flush_pending(self, error: str = "tool result never observed") -> int:
        """Close tool spans whose execution result never arrived.

        Called automatically on ``GroupChatTermination``; call it yourself
        before ``tracer.end_trace()`` when a run is aborted. Returns the
        number of spans closed.
        """
        pending, self._pending_tools = self._pending_tools, []
        for _call_id, _name, span in pending:
            tracer.end_span(span, error=error)
        return len(pending)

    # -- internals ----------------------------------------------------------

    def _bind_trace(self) -> None:
        if self._trace_id and not get_trace_id():
            set_trace_id(self._trace_id)

    @staticmethod
    def _sender_from(message_context: Any, kwargs: Dict[str, Any]) -> Any:
        sender = getattr(message_context, "sender", None)
        if sender is None:
            sender = kwargs.get("sender")
        return sender

    def _active_trace_id(self) -> Optional[str]:
        return get_trace_id() or self._trace_id

    def _remember(self, message_id: str) -> bool:
        """Return ``True`` when the id is new (and remember it)."""
        if message_id in self._seen_ids:
            return False
        self._seen_ids[message_id] = None
        while len(self._seen_ids) > self._MAX_SEEN_IDS:
            self._seen_ids.popitem(last=False)
        return True

    def _record(self, message: Any, meta: Dict[str, Any], direction: str) -> None:
        """Dispatch on the (duck-typed) message class."""
        if message is None:
            return
        kind = type(message).__name__
        if kind in _CONTROL_EVENTS or kind in _NOISE_EVENTS:
            return

        # --- team-level envelopes: unwrap ---------------------------------
        if kind == "GroupChatStart":
            for task_message in getattr(message, "messages", None) or []:
                self._record_chat_message(task_message, {**meta, "task": True}, "input")
            return
        if kind == "GroupChatMessage":
            self._record(getattr(message, "message", None), meta, "output")
            return
        if kind == "GroupChatAgentResponse":
            self._record_response(getattr(message, "response", None), meta)
            return
        if kind == "Response" and hasattr(message, "chat_message"):
            self._record_response(message, meta)
            return
        if kind in ("GroupChatTeamResponse", "TaskResult"):
            result = (
                getattr(message, "result", None) if kind == "GroupChatTeamResponse" else message
            )
            for m in getattr(result, "messages", None) or []:
                self._record(m, meta, "output")
            return
        if kind == "GroupChatTermination":
            self.flush_pending()
            stop = getattr(message, "message", None)
            error = getattr(message, "error", None)
            span = tracer.start_span(
                name="termination",
                span_type=SpanType.AGENT_STEP,
                metadata={**meta, "msg_type": kind},
                parent_span_id=None,
                set_current=False,
            )
            output = _content_repr(getattr(stop, "content", None))
            if output is None and stop is not None:
                output = _safe_str(stop)
            tracer.end_span(span, output=output, error=_safe_str(error) if error else None)
            return
        if kind == "GroupChatError":
            span = tracer.start_span(
                name="group_chat_error",
                span_type=SpanType.AGENT_STEP,
                metadata={**meta, "msg_type": kind},
                parent_span_id=None,
                set_current=False,
            )
            tracer.end_span(span, error=_safe_str(getattr(message, "error", None)) or kind)
            return

        # --- agentchat messages & events ----------------------------------
        if _is_chat_message(message):
            self._record_chat_message(message, meta, direction)
            return

        # --- anything else: legacy heuristic ------------------------------
        self._record_unknown(message, meta, direction)

    def _record_response(self, response: Any, meta: Dict[str, Any]) -> None:
        if response is None:
            return
        # Inner messages were normally logged already (dedup handles it);
        # walking them keeps a bare ``Response`` fully traced.
        for inner in getattr(response, "inner_messages", None) or []:
            self._record(inner, meta, "output")
        self._record(getattr(response, "chat_message", None), meta, "output")

    def _record_chat_message(self, message: Any, meta: Dict[str, Any], direction: str) -> None:
        if message is None:
            return
        if not _is_chat_message(message):
            self._record(message, meta, direction)
            return
        kind = type(message).__name__
        if kind in _NOISE_EVENTS:
            return

        message_id = getattr(message, "id", None)
        if isinstance(message_id, str) and message_id and not self._remember(message_id):
            return  # already traced under another envelope

        source = getattr(message, "source", None) or meta.get("sender") or "unknown"
        span_meta: Dict[str, Any] = {**meta, "msg_type": kind, "source": source}
        if isinstance(message_id, str) and message_id:
            span_meta["message_id"] = message_id

        # A model call's usage is attached to exactly one message.
        self._add_usage(getattr(message, "models_usage", None))

        if kind == "ToolCallRequestEvent":
            self._open_tool_calls(message, source, span_meta)
            return
        if kind == "ToolCallExecutionEvent":
            self._close_tool_calls(message, source, span_meta)
            return

        if kind == "ToolCallSummaryMessage":
            calls = getattr(message, "tool_calls", None) or []
            span_meta["tool_calls"] = [getattr(c, "name", None) for c in calls]
        elif kind == "HandoffMessage":
            span_meta["target"] = _safe_str(getattr(message, "target", None))

        # An agent's message is what it *produced* → output. Only the task
        # messages that start a run are input. (``direction`` matters for
        # unknown payloads only, where nothing says which side they are.)
        is_task = bool(meta.get("task"))
        span_type = SpanType.REASONING if kind == "ThoughtEvent" else SpanType.AGENT_STEP
        content = _content_repr(getattr(message, "content", None))
        span = tracer.start_span(
            name=f"{'task' if is_task else 'message'}:{source}",
            span_type=span_type,
            input_data=content if is_task else None,
            metadata=span_meta,
            parent_span_id=None,
            set_current=False,
        )
        tracer.end_span(span, output=None if is_task else content)

    def _open_tool_calls(self, message: Any, source: str, meta: Dict[str, Any]) -> None:
        calls = [c for c in (getattr(message, "content", None) or []) if _is_function_call(c)]
        request = tracer.start_span(
            name=f"tool_request:{source}",
            span_type=SpanType.AGENT_STEP,
            input_data=[_content_repr(c) for c in calls],
            metadata={**meta, "tool_calls": [c.name for c in calls]},
            parent_span_id=None,
            set_current=False,
        )
        tracer.end_span(request)
        parent_id = request.span_id if request is not None else None

        for call in calls:
            call_id = getattr(call, "id", None)
            call_id = call_id if isinstance(call_id, str) else ""
            span = tracer.start_span(
                name=call.name,
                span_type=SpanType.TOOL_CALL,
                input_data=_content_repr(call.arguments),
                metadata={"call_id": call_id, "source": source, "msg_type": type(message).__name__},
                parent_span_id=parent_id,
                set_current=False,
            )
            if span is None:
                continue
            self._pending_tools.append((call_id, call.name, span))

        while len(self._pending_tools) > self._MAX_PENDING_TOOLS:
            _cid, _name, stale = self._pending_tools.pop(0)
            tracer.end_span(stale, error="tool result never observed")

    def _pop_pending(self, call_id: Any, name: Any) -> Optional[TraceSpan]:
        """Match by ``call_id``; fall back to the oldest call of that name
        (some models return empty ids)."""
        if isinstance(call_id, str) and call_id:
            for i, (cid, _n, span) in enumerate(self._pending_tools):
                if cid == call_id:
                    del self._pending_tools[i]
                    return span
        if isinstance(name, str) and name:
            for i, (cid, n, span) in enumerate(self._pending_tools):
                if n == name and not cid:
                    del self._pending_tools[i]
                    return span
        return None

    def _close_tool_calls(self, message: Any, source: str, meta: Dict[str, Any]) -> None:
        for result in getattr(message, "content", None) or []:
            call_id = getattr(result, "call_id", None)
            name = getattr(result, "name", None)
            content = _content_repr(getattr(result, "content", None))
            is_error = getattr(result, "is_error", None) is True

            span = self._pop_pending(call_id, name)
            if span is None:
                span = tracer.start_span(
                    name=name if isinstance(name, str) and name else "tool",
                    span_type=SpanType.TOOL_CALL,
                    metadata={
                        "call_id": call_id if isinstance(call_id, str) else None,
                        "source": source,
                        "msg_type": type(message).__name__,
                        "unpaired": True,
                    },
                    parent_span_id=None,
                    set_current=False,
                )
            tracer.end_span(
                span,
                output=content,
                error=(_safe_str(content) or "tool error") if is_error else None,
            )

    def _record_unknown(self, message: Any, meta: Dict[str, Any], direction: str) -> None:
        kind = type(message).__name__
        hook = meta.get("hook", "message")
        sender = meta.get("sender", "unknown")
        text = _safe_str(message)
        span_meta = {**meta, "msg_type": kind}

        if "ToolCall" in kind or "FunctionCall" in kind:
            name = getattr(message, "name", None)
            span = tracer.start_span(
                name=name if isinstance(name, str) and name else f"tool_call:{sender}",
                span_type=SpanType.TOOL_CALL,
                input_data=text if direction == "input" else None,
                metadata=span_meta,
                parent_span_id=None,
                set_current=False,
            )
            tracer.end_span(span, output=text if direction == "output" else None)
            return

        if hook == "send":
            name = f"message:{sender}→{meta.get('recipient', 'unknown')}"
        elif hook == "publish":
            name = f"publish:{sender}"
        else:
            name = f"response:{sender}"
        span = tracer.start_span(
            name=name,
            span_type=SpanType.AGENT_STEP,
            input_data=text if direction == "input" else None,
            metadata=span_meta,
            parent_span_id=None,
            set_current=False,
        )
        tracer.end_span(span, output=text if direction == "output" else None)

    def _add_usage(self, usage: Any) -> None:
        pair = _token_pair(usage)
        if pair is None or self._usage_from_messages is False:
            return
        trace_id = self._active_trace_id()
        if self._usage_from_messages is None and _client_reported_usage(trace_id):
            return  # the model client already counted this call
        tracer.add_trace_usage(
            input_tokens=pair[0],
            output_tokens=pair[1],
            calls=1,
            trace_id=trace_id,
        )


# ---------------------------------------------------------------------------
# Model-client proxy
# ---------------------------------------------------------------------------


def _model_name(client: Any) -> Optional[str]:
    """Best-effort model id of an autogen model client (never its config)."""
    for attr in ("model", "model_name"):
        value = getattr(client, attr, None)
        if isinstance(value, str) and value:
            return value
    for attr in ("_create_args", "_raw_config"):
        cfg = getattr(client, attr, None)
        if isinstance(cfg, dict) and isinstance(cfg.get("model"), str) and cfg["model"]:
            return cfg["model"]
    try:
        info = client.model_info
        family = info.get("family") if isinstance(info, dict) else None
        if isinstance(family, str) and family and family != "unknown":
            return family
    except Exception:
        pass
    return None


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name") or "tool")
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) and name else type(tool).__name__


def _messages_repr(messages: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        entry: Dict[str, Any] = {"type": type(m).__name__}
        source = getattr(m, "source", None)
        if isinstance(source, str):
            entry["source"] = source
        if hasattr(m, "content"):
            entry["content"] = _content_repr(getattr(m, "content", None))
        else:
            entry["content"] = _safe_str(m)
        thought = getattr(m, "thought", None)
        if isinstance(thought, str) and thought:
            entry["thought"] = thought
        out.append(entry)
    return out


class TracedChatCompletionClient:
    """Proxy around an ``autogen_core.models.ChatCompletionClient``.

    Intercepts ``create()`` and ``create_stream()`` to open an ``LLM_CALL``
    span per model call (input = the messages, output =
    ``CreateResult.content``, metadata = model / finish reason / tokens)
    and to accumulate ``CreateResult.usage`` into the trace. Everything
    else (``model_info``, ``count_tokens``, ``close``, ``dump_component``…)
    is delegated to the wrapped client, so it drops in wherever a client
    is expected::

        model_client = TracedChatCompletionClient(OpenAIChatCompletionClient(model="gpt-4o"))
        agent = AssistantAgent("assistant", model_client=model_client)

    Args:
        client: The real model client.
        model: Model id for the spans; auto-detected from the client when
            omitted (``model`` attribute, the client's create args, or
            ``model_info["family"]``).
        trace_id: Trace to record into when the calling context has none
            (defaults to the trace active at construction time).
    """

    def __init__(self, client: Any, *, model: Optional[str] = None, trace_id: Optional[str] = None):
        self._client = client
        self._model = model or _model_name(client)
        self._trace_id = trace_id or get_trace_id()

    # -- delegation ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not defined on the proxy itself.
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"TracedChatCompletionClient({self._client!r})"

    @property
    def wrapped(self) -> Any:
        """The underlying client."""
        return self._client

    # -- traced calls -------------------------------------------------------

    async def create(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        span = self._start(messages, kwargs, stream=False)
        try:
            result = await self._client.create(messages, *args, **kwargs)
        except BaseException as e:
            self._fail(span, e)
            raise
        self._finish(span, result)
        return result

    async def create_stream(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        span = self._start(messages, kwargs, stream=True)
        result: Any = None
        try:
            async for chunk in self._client.create_stream(messages, *args, **kwargs):
                if not isinstance(chunk, str):
                    result = chunk  # the final CreateResult
                yield chunk
        except BaseException as e:
            self._fail(span, e)
            raise
        self._finish(span, result)

    # -- internals ----------------------------------------------------------

    def _bind_trace(self) -> None:
        if self._trace_id and not get_trace_id():
            set_trace_id(self._trace_id)

    def _start(self, messages: Any, kwargs: Dict[str, Any], stream: bool) -> Optional[TraceSpan]:
        try:
            self._bind_trace()
            meta: Dict[str, Any] = {
                "model": self._model,
                "provider": type(self._client).__name__,
                "stream": stream,
            }
            tools = kwargs.get("tools") or ()
            try:
                tool_names = [_tool_name(t) for t in tools]
            except TypeError:
                tool_names = []
            if tool_names:
                meta["tools"] = tool_names
            json_output = kwargs.get("json_output")
            if json_output is not None:
                meta["json_output"] = (
                    json_output
                    if isinstance(json_output, bool)
                    else getattr(json_output, "__name__", _safe_str(json_output))
                )
            return tracer.start_span(
                name=f"llm:{self._model or 'chat'}",
                span_type=SpanType.LLM_CALL,
                input_data=_messages_repr(messages),
                metadata=meta,
                set_current=False,
            )
        except Exception as e:
            logger.warning("eval_lib.tracing: AutoGen model-call span start failed: %r", e)
            return None

    def _fail(self, span: Optional[TraceSpan], error: BaseException) -> None:
        try:
            tracer.end_span(span, error=error)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("eval_lib.tracing: AutoGen model-call span end failed: %r", e)

    def _finish(self, span: Optional[TraceSpan], result: Any) -> None:
        try:
            pair = _token_pair(getattr(result, "usage", None))
            if span is not None:
                for key in ("finish_reason", "cached"):
                    value = getattr(result, key, None)
                    if value is not None:
                        span.metadata[key] = value
                thought = getattr(result, "thought", None)
                if isinstance(thought, str) and thought:
                    span.metadata["thought"] = thought
                if pair is not None:
                    span.metadata["input_tokens"] = pair[0]
                    span.metadata["output_tokens"] = pair[1]
            output = _content_repr(getattr(result, "content", None)) if result is not None else None
            tracer.end_span(span, output=output)

            trace_id = get_trace_id() or self._trace_id
            if pair is not None:
                _mark_client_usage(trace_id)
                tracer.add_trace_usage(
                    input_tokens=pair[0],
                    output_tokens=pair[1],
                    calls=1,
                    trace_id=trace_id,
                )
            if self._model and trace_id:
                tracer.set_trace_metadata(model=self._model)
        except Exception as e:
            logger.warning("eval_lib.tracing: AutoGen model-call span end failed: %r", e)
