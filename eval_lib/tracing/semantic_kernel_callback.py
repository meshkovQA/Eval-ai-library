# eval_lib/tracing/semantic_kernel_callback.py
"""Semantic Kernel trace collector via Kernel Filters.

Creates TraceSpans from Semantic Kernel function invocations, prompt
renderings, auto-invoked (LLM-initiated) tool calls and — when a chat
completion service is wrapped with :func:`trace_chat_completion` — the
LLM calls themselves.

Span tree produced for a ``ChatCompletionAgent`` with auto function calling::

    chat_completion            LLM_CALL   (from trace_chat_completion)
    ├── MathPlugin.add         TOOL_CALL  (auto invocation, args + result)
    └── MathPlugin.multiply    TOOL_CALL

and for ``kernel.invoke(prompt_function)``::

    Plugin.summarize           LLM_CALL   (is_prompt=True, usage + model)
    └── prompt_rendering       CUSTOM     (rendered prompt, real duration)

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.semantic_kernel_callback import (
        install_sk_tracing, trace_chat_completion,
    )

    trace_id = tracer.start_trace("semantic_kernel")

    kernel = sk.Kernel()
    service = OpenAIChatCompletion(ai_model_id="gpt-4o-mini")
    kernel.add_service(service)
    install_sk_tracing(kernel, services=[service])

    # Run your SK agent / function
    result = await kernel.invoke(function, input="query")

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

import functools
import inspect
import logging
import weakref
from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .context import get_parent_span_id, get_trace_id
from .trace_utils import safe_str
from .tracer import tracer
from .types import SpanType, TraceSpan
from .usage import as_mapping, usage_from_mapping

logger = logging.getLogger("eval_lib.tracing.semantic_kernel")

# Marker SK puts in a tool's FunctionResult when it swallows the exception
# (Kernel._inner_auto_function_invoke_handler).
_SK_ERROR_PREFIX = "An error occurred while invoking the function"

# Stack of open span ids for this task — explicit parenting, independent
# of the tracer's shared "current span" pointer.
_span_stack: ContextVar[Tuple[str, ...]] = ContextVar("eval_lib_sk_span_stack", default=())

# Active auto-invocation (set by the AUTO_FUNCTION_INVOCATION filter) so the
# nested FUNCTION_INVOCATION filter enriches that span instead of opening a
# second one for the same function.
_auto_record: ContextVar[Optional[Dict[str, Any]]] = ContextVar("eval_lib_sk_auto", default=None)

# Enclosing prompt-function span, so a wrapped service call nested in it
# does not report the same usage twice.
_prompt_scope: ContextVar[Optional[Dict[str, Any]]] = ContextVar("eval_lib_sk_prompt", default=None)

# Service instances whose wrapped method is currently executing (by id) —
# ``get_chat_message_content`` delegates to ``get_chat_message_contents``
# on the same instance, which must not open a second span.
_active_services: ContextVar[FrozenSet[int]] = ContextVar(
    "eval_lib_sk_services", default=frozenset()
)

# trace_id for which the model has already been declared.
_model_declared_for: Optional[str] = None

_PRIMITIVES = (str, int, float, bool)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parent_id() -> Optional[str]:
    stack = _span_stack.get()
    return stack[-1] if stack else get_parent_span_id()


def _push(span: Optional[TraceSpan]) -> Optional[Token]:
    if span is None:
        return None
    return _span_stack.set(_span_stack.get() + (span.span_id,))


def _pop(token: Optional[Token]) -> None:
    if token is None:
        return
    try:
        _span_stack.reset(token)
    except (ValueError, RuntimeError):
        stack = _span_stack.get()
        if stack:
            _span_stack.set(stack[:-1])


def _reset(var: ContextVar, token: Optional[Token]) -> None:
    if token is None:
        return
    try:
        var.reset(token)
    except (ValueError, RuntimeError):
        pass


def _function_name(func: Any) -> str:
    if func is None:
        return "unknown"
    plugin = getattr(func, "plugin_name", None) or ""
    name = getattr(func, "name", None) or getattr(func, "fully_qualified_name", None) or "unknown"
    if not isinstance(name, str):
        name = str(name)
    if plugin and isinstance(plugin, str) and not name.startswith(f"{plugin}."):
        return f"{plugin}.{name}"
    return name


def _arguments_dict(arguments: Any) -> Any:
    if arguments is None:
        return None
    try:
        data = dict(arguments)
    except Exception:
        return safe_str(arguments)
    return {
        str(k): (v if isinstance(v, _PRIMITIVES + (dict, list)) else safe_str(v))
        for k, v in data.items()
    }


def _result_output(result: Any) -> Any:
    """Plain output for a ``FunctionResult`` (or a raw value)."""
    if result is None:
        return None
    value = (
        result
        if isinstance(result, _PRIMITIVES + (dict, list))
        else getattr(result, "value", result)
    )
    if value is None:
        return None
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        items = [v if isinstance(v, _PRIMITIVES + (dict,)) else safe_str(v) for v in value]
        return items[0] if len(items) == 1 else items
    return safe_str(value)


def _sk_error_text(result: Any) -> Optional[str]:
    """Error message SK stored in a tool result after swallowing the exception."""
    value = getattr(result, "value", result)
    if isinstance(value, str) and value.startswith(_SK_ERROR_PREFIX):
        return value
    return None


def _same_function(record: Dict[str, Any], func: Any, full_name: str) -> bool:
    recorded = record.get("function")
    if recorded is not None and func is not None and recorded is func:
        return True
    return record.get("name") == full_name


def _declare_model(span: Optional[TraceSpan], model: Optional[str]) -> None:
    global _model_declared_for
    if not model:
        return
    if span is not None:
        span.metadata["model"] = model
    trace_id = get_trace_id()
    if trace_id and _model_declared_for != trace_id:
        _model_declared_for = trace_id
        tracer.set_trace_metadata(model=model)


def _record_usage(span: Optional[TraceSpan], usage: Optional[Dict[str, int]]) -> None:
    if not usage:
        return
    if span is not None:
        span.metadata["usage"] = dict(usage)
    tracer.add_trace_usage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=usage.get("cached_tokens", 0),
        reasoning_tokens=usage.get("reasoning_tokens", 0),
        calls=1,
    )


def _merge_usage(
    total: Optional[Dict[str, int]], usage: Optional[Dict[str, int]], how: str = "sum"
) -> Optional[Dict[str, int]]:
    if not usage:
        return total
    if total is None:
        return dict(usage)
    for key, value in usage.items():
        total[key] = (total.get(key, 0) + value) if how == "sum" else max(total.get(key, 0), value)
    return total


def _content_usage(content: Any) -> Optional[Dict[str, int]]:
    """Usage from one ``ChatMessageContent`` — ``metadata["usage"]`` (CompletionUsage or dict)."""
    meta = as_mapping(getattr(content, "metadata", None))
    return usage_from_mapping(meta) if meta else None


def _content_model(content: Any) -> Optional[str]:
    model = getattr(content, "ai_model_id", None)
    return str(model) if model else None


def _contents_text(contents: List[Any]) -> Any:
    texts = []
    for c in contents:
        if isinstance(c, _PRIMITIVES + (dict,)):
            texts.append(c)
        else:
            texts.append(safe_str(c))
    if not texts:
        return None
    return texts[0] if len(texts) == 1 else texts


def _prompt_result_facts(result: Any) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """``(summed usage, model)`` for a prompt function's ``FunctionResult``."""
    value = getattr(result, "value", None)
    items = value if isinstance(value, (list, tuple)) else ([value] if value is not None else [])
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    for item in items:
        usage = _merge_usage(usage, _content_usage(item))
        model = model or _content_model(item)
    if usage is None:
        # Fallback: FunctionResult.metadata["metadata"] = [completion.metadata, ...]
        outer = as_mapping(getattr(result, "metadata", None))
        inner = outer.get("metadata") if outer else None
        if isinstance(inner, (list, tuple)):
            for meta in inner:
                usage = _merge_usage(usage, usage_from_mapping(as_mapping(meta)))
    return usage, model


# ---------------------------------------------------------------------------
# Filters (SK-free — usable with any object honouring the filter protocol)
# ---------------------------------------------------------------------------


def build_sk_filters() -> Dict[str, Callable]:
    """Build the three filter coroutines without importing Semantic Kernel.

    Returns ``{"function_invocation", "prompt_rendering",
    "auto_function_invocation"}`` → ``async (context, next)``. SK passes the
    continuation as the keyword ``next`` (``partial(filter, next=...)``), so
    that is the parameter name.
    """

    async def function_invocation_filter(context: Any, next: Callable) -> None:  # noqa: A002
        """Trace every function/plugin invocation."""
        func = getattr(context, "function", None)
        full_name = _function_name(func)
        args = _arguments_dict(getattr(context, "arguments", None))

        record = _auto_record.get()
        if (
            record is not None
            and not record.get("claimed")
            and _same_function(record, func, full_name)
        ):
            # Inner invocation of an auto-invoked function: enrich the
            # outer TOOL_CALL span instead of nesting a duplicate.
            record["claimed"] = True
            span = record.get("span")
            if span is not None and span.input is None and args is not None:
                span.input = args
            try:
                await next(context)
            except BaseException as e:
                record["error"] = e
                raise
            result = getattr(context, "result", None)
            if result is not None:
                record["output"] = _result_output(result)
            return

        is_prompt = getattr(func, "is_prompt", False) is True
        span_type = SpanType.LLM_CALL if is_prompt else SpanType.TOOL_CALL
        metadata: Dict[str, Any] = {}
        if getattr(context, "is_streaming", False) is True:
            metadata["streaming"] = True

        span = tracer.start_span(
            name=full_name,
            span_type=span_type,
            input_data=args,
            metadata=metadata,
            parent_span_id=_parent_id(),
            # A tool span is the context span while the function runs, so a
            # decorated implementation nests under it (no sibling duplicate).
            set_current=(span_type == SpanType.TOOL_CALL),
        )
        token = _push(span)
        scope_token = None
        scope: Optional[Dict[str, Any]] = None
        if is_prompt:
            scope = {"span": span, "usage_recorded": False}
            scope_token = _prompt_scope.set(scope)
        try:
            await next(context)
        except BaseException as e:
            _reset(_prompt_scope, scope_token)
            _pop(token)
            tracer.end_span(span, error=e)
            raise
        _reset(_prompt_scope, scope_token)
        _pop(token)

        result = getattr(context, "result", None)
        output = _result_output(result) if result is not None else None
        if is_prompt and span is not None:
            usage, model = _prompt_result_facts(result)
            rendered = getattr(result, "rendered_prompt", None) if result is not None else None
            if rendered and "rendered_prompt" not in span.metadata:
                span.metadata["rendered_prompt"] = safe_str(rendered)
            _declare_model(span, model)
            # A wrapped service beneath us already reported this call.
            if not (scope and scope.get("usage_recorded")):
                _record_usage(span, usage)
        tracer.end_span(span, output=output)

    async def prompt_rendering_filter(context: Any, next: Callable) -> None:  # noqa: A002
        """Trace prompt rendering with its real duration."""
        func = getattr(context, "function", None)
        span = tracer.start_span(
            name="prompt_rendering",
            span_type=SpanType.CUSTOM,
            input_data=_arguments_dict(getattr(context, "arguments", None)),
            metadata={"function": _function_name(func)},
            parent_span_id=_parent_id(),
            set_current=False,
        )
        token = _push(span)
        try:
            await next(context)
        except BaseException as e:
            _pop(token)
            tracer.end_span(span, error=e)
            raise
        _pop(token)
        rendered = getattr(context, "rendered_prompt", None)
        rendered_text = safe_str(rendered) if rendered is not None else None
        scope = _prompt_scope.get()
        if scope is not None and scope.get("span") is not None and rendered_text:
            scope["span"].metadata["rendered_prompt"] = rendered_text
        tracer.end_span(span, output=rendered_text)

    async def auto_function_invocation_filter(context: Any, next: Callable) -> None:  # noqa: A002
        """Trace AI-initiated (auto-invoked) function calls as one TOOL_CALL span."""
        func = getattr(context, "function", None)
        full_name = _function_name(func)
        args = _arguments_dict(getattr(context, "arguments", None))

        metadata: Dict[str, Any] = {"auto_invocation": True}
        for attr in ("request_sequence_index", "function_sequence_index", "function_count"):
            value = getattr(context, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                metadata[attr] = value
        call_content = getattr(context, "function_call_content", None)
        call_id = getattr(call_content, "id", None)
        if isinstance(call_id, str) and call_id:
            metadata["tool_call_id"] = call_id

        span = tracer.start_span(
            name=full_name,
            span_type=SpanType.TOOL_CALL,
            input_data=args,
            metadata=metadata,
            parent_span_id=_parent_id(),
            set_current=True,  # see the function-invocation filter above
        )
        record: Dict[str, Any] = {
            "function": func,
            "name": full_name,
            "span": span,
            "claimed": False,
            "error": None,
            "output": None,
        }
        token = _push(span)
        record_token = _auto_record.set(record)
        try:
            await next(context)
        except BaseException as e:
            _reset(_auto_record, record_token)
            _pop(token)
            tracer.end_span(span, output=record.get("output"), error=e)
            raise
        _reset(_auto_record, record_token)
        _pop(token)

        if span is not None and getattr(context, "terminate", False) is True:
            span.metadata["terminate"] = True

        function_result = getattr(context, "function_result", None)
        output = record.get("output")
        if output is None and function_result is not None:
            output = _result_output(function_result)

        error: Any = record.get("error")
        if error is None:
            exc = getattr(context, "exception", None)
            if isinstance(exc, BaseException):
                error = exc
        if error is None:
            # SK swallowed the exception and put a message in the result.
            swallowed = _sk_error_text(function_result)
            if swallowed:
                tracer.end_span(
                    span, output=output, error=swallowed, error_type="FunctionExecutionError"
                )
                return
        tracer.end_span(span, output=output, error=error)

    for fn in (
        function_invocation_filter,
        prompt_rendering_filter,
        auto_function_invocation_filter,
    ):
        fn._evallib_sk_filter = True  # type: ignore[attr-defined]

    return {
        "function_invocation": function_invocation_filter,
        "prompt_rendering": prompt_rendering_filter,
        "auto_function_invocation": auto_function_invocation_filter,
    }


# ---------------------------------------------------------------------------
# Chat completion service wrapper
# ---------------------------------------------------------------------------


def _set_attr(obj: Any, name: str, value: Any) -> None:
    """Set an instance attribute, bypassing pydantic's field validation."""
    try:
        object.__setattr__(obj, name, value)
    except Exception:
        setattr(obj, name, value)


def _find_history(args: tuple, kwargs: dict) -> Any:
    if "chat_history" in kwargs:
        return kwargs["chat_history"]
    return args[0] if args else None


def _history_messages_list(history: Any) -> Optional[list]:
    if history is None:
        return None
    if isinstance(history, list):
        return history
    messages = getattr(history, "messages", None)
    return messages if isinstance(messages, list) else None


def _message_to_plain(message: Any) -> Any:
    if isinstance(message, dict):
        return message
    role = getattr(message, "role", None)
    role = getattr(role, "value", role)
    return {"role": str(role) if role is not None else None, "content": safe_str(message)}


def _history_input(history: Any) -> Any:
    messages = _history_messages_list(history)
    if messages is None:
        return safe_str(history) if history is not None else None
    return [_message_to_plain(m) for m in messages]


def _collect_usage(
    contents: Iterable[Any], history: Any, before: Optional[int], how: str = "sum"
) -> Optional[Dict[str, int]]:
    """Usage of the returned contents plus any assistant messages the
    auto-invoke loop appended to ``history`` during the call (intermediate
    tool-calling rounds carry their own usage)."""
    usage: Optional[Dict[str, int]] = None
    for content in contents:
        usage = _merge_usage(usage, _content_usage(content), how)
    messages = _history_messages_list(history)
    if messages is not None and before is not None:
        for message in messages[before:]:
            usage = _merge_usage(usage, _content_usage(message), "sum")
    return usage


def _flatten_contents(result: Any) -> List[Any]:
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return list(result)
    return [result]


def _wrap_chat_method(
    orig: Callable, method_name: str, service: Any, model_id: Optional[str]
) -> Callable:
    service_id = id(service)
    service_name = type(service).__name__

    @functools.wraps(orig)
    async def traced(*args: Any, **kwargs: Any) -> Any:
        if service_id in _active_services.get():
            return await orig(*args, **kwargs)

        history = _find_history(args, kwargs)
        messages = _history_messages_list(history)
        before = len(messages) if messages is not None else None

        span = tracer.start_span(
            name="chat_completion",
            span_type=SpanType.LLM_CALL,
            input_data=_history_input(history),
            metadata={
                "service": service_name,
                "method": method_name,
                **({"model": model_id} if model_id else {}),
            },
            parent_span_id=_parent_id(),
            set_current=False,
        )
        token = _push(span)
        active_token = _active_services.set(_active_services.get() | {service_id})
        try:
            result = await orig(*args, **kwargs)
        except BaseException as e:
            _reset(_active_services, active_token)
            _pop(token)
            tracer.end_span(span, error=e)
            raise
        _reset(_active_services, active_token)
        _pop(token)

        contents = _flatten_contents(result)
        usage = _collect_usage(contents, history, before)
        model = model_id or next((m for m in (_content_model(c) for c in contents) if m), None)
        _declare_model(span, model)
        _record_usage(span, usage)
        scope = _prompt_scope.get()
        if scope is not None and usage:
            scope["usage_recorded"] = True
        tracer.end_span(span, output=_contents_text(contents))
        return result

    return traced


def _wrap_stream_method(
    orig: Callable, method_name: str, service: Any, model_id: Optional[str]
) -> Callable:
    service_id = id(service)
    service_name = type(service).__name__

    @functools.wraps(orig)
    async def traced(*args: Any, **kwargs: Any):
        if service_id in _active_services.get():
            async for item in orig(*args, **kwargs):
                yield item
            return

        history = _find_history(args, kwargs)
        messages = _history_messages_list(history)
        before = len(messages) if messages is not None else None

        span = tracer.start_span(
            name="chat_completion",
            span_type=SpanType.LLM_CALL,
            input_data=_history_input(history),
            metadata={
                "service": service_name,
                "method": method_name,
                "streaming": True,
                **({"model": model_id} if model_id else {}),
            },
            parent_span_id=_parent_id(),
            set_current=False,
        )
        token = _push(span)
        active_token = _active_services.set(_active_services.get() | {service_id})
        chunks: List[Any] = []

        def _finish(truncated: bool = False) -> None:
            _reset(_active_services, active_token)
            _pop(token)
            contents = [c for item in chunks for c in _flatten_contents(item)]
            # Streaming usage is cumulative (Anthropic) or final-chunk-only
            # (OpenAI): per-field max across chunks fits both; history rounds sum.
            usage = _collect_usage(contents, history, before, how="max")
            model = model_id or next((m for m in (_content_model(c) for c in contents) if m), None)
            _declare_model(span, model)
            _record_usage(span, usage)
            scope = _prompt_scope.get()
            if scope is not None and usage:
                scope["usage_recorded"] = True
            if span is not None and truncated:
                span.metadata["truncated"] = True
            text = "".join(safe_str(c) or "" for c in contents)
            tracer.end_span(span, output=text or None)

        try:
            async for item in orig(*args, **kwargs):
                chunks.append(item)
                yield item
        except GeneratorExit:
            _finish(truncated=True)
            raise
        except BaseException as e:
            _reset(_active_services, active_token)
            _pop(token)
            tracer.end_span(span, error=e)
            raise
        _finish()

    return traced


def trace_chat_completion(service: Any, model: Optional[str] = None) -> Any:
    """Wrap a chat completion service so its LLM calls become LLM_CALL spans.

    ``ChatCompletionAgent`` (and any code that talks to the service
    directly) bypasses the kernel's FUNCTION_INVOCATION filter, so those
    calls are otherwise invisible. This monkeypatches the *instance's*
    ``get_chat_message_contents``, ``get_chat_message_content`` and
    ``get_streaming_chat_message_contents`` (whichever exist). Duck-typed —
    no Semantic Kernel import.

    Each span records: input = chat history messages, output = returned
    content text, ``metadata.model`` = ``service.ai_model_id`` (or the
    model reported by the response), and token usage from every returned
    content's ``metadata["usage"]`` (plus the intermediate tool-calling
    rounds the auto-invoke loop appended to the history), accumulated via
    :meth:`AgentTracer.add_trace_usage`.

    Idempotent per instance. Returns the same service.
    """
    if getattr(service, "_evallib_traced", False) is True:
        return service

    model_id = model or getattr(service, "ai_model_id", None) or getattr(service, "model_id", None)
    model_id = str(model_id) if model_id else None

    wrapped = 0
    for method_name in ("get_chat_message_contents", "get_chat_message_content"):
        orig = getattr(service, method_name, None)
        if callable(orig) and inspect.iscoroutinefunction(orig):
            _set_attr(service, method_name, _wrap_chat_method(orig, method_name, service, model_id))
            wrapped += 1

    stream_name = "get_streaming_chat_message_contents"
    stream_orig = getattr(service, stream_name, None)
    if callable(stream_orig) and inspect.isasyncgenfunction(stream_orig):
        _set_attr(
            service, stream_name, _wrap_stream_method(stream_orig, stream_name, service, model_id)
        )
        wrapped += 1

    if wrapped == 0:
        logger.warning(
            "eval_lib.tracing: trace_chat_completion(%s) found no chat completion "
            "methods to wrap",
            type(service).__name__,
        )
        return service

    _set_attr(service, "_evallib_traced", True)
    return service


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

_installed_kernels: "weakref.WeakSet[Any]" = weakref.WeakSet()
_installed_kernel_ids: set = set()


def _is_installed(kernel: Any) -> bool:
    try:
        return kernel in _installed_kernels
    except TypeError:
        return id(kernel) in _installed_kernel_ids


def _mark_installed(kernel: Any) -> None:
    try:
        _installed_kernels.add(kernel)
    except TypeError:
        _installed_kernel_ids.add(id(kernel))


def _register(kernel: Any, filter_type: Any, fn: Callable) -> None:
    add_filter = getattr(kernel, "add_filter", None)
    if callable(add_filter):
        add_filter(filter_type, fn)
        return
    kernel.filter(filter_type=filter_type)(fn)


def install_sk_tracing(kernel: Any, services: Optional[Iterable[Any]] = None) -> Any:
    """Install eval-lib tracing filters on a Semantic Kernel instance.

    Registers FUNCTION_INVOCATION, PROMPT_RENDERING and
    AUTO_FUNCTION_INVOCATION filters (see :func:`build_sk_filters`).
    Idempotent: installing twice on the same kernel registers once.

    LLM calls made by ``ChatCompletionAgent`` (or by your own code calling
    the service directly) do not pass through kernel filters. Pass the chat
    completion service(s) in ``services`` — or call
    :func:`trace_chat_completion` yourself — to record them as LLM_CALL
    spans with token usage and model.

    Args:
        kernel: A ``semantic_kernel.Kernel`` instance.
        services: Optional iterable of chat completion service instances to
            wrap with :func:`trace_chat_completion`.

    Returns:
        The kernel.
    """
    try:
        from semantic_kernel.filters.filter_types import FilterTypes
    except ImportError:
        raise ImportError("Semantic Kernel is required. Install with: pip install semantic-kernel")

    if _is_installed(kernel):
        logger.debug("eval_lib.tracing: SK tracing already installed on kernel %r", id(kernel))
    else:
        filters = build_sk_filters()
        _register(kernel, FilterTypes.FUNCTION_INVOCATION, filters["function_invocation"])
        _register(kernel, FilterTypes.PROMPT_RENDERING, filters["prompt_rendering"])
        try:
            _register(
                kernel, FilterTypes.AUTO_FUNCTION_INVOCATION, filters["auto_function_invocation"]
            )
        except Exception as e:
            logger.warning(
                "eval_lib.tracing: could not register AUTO_FUNCTION_INVOCATION filter "
                "(auto-invoked tool calls will not be traced): %r",
                e,
            )
        _mark_installed(kernel)

    for service in services or ():
        trace_chat_completion(service)

    return kernel
