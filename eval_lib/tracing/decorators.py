"""``@trace_llm`` / ``@trace_tool`` / ``@trace_step`` decorators.

One factory backs all three so a fix lands in every variant. It handles
coroutine functions, plain functions and — new — sync/async **generator**
functions: a decorated streaming call used to be dispatched to the sync
wrapper, which closed the span immediately with the generator object as
its output. Now the span stays open until the stream is exhausted and its
output is the concatenated chunks.

Failures are caught as ``BaseException`` so that ``asyncio.CancelledError``
and ``KeyboardInterrupt`` still close the span; before, a cancelled call
simply vanished from the trace.
"""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from .tracer import tracer
from .types import SpanType


def _capture_input(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Arguments as span input, minus a leading ``self``/``cls``.

    The bound instance is not an argument the reader cares about, and
    serialising a whole client object is how API keys ended up in traces.
    """
    positional = list(args)
    try:
        params = list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        params = []
    if positional and params and params[0] in ("self", "cls"):
        positional = positional[1:]
    return {"args": positional, "kwargs": kwargs}


def _join_chunks(chunks: List[Any]) -> Any:
    """Combine streamed chunks into one output value."""
    if not chunks:
        return None
    if all(isinstance(c, str) for c in chunks):
        return "".join(chunks)
    return chunks


def _make_decorator(
    span_type: SpanType,
    default_prefix: str,
    name: Optional[str],
    capture_input: bool,
    capture_output: bool,
    metadata: Optional[Dict[str, Any]],
) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{default_prefix}_{func.__name__}"

        def _start(args: tuple, kwargs: dict):
            return tracer.start_span(
                name=span_name,
                span_type=span_type,
                input_data=_capture_input(func, args, kwargs) if capture_input else None,
                metadata=metadata,
            )

        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                span = _start(args, kwargs)
                chunks: List[Any] = []
                try:
                    async for chunk in func(*args, **kwargs):
                        if capture_output:
                            chunks.append(chunk)
                        yield chunk
                except BaseException as e:
                    tracer.end_span(span, output=_join_chunks(chunks), error=e)
                    raise
                else:
                    tracer.end_span(span, output=_join_chunks(chunks))

            return async_gen_wrapper

        if inspect.isgeneratorfunction(func):

            @wraps(func)
            def gen_wrapper(*args, **kwargs):
                span = _start(args, kwargs)
                chunks: List[Any] = []
                try:
                    for chunk in func(*args, **kwargs):
                        if capture_output:
                            chunks.append(chunk)
                        yield chunk
                except BaseException as e:
                    tracer.end_span(span, output=_join_chunks(chunks), error=e)
                    raise
                else:
                    tracer.end_span(span, output=_join_chunks(chunks))

            return gen_wrapper

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = _start(args, kwargs)
                try:
                    result = await func(*args, **kwargs)
                except BaseException as e:
                    tracer.end_span(span, error=e)
                    raise
                tracer.end_span(span, output=result if capture_output else None)
                return result

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span = _start(args, kwargs)
            try:
                result = func(*args, **kwargs)
            except BaseException as e:
                tracer.end_span(span, error=e)
                raise
            tracer.end_span(span, output=result if capture_output else None)
            return result

        return sync_wrapper

    return decorator


def trace_llm(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for tracing LLM calls
    Usage:
        @trace_llm(name="openai_chat_completion")
        async def get_completion(prompt: str) -> str:
            return await openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    """
    return _make_decorator(SpanType.LLM_CALL, "llm", name, capture_input, capture_output, metadata)


def trace_tool(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator for tracing tool calls
    Usage:
        @trace_tool(name="web_search_tool")
        async def search_web(query: str) -> List[SearchResult]:
            return await web_search_api.search(query)
    """
    return _make_decorator(SpanType.TOOL_CALL, "tool", name, capture_input, capture_output, None)


def trace_step(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator for tracing agent steps
    Usage:
        @trace_step(name="reasoning_step")
        def reason_about_input(input_data: Any) -> str:
            # reasoning logic here
            return reasoning_result
    """
    return _make_decorator(SpanType.AGENT_STEP, "step", name, capture_input, capture_output, None)
