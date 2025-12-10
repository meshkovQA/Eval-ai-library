from functools import wraps
from typing import Callable, Optional, Dict, Any
from .tracer import tracer
from .types import SpanType


def trace_llm(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Decorator for tracing LLM calls
    Usage:
        @trace_llm(name="openai_chat_completion")
        async def get_completion(prompt: str) -> str:
            return await openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"llm_{func.__name__}"
            input_data = {"args": args,
                          "kwargs": kwargs} if capture_input else None

            span = tracer.start_span(
                name=span_name,
                span_type=SpanType.LLM_CALL,
                input_data=input_data,
                metadata=metadata
            )

            try:
                result = await func(*args, **kwargs)
                output = result if capture_output else None
                if span is not None:
                    tracer.end_span(span, output=output)
                return result
            except Exception as e:
                if span is not None:
                    tracer.end_span(span, error=e)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"llm_{func.__name__}"
            input_data = {"args": args,
                          "kwargs": kwargs} if capture_input else None

            span = tracer.start_span(
                name=span_name,
                span_type=SpanType.LLM_CALL,
                input_data=input_data,
                metadata=metadata
            )

            try:
                result = func(*args, **kwargs)
                output = result if capture_output else None
                if span is not None:
                    tracer.end_span(span, output=output)
                return result
            except Exception as e:
                if span is not None:
                    tracer.end_span(span, error=e)
                raise

        # Determine if the function is async or sync
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def trace_tool(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True
):
    """
    Decorator for tracing tool calls
    Usage:
        @trace_tool(name="web_search_tool")
        async def search_web(query: str) -> List[SearchResult]:
            return await web_search_api.search(query)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"tool_{func.__name__}"
            input_data = {"args": args,
                          "kwargs": kwargs} if capture_input else None

            span = tracer.start_span(
                name=span_name,
                span_type=SpanType.TOOL_CALL,
                input_data=input_data
            )

            try:
                result = await func(*args, **kwargs)
                output = result if capture_output else None
                if span is not None:
                    tracer.end_span(span, output=output)
                return result
            except Exception as e:
                if span is not None:
                    tracer.end_span(span, error=e)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"tool_{func.__name__}"
            input_data = {"args": args,
                          "kwargs": kwargs} if capture_input else None

            span = tracer.start_span(
                name=span_name,
                span_type=SpanType.TOOL_CALL,
                input_data=input_data
            )

            try:
                result = func(*args, **kwargs)
                output = result if capture_output else None
                if span is not None:
                    tracer.end_span(span, output=output)
                return result
            except Exception as e:
                if span is not None:
                    tracer.end_span(span, error=e)
                raise

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def trace_step(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True
):
    """
    Decorator for tracing agent steps
    Usage:
        @trace_step(name="reasoning_step")
        def reason_about_input(input_data: Any) -> str:
            # reasoning logic here
            return reasoning_result
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"step_{func.__name__}"
            input_data = {"args": args,
                          "kwargs": kwargs} if capture_input else None

            span = tracer.start_span(
                name=span_name,
                span_type=SpanType.AGENT_STEP,
                input_data=input_data
            )

            try:
                result = await func(*args, **kwargs)
                output = result if capture_output else None
                if span is not None:
                    tracer.end_span(span, output=output)
                return result
            except Exception as e:
                if span is not None:
                    tracer.end_span(span, error=e)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"step_{func.__name__}"
            input_data = {"args": args,
                          "kwargs": kwargs} if capture_input else None

            span = tracer.start_span(
                name=span_name,
                span_type=SpanType.AGENT_STEP,
                input_data=input_data
            )

            try:
                result = func(*args, **kwargs)
                output = result if capture_output else None
                if span is not None:
                    tracer.end_span(span, output=output)
                return result
            except Exception as e:
                if span is not None:
                    tracer.end_span(span, error=e)
                raise

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
