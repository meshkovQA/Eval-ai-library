# eval_lib/tracing/semantic_kernel_callback.py
"""Semantic Kernel trace collector via Kernel Filters.

Creates TraceSpans from Semantic Kernel function invocations
and prompt renderings.

Usage:
    from eval_lib.tracing import tracer
    from eval_lib.tracing.semantic_kernel_callback import install_sk_tracing

    trace_id = tracer.start_trace("semantic_kernel")

    kernel = sk.Kernel()
    install_sk_tracing(kernel)

    # Run your SK agent
    result = await kernel.invoke(function, input="query")

    from eval_lib.tracing.trace_utils import extract_test_case_data
    data = extract_test_case_data(trace_id)
    tracer.end_trace()
"""

from typing import Any
from .types import SpanType
from .tracer import tracer


def install_sk_tracing(kernel: Any):
    """Install eval-lib tracing filters on a Semantic Kernel instance.

    Args:
        kernel: A semantic_kernel.Kernel instance.
    """
    try:
        from semantic_kernel.filters.filter_types import FilterTypes
    except ImportError:
        raise ImportError(
            "Semantic Kernel is required. Install with: pip install semantic-kernel"
        )

    @kernel.filter(filter_type=FilterTypes.FUNCTION_INVOCATION)
    async def function_filter(context: Any, next_fn: Any):
        """Trace every function/plugin invocation."""
        func = context.function
        plugin_name = getattr(func, "plugin_name", "") or ""
        func_name = getattr(func, "name", "unknown")
        full_name = f"{plugin_name}.{func_name}" if plugin_name else func_name

        # Determine span type
        name_lower = full_name.lower()
        if any(kw in name_lower for kw in ("chat", "complete", "generate")):
            span_type = SpanType.LLM_CALL
        elif any(kw in name_lower for kw in ("search", "retriev")):
            span_type = SpanType.RETRIEVAL
        else:
            span_type = SpanType.TOOL_CALL

        args = None
        if context.arguments:
            try:
                args = dict(context.arguments)
            except Exception:
                args = str(context.arguments)

        span = tracer.start_span(
            name=full_name,
            span_type=span_type,
            input_data=args,
        )

        try:
            await next_fn(context)
            if span:
                result = None
                if context.result:
                    result = str(context.result.value)[:1000] if hasattr(context.result, 'value') else str(context.result)[:1000]
                tracer.end_span(span, output=result)
        except Exception as e:
            if span:
                tracer.end_span(span, error=e)
            raise

    @kernel.filter(filter_type=FilterTypes.PROMPT_RENDERING)
    async def prompt_filter(context: Any, next_fn: Any):
        """Trace prompt rendering."""
        await next_fn(context)
        rendered = getattr(context, "rendered_prompt", None)
        if rendered:
            span = tracer.start_span(
                name="prompt_rendering",
                span_type=SpanType.REASONING,
            )
            if span:
                tracer.end_span(span, output=str(rendered)[:1000])

    try:
        # Also install auto-function filter if available
        @kernel.filter(filter_type=FilterTypes.AUTO_FUNCTION_INVOCATION)
        async def auto_function_filter(context: Any, next_fn: Any):
            """Trace AI-initiated function calls."""
            func_name = "auto_function"
            if context.function:
                func_name = getattr(context.function, "name", func_name)

            span = tracer.start_span(
                name=f"auto:{func_name}",
                span_type=SpanType.TOOL_CALL,
                metadata={"auto_invocation": True},
            )

            try:
                await next_fn(context)
                if span:
                    result = str(context.function_result)[:1000] if hasattr(context, 'function_result') else None
                    tracer.end_span(span, output=result)
            except Exception as e:
                if span:
                    tracer.end_span(span, error=e)
                raise
    except Exception:
        pass  # AUTO_FUNCTION_INVOCATION may not be available in all versions
