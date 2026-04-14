from .tracer import tracer, AgentTracer
from .decorators import trace_llm, trace_tool, trace_step
from .types import SpanType, TraceSpan
from .config import TracingConfig

# EvalLibCallbackHandler depends on langchain_core, which lives in the
# [langchain] extra. Lazy-load via __getattr__ so that `from eval_lib.tracing
# import tracer` works on a slim install without langchain installed.


def __getattr__(name):
    if name in ("EvalLibCallbackHandler", "callback_handler"):
        try:
            from .langchain_callback import (
                EvalLibCallbackHandler,
                callback_handler,
            )
        except ImportError as e:
            raise ImportError(
                "EvalLibCallbackHandler requires the 'langchain' extra "
                "(langchain-core).\n"
                "Install with: pip install eval-ai-library[langchain]\n"
                f"Underlying error: {e}"
            ) from e
        return EvalLibCallbackHandler if name == "EvalLibCallbackHandler" else callback_handler
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "tracer",
    "AgentTracer",
    "trace_llm",
    "trace_tool",
    "trace_step",
    "SpanType",
    "TraceSpan",
    "TracingConfig",
    "EvalLibCallbackHandler",
    "callback_handler",
]
