from .tracer import tracer, AgentTracer
from .decorators import trace_llm, trace_tool, trace_step
from .types import SpanType, TraceSpan
from .config import TracingConfig
from .trace_utils import extract_test_case_data


# Framework-specific callbacks each depend on their own SDK (langchain_core,
# crewai, llama_index, ...). Keep the base tracing lightweight: lazy-load every
# optional integration via __getattr__ so that `from eval_lib.tracing import
# tracer` works on a slim install.

_LAZY = {
    "EvalLibCallbackHandler":         (".langchain_callback",     ["langchain_core"]),
    "callback_handler":               (".langchain_callback",     ["langchain_core"]),
    "ClaudeAgentTraceCollector":      (".claude_agent_callback",  ["claude_agent_sdk"]),
    "OpenAIAssistantsTraceCollector": (".openai_assistants_callback", ["openai"]),
    "EvalLibSpanExporter":            (".otel_collector",         ["opentelemetry"]),
    "CrewAITraceCollector":           (".crewai_callback",        ["crewai"]),
    "smolagents_step_callback":       (".smolagents_callback",    ["smolagents"]),
    "install_llamaindex_tracing":     (".llamaindex_callback",    ["llama_index"]),
    "EvalLibEventHandler":            (".llamaindex_callback",    ["llama_index"]),
    "EvalLibSpanHandler":             (".llamaindex_callback",    ["llama_index"]),
    "AutoGenTraceHandler":            (".autogen_callback",       ["autogen"]),
    "install_sk_tracing":             (".semantic_kernel_callback", ["semantic_kernel"]),
    "install_haystack_tracing":       (".haystack_callback",      ["haystack"]),
    "EvalLibHaystackTracer":          (".haystack_callback",      ["haystack"]),
    "PhidataTraceCollector":          (".phidata_callback",       ["phi"]),
}


def __getattr__(name):
    if name in _LAZY:
        module_name, deps = _LAZY[name]
        try:
            import importlib
            mod = importlib.import_module(module_name, package=__name__)
        except ImportError as e:
            raise ImportError(
                f"{name} requires {deps!r} — install the matching extra.\n"
                f"Underlying error: {e}"
            ) from e
        return getattr(mod, name)
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
    "extract_test_case_data",
    *_LAZY.keys(),
]
