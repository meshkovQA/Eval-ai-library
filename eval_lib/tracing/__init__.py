from .tracer import tracer, AgentTracer
from .decorators import trace_llm, trace_tool, trace_step
from .types import SpanType, TraceSpan
from .config import TracingConfig
from .langchain_callback import EvalLibCallbackHandler, callback_handler
from .trace_utils import extract_test_case_data
from .claude_agent_callback import ClaudeAgentTraceCollector
from .openai_assistants_callback import OpenAIAssistantsTraceCollector
from .otel_collector import EvalLibSpanExporter
from .crewai_callback import CrewAITraceCollector
from .smolagents_callback import smolagents_step_callback
from .llamaindex_callback import EvalLibEventHandler, EvalLibSpanHandler, install_llamaindex_tracing
from .autogen_callback import AutoGenTraceHandler
from .semantic_kernel_callback import install_sk_tracing
from .haystack_callback import EvalLibHaystackTracer, install_haystack_tracing
from .phidata_callback import PhidataTraceCollector

__all__ = [
    # Core
    "tracer",
    "AgentTracer",
    "trace_llm",
    "trace_tool",
    "trace_step",
    "SpanType",
    "TraceSpan",
    "TracingConfig",
    "extract_test_case_data",

    # Universal
    "EvalLibSpanExporter",  # OpenTelemetry — works with any OTEL-instrumented framework

    # SDK integrations
    "EvalLibCallbackHandler",  # LangChain / LangGraph
    "callback_handler",        # LangChain singleton instance
    "ClaudeAgentTraceCollector",  # Anthropic Claude Agent SDK
    "OpenAIAssistantsTraceCollector",  # OpenAI Assistants API

    # Framework integrations
    "CrewAITraceCollector",     # CrewAI
    "smolagents_step_callback", # Smolagents (Hugging Face)
    "install_llamaindex_tracing",  # LlamaIndex
    "EvalLibEventHandler",      # LlamaIndex event handler
    "EvalLibSpanHandler",       # LlamaIndex span handler
    "AutoGenTraceHandler",      # AutoGen (Microsoft)
    "install_sk_tracing",       # Semantic Kernel
    "install_haystack_tracing", # Haystack (deepset)
    "EvalLibHaystackTracer",    # Haystack tracer class
    "PhidataTraceCollector",    # Phidata / Agno
]
