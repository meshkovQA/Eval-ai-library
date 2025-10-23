# eval_lib/__init__.py

"""
Eval AI Library - Comprehensive AI Model Evaluation Framework

A powerful library for evaluating AI models with support for multiple LLM providers
and a wide range of evaluation metrics for RAG systems and AI agents.
"""

__version__ = "0.3.0"
__author__ = "Aleksandr Meshkov"

# Core evaluation functions
from eval_lib.evaluate import evaluate, evaluate_conversations
from eval_lib.utils import score_agg, extract_json_block

# Test case schemas
from eval_lib.testcases_schema import (
    EvalTestCase,
    ConversationalEvalTestCase,
    ToolCall
)

# Evaluation schemas
from eval_lib.evaluation_schema import (
    MetricResult,
    TestCaseResult,
    ConversationalTestCaseResult
)

# Base patterns
from eval_lib.metric_pattern import (
    MetricPattern,
    ConversationalMetricPattern
)

# LLM client
from eval_lib.llm_client import (
    chat_complete,
    get_embeddings,
    LLMDescriptor,
    Provider
)

# RAG Metrics
from eval_lib.metrics import (
    AnswerRelevancyMetric,
    AnswerPrecisionMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    BiasMetric,
    ToxicityMetric,
    RestrictedRefusalMetric,
    GEval,
    CustomEvalMetric
)

# Agent Metrics
from eval_lib.agent_metrics import (
    ToolCorrectnessMetric,
    TaskSuccessRateMetric,
    RoleAdherenceMetric,
    KnowledgeRetentionMetric
)


def __getattr__(name):
    """
    Ленивый импорт для модулей с тяжёлыми зависимостями.
    DataGenerator импортируется только когда реально используется.
    """
    if name == "DataGenerator":
        from eval_lib.datagenerator.datagenerator import DataGenerator
        return DataGenerator
    if name == "DocumentLoader":
        from eval_lib.datagenerator.document_loader import DocumentLoader
        return DocumentLoader
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",

    # Core functions
    "evaluate",
    "evaluate_conversations",

    # Schemas
    "EvalTestCase",
    "ConversationalEvalTestCase",
    "ToolCall",
    "MetricResult",
    "TestCaseResult",
    "ConversationalTestCaseResult",

    # Patterns
    "MetricPattern",
    "ConversationalMetricPattern",

    # LLM
    "chat_complete",
    "get_embeddings",
    "LLMDescriptor",
    "Provider",

    # RAG Metrics
    "AnswerRelevancyMetric",
    "AnswerPrecisionMetric",
    "FaithfulnessMetric",
    "ContextualRelevancyMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "BiasMetric",
    "ToxicityMetric",
    "RestrictedRefusalMetric",
    "GEval",
    "CustomEvalMetric",

    # Agent Metrics
    "ToolCorrectnessMetric",
    "TaskSuccessRateMetric",
    "RoleAdherenceMetric",
    "KnowledgeRetentionMetric",

    # Data Generation
    "DataGenerator",
    "DocumentLoader",

    # Utils
    "score_agg",
    "extract_json_block",
]
