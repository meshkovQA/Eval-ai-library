# eval_lib/__init__.py

"""
Eval AI Library - Comprehensive AI Model Evaluation Framework

A powerful library for evaluating AI models with support for multiple LLM providers
and a wide range of evaluation metrics for RAG systems and AI agents.
"""

__version__ = "0.4.4"
__author__ = "Aleksandr Meshkov"

# Core evaluation functions
from eval_lib.evaluate import evaluate, evaluate_conversations
from eval_lib.utils import score_agg, extract_json_block
from eval_lib.price import model_pricing

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
    CustomLLMClient,
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
    KnowledgeRetentionMetric,
    ToolsErrorMetric
)

# Security Metrics - Detection (confidence score 0.0-1.0)
from eval_lib.security_metrics import (
    PromptInjectionDetectionMetric,
    JailbreakDetectionMetric,
    PIILeakageMetric,
    HarmfulContentMetric,
    # Security Metrics - Resistance (binary score 0 or 1)
    PromptInjectionResistanceMetric,
    JailbreakResistanceMetric,
    PolicyComplianceMetric,
)

from .dashboard_server import (
    DashboardCache
)


def __getattr__(name):
    """
    Lazy loading for data generation components.
    """
    if name == "DatasetGenerator":
        from eval_lib.datagenerator.datagenerator import DatasetGenerator
        return DatasetGenerator
    if name == "DataGenerator":  # Alias for DatasetGenerator
        from eval_lib.datagenerator.datagenerator import DatasetGenerator
        return DatasetGenerator
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
    "model_pricing",

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
    "CustomLLMClient",
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
    "ToolsErrorMetric",

    # Security Metrics - Detection (confidence 0.0-1.0)
    "PromptInjectionDetectionMetric",
    "JailbreakDetectionMetric",
    "PIILeakageMetric",
    "HarmfulContentMetric",

    # Security Metrics - Resistance (binary 0 or 1)
    "PromptInjectionResistanceMetric",
    "JailbreakResistanceMetric",
    "PolicyComplianceMetric",

    # Data Generation
    "DataGenerator",
    "DocumentLoader",

    # Utils
    "score_agg",
    "extract_json_block",

    # Dashboard
    'start_dashboard',
    'DashboardCache',
]
