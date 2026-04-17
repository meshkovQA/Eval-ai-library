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
    CustomEvalMetric,
)
from eval_lib.agent_metrics import (
    ToolCorrectnessMetric,
    TaskSuccessRateMetric,
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
    ToolsErrorMetric,
    ConversationalFlowRateMetric,
    FailureRateMetric,
    GoalAchievementRateMetric,
    RepetitivePatternDetectionMetric,
)
from eval_lib.security_metrics import (
    PromptInjectionDetectionMetric,
    JailbreakDetectionMetric,
    PIILeakageMetric,
    HarmfulContentMetric,
    PromptInjectionResistanceMetric,
    JailbreakResistanceMetric,
    PolicyComplianceMetric,
)
from eval_lib.deterministic_metrics import (
    RegexMatchMetric,
    JsonSchemaMetric,
    LengthCheckMetric,
    ContainsMetric,
    StartsWithMetric,
    EndsWithMetric,
    ExactMatchMetric,
    NonEmptyMetric,
    FormatCheckMetric,
    LanguageDetectionMetric,
)
from eval_lib.vector_metrics import (
    SemanticSimilarityMetric,
    ReferenceMatchMetric,
)


METRIC_REGISTRY = {
    # RAG Metrics
    "AnswerRelevancyMetric": {
        "class": AnswerRelevancyMetric,
        "category": "rag",
        "description": "Multi-step relevancy assessment of answers",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.6, "min": 0, "max": 1},
            {"name": "temperature", "type": "float", "default": 0.5, "min": 0, "max": 2},
        ],
    },
    "AnswerPrecisionMetric": {
        "class": AnswerPrecisionMetric,
        "category": "rag",
        "description": "Precision of answer content vs expected output",
        "requires_model": True,
        "required_fields": ["input", "actual_output", "expected_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.8, "min": 0, "max": 1},
        ],
    },
    "FaithfulnessMetric": {
        "class": FaithfulnessMetric,
        "category": "rag",
        "description": "Factuality of answer vs retrieval context",
        "requires_model": True,
        "required_fields": ["actual_output", "retrieval_context"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "temperature", "type": "float", "default": 0.5, "min": 0, "max": 2},
        ],
    },
    "ContextualRelevancyMetric": {
        "class": ContextualRelevancyMetric,
        "category": "rag",
        "description": "Relevance of retrieved context to the query",
        "requires_model": True,
        "required_fields": ["input", "retrieval_context"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.6, "min": 0, "max": 1},
            {"name": "temperature", "type": "float", "default": 0.5, "min": 0, "max": 2},
        ],
    },
    "ContextualPrecisionMetric": {
        "class": ContextualPrecisionMetric,
        "category": "rag",
        "description": "Precision of retrieved context",
        "requires_model": True,
        "required_fields": ["input", "actual_output", "expected_output", "retrieval_context"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "top_k", "type": "int", "default": None},
        ],
    },
    "ContextualRecallMetric": {
        "class": ContextualRecallMetric,
        "category": "rag",
        "description": "Completeness of retrieval context",
        "requires_model": True,
        "required_fields": ["expected_output", "retrieval_context"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
        ],
    },
    "BiasMetric": {
        "class": BiasMetric,
        "category": "rag",
        "description": "Bias detection in AI responses",
        "requires_model": True,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.8, "min": 0, "max": 1},
        ],
    },
    "ToxicityMetric": {
        "class": ToxicityMetric,
        "category": "rag",
        "description": "Toxicity level detection",
        "requires_model": True,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
        ],
    },
    "RestrictedRefusalMetric": {
        "class": RestrictedRefusalMetric,
        "category": "rag",
        "description": "Proper refusal of harmful requests",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
        ],
    },
    "GEval": {
        "class": GEval,
        "category": "rag",
        "description": "G-Eval with probability-weighted scoring and custom criteria",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "criteria", "type": "text", "default": ""},
            {"name": "n_samples", "type": "int", "default": 20},
            {"name": "sampling_temperature", "type": "float", "default": 2.0, "min": 0, "max": 5},
        ],
    },
    "CustomEvalMetric": {
        "class": CustomEvalMetric,
        "category": "rag",
        "description": "Custom verdict-based evaluation with user-defined criteria",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "name", "type": "string", "default": "CustomMetric"},
            {"name": "criteria", "type": "text", "default": ""},
            {"name": "temperature", "type": "float", "default": 0.8, "min": 0, "max": 2},
        ],
    },
    # Agent Metrics
    "ToolCorrectnessMetric": {
        "class": ToolCorrectnessMetric,
        "category": "agent",
        "description": "Evaluates tool selection appropriateness",
        "requires_model": False,
        "required_fields": ["tools_called", "expected_tools"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "exact_match", "type": "bool", "default": False},
            {"name": "check_ordering", "type": "bool", "default": False},
        ],
    },
    "TaskSuccessRateMetric": {
        "class": TaskSuccessRateMetric,
        "category": "agent",
        "description": "Measures task completion success",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "temperature", "type": "float", "default": 0.5, "min": 0, "max": 2},
        ],
    },
    "RoleAdherenceMetric": {
        "class": RoleAdherenceMetric,
        "category": "agent",
        "description": "Adherence to assigned role/persona",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "temperature", "type": "float", "default": 0.5, "min": 0, "max": 2},
            {"name": "chatbot_role", "type": "string", "default": ""},
        ],
    },
    "KnowledgeRetentionMetric": {
        "class": KnowledgeRetentionMetric,
        "category": "agent",
        "description": "Memory and context retention in conversations",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "temperature", "type": "float", "default": 0.5, "min": 0, "max": 2},
        ],
    },
    "ToolsErrorMetric": {
        "class": ToolsErrorMetric,
        "category": "agent",
        "description": "Error detection in tool usage",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "error_types", "type": "list", "default": None},
        ],
    },
    # Agent Metrics - Conversational quality (0.7.1+)
    "ConversationalFlowRateMetric": {
        "class": ConversationalFlowRateMetric,
        "category": "agent",
        "description": "Grades overall dialogue naturalness and coherence",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
        ],
    },
    "FailureRateMetric": {
        "class": FailureRateMetric,
        "category": "agent",
        "description": "Evaluates how the agent handles uncertainty — hallucination vs honest fallback",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
        ],
    },
    "GoalAchievementRateMetric": {
        "class": GoalAchievementRateMetric,
        "category": "agent",
        "description": "Measures whether the user actually got what they wanted (outcome-oriented)",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "user_goal", "type": "string", "default": None},
        ],
    },
    "RepetitivePatternDetectionMetric": {
        "class": RepetitivePatternDetectionMetric,
        "category": "agent",
        "description": "Detects loops where the agent repeats same actions/responses without progress",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
        ],
    },
    # Security Metrics - Detection
    "PromptInjectionDetectionMetric": {
        "class": PromptInjectionDetectionMetric,
        "category": "security",
        "description": "Detects prompt injection attempts",
        "requires_model": True,
        "required_fields": ["input"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "detection_method", "type": "select", "default": "llm_judge", "options": ["model", "llm_judge"]},
        ],
    },
    "JailbreakDetectionMetric": {
        "class": JailbreakDetectionMetric,
        "category": "security",
        "description": "Detects jailbreak attempts",
        "requires_model": True,
        "required_fields": ["input"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "detection_method", "type": "select", "default": "llm_judge", "options": ["model", "llm_judge"]},
        ],
    },
    "PIILeakageMetric": {
        "class": PIILeakageMetric,
        "category": "security",
        "description": "Detects personally identifiable information leakage",
        "requires_model": True,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "detection_method", "type": "select", "default": "llm_judge", "options": ["model", "llm_judge"]},
            {"name": "pii_types", "type": "list", "default": None},
        ],
    },
    "HarmfulContentMetric": {
        "class": HarmfulContentMetric,
        "category": "security",
        "description": "Detects harmful/dangerous content",
        "requires_model": True,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "detection_method", "type": "select", "default": "llm_judge", "options": ["model", "llm_judge"]},
            {"name": "harm_categories", "type": "list", "default": None},
        ],
    },
    # Security Metrics - Resistance
    "PromptInjectionResistanceMetric": {
        "class": PromptInjectionResistanceMetric,
        "category": "security",
        "description": "Evaluates resistance to prompt injection",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
        ],
    },
    "JailbreakResistanceMetric": {
        "class": JailbreakResistanceMetric,
        "category": "security",
        "description": "Evaluates jailbreak resistance",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
        ],
    },
    "PolicyComplianceMetric": {
        "class": PolicyComplianceMetric,
        "category": "security",
        "description": "Evaluates policy adherence with custom rules",
        "requires_model": True,
        "required_fields": ["input", "actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "policy_rules", "type": "list", "default": None},
        ],
    },
    # Deterministic Metrics
    "RegexMatchMetric": {
        "class": RegexMatchMetric,
        "category": "deterministic",
        "description": "Check output matches regex pattern",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "pattern", "type": "string", "default": ""},
            {"name": "full_match", "type": "bool", "default": False},
        ],
    },
    "JsonSchemaMetric": {
        "class": JsonSchemaMetric,
        "category": "deterministic",
        "description": "Validate output against JSON schema",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "schema", "type": "text", "default": "{}"},
        ],
    },
    "LengthCheckMetric": {
        "class": LengthCheckMetric,
        "category": "deterministic",
        "description": "Check output length in chars or words",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "min_length", "type": "int", "default": 0},
            {"name": "max_length", "type": "int", "default": 10000},
            {"name": "unit", "type": "select", "default": "chars", "options": ["chars", "words"]},
        ],
    },
    "ContainsMetric": {
        "class": ContainsMetric,
        "category": "deterministic",
        "description": "Check presence/absence of keywords",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "keywords", "type": "list", "default": []},
            {"name": "mode", "type": "select", "default": "any", "options": ["any", "all", "none"]},
            {"name": "case_sensitive", "type": "bool", "default": False},
        ],
    },
    "StartsWithMetric": {
        "class": StartsWithMetric,
        "category": "deterministic",
        "description": "Check output starts with prefix",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "prefix", "type": "string", "default": ""},
            {"name": "case_sensitive", "type": "bool", "default": True},
        ],
    },
    "EndsWithMetric": {
        "class": EndsWithMetric,
        "category": "deterministic",
        "description": "Check output ends with suffix",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "suffix", "type": "string", "default": ""},
            {"name": "case_sensitive", "type": "bool", "default": True},
        ],
    },
    "ExactMatchMetric": {
        "class": ExactMatchMetric,
        "category": "deterministic",
        "description": "Compare actual output with expected output",
        "requires_model": False,
        "required_fields": ["actual_output", "expected_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "case_sensitive", "type": "bool", "default": True},
            {"name": "strip_whitespace", "type": "bool", "default": True},
        ],
    },
    "NonEmptyMetric": {
        "class": NonEmptyMetric,
        "category": "deterministic",
        "description": "Check output is not empty or whitespace",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
        ],
    },
    "FormatCheckMetric": {
        "class": FormatCheckMetric,
        "category": "deterministic",
        "description": "Validate output format (email, url, phone, date)",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "format_type", "type": "select", "default": "email", "options": ["email", "url", "phone", "date"]},
        ],
    },
    "LanguageDetectionMetric": {
        "class": LanguageDetectionMetric,
        "category": "deterministic",
        "description": "Check response language",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "expected_language", "type": "string", "default": "en"},
        ],
    },
    # Vector Metrics
    "SemanticSimilarityMetric": {
        "class": SemanticSimilarityMetric,
        "category": "vector",
        "description": "Cosine similarity between output and expected output embeddings",
        "requires_model": False,
        "required_fields": ["actual_output", "expected_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "embedding_provider", "type": "select", "default": "openai", "options": ["openai", "local"]},
            {"name": "model_name", "type": "string", "default": "text-embedding-3-small"},
        ],
    },
    "ReferenceMatchMetric": {
        "class": ReferenceMatchMetric,
        "category": "vector",
        "description": "Similarity against multiple reference texts",
        "requires_model": False,
        "required_fields": ["actual_output"],
        "params": [
            {"name": "threshold", "type": "float", "default": 0.7, "min": 0, "max": 1},
            {"name": "references", "type": "list", "default": []},
            {"name": "aggregation", "type": "select", "default": "max", "options": ["max", "mean"]},
            {"name": "embedding_provider", "type": "select", "default": "openai", "options": ["openai", "local"]},
            {"name": "model_name", "type": "string", "default": "text-embedding-3-small"},
        ],
    },
}


def get_metrics_info():
    """Return metric info for the frontend (without class references)."""
    result = []
    for name, info in METRIC_REGISTRY.items():
        result.append({
            "name": name,
            "category": info["category"],
            "description": info["description"],
            "requires_model": info["requires_model"],
            "required_fields": info["required_fields"],
            "params": info["params"],
        })
    return result


def instantiate_metric(name, eval_model, params):
    """Create a metric instance from name and params."""
    info = METRIC_REGISTRY.get(name)
    if not info:
        raise ValueError(f"Unknown metric: {name}")

    cls = info["class"]
    kwargs = {}

    if info["requires_model"]:
        kwargs["model"] = eval_model

    for p in info["params"]:
        pname = p["name"]
        if pname in params and params[pname] is not None:
            kwargs[pname] = params[pname]
        elif p["default"] is not None:
            kwargs[pname] = p["default"]

    return cls(**kwargs)
