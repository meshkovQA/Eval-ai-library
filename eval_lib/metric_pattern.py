# metric_pattern.py
"""
Base classes for evaluation metrics with beautiful console logging.
"""
from typing import Optional

from eval_lib.testcases_schema import EvalTestCase, ConversationalEvalTestCase
from eval_lib.llm_client import chat_complete


class MetricPattern:
    """
    Base class for metrics that use a pattern-based approach to evaluation.
    This class is designed to be subclassed for specific metrics.
    """
    name: str  # name of the metric

    def __init__(self, model: str, threshold: float):
        self.model = model
        self.threshold = threshold


class ConversationalMetricPattern:
    """
    Base class for conversational metrics (evaluating full dialogues).
    Used for metrics like RoleAdherence, DialogueCoherence, etc.
    """
    name: str

    def __init__(self, model: str, threshold: float):
        self.model = model
        self.threshold = threshold
