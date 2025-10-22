# evaluation_schema.py

from dataclasses import dataclass
from typing import List, Optional, Any, Dict


@dataclass
class MetricResult:
    """
    Result of one metric evaluation.
    In pair with TestCaseResult used in save_results in the form of (MetricResult, None)
    - name: str
    - score: float
    - threshold: float
    - success: bool
    - evaluation_cost: Optional[float]
    - reason: Optional[str]
    - evaluation_model: str
    """
    name: str
    score: float
    threshold: float
    success: bool
    evaluation_cost: Optional[float]
    reason: Optional[str]
    evaluation_model: str
    # can be a dict or any other type depending on the metric
    evaluation_log: Optional[Any] = None


@dataclass
class TestCaseResult:
    """
    Final result of one test case evaluation.
    In pair with MetricResult used in save_results in the form of (None, [TestCaseResult])
    - input:             str
    - actual_output:     str    
    - expected_output:   Optional[str]
    - retrieval_context: Optional[List[str]]
    - success:           bool               — General success of the test case
    - metrics_data:      List[MetricResult] — List of metric results
    """
    input: str
    actual_output: str
    expected_output: Optional[str]
    retrieval_context: Optional[List[str]]
    success: bool
    metrics_data: List[MetricResult]
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None


@dataclass
class ConversationalTestCaseResult:
    """
    Result of a conversational test case evaluation.
    - turns: List[TestCaseResult] — List of individual test case results for each turn
    - chatbot_role: Optional[str] — Role of the chatbot in the conversation
    - name: Optional[str] — Name of the conversational test case
    """
    dialogue: List[Dict[str, str]]
    success: bool
    metrics_data: List[MetricResult]
