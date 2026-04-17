# testcases_schema.py
from pydantic import BaseModel, Field

from typing import List, Optional, Dict, Any


class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    reasoning: Optional[str] = None


class TraceStep(BaseModel):
    """A single step in an agent's execution trace."""
    step_id: Optional[str] = None
    type: str  # "tool_call", "llm_call", "reasoning", "retrieval", "agent_step"
    name: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    duration_ms: Optional[float] = None
    timestamp: Optional[float] = None
    status: Optional[str] = None  # "success", "error"
    error: Optional[str] = None
    error_type: Optional[str] = None
    parent_step_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ResourceUsage(BaseModel):
    """Resource consumption metrics for an agent run."""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    duration_ms: Optional[float] = None
    cost: Optional[float] = None
    model: Optional[str] = None


class EvalTestCase(BaseModel):
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    reasoning: Optional[str] = None
    name: Optional[str] = None

    # Reliability evaluation fields (Rabanser + Meshkov)
    execution_trace: Optional[List[TraceStep]] = None
    agent_confidence: Optional[float] = None
    perturbation_group: Optional[str] = None
    planning_steps: Optional[List[str]] = None
    resource_usage: Optional[ResourceUsage] = None


class ConversationalEvalTestCase(BaseModel):
    turns: List[EvalTestCase]
    chatbot_role: Optional[str] = None
    name: Optional[str] = Field(default=None)
