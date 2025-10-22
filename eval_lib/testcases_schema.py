# testcases_schema.py
from pydantic import BaseModel, Field

from typing import List, Optional


class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    reasoning: Optional[str] = None


class EvalTestCase(BaseModel):
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    reasoning: Optional[str] = None
    name: Optional[str] = None


class ConversationalEvalTestCase(BaseModel):
    turns: List[EvalTestCase]
    chatbot_role: Optional[str] = None
    name: Optional[str] = Field(default=None)
