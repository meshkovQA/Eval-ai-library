# eval_lib/tracing/types.py
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, List
from datetime import datetime
import uuid


class SpanType(str, Enum):
    """Типы операций агента"""
    LLM_CALL = "llm_call"          # LLM Call
    TOOL_CALL = "tool_call"        # Tool Call
    AGENT_STEP = "agent_step"      # Agent Step
    REASONING = "reasoning"        # Reasoning Step
    RETRIEVAL = "retrieval"        # Knowledge Retrieval
    EVALUATION = "evaluation"      # Result Evaluation
    CUSTOM = "custom"              # Custom Type


@dataclass
class TraceSpan:
    """A single unit of a trace - one span"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""  # Set by the tracer
    parent_span_id: Optional[str] = None

    # Main fields
    span_type: SpanType = SpanType.CUSTOM
    name: str = ""
    start_time: float = field(
        default_factory=lambda: datetime.now().timestamp())
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None

    # Metadata
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "running"  # running, success, error
    error: Optional[str] = None
    error_type: Optional[str] = None

    def finish(self, output: Any = None, error: Optional[Exception] = None):
        """Finish the span"""
        self.end_time = datetime.now().timestamp()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)

        if error:
            self.status = "error"
            self.error = str(error)
            self.error_type = type(error).__name__
        else:
            self.status = "success"
            if output is not None:
                self.output = output

    def to_dict(self) -> dict:
        """Convert to dict for sending"""
        return {k: v for k, v in asdict(self).items() if v is not None}
