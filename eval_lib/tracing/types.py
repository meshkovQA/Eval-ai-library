# eval_lib/tracing/types.py
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timezone
import time
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

    # Monotonic clock captured at creation, so duration is immune to
    # wall-clock adjustments (NTP steps, DST) mid-span. Private: excluded
    # from to_dict(). `_created_at` remembers the original start_time so
    # finish() can tell when a caller backdated the span on purpose.
    _created_at: float = field(default=0.0, repr=False, compare=False)
    _start_perf: float = field(default=0.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._created_at = self.start_time
        self._start_perf = time.perf_counter()

    def finish(
        self,
        output: Any = None,
        error: Optional[Union[str, Exception]] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        """Finish the span.

        Args:
            output: Result payload. Recorded even on the error path — a
                failing tool's response body is usually the most useful
                thing in the span.
            error: Either an ``Exception`` or a plain string. In-band
                failures (a tool returning ``is_error=True`` with a message)
                no longer have to fabricate an exception just to be
                recorded as failed.
            status: Explicit status override (``running``/``success``/
                ``error``). Inferred from ``error`` when not given.
            error_type: Explicit error classifier. Defaults to the
                exception class name, or ``"ToolError"`` for string errors.
        """
        if self._start_perf and self.start_time == self._created_at:
            # Normal path: measure with the monotonic clock and derive
            # end_time from it so start/end/duration stay consistent.
            elapsed = time.perf_counter() - self._start_perf
            self.end_time = self.start_time + elapsed
            self.duration_ms = round(elapsed * 1000, 2)
        else:
            # The caller moved start_time (framework-supplied timestamp);
            # the monotonic reference no longer applies — use wall clock.
            self.end_time = datetime.now().timestamp()
            self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)

        if output is not None:
            self.output = output

        if error:
            self.status = status or "error"
            self.error = str(error)
            if isinstance(error, BaseException):
                self.error_type = error_type or type(error).__name__
            else:
                self.error_type = error_type or "ToolError"
        else:
            self.status = status or "success"
            if error_type:
                self.error_type = error_type

    def to_dict(self) -> dict:
        """Convert to dict for sending.

        Emits both the epoch floats (``start_time``/``end_time``) and
        ISO-8601 UTC strings (``started_at``/``ended_at``) so receivers can
        read whichever shape they model without a translation layer.
        """
        data = {
            k: v for k, v in asdict(self).items()
            if v is not None and not k.startswith("_")
        }
        if self.start_time is not None:
            data["started_at"] = _to_iso(self.start_time)
        if self.end_time is not None:
            data["ended_at"] = _to_iso(self.end_time)
        return data


def _to_iso(epoch: Optional[float]) -> Optional[str]:
    """Render an epoch timestamp as an ISO-8601 UTC string."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None
