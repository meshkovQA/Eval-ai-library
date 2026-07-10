# eval_lib/tracing/tracer.py
import uuid
from typing import Optional, Dict, Any
from contextlib import contextmanager
from .types import TraceSpan, SpanType
from .config import TracingConfig
from .context import (
    get_trace_id, set_trace_id,
    get_parent_span_id, set_current_span_id,
    clear_context
)
from .sender import TraceSender


class AgentTracer:
    """Singleton tracer for managing traces and spans"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.enabled = TracingConfig.is_enabled()
        self.sender = TraceSender() if self.enabled else None
        self._initialized = True

    def start_trace(self, name: str = "agent_trace") -> str:
        """Start a new trace and return its ID.

        The identifier is bound to the current async task / thread via
        :mod:`contextvars` — concurrent ``asyncio.gather`` traces don't
        clobber each other.
        """
        if not self.enabled:
            return ""

        trace_id = str(uuid.uuid4())
        set_trace_id(trace_id)
        set_current_span_id(None)
        return trace_id

    def end_trace(self):
        """End the current trace and send all its spans."""
        if not self.enabled:
            return

        trace_id = get_trace_id()
        if trace_id and self.sender:
            # Send complete trace with all spans
            self.sender.flush_trace(trace_id)

        clear_context()

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TraceSpan]:
        """Start a new span within the current trace"""
        if not self.enabled:
            return TraceSpan(name=name, span_type=span_type)

        trace_id = get_trace_id()
        if not trace_id:
            # No active trace - return None to skip tracing this span
            return None

        parent_span_id = get_parent_span_id()

        span = TraceSpan(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            input=input_data,
            metadata=metadata or {}
        )

        # Set the current span as the parent for the next spans
        set_current_span_id(span.span_id)

        return span

    def end_span(
        self,
        span: Optional[TraceSpan],
        output: Optional[Any] = None,
        error: Optional[Exception] = None
    ):
        """Finish the span and add it to the trace"""
        if not self.enabled or span is None:
            return

        span.finish(output=output, error=error)

        if self.sender:
            self.sender.add_span(span)
            # In streaming mode (TRACING_STREAM=true) ship the span
            # immediately so a crash later in the run doesn't cost the
            # data we already have.
            self.sender.flush_span(span)

        # Restore the parent span — including the case where the parent
        # is ``None`` (span was a root). Skipping this branch used to
        # cause the next sibling span to nest under the just-closed one
        # (see the ``tools_called`` regression test).
        set_current_span_id(span.parent_span_id)

    @contextmanager
    def trace(
        self,
        name: str,
        span_type: SpanType = SpanType.AGENT_STEP,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing a block of code"""
        span = self.start_span(name, span_type, input_data, metadata)
        try:
            yield span
            self.end_span(span)
        except Exception as e:
            self.end_span(span, error=e)
            raise

    def set_trace_metadata(
        self,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        response_time: Optional[float] = None,
        cost_usd: Optional[float] = None,
        cost_source: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Set trace-level metadata (model, tokens, input/output, timing, cost).
        Call this before end_trace() to include metadata in the trace.

        Args:
            model: The model name used (e.g., "gpt-4o-mini").
            input: The input/prompt sent to the agent.
            output: The final output/response of the agent.
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
            total_tokens: Total tokens (input + output).
            response_time: Response time in seconds.
            cost_usd: Total run cost in USD — first-class field in the
                trace payload. When the caller has an authoritative cost
                (``total_cost_usd`` from ``claude_agent_sdk``), pass it
                here so consumers can rank runs by spend without having
                to reconstruct pricing.
            cost_source: Provenance tag for ``cost_usd``. Typical values:
                ``"reported"`` (from the SDK) or ``"estimated"`` (derived
                from model + tokens via ``eval_lib.model_catalog``).
            session_id: Groups multiple traces into one logical session.
                Use the same value across all sub-agent traces that are
                part of the same user request — downstream (evalix
                Runtime eval) will render them as one session with a
                combined timeline.
            user_id: Owner of the session — application-defined stable
                identifier for the human running the agent. Enables the
                "all sessions of user X" view. Optional.
            **kwargs: Any additional metadata to include.
        """
        if not self.enabled or not self.sender:
            return

        trace_id = get_trace_id()
        if not trace_id:
            return

        metadata: Dict[str, Any] = {}
        if model is not None:
            metadata["model"] = model
        if input is not None:
            metadata["input"] = input
        if output is not None:
            metadata["output"] = output
        if input_tokens is not None:
            metadata["input_tokens"] = input_tokens
        if output_tokens is not None:
            metadata["output_tokens"] = output_tokens
        if total_tokens is not None:
            metadata["total_tokens"] = total_tokens
        if response_time is not None:
            metadata["response_time"] = response_time
        if cost_usd is not None:
            metadata["cost_usd"] = cost_usd
        if cost_source is not None:
            metadata["cost_source"] = cost_source
        if session_id is not None:
            metadata["session_id"] = session_id
        if user_id is not None:
            metadata["user_id"] = user_id
        metadata.update(kwargs)

        self.sender.set_trace_metadata(trace_id, metadata)

    def flush(self):
        """Force sending all accumulated traces"""
        if self.sender:
            self.sender.flush()


# Global singleton
tracer = AgentTracer()