# eval_lib/tracing/tracer.py
import atexit
import contextvars
import functools
import logging
import uuid
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
from .types import TraceSpan, SpanType
from .config import TracingConfig
from .context import (
    get_trace_id, set_trace_id,
    get_parent_span_id, set_current_span_id,
    clear_context
)
from .sender import TraceSender

# Sentinel: "no explicit parent given — inherit from the async context".
# Distinct from None, which is a valid explicit value meaning "root span".
_USE_CONTEXT = object()

logger = logging.getLogger("eval_lib.tracing")


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
        if self.enabled:
            # Ship whatever is still buffered when the interpreter exits.
            # Without this, a run that never reached end_trace() (crash,
            # sys.exit, forgotten call) left its spans in memory and lost
            # them silently.
            atexit.register(self._flush_at_exit)

    def _flush_at_exit(self) -> None:
        try:
            self.flush()
        except Exception as e:  # pragma: no cover - best effort at shutdown
            logger.warning("eval_lib.tracing: flush at exit failed: %r", e)

    def start_trace(self, name: str = "agent_trace", trace_id: Optional[str] = None) -> str:
        """Start a new trace and return its ID.

        The identifier is bound to the current async task / thread via
        :mod:`contextvars` — concurrent ``asyncio.gather`` traces don't
        clobber each other.

        Args:
            name: Human-readable label (kept for API compatibility).
            trace_id: Use this id instead of minting one — lets a caller
                correlate the trace with an upstream request id.

        If a trace is already active in this context it is flushed first
        and a warning is logged: previously its spans stayed in the buffer
        forever, unsent and unreported.
        """
        if not self.enabled:
            return ""

        previous = get_trace_id()
        if previous and self.sender and self.sender.has_trace(previous):
            logger.warning(
                "eval_lib.tracing: start_trace() called while trace %s is still "
                "active — flushing it. Call end_trace() first to silence this.",
                previous,
            )
            self.sender.flush_trace(previous)

        trace_id = trace_id or str(uuid.uuid4())
        set_trace_id(trace_id)
        set_current_span_id(None)
        return trace_id

    def end_trace(self, trace_id: Optional[str] = None):
        """End a trace and send everything buffered for it.

        Args:
            trace_id: End this specific trace. Required when ending from a
                different thread than the one that started it (a plain
                ``threading.Thread`` does not inherit contextvars), or from
                a place that never had the trace context. Defaults to the
                trace active in the current context.
        """
        if not self.enabled:
            return

        active = get_trace_id()
        target = trace_id or active
        if not target:
            logger.warning(
                "eval_lib.tracing: end_trace() called with no active trace — "
                "nothing flushed. Pass trace_id=... when ending from another "
                "thread or context."
            )
            return

        if self.sender:
            # Send complete trace with all spans
            self.sender.flush_trace(target)

        if target == active:
            clear_context()

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_span_id: Any = _USE_CONTEXT,
        set_current: bool = True,
    ) -> Optional[TraceSpan]:
        """Start a new span within the current trace.

        Args:
            parent_span_id: Explicit parent. Defaults to the span currently
                open in this async context. Pass an explicit value (including
                ``None`` for a root) when the caller tracks its own hierarchy
                — e.g. a framework that reports parentage itself.
            set_current: Whether this span becomes the context parent for
                subsequently-created spans. Integrations that supply their own
                ``parent_span_id`` should pass ``False``: moving the shared
                contextvar *and* overriding parentage are two competing
                mechanisms, and mixing them mis-nests concurrent spans.
        """
        if not self.enabled:
            return TraceSpan(name=name, span_type=span_type)

        trace_id = get_trace_id()
        if not trace_id:
            # No active trace - return None to skip tracing this span
            return None

        if parent_span_id is _USE_CONTEXT:
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
        if set_current:
            set_current_span_id(span.span_id)

        return span

    def end_span(
        self,
        span: Optional[TraceSpan],
        output: Optional[Any] = None,
        error: Optional[Union[str, Exception]] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        """Finish the span and add it to the trace.

        ``error`` accepts a string as well as an ``Exception``, so a tool
        that signals failure in-band (``is_error=True``) can be recorded
        without inventing an artificial exception. ``status`` overrides the
        inferred value when the caller knows better.
        """
        if not self.enabled or span is None:
            return

        span.finish(output=output, error=error, status=status, error_type=error_type)

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
        #
        # Only rewind when this span is the one currently on top. Spans
        # created with ``set_current=False`` never took the pointer, and
        # under concurrency spans can finish out of order — in both cases
        # rewinding would hijack an unrelated span's parent.
        if get_parent_span_id() == span.span_id:
            set_current_span_id(span.parent_span_id)

    @contextmanager
    def trace(
        self,
        name: str,
        span_type: SpanType = SpanType.AGENT_STEP,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing a block of code.

        Catches ``BaseException`` so that ``asyncio.CancelledError`` and
        ``KeyboardInterrupt`` still close the span (recorded as an error
        with the real class name) instead of leaving it unfinished and
        the context pointer stuck on it.
        """
        span = self.start_span(name, span_type, input_data, metadata)
        try:
            yield span
        except BaseException as e:
            self.end_span(span, error=e)
            raise
        else:
            self.end_span(span)

    def set_trace_metadata(
        self,
        model: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
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
            cached_tokens: Prompt tokens served from the provider's cache.
                Billed at a fraction of the input rate — in a multi-agent
                loop that resends the whole history each turn, ignoring
                this overstates cost several-fold.
            reasoning_tokens: Reasoning/thinking tokens, billed at the
                output rate and already included in ``output_tokens``.
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
        if cached_tokens is not None:
            metadata["cached_tokens"] = cached_tokens
        if reasoning_tokens is not None:
            metadata["reasoning_tokens"] = reasoning_tokens
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

    def add_trace_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
        calls: int = 1,
        trace_id: Optional[str] = None,
    ) -> None:
        """Accumulate usage for the active trace. Call once per LLM call.

        Unlike :meth:`set_trace_metadata` — which overwrites — every value
        here is **added** to a running total that is emitted as the
        ``usage`` block of the trace payload. This is the method framework
        callbacks should use for per-call token counts.

        Args:
            input_tokens: Prompt tokens for this call (including cached).
            output_tokens: Completion tokens (including reasoning).
            cached_tokens: Prompt tokens served from cache.
            reasoning_tokens: Thinking tokens (subset of ``output_tokens``).
            cost_usd: Cost of this call, if known.
            calls: Number of LLM calls this represents (default 1).
            trace_id: Target trace; defaults to the active one.
        """
        if not self.enabled or not self.sender:
            return
        target = trace_id or get_trace_id()
        if not target:
            return
        self.sender.add_trace_usage(
            target,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost_usd,
            calls=calls,
        )

    def wrap(self, fn):
        """Carry the current trace context into another thread.

        A plain ``threading.Thread`` / ``ThreadPoolExecutor`` worker starts
        with an empty context, so spans created there silently belong to no
        trace. Wrap the callable at submission time — inside the context
        that owns the trace — and each invocation runs in a private copy of
        it::

            executor.submit(tracer.wrap(fetch_page), url)

        Each call gets its own copy, so the same wrapped function may be
        submitted many times concurrently.
        """
        ctx = contextvars.copy_context()

        @functools.wraps(fn)
        def runner(*args, **kwargs):
            return ctx.copy().run(fn, *args, **kwargs)

        return runner

    def flush(self):
        """Force sending all accumulated traces.

        Schedules delivery; it does not wait for it. In an async process
        prefer :meth:`aflush`, which awaits the sends — otherwise a
        shutdown right after ``flush()`` can abandon them.
        """
        if self.sender:
            self.sender.flush()

    async def aflush(self):
        """Flush traces and await every in-flight delivery."""
        if self.sender:
            await self.sender.aflush()

    async def aclose(self):
        """Await pending deliveries, then release transport resources.

        Call once on application shutdown.
        """
        if self.sender:
            await self.sender.aclose()

    @property
    def stats(self) -> Optional[Dict[str, int]]:
        """Delivery counters (``sent``/``failed``/``retried``/``dropped``).

        ``None`` when tracing is disabled. Use it to assert in tests or a
        health check that traces are actually landing.
        """
        if not self.sender:
            return None
        return self.sender.stats.as_dict()


# Global singleton
tracer = AgentTracer()