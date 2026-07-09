"""Async-/thread-safe context storage for the active trace and span.

All trace/span identifiers live in :class:`~contextvars.ContextVar` so that
concurrent ``asyncio`` tasks each see their own value — a prerequisite for
running multiple traces in parallel (see ``tests/test_tracer_concurrency.py``).
"""

from contextvars import ContextVar
from typing import Optional

_current_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)


def get_trace_id() -> Optional[str]:
    """Return the trace_id for the current async task / thread, or ``None``."""
    return _current_trace_id.get()


def set_trace_id(trace_id: Optional[str]) -> None:
    """Bind ``trace_id`` to the current async task / thread."""
    _current_trace_id.set(trace_id)


def get_parent_span_id() -> Optional[str]:
    """Return the currently-open span for the current task, or ``None``."""
    return _current_span_id.get()


def set_current_span_id(span_id: Optional[str]) -> None:
    """Bind ``span_id`` as the current parent for the current task."""
    _current_span_id.set(span_id)


def clear_context() -> None:
    """Reset trace + span binding for the current task."""
    _current_trace_id.set(None)
    _current_span_id.set(None)
