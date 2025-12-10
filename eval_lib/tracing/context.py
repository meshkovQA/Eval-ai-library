from contextvars import ContextVar
from typing import Optional
from threading import Lock

# Context variables for thread-safe operation
_current_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)

# Global fallback storage for cross-thread access
_global_lock = Lock()
_global_trace_id: Optional[str] = None
_global_span_id: Optional[str] = None


def get_trace_id() -> Optional[str]:
    """Get current trace_id, with fallback to global storage"""
    trace_id = _current_trace_id.get()
    if trace_id is None:
        with _global_lock:
            return _global_trace_id
    return trace_id


def set_trace_id(trace_id: Optional[str]):
    """Set trace_id in both context var and global storage"""
    global _global_trace_id
    _current_trace_id.set(trace_id)
    with _global_lock:
        _global_trace_id = trace_id


def get_parent_span_id() -> Optional[str]:
    """Get current parent span_id, with fallback to global storage"""
    span_id = _current_span_id.get()
    if span_id is None:
        with _global_lock:
            return _global_span_id
    return span_id


def set_current_span_id(span_id: Optional[str]):
    """Set current span_id in both context var and global storage"""
    global _global_span_id
    _current_span_id.set(span_id)
    with _global_lock:
        _global_span_id = span_id


def clear_context():
    """Clear both context vars and global storage"""
    global _global_trace_id, _global_span_id
    _current_trace_id.set(None)
    _current_span_id.set(None)
    with _global_lock:
        _global_trace_id = None
        _global_span_id = None