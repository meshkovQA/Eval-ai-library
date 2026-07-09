"""Trace transport — buffers spans per trace, ships them to a Sink.

The pluggable :class:`Sink` layer keeps ``TraceSender`` transport-agnostic:

* :class:`HTTPSink` — default. POSTs each payload to ``TRACING_URL``.
* :class:`InMemorySink` — captures payloads in a list for tests.
* :class:`FileSink` — appends payloads as JSONL for local development.

Choose the default sink with the ``TRACING_SINK`` env
(``http`` / ``memory`` / ``file``) or by passing an instance to
``TraceSender(sink=...)``.

Delivery failures are always visible: :class:`HTTPSink` logs at
``WARNING`` level via ``logging.getLogger("eval_lib.tracing")``. Setting
``TRACING_STRICT=true`` re-raises instead of logging (useful in CI).
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import aiohttp

from .config import TracingConfig
from .types import TraceSpan

logger = logging.getLogger("eval_lib.tracing")


def _safe_serialize(obj: Any, seen: set = None) -> Any:
    """Recursively serialize an object to JSON-safe types"""
    if seen is None:
        seen = set()

    # Handle None and primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Prevent infinite recursion
    obj_id = id(obj)
    if obj_id in seen:
        return f"<circular ref: {type(obj).__name__}>"
    seen.add(obj_id)

    try:
        # Handle UUID
        if hasattr(obj, 'hex'):
            return str(obj)

        # Handle dict
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v, seen) for k, v in obj.items()}

        # Handle list/tuple
        if isinstance(obj, (list, tuple)):
            return [_safe_serialize(item, seen) for item in obj]

        # Handle Pydantic models (v1 and v2)
        if hasattr(obj, 'model_dump'):
            try:
                return _safe_serialize(obj.model_dump(), seen)
            except Exception:
                pass
        if hasattr(obj, 'dict') and callable(obj.dict):
            try:
                return _safe_serialize(obj.dict(), seen)
            except Exception:
                pass

        # Handle dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            try:
                from dataclasses import asdict
                return _safe_serialize(asdict(obj), seen)
            except Exception:
                pass

        # Handle LangChain objects (AgentAction, ToolAgentAction, etc.)
        if hasattr(obj, 'to_dict'):
            try:
                return _safe_serialize(obj.to_dict(), seen)
            except Exception:
                pass

        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            try:
                result = {"_type": type(obj).__name__}
                for k, v in obj.__dict__.items():
                    if not k.startswith('_'):
                        result[k] = _safe_serialize(v, seen)
                return result
            except Exception:
                pass

        # Handle iterables
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return [_safe_serialize(item, seen) for item in obj]
            except Exception:
                pass

        # Fallback to string representation
        return str(obj)
    finally:
        seen.discard(obj_id)


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects gracefully"""

    def default(self, obj: Any) -> Any:
        return _safe_serialize(obj)


# =========================================================================
# Sink abstraction
# =========================================================================


class Sink(ABC):
    """Abstract trace transport.

    Implementations must be safe to invoke from an async context;
    :meth:`send` may be scheduled onto a running loop rather than awaited
    directly.
    """

    @abstractmethod
    async def send(self, payload: Dict[str, Any]) -> None:
        """Persist / forward a single trace payload."""


class HTTPSink(Sink):
    """POST payloads to ``TRACING_URL`` as JSON.

    Adds the configured API key as ``Authorization: Bearer …``. On
    non-2xx responses or transport failures logs a WARNING with the URL,
    status code and truncated response body — silent failures are the
    single most confusing tracing bug and we refuse to hide them.
    ``TRACING_STRICT=true`` upgrades those to raised exceptions.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        strict: Optional[bool] = None,
    ):
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._strict = strict

    def _resolve_url(self) -> str:
        return self._url if self._url is not None else TracingConfig.get_url()

    def _resolve_api_key(self) -> Optional[str]:
        return self._api_key if self._api_key is not None else TracingConfig.get_api_key()

    def _resolve_strict(self) -> bool:
        return self._strict if self._strict is not None else TracingConfig.is_strict()

    async def send(self, payload: Dict[str, Any]) -> None:
        url = self._resolve_url()
        if not url:
            # No URL configured — nothing to do. This isn't an error;
            # tracing without a URL is used e.g. when the user is only
            # exercising the collector.
            return

        headers = {"Content-Type": "application/json"}
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            data = json.dumps(payload, cls=SafeJSONEncoder)
        except Exception as e:
            msg = f"eval_lib.tracing: failed to serialize payload for {url!r}: {e}"
            if self._resolve_strict():
                raise
            logger.warning(msg)
            return

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers, timeout=timeout) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        msg = (
                            f"eval_lib.tracing: POST {url} returned {resp.status}: "
                            f"{body[:500]}"
                        )
                        if self._resolve_strict():
                            raise RuntimeError(msg)
                        logger.warning(msg)
        except aiohttp.ClientError as e:
            msg = f"eval_lib.tracing: POST {url} failed: {type(e).__name__}: {e}"
            if self._resolve_strict():
                raise
            logger.warning(msg)
        except asyncio.TimeoutError:
            msg = f"eval_lib.tracing: POST {url} timed out after {self._timeout}s"
            if self._resolve_strict():
                raise
            logger.warning(msg)


class InMemorySink(Sink):
    """Collect payloads in a list. Intended for unit tests."""

    def __init__(self):
        self.payloads: List[Dict[str, Any]] = []
        self._lock = Lock()

    async def send(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.payloads.append(payload)

    def clear(self) -> None:
        with self._lock:
            self.payloads.clear()


class FileSink(Sink):
    """Append each payload as one JSON line to ``path`` — local dev / offline runs."""

    def __init__(self, path: Optional[str] = None):
        self._path = Path(path or TracingConfig.get_sink_path())
        self._lock = Lock()

    async def send(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, cls=SafeJSONEncoder)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def _build_default_sink() -> Sink:
    kind = TracingConfig.get_sink_kind()
    if kind == "memory":
        return InMemorySink()
    if kind == "file":
        return FileSink()
    return HTTPSink()


# =========================================================================
# TraceSender
# =========================================================================


class TraceSender:
    """Buffer spans per trace, hand complete traces off to a Sink.

    Backwards-compatible: ``TraceSender()`` still uses HTTP by default,
    honours ``TRACING_URL`` + ``TRACING_API_KEY`` and preserves the same
    payload shape (``{"project": …, "trace": {…}}``).

    Two new behaviours:

    * ``TRACING_STREAM=true`` — each :meth:`flush_span` invocation ships
      one span as ``{"project": …, "trace_id": …, "partial_span": {…}}``
      so that a long-running session with a crash still leaves spans
      on the receiver.
    * A caller may pass a custom :class:`Sink` instance for tests or
      offline development.
    """

    def __init__(self, sink: Optional[Sink] = None):
        self._lock = Lock()
        # Store spans grouped by trace_id
        self._traces: Dict[str, List[TraceSpan]] = {}
        # Store trace-level metadata (model, tokens, output)
        self._trace_metadata: Dict[str, Dict[str, Any]] = {}
        self.sink: Sink = sink if sink is not None else _build_default_sink()

    # ------------------------------------------------------------------ API

    def set_trace_metadata(self, trace_id: str, metadata: Dict[str, Any]):
        """Set trace-level metadata (model, tokens, final output, cost)."""
        with self._lock:
            if trace_id not in self._trace_metadata:
                self._trace_metadata[trace_id] = {}
            self._trace_metadata[trace_id].update(metadata)

    def add_span(self, span: TraceSpan):
        """Add a span to its trace group."""
        with self._lock:
            trace_id = span.trace_id
            if trace_id not in self._traces:
                self._traces[trace_id] = []
            self._traces[trace_id].append(span)

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Return spans for ``trace_id`` (copy, non-destructive)."""
        with self._lock:
            return list(self._traces.get(trace_id, []))

    def get_trace_metadata(self, trace_id: str) -> Dict[str, Any]:
        """Return trace-level metadata for ``trace_id`` (copy, non-destructive)."""
        with self._lock:
            return dict(self._trace_metadata.get(trace_id, {}))

    def flush_trace(self, trace_id: str):
        """Ship all spans for a specific trace to the sink."""
        with self._lock:
            if trace_id not in self._traces:
                return
            spans = self._traces.pop(trace_id)
            trace_meta = self._trace_metadata.pop(trace_id, {})

        if not spans:
            return

        trace_data = self._build_trace_structure(trace_id, spans, trace_meta or {})
        payload = {"project": TracingConfig.get_project(), "trace": trace_data}
        self._dispatch(payload)

    def flush_span(self, span: TraceSpan):
        """Ship a single span as ``partial_span`` (streaming mode).

        No-op unless ``TRACING_STREAM=true``. Removes the span from the
        buffered trace so that a later ``flush_trace`` doesn't duplicate
        it.
        """
        if not TracingConfig.is_stream():
            return
        if span is None or not span.trace_id:
            return

        with self._lock:
            spans = self._traces.get(span.trace_id, [])
            try:
                spans.remove(span)
            except ValueError:
                # add_span may not have been called yet — that's fine, we
                # still ship the span out-of-band.
                pass

        payload = {
            "project": TracingConfig.get_project(),
            "trace_id": span.trace_id,
            "partial_span": span.to_dict(),
        }
        self._dispatch(payload)

    def flush(self):
        """Ship every buffered trace."""
        with self._lock:
            trace_ids = list(self._traces.keys())
        for trace_id in trace_ids:
            self.flush_trace(trace_id)

    def stop(self):
        """Stop the sender and flush remaining traces."""
        self.flush()

    # ------------------------------------------------------------- internals

    def _dispatch(self, payload: Dict[str, Any]) -> None:
        """Route ``payload`` to the configured sink.

        If an event loop is already running we schedule the coroutine;
        otherwise we spin up a temporary loop and await synchronously.
        """
        coro = self.sink.send(payload)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()
            return
        asyncio.ensure_future(coro)

    def _build_trace_structure(
        self,
        trace_id: str,
        spans: List[TraceSpan],
        trace_meta: Dict[str, Any] = None,
    ) -> dict:
        """Build hierarchical trace structure from flat spans list."""
        trace_meta = trace_meta or {}

        # Create lookup by span_id
        span_map = {span.span_id: span.to_dict() for span in spans}

        # Find root spans (no parent or parent not in this trace)
        root_spans: List[dict] = []
        child_map: Dict[str, List[dict]] = {}

        for span in spans:
            span_dict = span_map[span.span_id]
            parent_id = span.parent_span_id

            if parent_id is None or parent_id not in span_map:
                root_spans.append(span_dict)
            else:
                child_map.setdefault(parent_id, []).append(span_dict)

        # Build tree recursively
        def attach_children(span_dict: dict):
            span_id = span_dict.get("span_id")
            if span_id in child_map:
                span_dict["children"] = child_map[span_id]
                for child in span_dict["children"]:
                    attach_children(child)

        for root in root_spans:
            attach_children(root)

        # Calculate trace-level metadata
        all_times = [s.start_time for s in spans if s.start_time]
        end_times = [s.end_time for s in spans if s.end_time]

        # Calculate response_time from root span's duration_ms if available
        response_time = None
        if root_spans:
            root_duration_ms = root_spans[0].get("duration_ms")
            if root_duration_ms:
                response_time = round(root_duration_ms / 1000, 3)

        # Collect tools called during the trace (only top-level tool calls, not nested)
        tool_span_ids = {span.span_id for span in spans if span.span_type and span.span_type.value == "tool_call"}

        tools_called = []
        for span in spans:
            if span.span_type and span.span_type.value == "tool_call":
                if span.parent_span_id not in tool_span_ids:
                    tools_called.append(span.name)

        # Try to extract tokens from LLM spans if not in trace_meta
        extracted_input_tokens = 0
        extracted_output_tokens = 0
        for span in spans:
            if span.span_type and span.span_type.value == "llm_call" and span.output:
                output = span.output
                if isinstance(output, dict):
                    llm_output = output.get("llm_output", {})
                    if llm_output:
                        token_usage = llm_output.get("token_usage", {})
                        if token_usage:
                            extracted_input_tokens += token_usage.get("prompt_tokens", 0)
                            extracted_output_tokens += token_usage.get("completion_tokens", 0)

        result: Dict[str, Any] = {
            "trace_id": trace_id,
            "start_time": min(all_times) if all_times else None,
            "end_time": max(end_times) if end_times else None,
            "response_time": response_time,
            "tools_called": tools_called if tools_called else None,
            "spans": root_spans,
            "span_count": len(spans),
        }

        # First-class trace-level fields, promoted out of metadata so
        # downstream consumers (evalix runtime-eval) can slice / aggregate
        # without having to guess where they landed.
        FIRST_CLASS = (
            "model", "input", "output",
            "input_tokens", "output_tokens", "total_tokens",
            "response_time", "cost_usd", "cost_source",
            "num_turns", "session_id",
        )
        if trace_meta:
            for key in FIRST_CLASS:
                if key in trace_meta:
                    result[key] = trace_meta[key]
            # Preserve any remaining custom keys — first-class fields already
            # placed above win over anything in `result`.
            for key, value in trace_meta.items():
                if key not in result:
                    result[key] = value

        return result
