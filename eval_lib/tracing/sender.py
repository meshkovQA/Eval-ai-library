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
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import aiohttp

from .config import TracingConfig
from .spans import top_level_tool_calls
from .types import TraceSpan, _to_iso
from .usage import span_token_usage as _span_token_usage

logger = logging.getLogger("eval_lib.tracing")


# Key names whose values are secrets. Matched case-insensitively as a
# substring of the key, so `openai_api_key`, `X-Api-Key`, `authToken` and
# `AWS_SECRET_ACCESS_KEY` are all caught. Captured `self` objects from
# decorated methods routinely carry a live client with its api_key.
_SECRET_KEY_FRAGMENTS = (
    "api_key", "apikey", "api-key", "authorization", "auth_token", "authtoken",
    "access_token", "refresh_token", "secret", "password", "passwd",
    "private_key", "client_secret", "bearer", "credential", "session_token",
    "signing_key", "x-api-key",
)
# Value shapes that are secrets regardless of the key they sit under.
_SECRET_VALUE_RE = re.compile(
    r"^(?:Bearer\s+\S+|sk-[A-Za-z0-9_\-]{8,}|sk-ant-[A-Za-z0-9_\-]{8,}"
    r"|AKIA[0-9A-Z]{16}|gsk_[A-Za-z0-9]{8,}|xai-[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_\-]{20,})$"
)
_REDACTED = "***REDACTED***"


def _is_secret_key(key: Any) -> bool:
    k = str(key).lower()
    return any(fragment in k for fragment in _SECRET_KEY_FRAGMENTS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str) and _SECRET_VALUE_RE.match(value.strip()):
        return _REDACTED
    return value


def _safe_serialize(obj: Any, seen: set = None, *, redact: Optional[bool] = None) -> Any:
    """Recursively serialize an object to JSON-safe types.

    Secrets are redacted on the way out (``TRACING_REDACT=false`` disables):
    any mapping/attribute whose name looks like a credential, and any
    string value shaped like a well-known token. A trace is shipped to a
    collector and stored; an API key must never ride along.
    """
    if seen is None:
        seen = set()
    if redact is None:
        redact = TracingConfig.is_redact_enabled()

    # Handle None and primitives
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _redact_value(obj) if redact else obj

    # Prevent infinite recursion
    obj_id = id(obj)
    if obj_id in seen:
        return f"<circular ref: {type(obj).__name__}>"
    seen.add(obj_id)

    def _item(key: Any, value: Any) -> Any:
        if redact and _is_secret_key(key) and value is not None:
            return _REDACTED
        return _safe_serialize(value, seen, redact=redact)

    try:
        # Handle UUID (an explicit type check — `hasattr(obj, "hex")` also
        # matched bytes and any object with a field called `hex`).
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, (bytes, bytearray)):
            return f"<{len(obj)} bytes>"

        # Handle dict
        if isinstance(obj, dict):
            return {str(k): _item(k, v) for k, v in obj.items()}

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
                        result[k] = _item(k, v)
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


class DeliveryStats:
    """Counters describing what actually happened to trace payloads.

    Tracing that fails silently is indistinguishable from tracing that
    works, so every sink keeps a tally the caller can assert on:
    ``tracer.sender.stats.as_dict()``.
    """

    __slots__ = ("sent", "failed", "retried", "dropped", "_lock")

    def __init__(self) -> None:
        self.sent = 0
        self.failed = 0
        self.retried = 0
        self.dropped = 0
        self._lock = Lock()

    def record_sent(self) -> None:
        with self._lock:
            self.sent += 1

    def record_failed(self) -> None:
        with self._lock:
            self.failed += 1

    def record_retried(self) -> None:
        with self._lock:
            self.retried += 1

    def record_dropped(self) -> None:
        with self._lock:
            self.dropped += 1

    def as_dict(self) -> Dict[str, int]:
        with self._lock:
            return {
                "sent": self.sent,
                "failed": self.failed,
                "retried": self.retried,
                "dropped": self.dropped,
            }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DeliveryStats({self.as_dict()})"


class Sink(ABC):
    """Abstract trace transport.

    Implementations must be safe to invoke from an async context;
    :meth:`send` may be scheduled onto a running loop rather than awaited
    directly.
    """

    def __init__(self) -> None:
        self.stats = DeliveryStats()

    @abstractmethod
    async def send(self, payload: Dict[str, Any]) -> None:
        """Persist / forward a single trace payload."""

    async def aclose(self) -> None:
        """Release transport resources. No-op unless overridden."""
        return None


class HTTPSink(Sink):
    """POST payloads to ``TRACING_URL`` as JSON.

    Adds the configured API key as ``Authorization: Bearer …``. On
    non-2xx responses or transport failures logs a WARNING with the URL,
    status code and truncated response body — silent failures are the
    single most confusing tracing bug and we refuse to hide them.
    ``TRACING_STRICT=true`` upgrades those to raised exceptions.
    """

    # Statuses worth another attempt: the server is busy or broken, not
    # the payload. Everything else in 4xx is a contract bug — retrying it
    # just burns time and hides the real problem.
    RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        strict: Optional[bool] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
    ):
        super().__init__()
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._strict = strict
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        # One session per event loop, reused across traces. Creating a
        # ClientSession per payload meant a fresh TCP+TLS handshake for
        # every trace; a pooled session keeps connections warm.
        self._sessions: Dict[int, "aiohttp.ClientSession"] = {}

    def _resolve_url(self) -> str:
        return self._url if self._url is not None else TracingConfig.get_url()

    def _resolve_api_key(self) -> Optional[str]:
        return self._api_key if self._api_key is not None else TracingConfig.get_api_key()

    def _resolve_strict(self) -> bool:
        return self._strict if self._strict is not None else TracingConfig.is_strict()

    def _resolve_max_retries(self) -> int:
        return self._max_retries if self._max_retries is not None else TracingConfig.get_max_retries()

    def _resolve_backoff(self) -> float:
        return self._retry_backoff if self._retry_backoff is not None else TracingConfig.get_retry_backoff()

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Return a pooled session bound to the running loop.

        Sessions are loop-affine, so they are keyed by loop identity and
        replaced whenever the cached one has been closed.
        """
        loop = asyncio.get_running_loop()
        key = id(loop)
        session = self._sessions.get(key)
        # getattr: test doubles and older clients may not expose `closed`.
        if session is not None and not getattr(session, "closed", False):
            return session
        session = aiohttp.ClientSession()
        self._sessions[key] = session
        return session

    async def aclose(self) -> None:
        """Close every pooled session."""
        sessions, self._sessions = self._sessions, {}
        for session in sessions.values():
            if getattr(session, "closed", False):
                continue
            close = getattr(session, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def _report(self, msg: str) -> None:
        """Record a delivery failure — loudly, or fatally under strict."""
        self.stats.record_failed()
        if self._resolve_strict():
            raise RuntimeError(msg)
        logger.warning(msg)

    async def send(self, payload: Dict[str, Any]) -> None:
        url = self._resolve_url()
        if not url:
            # No URL configured — nothing to do. This isn't an error;
            # tracing without a URL is used e.g. when the user is only
            # exercising the collector.
            self.stats.record_dropped()
            return

        headers = {"Content-Type": "application/json"}
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            data = json.dumps(payload, cls=SafeJSONEncoder)
        except Exception as e:
            self._report(f"eval_lib.tracing: failed to serialize payload for {url!r}: {e}")
            return

        attempts = self._resolve_max_retries() + 1
        delay = self._resolve_backoff()
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        last_error = f"eval_lib.tracing: POST {url} failed"

        for attempt in range(1, attempts + 1):
            try:
                session = await self._get_session()
                async with session.post(url, data=data, headers=headers, timeout=timeout) as resp:
                    if resp.status < 400:
                        self.stats.record_sent()
                        return

                    body = await resp.text()
                    last_error = (
                        f"eval_lib.tracing: POST {url} returned {resp.status}: {body[:500]}"
                    )
                    if resp.status not in self.RETRYABLE_STATUSES:
                        # Client error — the payload or auth is wrong.
                        # Retrying cannot help.
                        self._report(last_error)
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = (
                    f"eval_lib.tracing: POST {url} failed: {type(e).__name__}: {e}"
                )

            if attempt < attempts:
                self.stats.record_retried()
                logger.debug("%s — retrying (%d/%d)", last_error, attempt, attempts - 1)
                if delay > 0:
                    await asyncio.sleep(delay)
                    delay *= 2

        self._report(f"{last_error} (gave up after {attempts} attempt(s))")


class InMemorySink(Sink):
    """Collect payloads in a list. Intended for unit tests."""

    def __init__(self):
        super().__init__()
        self.payloads: List[Dict[str, Any]] = []
        self._lock = Lock()

    async def send(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.payloads.append(payload)
        self.stats.record_sent()

    def clear(self) -> None:
        with self._lock:
            self.payloads.clear()


class FileSink(Sink):
    """Append each payload as one JSON line to ``path`` — local dev / offline runs."""

    def __init__(self, path: Optional[str] = None):
        super().__init__()
        self._path = Path(path or TracingConfig.get_sink_path())
        self._lock = Lock()

    async def send(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, cls=SafeJSONEncoder)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        self.stats.record_sent()


_TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")


def _empty_usage() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "llm_calls": 0,
    }


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
        # Accumulated usage per trace — see add_trace_usage().
        self._trace_usage: Dict[str, Dict[str, Any]] = {}
        # Span ids already shipped as partial_span (streaming mode).
        self._streamed: Dict[str, set] = {}
        # Strong references to scheduled deliveries — see _dispatch().
        self._pending: set = set()
        # Traces already warned about repeated token declarations.
        self._token_warned: set = set()
        self.sink: Sink = sink if sink is not None else _build_default_sink()

    # ------------------------------------------------------------------ API

    def set_trace_metadata(self, trace_id: str, metadata: Dict[str, Any]):
        """Declare trace-level facts (model, final input/output, session…).

        This *overwrites* — it states what a value **is**. For counters
        that grow across calls (tokens, cost) use :meth:`add_trace_usage`
        instead; calling this once per LLM call with that call's tokens
        keeps only the last call's numbers.
        """
        with self._lock:
            if trace_id not in self._trace_metadata:
                self._trace_metadata[trace_id] = {}
            existing = self._trace_metadata[trace_id]
            # Re-declaring token counts is almost always the per-call
            # overwrite antipattern (only the last call survives). Say so
            # once per trace instead of silently keeping the last value.
            if any(k in metadata for k in _TOKEN_KEYS) and any(k in existing for k in _TOKEN_KEYS):
                if trace_id not in self._token_warned:
                    self._token_warned.add(trace_id)
                    logger.warning(
                        "eval_lib.tracing: set_trace_metadata() received token counts "
                        "more than once for trace %s — the earlier values are being "
                        "overwritten. For per-call counts use tracer.add_trace_usage(), "
                        "which accumulates.",
                        trace_id,
                    )
            existing.update(metadata)

    def add_trace_usage(
        self,
        trace_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
        calls: int = 1,
    ) -> None:
        """Accumulate usage for a trace. Safe to call once per LLM call.

        Every argument is *added* to the running total; the result is
        emitted as the ``usage`` block of the trace payload and, unless the
        caller declared explicit totals via :meth:`set_trace_metadata`,
        also fills the top-level ``input_tokens``/``output_tokens``/… fields.
        """
        with self._lock:
            usage = self._trace_usage.setdefault(trace_id, _empty_usage())
            usage["input_tokens"] += int(input_tokens or 0)
            usage["output_tokens"] += int(output_tokens or 0)
            usage["cached_tokens"] += int(cached_tokens or 0)
            usage["reasoning_tokens"] += int(reasoning_tokens or 0)
            usage["cost_usd"] += float(cost_usd or 0.0)
            usage["llm_calls"] += int(calls or 0)

    def get_trace_usage(self, trace_id: str) -> Dict[str, Any]:
        """Return the accumulated usage for ``trace_id`` (copy)."""
        with self._lock:
            return dict(self._trace_usage.get(trace_id) or _empty_usage())

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

    def has_trace(self, trace_id: str) -> bool:
        """True when anything (spans, metadata or usage) is buffered for it."""
        with self._lock:
            return (
                trace_id in self._traces
                or trace_id in self._trace_metadata
                or trace_id in self._trace_usage
            )

    def flush_trace(self, trace_id: str):
        """Ship the complete trace — spans, metadata and usage — to the sink.

        A trace with metadata but no spans (e.g. a run that only reported
        cost/tokens, or a streaming run whose spans already went out as
        ``partial_span``) is still sent: the final trace payload is the
        authoritative record and a receiver upserts it by ``trace_id``.
        """
        with self._lock:
            spans = self._traces.pop(trace_id, None)
            trace_meta = self._trace_metadata.pop(trace_id, None)
            usage = self._trace_usage.pop(trace_id, None)
            self._streamed.pop(trace_id, None)

        if spans is None and trace_meta is None and usage is None:
            return

        trace_data = self._build_trace_structure(
            trace_id, spans or [], trace_meta or {}, usage
        )
        payload = {"project": TracingConfig.get_project(), "trace": trace_data}
        self._dispatch(payload)

    def flush_span(self, span: TraceSpan):
        """Ship a single span as ``partial_span`` (streaming mode).

        No-op unless ``TRACING_STREAM=true``. The span **stays** in the
        buffer so that ``extract_test_case_data`` keeps working and the
        final :meth:`flush_trace` carries the whole trace; the receiver
        reconciles by ``span_id``. It is only marked as already streamed.
        """
        if not TracingConfig.is_stream():
            return
        if span is None or not span.trace_id:
            return

        with self._lock:
            self._streamed.setdefault(span.trace_id, set()).add(span.span_id)

        payload = {
            "project": TracingConfig.get_project(),
            "trace_id": span.trace_id,
            "partial_span": span.to_dict(),
        }
        self._dispatch(payload)

    def flush(self):
        """Ship every buffered trace."""
        with self._lock:
            trace_ids = set(self._traces) | set(self._trace_metadata) | set(self._trace_usage)
        for trace_id in trace_ids:
            self.flush_trace(trace_id)

    def stop(self):
        """Stop the sender and flush remaining traces."""
        self.flush()

    # ------------------------------------------------------------- internals

    def _dispatch(self, payload: Dict[str, Any]) -> None:
        """Route ``payload`` to the configured sink.

        If an event loop is already running we schedule the coroutine and
        keep a strong reference to the task: a bare ``ensure_future`` may
        be garbage-collected mid-flight, and its exception would surface
        only as an "exception was never retrieved" warning. Awaiting the
        scheduled work is possible via :meth:`aflush`.

        Without a running loop we spin up a temporary one and await
        synchronously.
        """
        coro = self.sink.send(payload)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro)
            except Exception as e:
                if TracingConfig.is_strict():
                    raise
                logger.warning("eval_lib.tracing: trace delivery failed: %r", e)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
            return

        task = asyncio.ensure_future(coro)
        with self._lock:
            self._pending.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: "asyncio.Future") -> None:
        """Discard the finished task and make any failure visible."""
        with self._lock:
            self._pending.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "eval_lib.tracing: trace delivery failed: %s: %s",
                type(exc).__name__, exc,
            )

    async def aflush(self) -> None:
        """Flush buffered traces **and await** every in-flight send.

        Call this before the process exits (or at the end of a request in
        a long-lived server) — otherwise scheduled deliveries can be
        abandoned when the loop shuts down and the traces are lost with
        no diagnostic.
        """
        self.flush()
        while True:
            with self._lock:
                pending = [t for t in self._pending if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def aclose(self) -> None:
        """Await pending deliveries, then release sink resources."""
        await self.aflush()
        await self.sink.aclose()

    @property
    def stats(self) -> DeliveryStats:
        """Delivery counters from the active sink."""
        return self.sink.stats

    def _build_trace_structure(
        self,
        trace_id: str,
        spans: List[TraceSpan],
        trace_meta: Dict[str, Any] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Build the trace payload from flat spans + declared metadata + usage.

        Payload shape (top level of ``"trace"``):

        * identity/timing: ``trace_id``, ``start_time``/``end_time`` (epoch),
          ``started_at``/``ended_at`` (ISO-8601), ``response_time``
        * structure: ``spans`` (roots, nested ``children``), ``span_count``,
          ``tools_called``
        * ``usage`` — **accumulated** counters: ``input_tokens``,
          ``output_tokens``, ``total_tokens``, ``cached_tokens``,
          ``reasoning_tokens``, ``cost_usd``, ``llm_calls``, plus ``source``
          (``"accumulated"`` from :meth:`add_trace_usage`, ``"spans"`` when
          rolled up from LLM spans, ``"declared"`` when the caller stated
          totals).
        * ``metadata`` — everything the caller passed to
          :meth:`set_trace_metadata`, verbatim, as one object. Consumers
          with a dedicated metadata column store this directly.
        * first-class promotions (``model``, ``input``, ``output``,
          ``input_tokens``…, ``cost_usd``, ``session_id``, ``user_id``…) at
          top level for backwards compatibility.

        Precedence for the top-level token/cost fields: declared total
        (``set_trace_metadata``) > accumulated (``add_trace_usage``) >
        span roll-up.
        """
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

        # Calculate trace-level timing
        all_times = [s.start_time for s in spans if s.start_time]
        end_times = [s.end_time for s in spans if s.end_time]
        start_time = min(all_times) if all_times else None
        end_time = max(end_times) if end_times else None

        # Wall-clock span of the whole trace. Using only the first root's
        # duration undercounted every trace with more than one root.
        response_time = None
        if start_time is not None and end_time is not None and end_time >= start_time:
            response_time = round(end_time - start_time, 3)
        elif root_spans:
            root_duration_ms = root_spans[0].get("duration_ms")
            if root_duration_ms:
                response_time = round(root_duration_ms / 1000, 3)

        # Tools called during the trace — one entry per distinct invocation
        # (a nested or time-contained same-named tool span is the same call
        # recorded by a second layer; see eval_lib.tracing.spans).
        tool_spans = [s for s in spans if s.span_type and s.span_type.value == "tool_call"]
        tools_called = [
            s.name for s in top_level_tool_calls(
                tool_spans,
                get_id=lambda s: s.span_id,
                get_parent=lambda s: s.parent_span_id,
                get_name=lambda s: s.name,
                get_start=lambda s: s.start_time,
                get_end=lambda s: s.end_time,
            )
        ]

        # ---- usage: accumulated > span roll-up ---------------------------
        rolled = _empty_usage()
        for span in spans:
            if not (span.span_type and span.span_type.value == "llm_call"):
                continue
            span_usage = _span_token_usage(span)
            if not span_usage:
                continue
            rolled["input_tokens"] += span_usage["input_tokens"]
            rolled["output_tokens"] += span_usage["output_tokens"]
            rolled["cached_tokens"] += span_usage["cached_tokens"]
            rolled["reasoning_tokens"] += span_usage["reasoning_tokens"]
            rolled["llm_calls"] += 1

        if usage and usage.get("llm_calls"):
            usage_block: Dict[str, Any] = dict(usage)
            usage_block["source"] = "accumulated"
        elif rolled["llm_calls"]:
            usage_block = rolled
            usage_block["source"] = "spans"
        else:
            usage_block = _empty_usage()
            usage_block["source"] = "none"

        # Declared totals override whatever was counted.
        declared = {
            k: trace_meta[k]
            for k in ("input_tokens", "output_tokens", "cached_tokens",
                      "reasoning_tokens", "cost_usd")
            if trace_meta.get(k) is not None
        }
        if declared:
            usage_block.update(declared)
            usage_block["source"] = "declared"
        usage_block["total_tokens"] = (
            trace_meta.get("total_tokens")
            if trace_meta.get("total_tokens") is not None
            else usage_block["input_tokens"] + usage_block["output_tokens"]
        )
        usage_block["cost_usd"] = round(float(usage_block.get("cost_usd") or 0.0), 6)

        result: Dict[str, Any] = {
            "trace_id": trace_id,
            "start_time": start_time,
            "end_time": end_time,
            "started_at": _to_iso(start_time),
            "ended_at": _to_iso(end_time),
            "response_time": response_time,
            "tools_called": tools_called if tools_called else None,
            "spans": root_spans,
            # `span_count` is every span in the trace; `spans` holds only the
            # roots (children are nested under them). `root_span_count` makes
            # that explicit so the two numbers stop looking contradictory.
            "span_count": len(spans),
            "root_span_count": len(root_spans),
            "usage": usage_block,
            # Everything the caller declared, as one object, verbatim.
            "metadata": dict(trace_meta),
        }

        # Promote usage counters to top level (backwards compatibility).
        # Zero counts are omitted so a consumer can tell "unknown" from "0".
        for key in ("input_tokens", "output_tokens", "total_tokens",
                    "cached_tokens", "reasoning_tokens"):
            if usage_block.get(key):
                result[key] = usage_block[key]
        if usage_block["cost_usd"]:
            result["cost_usd"] = usage_block["cost_usd"]
            result.setdefault(
                "cost_source",
                trace_meta.get("cost_source") or
                ("reported" if usage_block["source"] == "declared" else "estimated"),
            )

        # First-class trace-level fields, promoted out of metadata so
        # downstream consumers (evalix runtime-eval) can slice / aggregate
        # without having to guess where they landed.
        FIRST_CLASS = (
            "model", "input", "output",
            "response_time", "cost_source",
            "num_turns", "session_id", "user_id",
        )
        for key in FIRST_CLASS:
            if key in trace_meta:
                result[key] = trace_meta[key]
        # Custom keys stay at top level too so existing consumers keep
        # working; `metadata` above is the canonical home for them.
        for key, value in trace_meta.items():
            if key not in result:
                result[key] = value

        return result
