"""TraceStore — receives, stores, matches and evaluates traces from remote agents.

This is the core of the Trace Receiver system. It:
1. Receives trace payloads from agents via HTTP (TraceSender on agent side)
2. Matches incoming traces with queries from a pre-loaded dataset
3. Triggers evaluation when enough traces are collected
4. Stores results in DashboardCache for display

Storage is pluggable through :class:`~eval_lib.connector.trace_storage.TraceStorage`.
Default is in-memory (``InMemoryStorage``); passing a
``FileBackedStorage`` restores the historical file-cache behaviour, and
external services (evalix) can plug in Postgres / ClickHouse.

Thread-safe singleton (same pattern as ConnectorEngine).
"""

import os
import re
import hmac
import json
import hashlib
import asyncio
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime

from eval_lib.connector.trace_models import (
    TraceProjectConfig, StoredTrace, TraceProjectState, MatchingStrategy,
)
from eval_lib.connector.trace_storage import (
    TraceStorage, InMemoryStorage, FileBackedStorage,
)
from eval_lib.connector.metric_registry import instantiate_metric
from eval_lib.testcases_schema import EvalTestCase, TraceStep, ResourceUsage
from eval_lib.evaluate import evaluate


class TraceStore:
    """Singleton store for receiving and managing traces from remote agents.

    Instantiate with ``TraceStore(storage=…)`` on first use to plug in a
    custom persistence backend; subsequent ``TraceStore()`` calls return
    the same instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, storage: Optional[TraceStorage] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._projects: Dict[str, TraceProjectState] = {}
                    cls._instance._cache_dir = ".eval_cache"
                    cls._instance._initialized = False
                    cls._instance.storage = storage or InMemoryStorage()
                    # Warm cache from storage — matters for FileBackedStorage
                    # and any custom backend that survives process restarts.
                    for name in cls._instance.storage.list_projects():
                        loaded = cls._instance.storage.load_project(name)
                        if loaded is not None:
                            cls._instance._projects[name] = loaded
        return cls._instance

    def set_cache_dir(self, cache_dir: str):
        """Backward-compatible: swap in a FileBackedStorage rooted at ``cache_dir``.

        Call this once at startup (see ``cli.py``) to preserve on-disk
        JSON persistence of trace projects.
        """
        self._cache_dir = cache_dir
        self.storage = FileBackedStorage(cache_dir)
        self._projects.clear()
        for name in self.storage.list_projects():
            loaded = self.storage.load_project(name)
            if loaded is not None:
                self._projects[name] = loaded
        self._initialized = True

    # ---- Project management ----

    def create_project(
        self,
        config: TraceProjectConfig,
        dataset_rows: List[Dict[str, Any]],
    ) -> TraceProjectState:
        state = TraceProjectState(config=config, dataset_rows=dataset_rows)
        self._build_query_index(state)
        with self._lock:
            self._projects[config.project] = state
        self.storage.save_project(config.project, state)
        return state

    def get_project(self, project: str) -> Optional[TraceProjectState]:
        cached = self._projects.get(project)
        if cached is not None:
            return cached
        # Fall through to storage — supports persistent backends where
        # a fresh process hasn't warmed the in-memory cache yet.
        loaded = self.storage.load_project(project)
        if loaded is not None:
            self._projects[project] = loaded
        return loaded

    def delete_project(self, project: str):
        with self._lock:
            self._projects.pop(project, None)
        self.storage.delete_project(project)

    def list_projects(self) -> List[Dict[str, Any]]:
        result = []
        for name, state in self._projects.items():
            total_expected = len(state.query_index)
            traces_received = len(state.traces)
            matched = sum(
                1 for idx_traces in state.query_traces.values()
                if len(idx_traces) >= state.config.runs_per_query
            )
            result.append({
                "project": name,
                "dataset_id": state.config.dataset_id,
                "total_expected": total_expected,
                "traces_received": traces_received,
                "traces_matched": matched,
                "status": state.status,
                "created_at": state.config.created_at,
            })
        return result

    # ---- API key ----
    #
    # Project API keys are hashed before storage. We use scrypt — a memory-hard
    # KDF — with a per-key random salt. Stored format:
    #     scrypt$<salt_hex>$<derived_hex>
    # Legacy projects created before this change stored a bare SHA-256 hex
    # digest; validate_api_key still accepts those for backward compatibility.

    # scrypt cost parameters (CPU/memory). n must be a power of two.
    _SCRYPT_N = 2 ** 14
    _SCRYPT_R = 8
    _SCRYPT_P = 1
    _SCRYPT_DKLEN = 32

    @classmethod
    def hash_api_key(cls, key: str) -> str:
        """Derive a salted scrypt hash for storage. New format only."""
        salt = os.urandom(16)
        derived = hashlib.scrypt(
            key.encode("utf-8"),
            salt=salt,
            n=cls._SCRYPT_N,
            r=cls._SCRYPT_R,
            p=cls._SCRYPT_P,
            dklen=cls._SCRYPT_DKLEN,
        )
        return f"scrypt${salt.hex()}${derived.hex()}"

    # scrypt is deliberately slow (~50 ms, 16 MiB per call). Every ingest
    # presents the same key, so a verified (stored hash, sha256(key)) pair
    # is remembered. The raw key is never stored; changing the project's
    # hash naturally misses the cache.
    _VERIFY_CACHE_MAX = 256
    _verify_cache: "OrderedDict[tuple, bool]" = OrderedDict()
    _verify_cache_lock = threading.Lock()

    @classmethod
    def _verify_api_key(cls, key: str, stored: str) -> bool:
        """Constant-time check of a presented key against a stored hash.

        Handles both the new salted-scrypt format and the legacy bare SHA-256
        digest produced by older versions.
        """
        if stored.startswith("scrypt$"):
            cache_key = (stored, hashlib.sha256(key.encode("utf-8")).hexdigest())
            with cls._verify_cache_lock:
                cached = cls._verify_cache.get(cache_key)
                if cached is not None:
                    cls._verify_cache.move_to_end(cache_key)
                    return cached
            try:
                _, salt_hex, expected_hex = stored.split("$", 2)
                salt = bytes.fromhex(salt_hex)
                expected = bytes.fromhex(expected_hex)
            except (ValueError, TypeError):
                return False
            derived = hashlib.scrypt(
                key.encode("utf-8"),
                salt=salt,
                n=cls._SCRYPT_N,
                r=cls._SCRYPT_R,
                p=cls._SCRYPT_P,
                dklen=len(expected),
            )
            verdict = hmac.compare_digest(derived, expected)
            with cls._verify_cache_lock:
                cls._verify_cache[cache_key] = verdict
                cls._verify_cache.move_to_end(cache_key)
                while len(cls._verify_cache) > cls._VERIFY_CACHE_MAX:
                    cls._verify_cache.popitem(last=False)
            return verdict
        # Legacy SHA-256 digest — constant-time compare, no early exit.
        legacy = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return hmac.compare_digest(legacy, stored)

    def validate_api_key(self, project: str, api_key: str) -> bool:
        state = self._projects.get(project)
        if not state:
            return False
        if not state.config.api_key_hash:
            return True  # No key configured — open access
        return self._verify_api_key(api_key, state.config.api_key_hash)

    # ---- Trace ingestion ----

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Trace ``input``/``output`` as text.

        Structured values are serialised as JSON rather than ``str()``'d —
        a Python repr (``{'q': 'x'}``) never matches a dataset row and is
        not parseable by anything downstream.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def ingest_trace(self, project: str, trace_data: Dict[str, Any]) -> Optional[StoredTrace]:
        state = self._projects.get(project)
        if not state:
            return None

        usage = trace_data.get("usage") if isinstance(trace_data.get("usage"), dict) else None
        metadata = trace_data.get("metadata") if isinstance(trace_data.get("metadata"), dict) else None

        def _field(name: str) -> Any:
            """Top-level value, falling back to the usage / metadata blocks."""
            if trace_data.get(name) is not None:
                return trace_data.get(name)
            if usage and usage.get(name) is not None:
                return usage.get(name)
            if metadata and metadata.get(name) is not None:
                return metadata.get(name)
            return None

        trace = StoredTrace(
            trace_id=trace_data.get("trace_id", ""),
            project=project,
            input=self._coerce_text(_field("input")),
            output=self._coerce_text(_field("output")),
            model=_field("model"),
            input_tokens=self._int_or_none(_field("input_tokens")),
            output_tokens=self._int_or_none(_field("output_tokens")),
            total_tokens=self._int_or_none(_field("total_tokens")),
            cached_tokens=self._int_or_none(_field("cached_tokens")),
            reasoning_tokens=self._int_or_none(_field("reasoning_tokens")),
            response_time=trace_data.get("response_time"),
            started_at=trace_data.get("started_at"),
            ended_at=trace_data.get("ended_at"),
            tools_called=trace_data.get("tools_called"),
            spans=trace_data.get("spans"),
            span_count=trace_data.get("span_count", 0),
            cost_usd=_field("cost_usd"),
            cost_source=_field("cost_source"),
            usage=usage,
            metadata=metadata,
            session_id=_field("session_id"),
            user_id=_field("user_id"),
            num_turns=self._int_or_none(_field("num_turns")),
        )

        # Duplicate / upgrade check. A record assembled from streamed
        # partial spans is superseded by the final trace (authoritative);
        # a second full trace with the same id is a replay and is ignored.
        with self._lock:
            existing = next((t for t in state.traces if t.trace_id == trace.trace_id), None)
            if existing is not None:
                if not existing.is_partial:
                    return existing
                trace.received_at = existing.received_at
                # Keep any streamed spans the final payload does not carry.
                if not trace.spans and existing.spans:
                    trace.spans = existing.spans
                    trace.span_count = trace.span_count or len(existing.spans)
                state.traces.remove(existing)

        # Match with dataset
        query_idx = self._match_trace(state, trace)

        # Reserve the run slot and append under ONE lock acquisition —
        # doing them separately let two concurrent duplicates both pass the
        # check above and take two slots.
        with self._lock:
            if query_idx is not None:
                trace.matched_query_index = query_idx
                traces_list = state.query_traces.setdefault(str(query_idx), [])
                trace.run_index = len(traces_list)
                traces_list.append(trace.trace_id)
            state.traces.append(trace)

        # Persist both the trace and the updated project state — a
        # persistent backend can then reconstruct after a restart.
        self.storage.save_trace(project, trace)
        self.storage.save_project(project, state)

        # Check if auto-evaluation should trigger
        if query_idx is not None:
            self._check_and_trigger_evaluation(state)

        return trace

    def ingest_partial_span(
        self, project: str, trace_id: str, span: Dict[str, Any]
    ) -> Optional[StoredTrace]:
        """Accept one streamed span (``TRACING_STREAM=true`` on the agent).

        Spans accumulate on a placeholder record flagged ``is_partial`` and
        are de-duplicated by ``span_id``. The final trace payload later
        upgrades the record in :meth:`ingest_trace`. Previously this shape
        was answered with HTTP 400, so streaming lost every span.
        """
        state = self._projects.get(project)
        if not state or not trace_id or not isinstance(span, dict):
            return None

        with self._lock:
            existing = next((t for t in state.traces if t.trace_id == trace_id), None)
            if existing is None:
                existing = StoredTrace(
                    trace_id=trace_id, project=project, is_partial=True, spans=[]
                )
                state.traces.append(existing)
            elif not existing.is_partial:
                # Final trace already stored — a late partial adds nothing.
                return existing

            known = {s.get("span_id") for s in (existing.spans or [])}
            if span.get("span_id") not in known:
                existing.spans = list(existing.spans or []) + [span]
                existing.span_count = len(existing.spans)

        self.storage.save_project(project, state)
        return existing

    # ---- Span flattening ----

    @staticmethod
    def _iter_spans(spans: List[Dict[str, Any]], parent_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Depth-first walk over the nested ``children`` tree the sender emits.

        Yields every span with ``parent_span_id`` filled from the tree when
        the payload did not carry it. The receiver used to look only at the
        top-level list, so every nested llm/tool span was invisible to the
        reliability metrics.
        """
        for span in spans or []:
            if not isinstance(span, dict):
                continue
            node = dict(span)
            if node.get("parent_span_id") is None and parent_id is not None:
                node["parent_span_id"] = parent_id
            children = node.pop("children", None) or []
            yield node
            yield from TraceStore._iter_spans(children, node.get("span_id"))

    @classmethod
    def _steps_from_spans(cls, spans: Optional[List[Dict[str, Any]]]) -> Optional[List[TraceStep]]:
        if not spans:
            return None
        steps = [
            TraceStep(
                step_id=s.get("span_id"),
                type=s.get("span_type", s.get("type", "custom")),
                name=s.get("name"),
                input=s.get("input"),
                output=s.get("output"),
                duration_ms=s.get("duration_ms"),
                timestamp=s.get("start_time"),
                status=s.get("status"),
                error=s.get("error"),
                error_type=s.get("error_type"),
                parent_step_id=s.get("parent_span_id"),
                metadata=s.get("metadata") or None,
            )
            for s in cls._iter_spans(spans)
        ]
        # Chronological order is what loop/sequence metrics expect.
        steps.sort(key=lambda st: st.timestamp or 0)
        return steps or None

    # ---- Matching ----

    def _build_query_index(self, state: TraceProjectState):
        """Build normalized_input → [query_index] mapping from dataset."""
        state.query_index = {}
        input_col = state.config.input_column
        for idx, row in enumerate(state.dataset_rows):
            raw_input = str(row.get(input_col, ""))
            key = self._normalize_text(raw_input, state.config.matching_strategy)
            state.query_index.setdefault(key, []).append(idx)

    def _match_trace(self, state: TraceProjectState, trace: StoredTrace) -> Optional[int]:
        """Find matching dataset query index for a trace."""
        key = self._normalize_text(trace.input, state.config.matching_strategy)
        candidates = state.query_index.get(key, [])
        for idx in candidates:
            idx_key = str(idx)
            traces_for_idx = state.query_traces.get(idx_key, [])
            if len(traces_for_idx) < state.config.runs_per_query:
                return idx
        return None

    @staticmethod
    def _normalize_text(text: str, strategy: MatchingStrategy) -> str:
        if strategy == MatchingStrategy.EXACT:
            return text
        # NORMALIZED: strip, lowercase, collapse whitespace, strip punctuation
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.rstrip('?!.')
        return text

    # ---- Auto-evaluation trigger ----

    def _check_and_trigger_evaluation(self, state: TraceProjectState):
        if not state.config.auto_evaluate:
            return
        if state.status != "collecting":
            return

        total_queries = len(state.query_index)
        if total_queries == 0:
            return

        satisfied = sum(
            1 for idx_traces in state.query_traces.values()
            if len(idx_traces) >= state.config.runs_per_query
        )

        if satisfied >= total_queries:
            self.trigger_evaluation(state.config.project)

    def trigger_evaluation(self, project: str) -> Optional[str]:
        """Start evaluation in background thread. Returns job_id."""
        state = self._projects.get(project)
        if not state:
            return None

        state.status = "evaluating"
        job_id = f"trace_eval_{project}_{datetime.now().strftime('%H%M%S')}"
        state.evaluation_job_id = job_id

        thread = threading.Thread(
            target=self._run_evaluation_thread,
            args=(project,),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_evaluation_thread(self, project: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_evaluation(project))
        except Exception as e:
            state = self._projects.get(project)
            if state:
                state.status = f"failed: {e}"
        finally:
            loop.close()

    async def _run_evaluation(self, project: str):
        state = self._projects.get(project)
        if not state:
            return

        config = state.config
        test_cases = []
        multi_run_outputs: Dict[int, List[str]] = {}

        # Build test cases from matched traces
        for idx_key, trace_ids in state.query_traces.items():
            idx = int(idx_key)
            if idx >= len(state.dataset_rows):
                continue

            row = state.dataset_rows[idx]
            traces_for_query = [t for t in state.traces if t.trace_id in trace_ids]
            if not traces_for_query:
                continue

            primary = traces_for_query[0]

            # Build execution_trace from the full span tree
            execution_trace = self._steps_from_spans(primary.spans)

            resource_usage = None
            if (primary.input_tokens or primary.output_tokens
                    or primary.cost_usd or primary.response_time):
                total = primary.total_tokens
                if total is None and (primary.input_tokens or primary.output_tokens):
                    total = (primary.input_tokens or 0) + (primary.output_tokens or 0)
                resource_usage = ResourceUsage(
                    input_tokens=primary.input_tokens,
                    output_tokens=primary.output_tokens,
                    total_tokens=total,
                    duration_ms=round(primary.response_time * 1000, 2) if primary.response_time else None,
                    cost=primary.cost_usd,
                    model=primary.model,
                )

            tc = EvalTestCase(
                input=str(row.get(config.input_column, "")),
                actual_output=primary.output,
                expected_output=str(row.get(config.expected_output_column, "")) if config.expected_output_column else None,
                tools_called=primary.tools_called,
                execution_trace=execution_trace,
                resource_usage=resource_usage,
            )
            test_cases.append(tc)

            # Collect multi-run outputs
            if len(traces_for_query) > 1:
                multi_run_outputs[len(test_cases) - 1] = [t.output for t in traces_for_query]

        if not test_cases:
            state.status = "completed"
            return

        # Instantiate metrics
        metrics = []
        for mc in config.metrics:
            try:
                m = instantiate_metric(
                    mc.get("metric_class", ""),
                    config.eval_model,
                    mc.get("params", {}),
                )
                if mc.get("metric_class") == "OutcomeConsistencyMetric" and multi_run_outputs:
                    m.multi_outputs = [
                        multi_run_outputs.get(i, [tc.actual_output])
                        for i, tc in enumerate(test_cases)
                    ]
                metrics.append(m)
            except Exception:
                pass

        if not metrics:
            state.status = "failed: no valid metrics"
            return

        session_name = f"trace_{project}"
        try:
            await evaluate(
                test_cases,
                metrics,
                verbose=False,
                show_dashboard=True,
                session_name=session_name,
            )

            # Update trace statuses
            for trace in state.traces:
                if trace.matched_query_index is not None:
                    trace.evaluation_status = "completed"
                    trace.evaluation_session_id = session_name

            self.storage.save_eval_result(
                project,
                {
                    "session_name": session_name,
                    "completed_at": datetime.now().isoformat(),
                    "trace_count": len(test_cases),
                },
            )
            state.status = "completed"
        except Exception as e:
            state.status = f"failed: {e}"

        self.storage.save_project(project, state)
