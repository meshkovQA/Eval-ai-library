"""TraceStore — receives, stores, matches and evaluates traces from remote agents.

This is the core of the Trace Receiver system. It:
1. Receives trace payloads from agents via HTTP (TraceSender on agent side)
2. Matches incoming traces with queries from a pre-loaded dataset
3. Triggers evaluation when enough traces are collected
4. Stores results in DashboardCache for display

Thread-safe singleton (same pattern as ConnectorEngine).
"""

import re
import json
import hashlib
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from eval_lib.connector.trace_models import (
    TraceProjectConfig, StoredTrace, TraceProjectState, MatchingStrategy,
)
from eval_lib.connector.metric_registry import instantiate_metric
from eval_lib.testcases_schema import EvalTestCase, TraceStep, ResourceUsage
from eval_lib.evaluate import evaluate


class TraceStore:
    """Singleton store for receiving and managing traces from remote agents."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._projects: Dict[str, TraceProjectState] = {}
                    cls._instance._cache_dir = ".eval_cache"
                    cls._instance._initialized = False
        return cls._instance

    def set_cache_dir(self, cache_dir: str):
        self._cache_dir = cache_dir
        self._load_state()
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
        self._save_project(config.project)
        return state

    def get_project(self, project: str) -> Optional[TraceProjectState]:
        return self._projects.get(project)

    def delete_project(self, project: str):
        with self._lock:
            self._projects.pop(project, None)
        path = self._project_path(project)
        if path.exists():
            path.unlink()

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

    @staticmethod
    def hash_api_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_api_key(self, project: str, api_key: str) -> bool:
        state = self._projects.get(project)
        if not state:
            return False
        if not state.config.api_key_hash:
            return True  # No key configured — open access
        return self.hash_api_key(api_key) == state.config.api_key_hash

    # ---- Trace ingestion ----

    def ingest_trace(self, project: str, trace_data: Dict[str, Any]) -> Optional[StoredTrace]:
        state = self._projects.get(project)
        if not state:
            return None

        trace = StoredTrace(
            trace_id=trace_data.get("trace_id", ""),
            project=project,
            input=str(trace_data.get("input", "")),
            output=str(trace_data.get("output", "")),
            model=trace_data.get("model"),
            input_tokens=trace_data.get("input_tokens"),
            output_tokens=trace_data.get("output_tokens"),
            response_time=trace_data.get("response_time"),
            tools_called=trace_data.get("tools_called"),
            spans=trace_data.get("spans"),
            span_count=trace_data.get("span_count", 0),
        )

        # Check for duplicate trace_id
        with self._lock:
            existing_ids = {t.trace_id for t in state.traces}
            if trace.trace_id in existing_ids:
                return next(t for t in state.traces if t.trace_id == trace.trace_id)

        # Match with dataset
        query_idx = self._match_trace(state, trace)
        if query_idx is not None:
            trace.matched_query_index = query_idx
            idx_key = str(query_idx)
            with self._lock:
                traces_list = state.query_traces.setdefault(idx_key, [])
                trace.run_index = len(traces_list)
                traces_list.append(trace.trace_id)

        with self._lock:
            state.traces.append(trace)

        self._save_project(project)

        # Check if auto-evaluation should trigger
        if query_idx is not None:
            self._check_and_trigger_evaluation(state)

        return trace

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
            args=(project, self._cache_dir),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_evaluation_thread(self, project: str, cache_dir: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_evaluation(project, cache_dir))
        except Exception as e:
            state = self._projects.get(project)
            if state:
                state.status = f"failed: {e}"
        finally:
            loop.close()

    async def _run_evaluation(self, project: str, cache_dir: str):
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

            # Build execution_trace from spans
            execution_trace = None
            if primary.spans:
                execution_trace = [
                    TraceStep(
                        type=s.get("span_type", s.get("type", "custom")),
                        name=s.get("name"),
                        input=s.get("input"),
                        output=s.get("output"),
                        duration_ms=s.get("duration_ms"),
                        status=s.get("status"),
                        error=s.get("error"),
                    )
                    for s in primary.spans
                ]

            resource_usage = None
            if primary.input_tokens or primary.output_tokens:
                resource_usage = ResourceUsage(
                    input_tokens=primary.input_tokens,
                    output_tokens=primary.output_tokens,
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

            state.status = "completed"
        except Exception as e:
            state.status = f"failed: {e}"

        self._save_project(project)

    # ---- Persistence ----

    def _project_path(self, project: str) -> Path:
        return Path(self._cache_dir) / "trace_projects" / f"{project}.json"

    def _save_project(self, project: str):
        state = self._projects.get(project)
        if not state:
            return
        path = self._project_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.model_dump_json(indent=2))

    def _load_state(self):
        projects_dir = Path(self._cache_dir) / "trace_projects"
        if not projects_dir.exists():
            return
        for path in projects_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                state = TraceProjectState(**data)
                self._projects[state.config.project] = state
            except Exception:
                pass
