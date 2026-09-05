"""Storage abstraction for the trace receiver.

:class:`TraceStorage` is a thin ABC — an evalix service can plug in its
own Postgres / ClickHouse implementation without touching
:mod:`eval_lib.connector.trace_receiver` or the Flask routes. Default
implementation is :class:`InMemoryStorage`, kept in-process.

Ownership boundary — anything about how bytes land on disk / in a DB
lives in the storage; anything about matching traces to queries or
triggering evaluations lives in
:class:`~eval_lib.connector.trace_receiver.TraceStore`.
"""

from __future__ import annotations

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from eval_lib.connector.trace_models import StoredTrace, TraceProjectState


class TraceStorage(ABC):
    """Pluggable persistence backend for trace projects."""

    # ---- Project persistence ------------------------------------------

    @abstractmethod
    def save_project(self, project_name: str, state: TraceProjectState) -> None:
        """Persist project state (config + dataset + collected traces)."""

    @abstractmethod
    def load_project(self, project_name: str) -> Optional[TraceProjectState]:
        """Fetch project state by name, or ``None`` if unknown."""

    @abstractmethod
    def list_projects(self) -> List[str]:
        """Return all project names known to the store."""

    @abstractmethod
    def delete_project(self, project_name: str) -> None:
        """Remove a project. No-op if unknown."""

    # ---- Trace persistence --------------------------------------------

    @abstractmethod
    def save_trace(self, project_name: str, trace: StoredTrace) -> None:
        """Persist a single trace under ``project_name``."""

    @abstractmethod
    def load_traces_by_project(self, project_name: str) -> List[StoredTrace]:
        """Return every trace stored for ``project_name`` (order not defined)."""

    # ---- Dataset persistence (optional cross-cutting store) -----------

    @abstractmethod
    def save_dataset(self, dataset_id: str, dataset: Dict[str, Any]) -> None:
        """Store a dataset (rows + metadata) under ``dataset_id``."""

    @abstractmethod
    def load_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a dataset by id or ``None``."""

    # ---- Eval-result persistence --------------------------------------

    @abstractmethod
    def save_eval_result(self, project_name: str, result: Dict[str, Any]) -> None:
        """Append one evaluation result for ``project_name``."""

    @abstractmethod
    def load_eval_results_by_project(self, project_name: str) -> List[Dict[str, Any]]:
        """Return every eval result stored for ``project_name``."""


class InMemoryStorage(TraceStorage):
    """Thread-safe, process-local implementation of :class:`TraceStorage`.

    Used by default so that ``TraceStore()`` works with zero config.
    Data lives only for the lifetime of the process — swap in a
    persistent implementation for production use.
    """

    def __init__(self) -> None:
        # Re-entrant so a subclass can hold it across its own work *and*
        # the inherited in-memory update (FileBackedStorage.save_project).
        self._lock = RLock()
        self._projects: Dict[str, TraceProjectState] = {}
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._eval_results: Dict[str, List[Dict[str, Any]]] = {}

    # ---- projects -----------------------------------------------------

    def save_project(self, project_name: str, state: TraceProjectState) -> None:
        with self._lock:
            self._projects[project_name] = state

    def load_project(self, project_name: str) -> Optional[TraceProjectState]:
        with self._lock:
            return self._projects.get(project_name)

    def list_projects(self) -> List[str]:
        with self._lock:
            return list(self._projects.keys())

    def delete_project(self, project_name: str) -> None:
        with self._lock:
            self._projects.pop(project_name, None)
            self._eval_results.pop(project_name, None)

    # ---- traces -------------------------------------------------------

    def save_trace(self, project_name: str, trace: StoredTrace) -> None:
        with self._lock:
            state = self._projects.get(project_name)
            if state is None:
                return
            # Deduplicate by trace_id — a receiver may replay ingestion.
            existing = {t.trace_id for t in state.traces}
            if trace.trace_id not in existing:
                state.traces.append(trace)

    def load_traces_by_project(self, project_name: str) -> List[StoredTrace]:
        with self._lock:
            state = self._projects.get(project_name)
            if state is None:
                return []
            return list(state.traces)

    # ---- datasets -----------------------------------------------------

    def save_dataset(self, dataset_id: str, dataset: Dict[str, Any]) -> None:
        with self._lock:
            self._datasets[dataset_id] = dataset

    def load_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._datasets.get(dataset_id)

    # ---- eval results -------------------------------------------------

    def save_eval_result(self, project_name: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._eval_results.setdefault(project_name, []).append(result)

    def load_eval_results_by_project(self, project_name: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._eval_results.get(project_name, []))


# Project names become file names, so they must not contain path
# separators or traversal sequences.
_SAFE_PROJECT_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class FileBackedStorage(InMemoryStorage):
    """Extends :class:`InMemoryStorage` with JSON project snapshots on disk.

    Preserves the historical behaviour that ``TraceStore.set_cache_dir(dir)``
    activated. Projects are written as ``<cache_dir>/trace_projects/<name>.json``
    on every ``save_project`` call and reloaded at construction.
    """

    def __init__(self, cache_dir: str):
        super().__init__()
        self._cache_dir = Path(cache_dir)
        self._load_from_disk()

    def _projects_dir(self) -> Path:
        return (self._cache_dir / "trace_projects").resolve()

    def _project_path(self, project_name: str) -> Optional[Path]:
        """Return the on-disk path for a project, or ``None`` if unsafe."""
        if not isinstance(project_name, str) or not _SAFE_PROJECT_RE.match(project_name):
            return None
        base = self._projects_dir()
        candidate = (base / f"{project_name}.json").resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None
        return candidate

    def _load_from_disk(self) -> None:
        base = self._projects_dir()
        if not base.exists():
            return
        for path in base.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                state = TraceProjectState(**data)
                super().save_project(state.config.project, state)
            except Exception:
                # Corrupt / unreadable project file — skip.
                pass

    def save_project(self, project_name: str, state: TraceProjectState) -> None:
        """Snapshot the project to disk atomically.

        The JSON is written to a temporary file and moved into place with
        ``os.replace`` so a crash mid-write can never leave a truncated
        file behind — ``_load_from_disk`` silently skips unreadable files,
        which used to mean the whole project (config, dataset, every trace)
        disappeared on the next restart. The write happens under the lock
        so two concurrent ingests cannot interleave bytes in one file.
        """
        with self._lock:
            super().save_project(project_name, state)
            path = self._project_path(project_name)
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(
                f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    def delete_project(self, project_name: str) -> None:
        super().delete_project(project_name)
        path = self._project_path(project_name)
        if path is not None and path.exists():
            path.unlink()
