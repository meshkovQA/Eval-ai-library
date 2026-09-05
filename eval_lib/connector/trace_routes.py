"""Flask Blueprint for Trace Receiver HTTP endpoints.

Provides endpoints for:

* Receiving traces from remote agents (POST /api/traces/ingest)
* Managing trace projects (CRUD)
* Triggering evaluations

Two ways to mount:

* ``trace_bp`` — module-level default blueprint, backwards-compatible.
  Uses the singleton ``TraceStore()`` with :class:`InMemoryStorage`.

* :func:`create_trace_blueprint` — factory. Lets evalix (and any host
  service) inject a custom :class:`TraceStorage` implementation and an
  ``auth_verifier`` for JWT / workspace-scoped access.
"""

import hmac
import logging
import os
from typing import Any, Callable, Optional, Union

from flask import Blueprint, g, jsonify, request

from eval_lib.connector.trace_models import MatchingStrategy, TraceProjectConfig
from eval_lib.connector.trace_receiver import TraceStore
from eval_lib.connector.trace_storage import TraceStorage

_datasets = {}  # Shared reference — set by cli.py

logger = logging.getLogger("eval_lib.connector")


AuthVerifier = Callable[[Any], Union[bool, dict]]


def _bearer_token() -> str:
    auth_header = request.headers.get("Authorization", "")
    return auth_header[7:] if auth_header.startswith("Bearer ") else ""


def _admin_key() -> str:
    """Optional key guarding project create/list in legacy (no verifier) mode.

    Set ``TRACE_RECEIVER_ADMIN_KEY``. Without it those endpoints stay open,
    as before — but now with a startup warning instead of silence.
    """
    return os.getenv("TRACE_RECEIVER_ADMIN_KEY", "")


def _allowed_projects_from_context() -> Optional[set]:
    """Project scope carried by an ``auth_verifier`` verdict, if any.

    A verifier may return ``{"projects": [...]}`` or ``{"project": "..."}``
    to restrict the caller; when it does, every project-scoped endpoint
    enforces it. Verdicts without either key keep the historical
    "authenticated == allowed everywhere" behaviour.
    """
    ctx = getattr(g, "auth_context", None)
    if not isinstance(ctx, dict):
        return None
    if isinstance(ctx.get("projects"), (list, set, tuple)):
        return {str(p) for p in ctx["projects"]}
    if ctx.get("project"):
        return {str(ctx["project"])}
    return None


def create_trace_blueprint(
    storage: Optional[TraceStorage] = None,
    auth_verifier: Optional[AuthVerifier] = None,
    name: str = "traces",
) -> Blueprint:
    """Build a trace-receiver Blueprint bound to a storage + auth policy.

    Args:
        storage: A custom :class:`TraceStorage` (evalix passes its
            Postgres implementation here). Default: reuse
            ``TraceStore()`` singleton with :class:`InMemoryStorage`.
        auth_verifier: Optional callable invoked on every request; must
            return one of:

            * ``True`` / ``dict`` — request authorised. If a dict is
              returned it is stashed on ``flask.g.auth_context`` so
              downstream code can pull e.g. ``workspace_id`` off it.
            * ``False`` — 401.
            * Raise — 401 (the exception message is included in the
              response body).

            When ``None``, the endpoints fall back to the historical
            static ``Bearer <project_api_key>`` check.
        name: Blueprint name; must be unique per Flask app when mounting
            multiple copies.
    """
    bp = Blueprint(name, __name__)

    store = TraceStore(storage=storage) if storage is not None else TraceStore()

    if auth_verifier is None and not _admin_key():
        logger.warning(
            "trace receiver: no auth_verifier and TRACE_RECEIVER_ADMIN_KEY unset — "
            "project create/list endpoints are unauthenticated."
        )

    def _authorise(project_name: Optional[str], *, admin: bool = False) -> Optional[tuple]:
        """Return an error response tuple if unauthorised, else ``None``.

        ``admin=True`` marks project-management endpoints; in legacy mode
        they are guarded by ``TRACE_RECEIVER_ADMIN_KEY`` when it is set.
        """
        if auth_verifier is not None:
            try:
                verdict = auth_verifier(request)
            except Exception as e:
                return jsonify({"error": f"Unauthorised: {e}"}), 401
            if verdict is False or verdict is None:
                return jsonify({"error": "Unauthorised"}), 401
            if isinstance(verdict, dict):
                g.auth_context = verdict
            # Project isolation: honour a scope the verifier handed back.
            allowed = _allowed_projects_from_context()
            if project_name is not None and allowed is not None and project_name not in allowed:
                return jsonify({"error": "Forbidden for this project"}), 403
            return None

        # Legacy static-Bearer path.
        if project_name is None:
            admin_key = _admin_key()
            if admin and admin_key and not hmac.compare_digest(_bearer_token(), admin_key):
                return jsonify({"error": "Invalid admin key"}), 401
            return None
        if not store.validate_api_key(project_name, _bearer_token()):
            return jsonify({"error": "Invalid API key"}), 401
        return None

    # ---- Trace Ingestion ----

    @bp.route("/api/traces/ingest", methods=["POST"])
    def ingest_trace():
        """Receive a trace — or one streamed span — from a remote agent.

        Accepts both payload shapes ``TraceSender`` emits:

        * ``{"project", "trace": {...}}`` — the complete trace.
        * ``{"project", "trace_id", "partial_span": {...}}`` — one span
          shipped early under ``TRACING_STREAM=true``. These were rejected
          with 400 before, which made streaming mode lose everything.
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        project_name = data.get("project", "")
        trace_data = data.get("trace")
        partial_span = data.get("partial_span")

        if not project_name:
            return jsonify({"error": "Missing 'project' field"}), 400
        if not trace_data and not partial_span:
            return jsonify({"error": "Missing 'trace' or 'partial_span' field"}), 400

        # Authorise BEFORE revealing whether the project exists — in legacy
        # mode an unknown project simply fails the key check, so callers can
        # no longer enumerate project names via 404-vs-401.
        unauth = _authorise(project_name)
        if unauth is not None:
            return unauth

        state = store.get_project(project_name)
        if not state:
            return jsonify({"error": f"Project '{project_name}' not found"}), 404

        if partial_span and not trace_data:
            trace_id = data.get("trace_id") or partial_span.get("trace_id") or ""
            stored = store.ingest_partial_span(project_name, trace_id, partial_span)
            if not stored:
                return jsonify({"error": "Failed to ingest partial span"}), 500
            return jsonify({
                "ok": True,
                "trace_id": stored.trace_id,
                "project": project_name,
                "partial": True,
                "span_count": stored.span_count,
            }), 202

        # Ingest trace
        trace = store.ingest_trace(project_name, trace_data)
        if not trace:
            return jsonify({"error": "Failed to ingest trace"}), 500

        return jsonify({
            "ok": True,
            "trace_id": trace.trace_id,
            "project": project_name,
            "matched_query_index": trace.matched_query_index,
            "evaluation_triggered": state.status == "evaluating",
        }), 201

    # ---- Project Management ----

    @bp.route("/api/traces/projects", methods=["POST"])
    def create_project():
        """Create a trace receiver project."""
        unauth = _authorise(None, admin=True)
        if unauth is not None:
            return unauth

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        project_name = data.get("project", "")
        if not project_name:
            return jsonify({"error": "Missing 'project' field"}), 400

        dataset_id = data.get("dataset_id", "")
        if not dataset_id:
            return jsonify({"error": "Missing 'dataset_id'"}), 400

        # Load dataset from shared storage
        dataset = _datasets.get(dataset_id)
        if not dataset:
            return jsonify({"error": f"Dataset '{dataset_id}' not found. Upload it first via /api/connector/upload-dataset"}), 404

        dataset_rows = dataset.get("rows", [])

        # Hash API key if provided
        api_key = data.get("api_key", "")
        api_key_hash = TraceStore.hash_api_key(api_key) if api_key else ""

        strategy = data.get("matching_strategy", "normalized")
        try:
            matching_strategy = MatchingStrategy(strategy)
        except ValueError:
            matching_strategy = MatchingStrategy.NORMALIZED

        config = TraceProjectConfig(
            project=project_name,
            api_key_hash=api_key_hash,
            dataset_id=dataset_id,
            input_column=data.get("input_column", "input"),
            expected_output_column=data.get("expected_output_column"),
            context_column=data.get("context_column"),
            expected_tools_column=data.get("expected_tools_column"),
            matching_strategy=matching_strategy,
            metrics=data.get("metrics", []),
            eval_model=data.get("eval_model", "gpt-4o-mini"),
            auto_evaluate=data.get("auto_evaluate", True),
            runs_per_query=data.get("runs_per_query", 1),
            trace_timeout_seconds=data.get("trace_timeout_seconds", 300),
        )

        state = store.create_project(config, dataset_rows)

        return jsonify({
            "ok": True,
            "project": project_name,
            "expected_queries": len(state.query_index),
            "total_dataset_rows": len(dataset_rows),
            "api_key_set": bool(api_key),
        })

    @bp.route("/api/traces/projects", methods=["GET"])
    def list_projects():
        """List trace receiver projects (scoped to the caller when a verifier
        returned a project scope)."""
        unauth = _authorise(None, admin=True)
        if unauth is not None:
            return unauth
        projects = store.list_projects()
        allowed = _allowed_projects_from_context()
        if allowed is not None:
            projects = [p for p in projects if p.get("project") in allowed]
        return jsonify(projects)

    @bp.route("/api/traces/projects/<project>", methods=["GET"])
    def get_project(project: str):
        """Get project details with traces."""
        unauth = _authorise(project)
        if unauth is not None:
            return unauth

        state = store.get_project(project)
        if not state:
            return jsonify({"error": "Project not found"}), 404

        total_expected = len(state.query_index)
        satisfied = sum(
            1 for t in state.query_traces.values()
            if len(t) >= state.config.runs_per_query
        )

        traces_summary = []
        for trace in state.traces[-100:]:  # Last 100 traces
            traces_summary.append({
                "trace_id": trace.trace_id,
                "input": trace.input[:100],
                "output": trace.output[:200],
                "matched_query_index": trace.matched_query_index,
                "run_index": trace.run_index,
                "received_at": trace.received_at,
                "evaluation_status": trace.evaluation_status,
                "cost_usd": trace.cost_usd,
                "cost_source": trace.cost_source,
            })

        return jsonify({
            "project": project,
            "config": state.config.model_dump(),
            "status": state.status,
            "traces": traces_summary,
            "matching_summary": {
                "total_expected": total_expected,
                "satisfied": satisfied,
                "traces_received": len(state.traces),
                "pending": total_expected - satisfied,
            },
            "evaluation_job_id": state.evaluation_job_id,
        })

    @bp.route("/api/traces/projects/<project>/evaluate", methods=["POST"])
    def trigger_evaluation(project: str):
        """Manually trigger evaluation for a project."""
        unauth = _authorise(project)
        if unauth is not None:
            return unauth

        state = store.get_project(project)
        if not state:
            return jsonify({"error": "Project not found"}), 404

        job_id = store.trigger_evaluation(project)
        return jsonify({
            "ok": True,
            "job_id": job_id,
            "traces_to_evaluate": len([t for t in state.traces if t.matched_query_index is not None]),
        })

    @bp.route("/api/traces/projects/<project>", methods=["DELETE"])
    def delete_project(project: str):
        """Delete a project and its traces."""
        unauth = _authorise(project)
        if unauth is not None:
            return unauth
        store.delete_project(project)
        return jsonify({"ok": True})

    return bp


# Module-level default blueprint — preserves the pre-0.8.0 wiring where
# `cli.py` (and any external Flask app) does
#     from eval_lib.connector.trace_routes import trace_bp
#     app.register_blueprint(trace_bp)
trace_bp = create_trace_blueprint()
