"""Flask Blueprint for Trace Receiver HTTP endpoints.

Provides endpoints for:
- Receiving traces from remote agents (POST /api/traces/ingest)
- Managing trace projects (CRUD)
- Triggering evaluations
"""

from flask import Blueprint, request, jsonify
from eval_lib.connector.trace_receiver import TraceStore
from eval_lib.connector.trace_models import TraceProjectConfig, MatchingStrategy

trace_bp = Blueprint("traces", __name__)

_datasets = {}  # Shared reference — set by cli.py


def _get_store() -> TraceStore:
    return TraceStore()


# ---- Trace Ingestion ----

@trace_bp.route("/api/traces/ingest", methods=["POST"])
def ingest_trace():
    """Receive a trace from a remote agent (TraceSender)."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    project_name = data.get("project", "")
    trace_data = data.get("trace", {})

    if not project_name:
        return jsonify({"error": "Missing 'project' field"}), 400
    if not trace_data:
        return jsonify({"error": "Missing 'trace' field"}), 400

    store = _get_store()
    state = store.get_project(project_name)
    if not state:
        return jsonify({"error": f"Project '{project_name}' not found"}), 404

    # Validate API key
    auth_header = request.headers.get("Authorization", "")
    api_key = ""
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]

    if not store.validate_api_key(project_name, api_key):
        return jsonify({"error": "Invalid API key"}), 401

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

@trace_bp.route("/api/traces/projects", methods=["POST"])
def create_project():
    """Create a trace receiver project."""
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

    store = _get_store()
    state = store.create_project(config, dataset_rows)

    return jsonify({
        "ok": True,
        "project": project_name,
        "expected_queries": len(state.query_index),
        "total_dataset_rows": len(dataset_rows),
        "api_key_set": bool(api_key),
    })


@trace_bp.route("/api/traces/projects", methods=["GET"])
def list_projects():
    """List all trace receiver projects."""
    store = _get_store()
    return jsonify(store.list_projects())


@trace_bp.route("/api/traces/projects/<project>", methods=["GET"])
def get_project(project: str):
    """Get project details with traces."""
    store = _get_store()
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


@trace_bp.route("/api/traces/projects/<project>/evaluate", methods=["POST"])
def trigger_evaluation(project: str):
    """Manually trigger evaluation for a project."""
    store = _get_store()
    state = store.get_project(project)
    if not state:
        return jsonify({"error": "Project not found"}), 404

    job_id = store.trigger_evaluation(project)
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "traces_to_evaluate": len([t for t in state.traces if t.matched_query_index is not None]),
    })


@trace_bp.route("/api/traces/projects/<project>", methods=["DELETE"])
def delete_project(project: str):
    """Delete a project and its traces."""
    store = _get_store()
    store.delete_project(project)
    return jsonify({"ok": True})
