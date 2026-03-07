import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any

from flask import Blueprint, request, jsonify, Response

from eval_lib.connector.models import (
    ApiConnectionConfig, HeaderEntry, EvalJobConfig,
    ResponseMapping, DatasetColumnMapping, MetricConfig,
)
from eval_lib.connector.dataset_parser import parse_dataset
from eval_lib.connector.metric_registry import get_metrics_info
from eval_lib.connector.engine import ConnectorEngine, test_api_connection

connector_bp = Blueprint("connector", __name__)

# In-memory dataset storage (keyed by dataset_id)
_datasets: Dict[str, Dict[str, Any]] = {}

# Cache directory (set during blueprint registration)
_cache_dir = ".eval_cache"


def set_cache_dir(cache_dir: str):
    global _cache_dir
    _cache_dir = cache_dir


def _get_datasets_dir() -> Path:
    d = Path(_cache_dir) / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_configs_dir() -> Path:
    d = Path(_cache_dir) / "connector_configs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Dataset endpoints ---

@connector_bp.route("/api/connector/upload-dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    content = f.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        return jsonify({"error": "File too large (max 50MB)"}), 400

    try:
        columns, rows = parse_dataset(content, f.filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    dataset_id = str(uuid.uuid4())[:8]
    _datasets[dataset_id] = {"columns": columns, "rows": rows}

    # Persist to disk
    ds_path = _get_datasets_dir() / f"{dataset_id}.json"
    ds_path.write_text(json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False), encoding="utf-8")

    preview = rows[:10]
    return jsonify({
        "dataset_id": dataset_id,
        "columns": columns,
        "preview": preview,
        "row_count": len(rows),
    })


@connector_bp.route("/api/connector/dataset/<dataset_id>")
def get_dataset(dataset_id):
    data = _load_dataset(dataset_id)
    if not data:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(data)


@connector_bp.route("/api/connector/dataset/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id):
    _datasets.pop(dataset_id, None)
    ds_path = _get_datasets_dir() / f"{dataset_id}.json"
    if ds_path.exists():
        ds_path.unlink()
    return jsonify({"ok": True})


def _load_dataset(dataset_id):
    if dataset_id in _datasets:
        return _datasets[dataset_id]
    ds_path = _get_datasets_dir() / f"{dataset_id}.json"
    if ds_path.exists():
        data = json.loads(ds_path.read_text(encoding="utf-8"))
        _datasets[dataset_id] = data
        return data
    return None


# --- Test connection ---

@connector_bp.route("/api/connector/test-connection", methods=["POST"])
def api_test_connection():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    try:
        api_config = ApiConnectionConfig(
            base_url=body.get("base_url", ""),
            method=body.get("method", "POST"),
            headers=[HeaderEntry(**h) for h in body.get("headers", [])],
            query_params=body.get("query_params", {}),
            body_template=body.get("body_template", ""),
            timeout_seconds=body.get("timeout_seconds", 60),
        )
    except Exception as e:
        return jsonify({"error": f"Invalid config: {e}"}), 400

    sample_row = body.get("sample_row", {})
    variable_map = body.get("variable_map", {})

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            test_api_connection(api_config, sample_row, variable_map)
        )
    finally:
        loop.close()

    return Response(
        json.dumps(result, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


# --- Metrics ---

@connector_bp.route("/api/connector/metrics")
def list_metrics():
    return jsonify(get_metrics_info())


# --- Job execution ---

@connector_bp.route("/api/connector/start-job", methods=["POST"])
def start_job():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    dataset_id = body.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "dataset_id required"}), 400

    data = _load_dataset(dataset_id)
    if not data:
        return jsonify({"error": "Dataset not found"}), 404

    try:
        config = _parse_job_config(body.get("config", {}))
    except Exception as e:
        return jsonify({"error": f"Invalid config: {e}"}), 400

    engine = ConnectorEngine()
    try:
        job_id = engine.start_job(config, data["rows"], cache_dir=_cache_dir)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409

    return jsonify({"job_id": job_id})


@connector_bp.route("/api/connector/job/<job_id>/progress")
def job_progress(job_id):
    engine = ConnectorEngine()
    progress = engine.get_progress(job_id)
    if not progress:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(progress.model_dump())


@connector_bp.route("/api/connector/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    engine = ConnectorEngine()
    ok = engine.cancel_job(job_id)
    if not ok:
        return jsonify({"error": "Job not found or not running"}), 404
    return jsonify({"ok": True})


# --- Config save/load ---

@connector_bp.route("/api/connector/save-config", methods=["POST"])
def save_config():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    config_id = body.get("id") or str(uuid.uuid4())[:8]
    body["id"] = config_id

    from datetime import datetime
    body["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_path = _get_configs_dir() / f"{config_id}.json"
    config_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    return jsonify({"config_id": config_id})


@connector_bp.route("/api/connector/configs")
def list_configs():
    configs_dir = _get_configs_dir()
    configs = []
    for p in sorted(configs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            configs.append({
                "id": data.get("id", p.stem),
                "name": data.get("name", "Untitled"),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            pass
    return jsonify(configs)


@connector_bp.route("/api/connector/config/<config_id>")
def load_config(config_id):
    config_path = _get_configs_dir() / f"{config_id}.json"
    if not config_path.exists():
        return jsonify({"error": "Config not found"}), 404
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return jsonify(data)


@connector_bp.route("/api/connector/config/<config_id>", methods=["DELETE"])
def delete_config(config_id):
    config_path = _get_configs_dir() / f"{config_id}.json"
    if config_path.exists():
        config_path.unlink()
    return jsonify({"ok": True})


# --- Provider / API key management ---

PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "gpt-4.5-preview", "o4-mini"],
    },
    "anthropic": {
        "name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-opus-4-6-20250619", "claude-opus-4-5-20250415", "claude-sonnet-4-6-20250619", "claude-sonnet-4-5-20250415", "claude-haiku-4-5-20251001"],
    },
    "google": {
        "name": "Google",
        "env_var": "GOOGLE_API_KEY",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    },
    "azure": {
        "name": "Azure OpenAI",
        "env_var": "AZURE_OPENAI_API_KEY",
        "extra_vars": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"],
        "models": ["gpt-4.1", "gpt-4.5-preview", "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-32k", "gpt-35-turbo"],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "env_var": "OLLAMA_API_KEY",
        "extra_vars": ["OLLAMA_API_BASE_URL"],
        "models": ["llama3.3", "llama3.1", "mistral", "mixtral", "phi4", "gemma2", "qwen2.5"],
        "key_optional": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_var": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-v3.2", "deepseek-v3.2-exp", "deepseek-v3.1", "deepseek-v3", "deepseek-reasoner", "deepseek-r1", "deepseek-r1-lite"],
    },
    "qwen": {
        "name": "Qwen (Alibaba)",
        "env_var": "DASHSCOPE_API_KEY",
        "models": ["qwen-max", "qwen-plus", "qwen-turbo", "qwen-long", "qwq-plus"],
    },
    "zhipu": {
        "name": "Zhipu GLM",
        "env_var": "ZHIPU_API_KEY",
        "models": ["glm-4-plus", "glm-4-air", "glm-4-airx", "glm-4-long", "glm-4-flash", "glm-4-flashx"],
    },
    "mistral": {
        "name": "Mistral AI",
        "env_var": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-small-latest", "codestral-latest", "pixtral-large-latest", "ministral-8b-latest"],
    },
    "groq": {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-90b-vision-preview", "mixtral-8x7b-32768", "gemma2-9b-it", "deepseek-r1-distill-llama-70b"],
    },
    "grok": {
        "name": "Grok (xAI)",
        "env_var": "XAI_API_KEY",
        "models": ["grok-4.1", "grok-4", "grok-4-heavy", "grok-4-fast", "grok-beta", "grok-3", "grok-2"],
    },
}


def _get_api_keys_path() -> Path:
    d = Path(_cache_dir) / "api_keys.json"
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


def _load_api_keys() -> dict:
    p = _get_api_keys_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_api_keys(keys: dict):
    p = _get_api_keys_path()
    p.write_text(json.dumps(keys, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_api_keys():
    """Set saved API keys as environment variables."""
    keys = _load_api_keys()
    for var, val in keys.items():
        if val:
            os.environ[var] = val


# Apply saved keys on module load
def _init_keys():
    try:
        _apply_api_keys()
    except Exception:
        pass

_init_keys()


@connector_bp.route("/api/connector/providers")
def list_providers():
    keys = _load_api_keys()
    result = []
    for pid, pinfo in PROVIDERS.items():
        env_var = pinfo["env_var"]
        has_key = bool(os.environ.get(env_var) or keys.get(env_var))
        extra = {}
        for ev in pinfo.get("extra_vars", []):
            extra[ev] = bool(os.environ.get(ev) or keys.get(ev))
        result.append({
            "id": pid,
            "name": pinfo["name"],
            "env_var": env_var,
            "extra_vars": pinfo.get("extra_vars", []),
            "models": pinfo["models"],
            "configured": has_key or pinfo.get("key_optional", False),
            "has_key": has_key,
            "key_optional": pinfo.get("key_optional", False),
            "extra_configured": extra,
        })
    return jsonify(result)


@connector_bp.route("/api/connector/save-api-key", methods=["POST"])
def save_api_key():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    env_var = body.get("env_var", "")
    value = body.get("value", "")

    if not env_var:
        return jsonify({"error": "env_var required"}), 400

    keys = _load_api_keys()
    if value:
        keys[env_var] = value
        os.environ[env_var] = value
    else:
        keys.pop(env_var, None)
        os.environ.pop(env_var, None)

    _save_api_keys(keys)

    # Clear cached LLM clients so new key is picked up
    try:
        from eval_lib.llm_client import _get_client
        _get_client.cache_clear()
    except Exception:
        pass

    return jsonify({"ok": True})


@connector_bp.route("/api/connector/delete-api-key", methods=["POST"])
def delete_api_key():
    body = request.get_json()
    env_var = body.get("env_var", "") if body else ""
    if not env_var:
        return jsonify({"error": "env_var required"}), 400

    keys = _load_api_keys()
    keys.pop(env_var, None)
    os.environ.pop(env_var, None)
    _save_api_keys(keys)

    try:
        from eval_lib.llm_client import _get_client
        _get_client.cache_clear()
    except Exception:
        pass

    return jsonify({"ok": True})


def _parse_job_config(data: dict) -> EvalJobConfig:
    api_data = data.get("api_config", {})
    api_config = ApiConnectionConfig(
        name=api_data.get("name", "Untitled"),
        base_url=api_data.get("base_url", ""),
        method=api_data.get("method", "POST"),
        headers=[HeaderEntry(**h) for h in api_data.get("headers", [])],
        query_params=api_data.get("query_params", {}),
        body_template=api_data.get("body_template", ""),
        timeout_seconds=api_data.get("timeout_seconds", 60),
        max_retries=api_data.get("max_retries", 1),
        delay_between_requests_ms=api_data.get("delay_between_requests_ms", 0),
    )

    resp_data = data.get("response_mapping", {})
    response_mapping = ResponseMapping(
        actual_output_path=resp_data.get("actual_output_path", ""),
        retrieval_context_path=resp_data.get("retrieval_context_path") or None,
        tools_called_path=resp_data.get("tools_called_path") or None,
        token_usage_path=resp_data.get("token_usage_path") or None,
        system_prompt_path=resp_data.get("system_prompt_path") or None,
    )

    col_data = data.get("dataset_column_mapping", {})
    column_mapping = DatasetColumnMapping(
        input_column=col_data.get("input_column", ""),
        expected_output_column=col_data.get("expected_output_column") or None,
        context_column=col_data.get("context_column") or None,
        tools_called_column=col_data.get("tools_called_column") or None,
        expected_tools_column=col_data.get("expected_tools_column") or None,
        template_variable_map=col_data.get("template_variable_map", {}),
    )

    metrics = []
    for mc in data.get("metrics", []):
        metrics.append(MetricConfig(
            metric_class=mc["metric_class"],
            params=mc.get("params", {}),
        ))

    return EvalJobConfig(
        name=data.get("name", "Untitled Job"),
        api_config=api_config,
        response_mapping=response_mapping,
        dataset_column_mapping=column_mapping,
        metrics=metrics,
        eval_model=data.get("eval_model", "gpt-4o-mini"),
        cost_per_1m_tokens=float(data.get("cost_per_1m_tokens", 0)),
    )
