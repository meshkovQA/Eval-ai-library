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
    CustomLLMConfig,
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
#
# PROVIDERS holds metadata only (display name, env vars, flags). The list of
# available models is no longer hand-maintained — it is fetched dynamically from
# eval_lib.model_catalog (which wraps litellm.models_by_provider) inside
# list_providers() below.
#
# The PROVIDERS dict itself is also built dynamically: a hand-curated list of
# "first-class" providers (openai/anthropic/azure/...) with friendly env-var
# conventions, plus an auto-generated entry for every additional LiteLLM
# provider that ships at least one chat model. This means new LiteLLM
# integrations show up in the connector UI as soon as you upgrade litellm,
# without touching this file.

from eval_lib.model_catalog import (
    get_all_litellm_chat_providers,
    get_models_for_provider,
    get_provider_display_name,
    get_provider_env_vars,
)

# First-class providers — hand-tuned display names, env-var sets and flags.
# These wrap eval-lib's own routing aliases (google → gemini, qwen → dashscope,
# grok → xai) and the native paths (ollama, mlx, custom, zhipu).
_FIRST_CLASS_PROVIDERS: dict[str, dict] = {
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
    },
    "google": {
        "name": "Google",
        "env_var": "GOOGLE_API_KEY",
    },
    "azure": {
        "name": "Azure OpenAI",
        "env_var": "AZURE_OPENAI_API_KEY",
        "extra_vars": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "env_var": "OLLAMA_API_KEY",
        "extra_vars": ["OLLAMA_API_BASE_URL"],
        "key_optional": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_var": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "name": "Qwen (Alibaba)",
        "env_var": "DASHSCOPE_API_KEY",
    },
    "zhipu": {
        "name": "Zhipu GLM",
        "env_var": "ZHIPU_API_KEY",
    },
    "mistral": {
        "name": "Mistral AI",
        "env_var": "MISTRAL_API_KEY",
    },
    "groq": {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
    },
    "grok": {
        "name": "Grok (xAI)",
        "env_var": "XAI_API_KEY",
    },
    "custom": {
        "name": "Custom LLM",
        "env_var": "CUSTOM_LLM_API_KEY",
        "extra_vars": ["CUSTOM_LLM_BASE_URL"],
        "key_optional": True,
        "is_custom_llm": True,
    },
}

# LiteLLM provider ids that are duplicates / aliases of a first-class provider.
# Skipped during auto-discovery so we don't show two entries for the same backend.
_FIRST_CLASS_LITELLM_ALIASES = {
    "openai",     # → first_class "openai"
    "anthropic",  # → first_class "anthropic"
    "gemini",     # → first_class "google"
    "azure",      # → first_class "azure"
    "deepseek",   # → first_class "deepseek"
    "dashscope",  # → first_class "qwen"
    "mistral",    # → first_class "mistral"
    "groq",       # → first_class "groq"
    "xai",        # → first_class "grok"
}


def _build_providers() -> dict[str, dict]:
    """
    Compose the full provider dictionary at import time. Order:
        1. First-class providers in their declared order (openai → custom).
        2. All other LiteLLM providers with chat models, alphabetically.
    """
    providers: dict[str, dict] = dict(_FIRST_CLASS_PROVIDERS)
    for pid in get_all_litellm_chat_providers():
        if pid in _FIRST_CLASS_LITELLM_ALIASES or pid in providers:
            continue
        env_vars = get_provider_env_vars(pid)
        primary = env_vars[0] if env_vars else f"{pid.upper()}_API_KEY"
        extras = env_vars[1:]
        providers[pid] = {
            "name": get_provider_display_name(pid),
            "env_var": primary,
            "extra_vars": extras,
        }
    return providers


PROVIDERS = _build_providers()


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
    # Also apply custom LLM config
    try:
        cfg = _load_custom_llm_config()
        if cfg.get("api_key"):
            os.environ["CUSTOM_LLM_API_KEY"] = cfg["api_key"]
        if cfg.get("base_url"):
            os.environ["CUSTOM_LLM_BASE_URL"] = cfg["base_url"]
    except Exception:
        pass


# Apply saved keys on module load
def _init_keys():
    try:
        _apply_api_keys()
    except Exception:
        pass

_init_keys()


def _get_custom_llm_config_path() -> Path:
    d = Path(_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / "custom_llm_config.json"


def _load_custom_llm_config() -> dict:
    p = _get_custom_llm_config_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_custom_llm_config(cfg: dict):
    p = _get_custom_llm_config_path()
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


@connector_bp.route("/api/connector/custom-llm-config", methods=["GET"])
def get_custom_llm_config():
    cfg = _load_custom_llm_config()
    return jsonify(cfg)


@connector_bp.route("/api/connector/custom-llm-config", methods=["POST"])
def save_custom_llm_config():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    cfg = _load_custom_llm_config()
    if "base_url" in body:
        cfg["base_url"] = body["base_url"]
    if "api_key" in body:
        cfg["api_key"] = body["api_key"]
    if "model_name" in body:
        cfg["model_name"] = body["model_name"]

    _save_custom_llm_config(cfg)

    # Set env vars so they are available for LLM client
    if cfg.get("api_key"):
        os.environ["CUSTOM_LLM_API_KEY"] = cfg["api_key"]
    if cfg.get("base_url"):
        os.environ["CUSTOM_LLM_BASE_URL"] = cfg["base_url"]

    # Clear cached LLM clients
    try:
        from eval_lib.llm_client import _get_client
        _get_client.cache_clear()
    except Exception:
        pass

    return jsonify({"ok": True})


@connector_bp.route("/api/connector/providers")
def list_providers():
    keys = _load_api_keys()
    custom_cfg = _load_custom_llm_config()
    result = []
    for pid, pinfo in PROVIDERS.items():
        env_var = pinfo["env_var"]
        has_key = bool(os.environ.get(env_var) or keys.get(env_var))
        extra = {}
        for ev in pinfo.get("extra_vars", []):
            extra[ev] = bool(os.environ.get(ev) or keys.get(ev))

        # Pull models dynamically from LiteLLM via model_catalog. This replaces
        # the old hand-maintained pinfo["models"] lists.
        models = get_models_for_provider(pid)

        item = {
            "id": pid,
            "name": pinfo["name"],
            "env_var": env_var,
            "extra_vars": pinfo.get("extra_vars", []),
            "models": models,
            "configured": has_key or pinfo.get("key_optional", False),
            "has_key": has_key,
            "key_optional": pinfo.get("key_optional", False),
            "extra_configured": extra,
        }

        # For custom_llm, inject saved config and dynamic model list
        if pinfo.get("is_custom_llm"):
            item["is_custom_llm"] = True
            item["custom_llm_config"] = {
                "base_url": custom_cfg.get("base_url", ""),
                "api_key": bool(custom_cfg.get("api_key")),
                "model_name": custom_cfg.get("model_name", ""),
            }
            # Build models list from saved model name
            model_name = custom_cfg.get("model_name", "")
            if model_name:
                item["models"] = [model_name]
            has_base_url = bool(custom_cfg.get("base_url"))
            item["configured"] = has_base_url
            item["has_key"] = bool(custom_cfg.get("api_key"))
            item["extra_configured"]["CUSTOM_LLM_BASE_URL"] = has_base_url

        result.append(item)
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

    # Parse custom LLM config if eval_model starts with custom_llm:
    custom_llm_cfg = None
    eval_model = data.get("eval_model", "gpt-4o-mini")
    if eval_model.startswith("custom:"):
        saved_cfg = _load_custom_llm_config()
        if saved_cfg.get("base_url"):
            custom_llm_cfg = CustomLLMConfig(
                base_url=saved_cfg.get("base_url", ""),
                api_key=saved_cfg.get("api_key", ""),
                model_name=saved_cfg.get("model_name", ""),
            )

    return EvalJobConfig(
        name=data.get("name", "Untitled Job"),
        api_config=api_config,
        response_mapping=response_mapping,
        dataset_column_mapping=column_mapping,
        metrics=metrics,
        eval_model=eval_model,
        custom_llm_config=custom_llm_cfg,
        cost_per_1m_tokens=float(data.get("cost_per_1m_tokens", 0)),
    )
