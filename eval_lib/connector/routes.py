import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

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

_log = logging.getLogger(__name__)

# In-memory dataset storage (keyed by dataset_id)
_datasets: Dict[str, Dict[str, Any]] = {}

# Cache directory (set during blueprint registration)
_cache_dir = ".eval_cache"

# Identifiers that become filesystem paths must be restricted to a safe
# character set, otherwise a value like "../../etc/passwd" would let a request
# escape the cache directory (path traversal).
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def set_cache_dir(cache_dir: str):
    global _cache_dir
    _cache_dir = cache_dir


def _safe_id(value: Any) -> Optional[str]:
    """Return the value if it is a safe path component, else None.

    Accepts only letters, digits, underscore and hyphen — enough for the
    uuid-derived dataset/config ids and user-chosen project names, while
    rejecting separators ('/', '\\'), '..' and absolute paths.
    """
    if not isinstance(value, str):
        return None
    return value if _SAFE_ID_RE.match(value) else None


def _resolve_within(base: Path, *parts: str) -> Optional[Path]:
    """Join parts onto base and confirm the result stays inside base.

    Defence-in-depth on top of _safe_id: even if the regex is loosened later,
    a resolved path that escapes the base directory is rejected.
    """
    base = base.resolve()
    candidate = (base / Path(*parts)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


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
    except Exception:
        # Full detail goes to the server log; the client gets a generic message
        # so internal paths / stack frames are not exposed.
        _log.exception("Failed to parse uploaded dataset")
        return jsonify({"error": "Could not parse dataset. Check the file format."}), 400

    dataset_id = str(uuid.uuid4())[:8]
    _datasets[dataset_id] = {"columns": columns, "rows": rows}

    # Persist to disk. dataset_id is uuid-derived (always safe), but route the
    # write through _resolve_within so the path is provably inside the cache dir.
    ds_path = _resolve_within(_get_datasets_dir(), f"{dataset_id}.json")
    if not ds_path:
        return jsonify({"error": "Could not persist dataset"}), 500
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
    if not _safe_id(dataset_id):
        return jsonify({"error": "Invalid dataset id"}), 400
    data = _load_dataset(dataset_id)
    if not data:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(data)


@connector_bp.route("/api/connector/dataset/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id):
    if not _safe_id(dataset_id):
        return jsonify({"error": "Invalid dataset id"}), 400
    _datasets.pop(dataset_id, None)
    ds_path = _resolve_within(_get_datasets_dir(), f"{dataset_id}.json")
    if ds_path and ds_path.exists():
        ds_path.unlink()
    return jsonify({"ok": True})


def _load_dataset(dataset_id):
    if not _safe_id(dataset_id):
        return None
    if dataset_id in _datasets:
        return _datasets[dataset_id]
    ds_path = _resolve_within(_get_datasets_dir(), f"{dataset_id}.json")
    if ds_path and ds_path.exists():
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
    except Exception:
        _log.exception("Invalid API connection config in test-connection")
        return jsonify({"error": "Invalid API connection config."}), 400

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
    except Exception:
        _log.exception("Invalid job config in start-job")
        return jsonify({"error": "Invalid job config."}), 400

    engine = ConnectorEngine()
    try:
        job_id = engine.start_job(config, data["rows"], cache_dir=_cache_dir)
    except RuntimeError:
        # The only RuntimeError start_job raises is "a job is already running".
        return jsonify({"error": "A job is already running."}), 409

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

    # config_id may be supplied by the client (updating an existing config), so
    # it must be validated before it becomes a file name.
    config_id = body.get("id") or str(uuid.uuid4())[:8]
    if not _safe_id(config_id):
        return jsonify({"error": "Invalid config id"}), 400
    body["id"] = config_id

    from datetime import datetime
    body["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_path = _resolve_within(_get_configs_dir(), f"{config_id}.json")
    if not config_path:
        return jsonify({"error": "Invalid config id"}), 400
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
    if not _safe_id(config_id):
        return jsonify({"error": "Invalid config id"}), 400
    config_path = _resolve_within(_get_configs_dir(), f"{config_id}.json")
    if not config_path or not config_path.exists():
        return jsonify({"error": "Config not found"}), 404
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return jsonify(data)


@connector_bp.route("/api/connector/config/<config_id>", methods=["DELETE"])
def delete_config(config_id):
    if not _safe_id(config_id):
        return jsonify({"error": "Invalid config id"}), 400
    config_path = _resolve_within(_get_configs_dir(), f"{config_id}.json")
    if config_path and config_path.exists():
        config_path.unlink()
    return jsonify({"ok": True})


# --- Provider / API key management ---
#
# There is NO hand-maintained "first-class" provider list. The full catalogue
# of providers, their display names, required env vars and model lists all
# come from LiteLLM (via eval_lib.model_catalog) — upgrade litellm and any
# new integration shows up in the connector UI automatically.
#
# The only hand-maintained piece is `_NATIVE_PROVIDERS`: providers that do NOT
# go through LiteLLM and have dedicated code paths in llm_client.py (Ollama,
# MLX, Zhipu, Custom LLM). They get appended to the end of the provider list.

from eval_lib.model_catalog import (
    get_all_litellm_chat_providers,
    get_models_for_provider,
    get_provider_display_name,
    get_provider_env_vars,
)

# Providers that do not go through LiteLLM. Each has a native helper in
# llm_client.py and its own config shape in the UI.
_NATIVE_PROVIDERS: dict[str, dict] = {
    "ollama": {
        "name": "Ollama (Local)",
        "env_var": "OLLAMA_API_KEY",
        "extra_vars": ["OLLAMA_API_BASE_URL"],
        "key_optional": True,
        "is_native": True,
    },
    "mlx": {
        "name": "MLX (Apple Silicon)",
        "env_var": "MLX_API_BASE_URL",
        "extra_vars": [],
        "key_optional": True,
        "is_native": True,
    },
    "zhipu": {
        "name": "Zhipu GLM",
        "env_var": "ZHIPU_API_KEY",
        "extra_vars": [],
        "is_native": True,
    },
    "custom": {
        "name": "Custom LLM",
        "env_var": "CUSTOM_LLM_API_KEY",
        "extra_vars": ["CUSTOM_LLM_BASE_URL"],
        "key_optional": True,
        "is_custom_llm": True,
        "is_native": True,
    },
}


_providers_cache: dict[str, dict] | None = None


def _build_providers() -> dict[str, dict]:
    """
    Compose the full provider dictionary on first access.

    Order:
        1. Every LiteLLM provider with chat models, in alphabetical order.
           Display name and env vars come from the model catalog (which in
           turn queries litellm.validate_environment).
        2. Native providers (Ollama, MLX, Zhipu, Custom) appended at the end.

    Result is memoised in `_providers_cache` so the first /api/connector/providers
    request pays the discovery cost (and the litellm.validate_environment probes
    that go with it), while subsequent requests are O(1).
    """
    providers: dict[str, dict] = {}

    for pid in get_all_litellm_chat_providers():
        # Skip litellm ids that would clash with a native provider entry.
        if pid in _NATIVE_PROVIDERS:
            continue
        env_vars = get_provider_env_vars(pid)
        # LiteLLM reports env vars in arbitrary order. Surface the API key
        # first so the UI shows credentials before endpoint/version fields.
        ordered = sorted(env_vars, key=lambda v: (0 if "API_KEY" in v else 1, v))
        primary = ordered[0] if ordered else f"{pid.upper()}_API_KEY"
        extras = ordered[1:]
        providers[pid] = {
            "name": get_provider_display_name(pid),
            "env_var": primary,
            "extra_vars": extras,
            "is_native": False,
        }

    # Native providers come after the LiteLLM catalogue so they're grouped at
    # the bottom of the UI list.
    for pid, pinfo in _NATIVE_PROVIDERS.items():
        providers[pid] = dict(pinfo)

    return providers


def get_providers() -> dict[str, dict]:
    """Return the (cached) provider dict, building it lazily on first call."""
    global _providers_cache
    if _providers_cache is None:
        _providers_cache = _build_providers()
    return _providers_cache


def __getattr__(name):
    # Module-level __getattr__ (PEP 562) lets us expose `PROVIDERS` as a
    # lazily-built attribute. `from eval_lib.connector.routes import PROVIDERS`
    # still works, but the build only happens on first access — `import
    # eval_lib.connector.routes` no longer triggers the litellm probes.
    if name == "PROVIDERS":
        return get_providers()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _write_secret_file(path: Path, content: str):
    """Write a file that contains credentials with owner-only permissions (0600).

    The connector is a local desktop tool, so API keys live on disk next to the
    cache. We cannot avoid storing them, but we restrict the file so other OS
    users / processes cannot read it. The file is created with 0600 from the
    start (via os.open) so there is no window where the secret is world-readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    # Re-assert permissions in case the file pre-existed with looser bits.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


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
    _write_secret_file(p, json.dumps(keys, ensure_ascii=False, indent=2))


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
    # Contains an api_key field — write with owner-only permissions.
    p = _get_custom_llm_config_path()
    _write_secret_file(p, json.dumps(cfg, ensure_ascii=False, indent=2))


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
    for pid, pinfo in get_providers().items():
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


@connector_bp.route("/api/connector/save-provider-config", methods=["POST"])
def save_provider_config():
    """Persist several env vars for one provider in a single call.

    Body: {"provider": "<id>", "values": {"ENV_VAR": "value", ...}}

    Empty string values clear the corresponding variable. Masked placeholders
    that start with "•" are ignored, so the UI can safely send back the
    rendered placeholder for fields the user did not touch.
    """
    body = request.get_json() or {}
    provider = body.get("provider", "")
    values = body.get("values") or {}
    if not provider:
        return jsonify({"error": "provider required"}), 400
    if not isinstance(values, dict):
        return jsonify({"error": "values must be an object"}), 400

    pinfo = get_providers().get(provider)
    if not pinfo:
        return jsonify({"error": f"unknown provider: {provider}"}), 404
    allowed = {pinfo["env_var"], *pinfo.get("extra_vars", [])}

    keys = _load_api_keys()
    for env_var, raw in values.items():
        if env_var not in allowed:
            continue
        value = (raw or "").strip()
        if value.startswith("•"):
            continue
        if value:
            keys[env_var] = value
            os.environ[env_var] = value
        else:
            keys.pop(env_var, None)
            os.environ.pop(env_var, None)

    _save_api_keys(keys)

    try:
        from eval_lib.llm_client import _get_client
        _get_client.cache_clear()
    except Exception:
        pass

    return jsonify({"ok": True})


@connector_bp.route("/api/connector/test-provider", methods=["POST"])
def test_provider():
    """Send a tiny chat_complete ping to verify the provider's credentials.

    Body: {"provider": "<id>", "model": "<deployment-or-model-name>"}

    Returns: {"ok": true, "latency_ms": <int>} on success or
             {"ok": false, "error": "<message>"} on failure.
    """
    body = request.get_json() or {}
    provider = body.get("provider", "")
    model = (body.get("model") or "").strip()
    if not provider:
        return jsonify({"ok": False, "error": "provider required"}), 400

    pinfo = get_providers().get(provider)
    if not pinfo:
        return jsonify({"ok": False, "error": f"unknown provider: {provider}"}), 404

    if not model:
        models = get_models_for_provider(provider)
        if not models:
            return jsonify({
                "ok": False,
                "error": "no model specified and no models known for this provider",
            }), 400
        model = models[0]

    spec = f"{provider}:{model}"

    import time
    from eval_lib.llm_client import chat_complete, LLMConfigurationError

    t0 = time.perf_counter()
    try:
        text, _ = asyncio.run(chat_complete(
            spec,
            [{"role": "user", "content": "ping"}],
            temperature=0.0,
        ))
    except LLMConfigurationError as e:
        # Our own controlled error type — its message is a user-facing hint
        # (e.g. "API key not set"), safe to surface.
        return jsonify({"ok": False, "error": str(e)}), 200
    except Exception as e:
        # Unknown provider/SDK error: log the detail server-side, return only
        # the exception class name so internals are not exposed to the client.
        _log.exception("test-provider failed for %s", spec)
        return jsonify({
            "ok": False,
            "error": f"Provider check failed ({type(e).__name__}).",
        }), 200

    latency_ms = int((time.perf_counter() - t0) * 1000)
    preview = (text or "").strip()
    if len(preview) > 120:
        preview = preview[:117] + "..."
    return jsonify({"ok": True, "latency_ms": latency_ms, "model": spec, "reply": preview})


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
