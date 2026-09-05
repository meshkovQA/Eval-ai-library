# model_catalog.py
"""
Thin facade over LiteLLM's built-in model registry.

LiteLLM ships with two big in-memory tables:
    - litellm.models_by_provider — {provider_name: set[model_id]}, ~2k entries
    - litellm.model_cost         — {model_id: {input_cost_per_token, output_cost_per_token,
                                                litellm_provider, mode, ...}}

This module exposes them under the eval-ai-library Provider taxonomy so that:
    1. The connector UI can populate model dropdowns dynamically (no more hand-maintained
       lists in routes.PROVIDERS).
    2. llm_client._calculate_cost() can fall back to LiteLLM's prices when our
       optional override table in price.py doesn't have the model.

Native providers (Ollama, MLX, Custom) keep a small hand-curated list because they
either run locally or are user-defined.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import litellm

from .price import model_pricing as _override_pricing

# Pretty display names for LiteLLM providers when we auto-discover them.
# Anything not in this map gets a Title-cased fallback (snake_case → "Snake Case").
_LITELLM_DISPLAY_NAMES: dict[str, str] = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "azure_ai": "Azure AI Foundry",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "vertex_ai": "Google Vertex AI",
    "bedrock": "AWS Bedrock",
    "cohere": "Cohere",
    "cohere_chat": "Cohere Chat",
    "deepseek": "DeepSeek",
    "dashscope": "Qwen (DashScope)",
    "mistral": "Mistral AI",
    "groq": "Groq",
    "xai": "Grok (xAI)",
    "openrouter": "OpenRouter",
    "fireworks_ai": "Fireworks AI",
    "together_ai": "Together AI",
    "perplexity": "Perplexity",
    "deepinfra": "DeepInfra",
    "cerebras": "Cerebras",
    "anyscale": "Anyscale",
    "cloudflare": "Cloudflare Workers AI",
    "watsonx": "IBM watsonx",
    "databricks": "Databricks",
    "snowflake": "Snowflake Cortex",
    "sambanova": "SambaNova",
    "moonshot": "Moonshot (Kimi)",
    "hyperbolic": "Hyperbolic",
    "lambda_ai": "Lambda Labs",
    "novita": "Novita AI",
    "nscale": "Nscale",
    "vercel_ai_gateway": "Vercel AI Gateway",
    "volcengine": "VolcEngine (ByteDance)",
    "meta_llama": "Meta Llama API",
    "friendliai": "FriendliAI",
    "featherless_ai": "Featherless",
    "heroku": "Heroku AI",
    "morph": "Morph",
    "v0": "Vercel v0",
    "oci": "Oracle OCI",
    "palm": "Google PaLM",
    "nlp_cloud": "NLP Cloud",
    "gradient_ai": "Gradient AI",
    "aleph_alpha": "Aleph Alpha",
    "codestral": "Mistral Codestral",
    "ollama": "Ollama (Local)",
    "ollama_chat": "Ollama Chat",
}

# Legacy eval-lib provider ids → their canonical LiteLLM names.
#
# These three aliases (google/qwen/grok) are the ONLY hand-maintained
# provider mapping left. They exist so user code using the old eval-lib
# prefixes ("google:gemini-2.0-flash", "qwen:qwen-max", "grok:grok-2")
# continues to resolve to the correct LiteLLM models. Every other provider
# goes through LiteLLM under its own name ("cohere", "bedrock", "vertex_ai",
# "openrouter" etc.) with no translation layer.
#
# Keep in sync with llm_client._LEGACY_PROVIDER_ALIASES.
_PROVIDER_TO_LITELLM: dict[str, str] = {
    "google": "gemini",
    "qwen": "dashscope",
    "grok": "xai",
    # zhipu has no first-class LiteLLM integration; we route it through
    # openai-compatible base_url. Models are listed manually below.
}

# Hand-curated lists for providers LiteLLM doesn't enumerate well.
_NATIVE_MODELS: dict[str, list[str]] = {
    "ollama": [
        "llama3.3",
        "llama3.1",
        "mistral",
        "mixtral",
        "phi4",
        "gemma2",
        "qwen2.5",
    ],
    "mlx": [],
    "custom": [],
    "zhipu": [
        "glm-4-plus",
        "glm-4-air",
        "glm-4-airx",
        "glm-4-long",
        "glm-4-flash",
        "glm-4-flashx",
    ],
}


def _strip_provider_prefix(model: str, litellm_provider: str) -> str:
    """
    LiteLLM stores some entries as "<provider>/<model>" (e.g. "xai/grok-2").
    We want the bare model id so that callers can use it as-is — _to_litellm_args()
    re-attaches the prefix on the way to litellm.acompletion().

    Only strip when the prefix actually matches this provider, otherwise leave it
    alone (e.g. "meta-llama/llama-4-maverick..." on Groq is NOT a provider prefix).
    """
    expected = f"{litellm_provider}/"
    if model.startswith(expected):
        return model[len(expected) :]
    return model


def _is_chat_model(model_id: str) -> bool:
    """Filter out image / audio / embedding / tts entries — only chat models."""
    info = litellm.model_cost.get(model_id)
    if not info:
        return False
    return info.get("mode") == "chat"


def get_models_for_provider(provider: str) -> list[str]:
    """
    Return the sorted list of chat-capable model ids known to LiteLLM for a
    given eval-lib provider id.

    Resolution order:
        1. Native curated lists (ollama, mlx, custom, zhipu).
        2. Aliased providers (google → gemini, qwen → dashscope, grok → xai)
           via _PROVIDER_TO_LITELLM mapping.
        3. Direct LiteLLM provider lookup — any provider id that exists in
           litellm.models_by_provider works as-is (e.g. "cohere", "bedrock",
           "fireworks_ai", "openrouter", etc.).
        4. Unknown providers return an empty list.
    """
    if provider in _NATIVE_MODELS:
        return list(_NATIVE_MODELS[provider])

    litellm_name = _PROVIDER_TO_LITELLM.get(provider, provider)

    raw = litellm.models_by_provider.get(litellm_name, set())
    cleaned: set[str] = set()
    for entry in raw:
        short = _strip_provider_prefix(entry, litellm_name)
        # Filter to chat models. Check both the short name and the original
        # entry — model_cost is keyed inconsistently across providers.
        if _is_chat_model(short) or _is_chat_model(entry):
            cleaned.add(short)

    return sorted(cleaned)


def get_provider_display_name(provider: str) -> str:
    """Pretty name for a provider id (falls back to Title-cased snake_case)."""
    if provider in _LITELLM_DISPLAY_NAMES:
        return _LITELLM_DISPLAY_NAMES[provider]
    return provider.replace("_", " ").title()


def _detect_env_vars(litellm_name: str) -> list[str]:
    """
    Ask LiteLLM which env vars authenticate this provider, by validating a
    representative chat model.

    LiteLLM's `validate_environment` only lists vars that are *currently missing
    in process env* — once a user saves a key, that variable disappears from
    the list. To get a stable answer we temporarily unset everything that looks
    like a provider env var, ask LiteLLM, then restore the originals. This keeps
    the UI panel rendering the same fields whether or not keys are already set.
    """
    sample = None
    for entry in litellm.models_by_provider.get(litellm_name, set()):
        short = entry.split("/", 1)[1] if "/" in entry else entry
        info = litellm.model_cost.get(short) or litellm.model_cost.get(entry)
        if info and info.get("mode") == "chat":
            sample = entry if "/" in entry else f"{litellm_name}/{entry}"
            break
    if not sample:
        return []
    stashed: dict[str, str] = {}
    for k in list(os.environ):
        if k.endswith(("_API_KEY", "_API_BASE", "_API_VERSION", "_ENDPOINT", "_PROJECT", "_LOCATION", "_REGION", "_REGION_NAME", "_ACCOUNT_ID", "_PROJECT_ID")) \
                or k.startswith(("AWS_", "AZURE_", "VERTEX_", "VERTEXAI_", "GOOGLE_", "GEMINI_")):
            stashed[k] = os.environ.pop(k)
    try:
        result = litellm.validate_environment(model=sample)
        return list(result.get("missing_keys") or [])
    except Exception:
        return []
    finally:
        os.environ.update(stashed)


def get_all_litellm_chat_providers() -> list[str]:
    """
    Return every LiteLLM provider id that has at least one chat-capable model.
    Used by the connector UI to auto-populate the provider dropdown.
    """
    out = []
    for p in litellm.models_by_provider.keys():
        # Skip duplicate / non-chat shells
        if p in {"ollama", "ollama_chat", "text-completion-openai", "text-completion-codestral", "azure_text"}:
            continue
        models = get_models_for_provider(p)
        if models:
            out.append(p)
    return sorted(out)


# Native providers that LiteLLM doesn't know about (local servers or
# OpenAI-compatible passthroughs we route ourselves). For everything else we
# ask LiteLLM via _detect_env_vars().
_NATIVE_ENV_VARS: dict[str, list[str]] = {
    "ollama": ["OLLAMA_API_BASE_URL"],
    "mlx": ["MLX_API_BASE_URL"],
    "zhipu": ["ZHIPU_API_KEY"],
    "custom": ["CUSTOM_LLM_API_KEY", "CUSTOM_LLM_BASE_URL"],
}


def get_provider_env_vars(provider: str) -> list[str]:
    """
    Return the env vars a provider needs, as reported by LiteLLM itself.

    For native providers (ollama/mlx/zhipu/custom) LiteLLM has no entry, so we
    keep a tiny hand-list. For every other id we resolve any legacy alias via
    _PROVIDER_TO_LITELLM and ask LiteLLM directly. Last-resort fallback is a
    generic <UPPER>_API_KEY guess for obscure providers LiteLLM can't introspect.
    """
    if provider in _NATIVE_ENV_VARS:
        return list(_NATIVE_ENV_VARS[provider])
    litellm_name = _PROVIDER_TO_LITELLM.get(provider, provider)
    detected = _detect_env_vars(litellm_name)
    if detected:
        return detected
    return [f"{provider.upper()}_API_KEY"]


def all_models_by_provider() -> dict[str, list[str]]:
    """
    Convenience: return {provider_id: [model_id, ...]} for every provider that
    eval-lib knows about — first-class aliases (openai/google/grok/...), native
    providers (ollama/mlx/custom/zhipu) and every auto-discovered LiteLLM
    provider with chat models.
    """
    out: dict[str, list[str]] = {}
    seen: set[str] = set()
    for pid in list(_PROVIDER_TO_LITELLM.keys()) + list(_NATIVE_MODELS.keys()):
        out[pid] = get_models_for_provider(pid)
        seen.add(pid)
    for pid in get_all_litellm_chat_providers():
        # Skip aliases that map to a first-class provider already in `out`.
        # We rely on the litellm name == eval-lib provider id for everything else.
        if pid in seen:
            continue
        if pid in {"gemini", "dashscope", "xai"}:
            continue  # already exposed via google/qwen/grok aliases
        out[pid] = get_models_for_provider(pid)
        seen.add(pid)
    return out


# Release-date suffixes providers append to an otherwise stable model id:
# "-2025-08-07" (OpenAI) and "-20250514" (Anthropic). Stripping one of these
# is safe because a date is never a distinct SKU — unlike, say, the "-mini"
# in "gpt-4o-mini", which must never be stripped.
_DATE_SUFFIX_RE = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$")


def _pricing_candidates(model: str) -> list[str]:
    """Model ids to try, most specific first.

    Handles the two ways a real-world id misses an exact table entry:
    a release-date suffix the table doesn't carry, and a ``provider/``
    routing prefix.
    """
    candidates: list[str] = []

    def add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    for base in (model, model.split("/", 1)[1] if "/" in model else ""):
        if not base:
            continue
        add(base)
        stripped = _DATE_SUFFIX_RE.sub("", base)
        add(stripped)
    return candidates


def _lookup_pricing(model: str) -> Optional[dict]:
    """Resolve one model id against the override table, then LiteLLM."""
    override = _override_pricing.get(model)
    if override:
        price = {"input": override["input"], "output": override["output"]}
        # Cache rates are optional in the override table; only surface the
        # keys when a value actually exists so the returned shape stays
        # exactly {"input", "output"} for the common case.
        for key in ("cache_read", "cache_write"):
            if override.get(key) is not None:
                price[key] = override[key]
        return price

    info = litellm.model_cost.get(model)
    if not info:
        return None

    in_per_token = info.get("input_cost_per_token")
    out_per_token = info.get("output_cost_per_token")
    if in_per_token is None and out_per_token is None:
        return None

    price = {
        "input": (in_per_token or 0.0) * 1_000_000,
        "output": (out_per_token or 0.0) * 1_000_000,
    }
    cache_read = info.get("cache_read_input_token_cost")
    if cache_read is not None:
        price["cache_read"] = cache_read * 1_000_000
    cache_write = info.get("cache_creation_input_token_cost")
    if cache_write is not None:
        price["cache_write"] = cache_write * 1_000_000
    return price


def get_cost_per_million(model: str) -> Optional[dict[str, float]]:
    """
    Return {"input": <usd_per_1M>, "output": <usd_per_1M>} for a model id,
    plus "cache_read"/"cache_write" when the source knows them (may be None).

    Resolution order, for each candidate id:
        1. Local override in price.py (model_pricing) — wins if present, lets us
           patch wrong / missing prices without waiting for a LiteLLM release.
        2. litellm.model_cost — the canonical table shipped with LiteLLM.

    Candidates are tried most-specific first: the id as given, then with a
    release-date suffix stripped (``gpt-5-nano-2025-08-07`` → ``gpt-5-nano``),
    then the same two with a ``provider/`` prefix removed. Without this a
    dated id that LiteLLM hasn't catalogued yet silently priced at zero.

    Returns None if no candidate is known.
    """
    for candidate in _pricing_candidates(model):
        price = _lookup_pricing(candidate)
        if price:
            return price
    return None
