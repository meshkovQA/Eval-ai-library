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

# Map our provider id (the value of Provider enum) to the prefix used inside
# LiteLLM's models_by_provider / model_cost tables.
#
# Keep in sync with llm_client._PROVIDER_TO_LITELLM_PREFIX. Duplicated here
# instead of imported to avoid a circular import (llm_client also reads from
# this module via _calculate_cost).
_PROVIDER_TO_LITELLM: dict[str, str] = {
    "openai": "openai",
    "azure": "azure",
    "google": "gemini",
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "qwen": "dashscope",
    "mistral": "mistral",
    "groq": "groq",
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
    Ask LiteLLM which env vars are required to authenticate against this
    provider, by validating a representative chat model. Returns a list of
    env var names — possibly empty if LiteLLM can't determine them.
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
    try:
        result = litellm.validate_environment(model=sample)
        missing = result.get("missing_keys") or []
        # validate_environment lists vars *currently missing in env*; if some
        # vars are already set we lose them. As a heuristic, also include any
        # *_API_KEY-shaped vars from os.environ that match the provider name.
        return list(missing)
    except Exception:
        return []


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


def get_provider_env_vars(provider: str) -> list[str]:
    """
    Best-effort detection of env vars needed for a provider. For known providers
    we have hand-curated answers; for the rest we ask LiteLLM.
    """
    known: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google": ["GOOGLE_API_KEY"],
        "gemini": ["GOOGLE_API_KEY"],
        "azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"],
        "azure_ai": ["AZURE_AI_API_KEY", "AZURE_AI_API_BASE"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "qwen": ["DASHSCOPE_API_KEY"],
        "dashscope": ["DASHSCOPE_API_KEY"],
        "zhipu": ["ZHIPU_API_KEY"],
        "mistral": ["MISTRAL_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "grok": ["XAI_API_KEY"],
        "xai": ["XAI_API_KEY"],
        "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
        "vertex_ai": ["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"],
        "cohere": ["COHERE_API_KEY"],
        "cohere_chat": ["COHERE_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "fireworks_ai": ["FIREWORKS_AI_API_KEY"],
        "together_ai": ["TOGETHERAI_API_KEY"],
        "perplexity": ["PERPLEXITYAI_API_KEY"],
        "deepinfra": ["DEEPINFRA_API_KEY"],
        "cerebras": ["CEREBRAS_API_KEY"],
        "cloudflare": ["CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID"],
        "watsonx": ["WATSONX_API_KEY", "WATSONX_PROJECT_ID"],
        "databricks": ["DATABRICKS_API_KEY", "DATABRICKS_API_BASE"],
        "ollama": ["OLLAMA_API_BASE_URL"],
        "mlx": ["MLX_API_BASE_URL"],
        "custom": ["CUSTOM_LLM_API_KEY", "CUSTOM_LLM_BASE_URL"],
    }
    if provider in known:
        return known[provider]
    # Auto-detect via LiteLLM
    detected = _detect_env_vars(provider)
    if detected:
        return detected
    # Last-resort guess: <UPPER>_API_KEY
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


def get_cost_per_million(model: str) -> Optional[dict[str, float]]:
    """
    Return {"input": <usd_per_1M>, "output": <usd_per_1M>} for a model id.

    Resolution order:
        1. Local override in price.py (model_pricing) — wins if present, lets us
           patch wrong / missing prices without waiting for a LiteLLM release.
        2. litellm.model_cost — the canonical table shipped with LiteLLM.

    Returns None if neither source knows the model.
    """
    override = _override_pricing.get(model)
    if override:
        return {"input": override["input"], "output": override["output"]}

    info = litellm.model_cost.get(model)
    if not info:
        return None

    in_per_token = info.get("input_cost_per_token")
    out_per_token = info.get("output_cost_per_token")
    if in_per_token is None and out_per_token is None:
        return None

    return {
        "input": (in_per_token or 0.0) * 1_000_000,
        "output": (out_per_token or 0.0) * 1_000_000,
    }
