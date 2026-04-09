"""
Manual price override table.

Since 0.6.0 model pricing is sourced from LiteLLM's built-in `litellm.model_cost`
table (~2600 entries, refreshed on every LiteLLM release). This file is kept as
an *optional* override layer: anything you put here wins over LiteLLM, which is
useful when:

    - LiteLLM doesn't ship the model (private, regional, brand-new release).
    - You have a private/custom model name that LiteLLM doesn't know about.
    - You want to override prices for billing/contract reasons.

Format is unchanged from earlier versions:

    model_pricing = {
        "my-model": {"input": 1.50, "output": 6.00},  # USD per 1M tokens
    }

Lookup goes through `eval_lib.model_catalog.get_cost_per_million()` which checks
this table first, then falls back to LiteLLM. If a model is in BOTH places, the
override here wins — so when LiteLLM catches up on a model with accurate pricing,
remove the entry from this file to delegate to the canonical source.
"""
from typing import Dict

# Models LiteLLM doesn't ship at the time of writing. Each entry is "USD per
# 1M tokens" for input and output. When LiteLLM adds one of these, drop it from
# this dict to stop overriding.
model_pricing: Dict[str, Dict[str, float]] = {
    # OpenAI — preview SKUs not in LiteLLM
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    # Google Gemini — preview SKUs
    "gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash-preview": {"input": 0.15, "output": 0.60},
    # Anthropic — date-suffixed model ids LiteLLM doesn't list
    "claude-opus-4-6-20250619": {"input": 15.00, "output": 75.00},
    "claude-opus-4-5-20250415": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6-20250619": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5-20250415": {"input": 3.00, "output": 15.00},
    # DeepSeek — newer SKUs
    "deepseek-v3.2": {"input": 0.28, "output": 0.42},
    "deepseek-v3.2-exp": {"input": 0.28, "output": 0.42},
    "deepseek-v3.1": {"input": 0.28, "output": 0.42},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
    "deepseek-r1": {"input": 0.55, "output": 2.19},
    "deepseek-r1-lite": {"input": 0.14, "output": 0.55},
    # Qwen (Alibaba) — USD approx from CNY
    "qwen-max": {"input": 2.78, "output": 11.11},
    "qwen-plus": {"input": 0.56, "output": 1.67},
    "qwen-turbo": {"input": 0.14, "output": 0.28},
    "qwen-long": {"input": 0.07, "output": 0.28},
    "qwq-plus": {"input": 0.56, "output": 1.67},
    # Zhipu GLM — USD approx from CNY (no LiteLLM coverage)
    "glm-4-plus": {"input": 6.94, "output": 6.94},
    "glm-4-air": {"input": 0.14, "output": 0.14},
    "glm-4-airx": {"input": 1.39, "output": 1.39},
    "glm-4-long": {"input": 0.14, "output": 0.14},
    "glm-4-flash": {"input": 0.00, "output": 0.00},
    "glm-4-flashx": {"input": 0.03, "output": 0.03},
    # Mistral — *-latest aliases
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-small-latest": {"input": 0.10, "output": 0.30},
    "codestral-latest": {"input": 0.30, "output": 0.90},
    "pixtral-large-latest": {"input": 2.00, "output": 6.00},
    "ministral-8b-latest": {"input": 0.10, "output": 0.10},
    "mistral-embed": {"input": 0.10, "output": 0.0},
    # Groq (inference hosting)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.2-90b-vision-preview": {"input": 0.90, "output": 0.90},
    "llama-3.2-11b-vision-preview": {"input": 0.18, "output": 0.18},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    "deepseek-r1-distill-llama-70b": {"input": 0.59, "output": 0.79},
    # Grok (xAI) — most recent SKUs
    "grok-4.1": {"input": 3.00, "output": 15.00},
    "grok-4": {"input": 3.00, "output": 15.00},
    "grok-4-heavy": {"input": 15.00, "output": 75.00},
    "grok-4-fast": {"input": 2.00, "output": 10.00},
    "grok-beta": {"input": 5.00, "output": 15.00},
    "grok-3": {"input": 3.00, "output": 15.00},
    "grok-2": {"input": 2.00, "output": 10.00},
}
