from typing import Dict

# Model pricing (USD per 1M tokens)
model_pricing: Dict[str, Dict[str, float]] = {
    # Embeddings
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    # OpenAI - GPT-5
    "gpt-5": {"input": 10.00, "output": 30.00},
    "gpt-5-mini": {"input": 1.50, "output": 6.00},
    "gpt-5-nano": {"input": 0.30, "output": 1.20},
    # OpenAI - GPT-4.5
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    # OpenAI - GPT-4.1
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # OpenAI - GPT-4o
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # OpenAI - o-series
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Google Gemini
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-preview": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    # Anthropic Claude
    "claude-opus-4-6-20250619": {"input": 15.00, "output": 75.00},
    "claude-opus-4-5-20250415": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6-20250619": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5-20250415": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    # DeepSeek
    "deepseek-chat": {"input": 0.28, "output": 0.42},
    "deepseek-v3.2": {"input": 0.28, "output": 0.42},
    "deepseek-v3.2-exp": {"input": 0.28, "output": 0.42},
    "deepseek-v3.1": {"input": 0.28, "output": 0.42},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "deepseek-r1": {"input": 0.55, "output": 2.19},
    "deepseek-r1-lite": {"input": 0.14, "output": 0.55},
    # Qwen (Alibaba) — USD approx from CNY
    "qwen-max": {"input": 2.78, "output": 11.11},
    "qwen-plus": {"input": 0.56, "output": 1.67},
    "qwen-turbo": {"input": 0.14, "output": 0.28},
    "qwen-long": {"input": 0.07, "output": 0.28},
    "qwq-plus": {"input": 0.56, "output": 1.67},
    # Zhipu GLM — USD approx from CNY
    "glm-4-plus": {"input": 6.94, "output": 6.94},
    "glm-4-air": {"input": 0.14, "output": 0.14},
    "glm-4-airx": {"input": 1.39, "output": 1.39},
    "glm-4-long": {"input": 0.14, "output": 0.14},
    "glm-4-flash": {"input": 0.00, "output": 0.00},
    "glm-4-flashx": {"input": 0.03, "output": 0.03},
    # Mistral
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
    # Grok (xAI)
    "grok-4.1": {"input": 3.00, "output": 15.00},
    "grok-4": {"input": 3.00, "output": 15.00},
    "grok-4-heavy": {"input": 15.00, "output": 75.00},
    "grok-4-fast": {"input": 2.00, "output": 10.00},
    "grok-beta": {"input": 5.00, "output": 15.00},
    "grok-3": {"input": 3.00, "output": 15.00},
    "grok-2": {"input": 2.00, "output": 10.00},
}
