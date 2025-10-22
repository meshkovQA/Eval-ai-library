# llm_client.py
import openai
import functools
import anthropic
from openai import AsyncAzureOpenAI
from google import genai
from google.genai.types import GenerateContentConfig
import os
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from types import SimpleNamespace
from .price import model_pricing


class LLMConfigurationError(Exception):
    """Raised when LLM client configuration is missing or invalid."""
    pass


class Provider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    GOOGLE = "google"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True, slots=True)
class LLMDescriptor:
    """'openai:gpt-4o'  →  provider=openai, model='gpt-4o'"""
    provider: Provider
    model: str

    @classmethod
    def parse(cls, spec: str | Tuple[str, str] | "LLMDescriptor") -> "LLMDescriptor":
        if isinstance(spec, LLMDescriptor):
            return spec
        if isinstance(spec, tuple):
            provider, model = spec
            return cls(Provider(provider), model)
        try:
            provider, model = spec.split(":", 1)
        except ValueError:
            return cls(Provider.OPENAI, spec)
        return cls(Provider(provider), model)

    def key(self) -> str:
        """Return a unique key for the LLM descriptor."""
        return f"{self.provider}:{self.model}"


def _check_env_var(var_name: str, provider: str, required: bool = True) -> Optional[str]:
    """
    Check if environment variable is set and return its value.

    Args:
        var_name: Name of the environment variable
        provider: Provider name for error message
        required: Whether this variable is required

    Returns:
        Value of the environment variable or None if not required

    Raises:
        LLMConfigurationError: If required variable is missing
    """
    value = os.getenv(var_name)
    if required and not value:
        raise LLMConfigurationError(
            f"❌ Missing {provider} configuration!\n\n"
            f"Environment variable '{var_name}' is not set.\n\n"
            f"To fix this, set the environment variable:\n"
            f"  export {var_name}='your-api-key-here'\n\n"
            f"Or add it to your .env file:\n"
            f"  {var_name}=your-api-key-here\n\n"
            f"📖 Documentation: https://github.com/meshkovQA/Eval-ai-library#environment-variables"
        )
    return value


@functools.cache
def _get_client(provider: Provider):
    """
    Get or create LLM client for the specified provider.

    Args:
        provider: LLM provider enum

    Returns:
        Configured client instance

    Raises:
        LLMConfigurationError: If required configuration is missing
        ValueError: If provider is not supported
    """
    if provider == Provider.OPENAI:
        _check_env_var("OPENAI_API_KEY", "OpenAI")
        return openai.AsyncOpenAI()

    if provider == Provider.AZURE:
        _check_env_var("AZURE_OPENAI_API_KEY", "Azure OpenAI")
        _check_env_var("AZURE_OPENAI_ENDPOINT", "Azure OpenAI")
        # AZURE_OPENAI_DEPLOYMENT проверяется при вызове, не обязателен здесь

        return AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

    if provider == Provider.GOOGLE:
        _check_env_var("GOOGLE_API_KEY", "Google Gemini")
        return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    if provider == Provider.OLLAMA:
        # Ollama может работать без ключа (локальный сервер)
        api_key = _check_env_var(
            "OLLAMA_API_KEY", "Ollama", required=False) or "ollama"
        base_url = _check_env_var(
            "OLLAMA_API_BASE_URL", "Ollama", required=False) or "http://localhost:11434/v1"

        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    if provider == Provider.ANTHROPIC:
        _check_env_var("ANTHROPIC_API_KEY", "Anthropic Claude")
        return anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    raise ValueError(f"Unsupported provider: {provider}")


async def _openai_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """OpenAI chat completion."""
    try:
        response = await client.chat.completions.create(
            model=llm.model,
            messages=messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip()
        cost = _calculate_cost(llm, response.usage)
        return text, cost
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ OpenAI API authentication failed!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check that your OPENAI_API_KEY is valid.\n"
                f"Get your API key at: https://platform.openai.com/api-keys"
            )
        raise


async def _azure_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """Azure OpenAI chat completion."""
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or llm.model

    if not deployment_name:
        raise LLMConfigurationError(
            f"❌ Missing Azure OpenAI deployment name!\n\n"
            f"Please set AZURE_OPENAI_DEPLOYMENT environment variable.\n"
            f"Example: export AZURE_OPENAI_DEPLOYMENT='gpt-4o'"
        )

    try:
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip()
        cost = _calculate_cost(llm, response.usage)
        return text, cost
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ Azure OpenAI authentication failed!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check your Azure OpenAI configuration:\n"
                f"  - AZURE_OPENAI_API_KEY\n"
                f"  - AZURE_OPENAI_ENDPOINT\n"
                f"  - AZURE_OPENAI_DEPLOYMENT"
            )
        raise


async def _google_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """Google GenAI / Gemini chat completion."""
    prompt = "\n".join(m["content"] for m in messages)

    try:
        response = await client.aio.models.generate_content(
            model=llm.model,
            contents=prompt,
            config=GenerateContentConfig(temperature=temperature),
        )

        text = response.text.strip()

        um = response.usage_metadata
        usage = SimpleNamespace(
            prompt_tokens=um.prompt_token_count,
            completion_tokens=um.candidates_token_count,
        )

        cost = _calculate_cost(llm, usage)
        return text, cost
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower() or "credentials" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ Google Gemini API authentication failed!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check that your GOOGLE_API_KEY is valid.\n"
                f"Get your API key at: https://aistudio.google.com/apikey"
            )
        raise


async def _ollama_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """Ollama (local) chat completion."""
    try:
        response = await client.chat.completions.create(
            model=llm.model,
            messages=messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip()
        cost = _calculate_cost(llm, response.usage)
        return text, cost
    except Exception as e:
        if "Connection" in str(e) or "refused" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ Cannot connect to Ollama server!\n\n"
                f"Error: {str(e)}\n\n"
                f"Make sure Ollama is running:\n"
                f"  1. Install Ollama: https://ollama.ai/download\n"
                f"  2. Start Ollama: ollama serve\n"
                f"  3. Pull model: ollama pull {llm.model}\n\n"
                f"Or set OLLAMA_API_BASE_URL to your Ollama server:\n"
                f"  export OLLAMA_API_BASE_URL='http://localhost:11434/v1'"
            )
        raise


async def _anthropic_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """Anthropic Claude chat completion."""
    try:
        response = await client.messages.create(
            model=llm.model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        if isinstance(response.content, list):
            text = "".join(
                block.text for block in response.content if block.type == "text").strip()
        else:
            text = response.content.strip()

        cost = _calculate_cost(llm, response.usage)
        return text, cost
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ Anthropic Claude API authentication failed!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check that your ANTHROPIC_API_KEY is valid.\n"
                f"Get your API key at: https://console.anthropic.com/settings/keys"
            )
        raise


_HELPERS = {
    Provider.OPENAI: _openai_chat_complete,
    Provider.AZURE: _azure_chat_complete,
    Provider.GOOGLE: _google_chat_complete,
    Provider.OLLAMA: _ollama_chat_complete,
    Provider.ANTHROPIC: _anthropic_chat_complete,
}


async def chat_complete(
    llm: str | tuple[str, str] | LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
):
    """
    Complete a chat conversation using the specified LLM.

    Args:
        llm: LLM specification (e.g., "gpt-4o-mini", "openai:gpt-4o", or LLMDescriptor)
        messages: List of message dicts with "role" and "content"
        temperature: Sampling temperature (0.0-2.0)

    Returns:
        Tuple of (response_text, cost_in_usd)

    Raises:
        LLMConfigurationError: If required API keys or configuration are missing
        ValueError: If provider is not supported
    """
    llm = LLMDescriptor.parse(llm)
    helper = _HELPERS.get(llm.provider)

    if helper is None:
        raise ValueError(f"Unsupported provider: {llm.provider}")

    client = _get_client(llm.provider)
    return await helper(client, llm, messages, temperature)


def _calculate_cost(llm: LLMDescriptor, usage) -> Optional[float]:
    """Calculate the cost of the LLM usage based on the model and usage data."""
    if llm.provider == Provider.OLLAMA:
        return 0.0
    if not usage:
        return None

    price = model_pricing.get(llm.model)
    if not price:
        return None

    prompt = getattr(usage, "prompt_tokens", 0)
    completion = getattr(usage, "completion_tokens", 0)

    return round(
        prompt * price["input"] / 1_000_000 +
        completion * price["output"] / 1_000_000,
        6
    )


async def get_embeddings(
    model: str | tuple[str, str] | LLMDescriptor,
    texts: list[str],
) -> tuple[list[list[float]], Optional[float]]:
    """
    Get embeddings for a list of texts using OpenAI models.

    Args:
        model: Model specification (e.g., "openai:text-embedding-3-small")
        texts: List of texts to embed

    Returns:
        Tuple of (embeddings_list, total_cost)

    Raises:
        LLMConfigurationError: If required API keys are missing
        ValueError: If non-OpenAI provider is specified
    """
    llm = LLMDescriptor.parse(model)

    if llm.provider != Provider.OPENAI:
        raise ValueError(
            f"Only OpenAI embedding models are supported, got {llm.provider}")

    client = _get_client(llm.provider)
    return await _openai_get_embeddings(client, llm, texts)


async def _openai_get_embeddings(
    client,
    llm: LLMDescriptor,
    texts: list[str],
) -> tuple[list[list[float]], Optional[float]]:
    """OpenAI embeddings implementation."""
    try:
        response = await client.embeddings.create(
            model=llm.model,
            input=texts,
            encoding_format="float"
        )

        embeddings = [data.embedding for data in response.data]
        cost = _calculate_embedding_cost(llm, response.usage)

        return embeddings, cost
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ OpenAI API authentication failed for embeddings!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check that your OPENAI_API_KEY is valid.\n"
                f"Get your API key at: https://platform.openai.com/api-keys"
            )
        raise


def _calculate_embedding_cost(llm: LLMDescriptor, usage) -> Optional[float]:
    """Calculate the cost of embedding usage for OpenAI models."""
    if not usage:
        return None

    price = model_pricing.get(llm.model)
    if not price:
        return None

    total_tokens = getattr(usage, 'total_tokens', 0)
    input_price = price.get("input", 0)

    return round(total_tokens * input_price / 1_000_000, 6)
