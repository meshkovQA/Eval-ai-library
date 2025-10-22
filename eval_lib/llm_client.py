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


@functools.cache
def _get_client(provider: Provider):
    if provider == Provider.OPENAI:
        return openai.AsyncOpenAI()

    if provider == Provider.AZURE:
        return AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

    if provider == Provider.GOOGLE:
        return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    if provider == Provider.OLLAMA:
        return openai.AsyncOpenAI(
            api_key=os.getenv("OLLAMA_API_KEY"),
            base_url=os.getenv("OLLAMA_API_BASE_URL")
        )

    if provider == Provider.ANTHROPIC:
        return anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    raise ValueError(f"Unsupported provider {provider}")


async def _openai_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """
    Обычный OpenAI.
    """
    response = await client.chat.completions.create(
        model=llm.model,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content.strip()
    cost = _calculate_cost(llm, response.usage)
    return text, cost


async def _azure_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):

    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or llm.model

    response = await client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content.strip()
    cost = _calculate_cost(llm, response.usage)
    return text, cost


async def _google_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """
    Google GenAI / Gemini 2.x 
    """
    prompt = "\n".join(m["content"] for m in messages)

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


async def _ollama_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    response = await client.chat.completions.create(
        model=llm.model,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content.strip()
    cost = _calculate_cost(llm, response.usage)
    return text, cost


async def _anthropic_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """
    Anthropic Claude chat completion.
    """
    response = await client.messages.create(
        model=llm.model,
        messages=messages,
        temperature=temperature,
        max_tokens=4096,  # Default max tokens for Claude
    )
    if isinstance(response.content, list):
        text = "".join(
            block.text for block in response.content if block.type == "text").strip()
    else:
        text = response.content.strip()

    cost = _calculate_cost(llm, response.usage)
    return text, cost


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
    llm = LLMDescriptor.parse(llm)
    helper = _HELPERS.get(llm.provider)

    if helper is None:
        raise ValueError(f"Unsupported provider {llm.provider}")

    client = _get_client(llm.provider)
    return await helper(client, llm, messages, temperature)


def _calculate_cost(llm: LLMDescriptor, usage) -> Optional[float]:
    """
    Calculate the cost of the LLM usage based on the model and usage data.
    """
    if llm.provider == Provider.OLLAMA:
        return 0.0
    if not usage:
        return None

    price = model_pricing.get(llm.model)
    if not price:
        return None

    prompt = getattr(usage, "prompt_tokens",     0)
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
    response = await client.embeddings.create(
        model=llm.model,
        input=texts,
        encoding_format="float"
    )

    embeddings = [data.embedding for data in response.data]
    cost = _calculate_embedding_cost(llm, response.usage)

    return embeddings, cost


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
