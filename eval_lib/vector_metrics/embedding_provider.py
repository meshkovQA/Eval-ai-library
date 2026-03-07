"""
Embedding provider abstraction for vector metrics.
Supports OpenAI API and local sentence-transformers models.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        """Returns (embeddings, cost_usd)."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name

    async def embed(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        from eval_lib.llm_client import get_embeddings
        embeddings, cost = await get_embeddings(self.model_name, texts)
        return embeddings, cost or 0.0


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install eval-ai-library[vectors] or pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)

    async def embed(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        import asyncio
        self._load_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts).tolist()
        )
        return embeddings, 0.0


def get_embedding_provider(provider: str, model_name: str = None) -> EmbeddingProvider:
    if provider == "openai":
        return OpenAIEmbeddingProvider(model_name or "text-embedding-3-small")
    elif provider == "local":
        return LocalEmbeddingProvider(model_name or "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'openai' or 'local'.")
