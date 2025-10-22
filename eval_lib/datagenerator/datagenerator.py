from typing import List
import re
import json
from eval_lib.llm_client import chat_complete
from .document_loader import load_documents, chunk_documents
import math
from eval_lib.llm_client import get_embeddings
import numpy as np
from .prompts import dataset_generation_prompt, dataset_generation_from_scratch_prompt
import asyncio
import random


async def retry_async(fn, *args, retries=4, base_delay=0.6, max_delay=6.0,
                      retriable_statuses=(429, 500, 502, 503, 504),
                      **kwargs):
    """
    fn — корутина, которая может бросить исключение вида:
    - HTTPException-like с .status_code
    - Exception с текстом, где встречается 'Service Unavailable' и т.п.
    """
    attempt = 0
    while True:
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            status = getattr(e, "status_code", None)
            msg = str(e).lower()

            retriable = (status in retriable_statuses) or any(
                s in msg for s in ["service unavailable", "temporarily unavailable",
                                   "gateway timeout", "bad gateway", "timeout"])
            if attempt > retries or not retriable:
                raise

            # экспоненциальный бэкофф + джиттер
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, 0.4)
            await asyncio.sleep(delay)


def _parse_json_response(raw_response: str) -> list[dict]:

    # Remove leading/trailing whitespace
    content = raw_response.strip()

    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        content = match.group(1).strip()

    # If no markdown blocks found, try to find JSON array directly
    if not match:
        # Look for JSON array pattern
        array_pattern = r'\[.*?\]'
        match = re.search(array_pattern, content, re.DOTALL)
        if match:
            content = match.group(0)

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Response is not a JSON array")
        return data
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse JSON response:\n{exc}\n\nRaw content:\n{content}")


class DatasetGenerator:

    def __init__(
        self,
        *,
        model: str,
        input_format: str,
        expected_output_format: str,
        agent_description: str,
        test_types: List[str],
        question_length: str = "mixed",
        question_openness: str = "mixed",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        temperature: float = 0.3,
        max_rows: int = 10,
        trap_density: float = 0.1,
        language: str = "en",
        max_chunks: int = 30,
        relevance_margin: float = 1.5,
        embedding_model: str = "openai:text-embedding-3-small",
    ):
        self.model = model
        self.input_format = input_format
        self.expected_output_format = expected_output_format
        self.agent_description = agent_description
        self.test_types = test_types
        self.question_length = question_length
        self.question_openness = question_openness
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        self.max_rows = max_rows
        self.trap_density = trap_density
        self.language = language
        self.max_chunks = max_chunks
        self.relevance_margin = relevance_margin
        self.embedding_model = embedding_model

    async def generate_from_scratch(self) -> List[dict]:
        prompt = dataset_generation_from_scratch_prompt(
            max_rows=self.max_rows,
            agent_description=self.agent_description,
            input_format=self.input_format,
            expected_output_format=self.expected_output_format,
            test_types=self.test_types,
            question_length=self.question_length,
            question_openness=self.question_openness,
            trap_density=self.trap_density,
            language=self.language
        )

        raw, _ = await chat_complete(
            llm=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        try:
            data = _parse_json_response(raw)
            assert isinstance(data, list), "not a JSON array"
            return data
        except Exception as exc:
            raise RuntimeError(f"Failed to parse dataset:\n{exc}\n\n{raw}")

    async def generate_from_documents(self, file_paths: List[str]) -> List[dict]:

        docs = load_documents(file_paths)
        doc_chunks = chunk_documents(docs,
                                     chunk_size=self.chunk_size,
                                     chunk_overlap=self.chunk_overlap)

        chunks_text = [d.page_content for d in doc_chunks]
        if not chunks_text:
            raise ValueError("No text extracted from documents.")

        ranked_chunks = await self._rank_chunks_by_relevance(chunks_text)

        total_chunks = len(ranked_chunks)
        rows_per_chunk = max(1, math.ceil(self.max_rows / total_chunks))

        needed_chunks = math.ceil(self.max_rows / rows_per_chunk)
        top_k = min(int(needed_chunks * self.relevance_margin),
                    self.max_chunks)
        selected_chunks = ranked_chunks[:top_k]

        dataset: list[dict] = []

        MAX_PROMPT_CHARS = 24_000

        for chunk in selected_chunks:

            safe_chunk = chunk if len(
                chunk) <= MAX_PROMPT_CHARS else chunk[:MAX_PROMPT_CHARS]

            prompt = dataset_generation_prompt(
                chunk=safe_chunk,
                rows_per_chunk=rows_per_chunk,
                agent_description=self.agent_description,
                input_format=self.input_format,
                expected_output_format=self.expected_output_format,
                test_types=self.test_types,
                question_length=self.question_length,
                question_openness=self.question_openness,
                trap_density=self.trap_density,
                language=self.language
            )

            raw, _ = await retry_async(
                chat_complete,
                llm=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            try:
                chunk_data = _parse_json_response(raw)
                assert isinstance(chunk_data, list)
                dataset.extend(chunk_data)
            except Exception as exc:
                raise RuntimeError(f"Chunk parsing error:\n{exc}\n\n{raw}")

            if len(dataset) >= self.max_rows:
                break

        return dataset[: self.max_rows]

    async def _rank_chunks_by_relevance(self, chunks: list[str]) -> list[str]:
        """
        Count token similarity between chunks and query.

        """
        # estimate tokens
        def approx_tokens(s: str) -> int:
            return max(1, len(s) // 4)

        # restrict length of each chunk for embedding (e.g., to ~8k tokens)
        MAX_EMBED_TOKENS_PER_INPUT = 8000
        MAX_EMBED_CHARS_PER_INPUT = MAX_EMBED_TOKENS_PER_INPUT * 4

        truncated_chunks = [
            c if len(
                c) <= MAX_EMBED_CHARS_PER_INPUT else c[:MAX_EMBED_CHARS_PER_INPUT]
            for c in chunks
        ]

        # limit tokens per request
        TOKEN_BUDGET_PER_REQUEST = 280_000

        # divide into batches by total tokens
        batches: list[list[str]] = []
        cur: list[str] = []
        cur_tokens = 0
        for c in truncated_chunks:
            t = approx_tokens(c)
            if cur and (cur_tokens + t) > TOKEN_BUDGET_PER_REQUEST:
                batches.append(cur)
                cur = [c]
                cur_tokens = t
            else:
                cur.append(c)
                cur_tokens += t
        if cur:
            batches.append(cur)

        # embedding for query
        query = self.agent_description + " " + " ".join(self.test_types)
        q_vec, _ = await retry_async(get_embeddings, model=self.embedding_model, texts=[query])
        q_vec = q_vec[0]

        # go through batches, accumulating embeddings
        all_vecs = []
        for batch in batches:
            vecs, _ = await retry_async(get_embeddings, model=self.embedding_model, texts=batch)
            all_vecs.extend(vecs)

        import numpy as np
        q_norm = np.linalg.norm(q_vec) + 1e-7
        sims = [
            float(np.dot(q_vec, v) / (q_norm * (np.linalg.norm(v) + 1e-7)))
            for v in all_vecs
        ]

        # sort
        ranked = [c for _, c in sorted(
            zip(sims, chunks), key=lambda x: x[0], reverse=True)]
        return ranked
