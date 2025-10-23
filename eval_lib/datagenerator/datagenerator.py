from typing import List
from eval_lib.llm_client import chat_complete
from .document_loader import load_documents, chunk_documents
import math
from eval_lib.llm_client import get_embeddings
import numpy as np
from .prompts import dataset_generation_prompt, dataset_generation_from_scratch_prompt
from eval_lib.utils import extract_json_block
import asyncio
import random
import json
import time

# Colors for beautiful console output


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


async def retry_async(fn, *args, retries=4, base_delay=0.6, max_delay=6.0,
                      retriable_statuses=(429, 500, 502, 503, 504),
                      **kwargs):
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

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, 0.4)
            await asyncio.sleep(delay)


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
        verbose: bool = False,
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
        self.verbose = verbose

    def _log(self, message: str, color: str = Colors.CYAN):
        """Log message with color if verbose mode is enabled"""
        if self.verbose:
            print(f"{color}{message}{Colors.ENDC}")

    def _log_step(self, step_name: str, step_num: int = None):
        """Log generation step"""
        if self.verbose:
            prefix = f"[{step_num}] " if step_num else ""
            print(f"{Colors.DIM}  {prefix}{step_name}...{Colors.ENDC}")

    def _log_progress(self, current: int, total: int, label: str = "Progress"):
        """Log progress bar"""
        if self.verbose:
            percentage = (current / total) * 100 if total > 0 else 0
            bar_length = 30
            filled = int(bar_length * current / total) if total > 0 else 0
            bar = '█' * filled + '░' * (bar_length - filled)
            print(
                f"{Colors.CYAN}  {label}: [{bar}] {current}/{total} ({percentage:.0f}%){Colors.ENDC}")

    def _print_header(self, title: str):
        """Print beautiful header"""
        if self.verbose:
            import shutil
            terminal_width = shutil.get_terminal_size().columns
            WIDTH = terminal_width // 2
            WIDTH = max(WIDTH, 60)

            border = "═" * WIDTH
            title_text = f"🎯 {title}"
            padding = WIDTH - len(title_text)
            left_pad = padding // 2
            right_pad = padding - left_pad
            centered_title = " " * left_pad + title_text + " " * right_pad

            print(f"\n{Colors.BOLD}{Colors.CYAN}╔{border}╗{Colors.ENDC}")
            print(
                f"{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}{centered_title}{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.CYAN}╚{border}╝{Colors.ENDC}\n")

    def _print_summary(self, dataset: List[dict], elapsed_time: float, total_cost: float = 0.0):
        """Print generation summary with full dataset in readable format"""
        if not self.verbose:
            return

        import shutil
        import textwrap
        terminal_width = shutil.get_terminal_size().columns
        WIDTH = terminal_width - 10
        WIDTH = max(WIDTH, 80)

        print(
            f"\n{Colors.BOLD}{Colors.GREEN}✅ Dataset Generation Complete{Colors.ENDC}\n")
        print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
        print(
            f"  📊 Total rows generated: {Colors.YELLOW}{len(dataset)}{Colors.ENDC}")
        print(
            f"  ⏱️  Time elapsed: {Colors.YELLOW}{elapsed_time:.2f}s{Colors.ENDC}")
        if total_cost > 0:
            print(
                f"  💰 Total cost: {Colors.BLUE}${total_cost:.6f}{Colors.ENDC}")

        # Show full dataset
        if dataset:
            print(f"\n{Colors.BOLD}Generated Dataset:{Colors.ENDC}\n")

            for idx, row in enumerate(dataset, 1):
                # Header
                print(f"{Colors.CYAN}{'─' * WIDTH}{Colors.ENDC}")
                print(
                    f"{Colors.CYAN}{Colors.BOLD}Row {idx}/{len(dataset)}:{Colors.ENDC}")
                print(f"{Colors.CYAN}{'─' * WIDTH}{Colors.ENDC}")

                # Fields
                for key, value in row.items():
                    value_str = str(value)

                    # Key with proper formatting
                    print(f"{Colors.BOLD}{key}:{Colors.ENDC}", end=" ")

                    # Wrap long text to fit terminal width
                    # Calculate available width (WIDTH - key length - 2 for ": ")
                    available_width = WIDTH - len(key) - 2

                    if len(value_str) <= available_width:
                        # Short value - print on same line
                        print(value_str)
                    else:
                        # Long value - wrap to multiple lines with proper indentation
                        print()  # New line after key
                        wrapped = textwrap.fill(
                            value_str,
                            width=WIDTH - 2,
                            initial_indent="  ",
                            subsequent_indent="  ",
                            break_long_words=False,
                            break_on_hyphens=False
                        )
                        print(f"{Colors.DIM}{wrapped}{Colors.ENDC}")

                print()  # Spacing after row

            print(f"{Colors.CYAN}{'─' * WIDTH}{Colors.ENDC}\n")

            # Add spacing between rows
            if idx < len(dataset):
                print()

    async def generate_from_scratch(self) -> List[dict]:

        start_time = time.time()

        if self.verbose:
            self._print_header("Dataset Generation from Scratch")
            self._log(f"Configuration:", Colors.BOLD)
            self._log(f"  Model: {self.model}")
            self._log(f"  Max rows: {self.max_rows}")
            self._log(f"  Test types: {', '.join(self.test_types)}")
            self._log(f"  Language: {self.language}")
            self._log("")

        self._log_step("Generating prompt", 1)

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

        self._log_step("Calling LLM to generate dataset", 2)

        raw, cost = await chat_complete(
            llm=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        self._log_step("Parsing response", 3)

        try:
            raw_json = extract_json_block(raw)
            data = json.loads(raw_json)
            assert isinstance(data, list), "not a JSON array"
            elapsed_time = time.time() - start_time
            self._print_summary(data, elapsed_time, cost or 0.0)

            return data
        except Exception as exc:
            if self.verbose:
                self._log(f"❌ Failed to parse dataset", Colors.RED)
            raise RuntimeError(f"Failed to parse dataset:\n{exc}\n\n{raw}")

    async def generate_from_documents(self, file_paths: List[str]) -> List[dict]:
        """Generate dataset from documents"""
        start_time = time.time()
        total_cost = 0.0

        if self.verbose:
            self._print_header("Dataset Generation from Documents")
            self._log(f"Configuration:", Colors.BOLD)
            self._log(f"  Model: {self.model}")
            self._log(f"  Max rows: {self.max_rows}")
            self._log(f"  Documents: {len(file_paths)}")
            self._log(f"  Chunk size: {self.chunk_size}")
            self._log(f"  Test types: {', '.join(self.test_types)}")
            self._log("")

        self._log_step("Loading documents", 1)
        docs = load_documents(file_paths)

        if self.verbose:
            self._log(
                f"  ✅ Loaded {len(file_paths)} file(s) → {len(docs)} page(s)/document(s)", Colors.GREEN)

        self._log_step("Chunking documents", 2)
        doc_chunks = chunk_documents(docs,
                                     chunk_size=self.chunk_size,
                                     chunk_overlap=self.chunk_overlap)

        chunks_text = [d.page_content for d in doc_chunks]
        if not chunks_text:
            raise ValueError("No text extracted from documents.")

        if self.verbose:
            self._log(f"  ✅ Created {len(chunks_text)} chunks", Colors.GREEN)

        self._log_step("Ranking chunks by relevance", 3)
        ranked_chunks = await self._rank_chunks_by_relevance(chunks_text)

        if self.verbose:
            self._log(f"  ✅ Ranked {len(ranked_chunks)} chunks", Colors.GREEN)

        total_chunks = len(ranked_chunks)
        rows_per_chunk = max(1, math.ceil(self.max_rows / total_chunks))

        needed_chunks = math.ceil(self.max_rows / rows_per_chunk)
        top_k = min(int(needed_chunks * self.relevance_margin),
                    self.max_chunks)
        selected_chunks = ranked_chunks[:top_k]

        if self.verbose:
            self._log(
                f"  📌 Selected top {len(selected_chunks)} chunks for generation", Colors.YELLOW)
            self._log("")

        dataset: list[dict] = []

        MAX_PROMPT_CHARS = 24_000

        self._log_step(f"Generating dataset from chunks", 4)

        for i, chunk in enumerate(selected_chunks):
            if self.verbose:
                self._log_progress(
                    i + 1, len(selected_chunks), "Processing chunks")

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

            raw, cost = await retry_async(
                chat_complete,
                llm=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            total_cost += cost or 0.0

            try:
                chunk_data = json.loads(extract_json_block(raw))
                assert isinstance(chunk_data, list)
                dataset.extend(chunk_data)

                if self.verbose:
                    self._log(
                        f"    ✅ Generated {len(chunk_data)} rows from chunk {i+1}", Colors.GREEN)

            except Exception as exc:
                if self.verbose:
                    self._log(
                        f"    ⚠️  Chunk {i+1} parsing failed, skipping", Colors.YELLOW)
                continue

            if len(dataset) >= self.max_rows:
                break

        final_dataset = dataset[: self.max_rows]
        elapsed_time = time.time() - start_time

        self._print_summary(final_dataset, elapsed_time, total_cost)

        return final_dataset

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
