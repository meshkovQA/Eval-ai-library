# context_precision_metric.py
'''
Context Precision Metric: Measures the precision of retrieved context chunks
in relation to a reference answer.

Score calculation: Weighted average of precision@k across relevant chunks
'''
from typing import List, Dict, Tuple, Any
import json
import re
from math import exp
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import extract_json_block


class ContextualPrecisionMetric(MetricPattern):
    name = "contextPrecisionMetric"

    def __init__(self, model: str, threshold: float = 0.7, top_k: int | None = None, verbose: bool = False):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    async def _is_chunk_relevant(                       # judgement = 0 / 1
        self, reference: str, chunk: str
    ) -> Tuple[int, float]:
        prompt = (
            "Determine whether the following CONTEXT CHUNK contains information "
            "that also appears in the REFERENCE ANSWER (even if wording differs).\n\n"
            f"REFERENCE ANSWER:\n{reference}\n\n"
            f"CONTEXT CHUNK:\n{chunk}\n\n"
            "Reply ONLY with JSON: {\"relevant\": 1 | 0}"
        )
        text, cost = await chat_complete(
            self.model, [{"role": "user", "content": prompt}], temperature=0.0
        )
        try:
            rel = int(json.loads(extract_json_block(text))["relevant"])
            return rel, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Bad LLM relevance JSON: {e}\n{text}")

    # ------------------------------------------------------------------ #
    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        """Compute Context Precision@K as mean(precision@k * v_k) / (#relevant)."""
        reference = tc.actual_output   # fallback if no reference
        chunks: List[str] = (tc.retrieval_context or [])[
            : self.top_k] if self.top_k else tc.retrieval_context or []

        llm_cost: float = 0.0
        tp, fp = 0, 0
        precisions: List[float] = []
        indicators: List[int] = []
        verdicts: List[Dict[str, Any]] = []

        for rank, chunk in enumerate(chunks, 1):
            rel, cost = await self._is_chunk_relevant(reference, chunk)
            llm_cost += cost
            indicators.append(rel)

            tp += rel
            fp += 1 - rel
            prec_k = tp / max(1, tp + fp)
            precisions.append(prec_k)

            verdicts.append({"rank": rank, "relevant": bool(
                rel), "precision@k": round(prec_k, 4)})

        if sum(indicators):
            ctx_precision = round(
                sum(p * v for p, v in zip(precisions, indicators)) /
                sum(indicators),
                4,
            )
        else:
            ctx_precision = 0.0

        success = ctx_precision >= self.threshold

        evaluation_log = {
            # --- required fields --------------------------------------
            "input_question": tc.input,
            "retrieval_context": chunks,
            "llm_answer": reference,
            "verdicts": verdicts,
            # --- meta -------------------------------------------------
            "final_score": ctx_precision,
            "comment_final_score": "Context Precision@K.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether precision meets threshold."
        }

        result = {
            "name": self.name,
            "score": ctx_precision,
            "success": success,
            "reason": f"Average precision across top-{len(chunks)} context chunks.",
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log,
        }

        self.print_result(result)

        return result
