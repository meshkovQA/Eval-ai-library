# faithfulness_metric.py
'''
Faithfulness Metric: Evaluates the factual consistency of a chatbot's answer
with respect to the retrieved context.
Score calculation: Softmax aggregation of verdicts on factual statements
'''
from typing import List, Dict, Tuple, Any
import json
import re
import numpy as np
from math import exp
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block

VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}


class FaithfulnessMetric(MetricPattern):
    name = "faithfulnessMetric"

    def __init__(
            self,
            model: str,
            threshold: float = 0.7,
            temperature: float = 0.5,
            verbose: bool = False
    ):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature

    async def _generate_statements(self, answer: str) -> Tuple[List[str], float]:
        prompt = (
            "Extract standalone factual claims from the following answer.\n"
            "Each statement must be a distinct, verifiable fact.\n\n"
            f"Answer:\n{answer}\n\n"
            "Return a JSON array of strings."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        statements = json.loads(raw_json)
        assert isinstance(statements, list)
        return statements, cost or 0.0

    async def _generate_verdicts(self, context: str, statements: List[str]) -> Tuple[List[Dict[str, str]], float, float]:
        prompt = (
            "Evaluate how well each statement is supported by the context.\n\n"
            "Levels:\n"
            "- fully: directly supported word-for-word\n"
            "- mostly: strongly supported but wording differs slightly\n"
            "- partial: partially supported but with some gaps\n"
            "- minor: tangentially related or ambiguous\n"
            "- none: clearly unsupported or contradicted\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"STATEMENTS (JSON array):\n{json.dumps(statements, ensure_ascii=False)}\n\n"
            "Return only a JSON array of objects like:\n"
            '[{"verdict": "fully|mostly|partial|minor|none", '
            '"reason": "<brief>", '
            '"support": "<exact context sentence(s)> or \'none\'"}]'
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        verdicts: List[Dict[str, Any]] = json.loads(raw_json)

        for v in verdicts:
            supp = v.get("support", "").strip().lower()
            if supp == "none" and v["verdict"] in ("fully", "mostly"):
                v["verdict"] = "partial"

        scores = [VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in verdicts]
        score = round(score_agg(scores, temperature=self.temperature), 4)
        return verdicts, score, cost or 0.0

    async def _summarize_reasons_via_llm(self, verdicts: List[Dict[str, str]]) -> Tuple[str, float]:
        grouped: Dict[str, List[str]] = {}
        for v in verdicts:
            grouped.setdefault(v["verdict"], []).append(v["reason"])
        bullets = []
        for tag in ("fully", "mostly", "partial", "none"):
            bullets.extend(f"- {r}" for r in grouped.get(tag, [])[:2])
        prompt = (
            "Summarize the following points from a factual consistency evaluation.\n"
            "Give one short paragraph (1-2 sentences) that explains whether the answer "
            "was well supported by the context, mentioning both strong and weak parts.\n\n"
            f"{chr(10).join(bullets)}\n\n"
            "Summary:"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return text.strip(), cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, any]:
        llm_cost = 0.0
        answer = test_case.actual_output
        context = "\n".join(test_case.retrieval_context or [])
        question = test_case.input

        # 1. Statements from answer
        statements, cost = await self._generate_statements(answer)
        llm_cost += cost

        # 2. Verdicts against context
        verdicts, verdict_score, cost = await self._generate_verdicts(context, statements)
        llm_cost += cost

        # 3. Reason summary
        summary_reason, cost = await self._summarize_reasons_via_llm(verdicts)
        llm_cost += cost

        success = verdict_score >= self.threshold

        evaluation_log = {
            "input_question": question,
            "retrieval_context": test_case.retrieval_context,
            "answer": answer,
            "statements": statements,
            "comment_statements": "Factual assertions extracted from the answer.",
            "verdicts": verdicts,
            "comment_verdicts": "Each verdict shows how well a statement is supported by the context.",
            "final_score": verdict_score,
            "comment_final_score": "Final score based on faithfulness of statements.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the score meets the required threshold.",
            "final_reason": summary_reason,
            "comment_reasoning": "Summary explanation based on all verdicts."
        }

        result = {
            "name": self.name,
            "score": verdict_score,
            "success": success,
            "reason": summary_reason,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log
        }
        self.print_result(result)

        return result
