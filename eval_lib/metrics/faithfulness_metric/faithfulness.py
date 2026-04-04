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
            "Extract the key factual claims from the following answer.\n\n"
            "Rules:\n"
            "- Each claim must be a single, verifiable factual statement.\n"
            "- Ignore greetings, meta-comments (\"Sure!\", \"Here's...\"), and stylistic phrases.\n"
            "- Do NOT split one sentence into micro-facts. Keep claims at sentence-level granularity.\n"
            "- Combine closely related details into one claim rather than listing separately.\n"
            "- Maximum 8 claims. Focus on the most important facts.\n\n"
            f"Answer:\n{answer}\n\n"
            "Return a JSON array of strings."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        statements = json.loads(raw_json)
        assert isinstance(statements, list)
        return statements, cost or 0.0

    async def _generate_verdicts(self, context: str, statements: List[str]) -> Tuple[List[Dict[str, str]], float, float]:
        """Single-step verdict with improved prompt.

        Key improvements over naive approach:
        - Paraphrasing = "fully" (not "mostly")
        - "none" ONLY for contradictions or zero related info
        - Chain-of-thought: find relevant passage first, then judge
        """
        prompt = (
            "Evaluate how well each statement is supported by the context.\n\n"
            "IMPORTANT: First, find the relevant passage in the context. Then assign a verdict.\n\n"
            "Verdict levels:\n"
            "- fully: The core meaning is clearly present in the context (exact wording NOT required).\n"
            "- mostly: The main idea is supported but with minor differences in details "
            "(e.g., approximate numbers, paraphrased names).\n"
            "- partial: Some parts are supported but key information is missing or incomplete.\n"
            "- minor: Only tangentially related; the context mentions the topic but not the specific claim.\n"
            "- none: The claim directly contradicts the context, OR the context contains "
            "absolutely no related information.\n\n"
            "Key distinctions:\n"
            "- Paraphrasing or using synonyms = \"fully\" (not \"mostly\").\n"
            "- Missing exact numbers/dates but correct overall = \"mostly\" (not \"partial\" or \"none\").\n"
            "- Use \"none\" ONLY when the context contradicts the claim or has zero relevant information.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"STATEMENTS (JSON array):\n{json.dumps(statements, ensure_ascii=False)}\n\n"
            "For each statement, first quote the relevant context passage, then assign a verdict.\n"
            "Return only a JSON array of objects like:\n"
            '[{"verdict": "fully|mostly|partial|minor|none", '
            '"reason": "<brief explanation>", '
            '"support": "<exact context sentence(s) that support/contradict this claim> or \'none\'"}]'
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        verdicts: List[Dict[str, Any]] = json.loads(raw_json)

        # Safety check: high verdict but no support → downgrade
        for v in verdicts:
            supp = v.get("support", "").strip().lower()
            if supp in ("none", "") and v["verdict"] in ("fully", "mostly"):
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
