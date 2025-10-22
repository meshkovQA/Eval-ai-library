# contextual_recall.py
from typing import List, Dict, Tuple, Any
import json
import re
from math import exp
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete


def extract_json_block(text: str) -> str:
    """
    Extracts the first JSON block from Markdown-like fenced code blocks.
    """
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    json_match = re.search(r"({.*?})", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    return text.strip()


class ContextualRecallMetric(MetricPattern):
    name = "contextualRecallMetric"
    template_cls = None

    def __init__(self, model: str, threshold: float = 0.7):
        super().__init__(model=model, threshold=threshold)

    async def _extract_claims(self, reference: str) -> Tuple[List[str], float]:
        prompt = (
            "Extract standalone factual claims from the following reference answer. "
            "Each statement must be atomic, verifiable, and distinct.\n\n"
            f"Reference:\n{reference}\n\nReturn a JSON array of strings."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        claims = json.loads(raw_json)
        assert isinstance(claims, list)
        return claims, cost or 0.0

    async def _check_claim_support(self, context: List[str], claims: List[str]) -> Tuple[List[Dict[str, str]], float, int]:
        ctx = "\n".join(context)
        prompt = (
            "For each claim, check if it is supported by the context. "
            "Respond with JSON array of objects: "
            '{"claim": "...", "supported": true|false, "reason": "..."}\n\n'
            f"CONTEXT:\n{ctx}\n\n"
            f"CLAIMS:\n{json.dumps(claims)}"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        results = json.loads(raw_json)
        supported = [r for r in results if r["supported"]]
        return results, cost or 0.0, len(supported)

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        question = test_case.input
        context = test_case.retrieval_context or []
        reference = test_case.expected_output

        # Step 1: Extract claims
        claims, cost = await self._extract_claims(reference)
        llm_cost += cost

        # Step 2: Check if each claim is supported by the retrieved context
        verdicts, cost, supported_count = await self._check_claim_support(context, claims)
        llm_cost += cost

        total_claims = len(claims)
        recall_score = round(supported_count / total_claims,
                             4) if total_claims else 0.0
        success = recall_score >= self.threshold

        evaluation_log = {
            "input_question": question,
            "expected_output": reference,
            "retrieval_context": context,
            "claims": claims,
            "comment_claims": "Claims extracted from reference answer.",
            "verdicts": verdicts,
            "comment_verdicts": "Each claim checked for support in context.",
            "final_score": recall_score,
            "comment_final_score": "Proportion of supported claims from reference.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the score exceeds the threshold.",
        }

        return {
            "score": recall_score,
            "success": success,
            "reason": f"{supported_count} out of {total_claims} reference claims supported by context.",
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log
        }
