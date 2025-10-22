# contextual_relevancy_llm.py

from typing import List, Dict, Tuple, Any
import json
import re
from math import exp
import numpy as np
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete

# weights for each verdict category
VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}


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


def score_agg(
        scores: List[float],
        temperature: float = 0.5,
        penalty: float = 0.1
) -> float:
    """
    Compute a softmax-weighted aggregate of scores, then
    apply a penalty proportional to the count of low-scoring items.
    """
    if not scores:
        return 0.0
    exp_scores = [exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    softmax_score = sum(s * e / total for s, e in zip(scores, exp_scores))
    # penalize if many statements have verdict ≤ minor
    irrelevant = sum(1 for s in scores if s <= 0.3)
    penalty_factor = max(0.0, 1 - penalty * irrelevant)
    return round(softmax_score * penalty_factor, 4)


class ContextualRelevancyMetric(MetricPattern):
    """
    Evaluates how relevant the retrieved context passages are to the user's intent.
    1) Infer intent from the question.
    2) For each context segment, ask the LLM to judge its relevance.
    3) Aggregate verdicts into a final score via softmax.
    """

    name = "contextualRelevancyMetric"
    template_cls = None

    def __init__(self, model: str, threshold: float = 0.6):
        super().__init__(model=model, threshold=threshold)

    async def _infer_user_intent(self, question: str) -> Tuple[str, float]:
        """
        Ask the LLM to summarize the user's intent in one sentence.
        """
        prompt = (
            "Determine the user's intent behind this question.\n"
            "Answer in one concise sentence.\n\n"
            f"Question: {question}"
        )
        resp, cost = await chat_complete(
            self.model,
            [{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.strip(), cost or 0.0

    async def _generate_verdicts(
        self,
        intent: str,
        context: List[str],
        question: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        """
        For each context segment, ask the LLM to classify its relevance 
        to the inferred intent with a 5-level verdict and a brief reason.
        """
        prompt = (
            "You are evaluating how well each CONTEXT segment serves both the user's explicit question and underlying intent.\n\n"
            f"USER QUESTION: {question}\n\n"
            f"USER INTENT: {intent}\n\n"
            "CONTEXT SEGMENTS (JSON array):\n"
            f"{json.dumps(context, ensure_ascii=False)}\n\n"
            "For each segment, evaluate its relevance to BOTH the specific question asked AND the user's broader intent.\n"
            "Return an object for each segment:\n"
            '{"verdict": "fully|mostly|partial|minor|none", "reason": "<one-sentence explaining relevance to question and intent>"}\n'
            "Respond with a JSON array ONLY.\n\n"
            "Verdict levels:\n"
            "- fully: directly answers the question and completely addresses the user's intent\n"
            "- mostly: addresses the question well and covers most of the user's intent with minor gaps\n"
            "- partial: partially relevant to the question or intent but missing key information\n"
            "- minor: tangentially related to either the question or intent\n"
            "- none: not relevant to the question or user's intent"
        )
        resp, cost = await chat_complete(
            self.model,
            [{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = extract_json_block(resp)
        verdicts = json.loads(raw)
        # compute weights list
        scores = [VERDICT_WEIGHTS.get(v["verdict"].lower(), 0.0)
                  for v in verdicts]
        agg = score_agg(scores, temperature=0.5)
        return verdicts, round(agg, 4), cost or 0.0

    async def _summarize_reasons(
        self,
        verdicts: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        """
        Take the top two and bottom one verdict reasons and ask the LLM
        to write a unified 1–2 sentence summary of context relevancy.
        """
        # sort by weight
        sorted_by_weight = sorted(
            verdicts,
            key=lambda v: VERDICT_WEIGHTS.get(v["verdict"].lower(), 0.0),
            reverse=True
        )
        top_reasons = [v["reason"] for v in sorted_by_weight[:2]]
        bottom_reasons = [v["reason"] for v in sorted_by_weight[-1:]]
        bullets = "\n".join(f"- {r}" for r in top_reasons + bottom_reasons)

        prompt = (
            "You are an expert evaluator."
            "Below are key points about how context segments matched the user's question and intent:\n\n"
            f"{bullets}\n\n"
            "Write a concise 1–2 sentence summary explaining overall how relevant "
            "the retrieved context is to answering the user's question and meeting their needs."
        )
        resp, cost = await chat_complete(
            self.model,
            [{"role": "user",   "content": prompt}],
            temperature=0.0
        )
        return resp.strip(), cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        question = test_case.input
        context = test_case.retrieval_context or []

        # 1) Infer intent
        intent, cost = await self._infer_user_intent(question)
        llm_cost += cost

        # 2) Generate verdicts for each context segment
        verdicts, score, cost = await self._generate_verdicts(intent, context, question)
        llm_cost += cost

        # 3) Summarize reasons
        summary, cost = await self._summarize_reasons(verdicts)
        llm_cost += cost

        success = score >= self.threshold

        evaluation_log = {
            "input_question": question,
            "user_intent": intent,
            "retrieval_context": context,
            "comment_verdicts": "Each shows if a statement is grounded in the context.",
            "verdicts": verdicts,
            "final_score": score,
            "comment_final_score": "Weighted support score from context.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the score exceeds the threshold.",
            "final_reason": summary,
            "comment_reasoning": "LLM-generated explanation based on verdict rationales."
        }

        return {
            "score": score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log
        }
