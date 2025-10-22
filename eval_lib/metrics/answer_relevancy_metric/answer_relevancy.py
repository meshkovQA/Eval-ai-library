# answer_relevancy.py
from typing import List, Dict, Any, Tuple
import numpy as np
import json
import re
from math import exp
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete

# Constants for verdict weights
VERDICT_WEIGHTS = {
    "fully": 1.0,    # Fully related
    "mostly": 0.9,   # Mostly related
    "partial": 0.7,  # Partially related
    "minor": 0.3,    # Weekly related
    "none": 0.0      # Not related
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


def softmax_agg(scores: List[float], temperature: float = 0.5) -> float:
    """
    Computes a softmax aggregation of scores with temperature scaling.
    This function applies the softmax formula to the input scores, allowing
    for a temperature parameter to control the sharpness of the distribution.
    If the scores list is empty, it returns 1.0 as a default score.
    The temperature parameter can be adjusted to make the distribution more or
    less sensitive to the differences in scores.
    """
    if not scores:
        return 1.0
    exp_scores = [exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    return sum(s * e / total for s, e in zip(scores, exp_scores))


class AnswerRelevancyMetric(MetricPattern):
    name = "answerRelevancyMetric"
    template_cls = None  # not used, LLM + embeddings logic are combined

    def __init__(
        self,
        model: str,
        threshold: float = 0.6,
    ):
        super().__init__(model=model, threshold=threshold)

    async def _infer_user_intent(self, question: str) -> str:
        prompt = (
            "Determine the user's intent behind the following question.\n"
            "Answer in ONE concise sentence without adding extra details.\n\n"
            f"Question: {question}"
        )
        response, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return response.strip(), cost or 0.0

    async def _generate_statements(self, intent: str, answer: str) -> Tuple[List[str], float]:
        prompt = (
            "You are extracting atomic facts from a chatbot answer.\n"
            f"User intent: {intent}\n\n"
            "Answer:\n"
            f"{answer}\n\n"
            "Instructions:\n"
            "• Extract ALL factual statements from the answer.\n"
            "• Include both relevant AND irrelevant statements.\n"
            "• Skip only greetings, disclaimers, offers to help.\n"
            "• 1-sentence per statement, no numbering.\n"
            "• Output as a JSON array of strings."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        try:
            raw_json = extract_json_block(text)
            statements = json.loads(raw_json)
            assert isinstance(statements, list)
            return statements, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse statements: {e}\n{text}")

    async def _generate_verdicts(self, question: str, intent: str, statements: List[str]) -> Tuple[List[Dict[str, str]], float, float]:

        prompt_user = (
            "You are an impartial evaluator.\n"
            "TASK\n"
            "For every statement below decide **how directly it fulfils the user intent**, using the 5-level scale:\n"
            "• fully   – Explicitly answers the intent with no missing info.\n"
            "• mostly  – Clearly supports the intent via concrete example or list item; small details may be missing.\n"
            "• partial – Related to the topic but only partially addresses the intent.\n"
            "• minor   – Weak or tangential relation.\n"
            "• none    – Irrelevant or off-topic.\n\n"
            "⚠️  Do NOT punish a statement just because it is an example or uses different wording; examples usually deserve **mostly**.\n"
            "⚠️  Ignore polite closings, greetings, offers to help.\n\n"
            f"USER INTENT: {intent}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"STATEMENTS (JSON array):\n{json.dumps(statements, ensure_ascii=False)}\n\n"
            "Return **only** a JSON array of objects in the form:\n"
            "[{\"verdict\": \"fully|mostly|partial|minor|none\", \"reason\": \"<one sentence>\"}, …]"
        )
        text, cost = await chat_complete(self.model, [{"role": "user",   "content": prompt_user}], temperature=0.0)

        try:
            raw_json = extract_json_block(text)
            verdicts = json.loads(raw_json)
            assert isinstance(verdicts, list)

            scores = [VERDICT_WEIGHTS.get(v.get("verdict", "").lower(), 0.0)
                      for v in verdicts]
            verdict_score = round(float(np.mean(scores)), 4) if scores else 0.0
            return verdicts, verdict_score, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse verdicts: {e}\n{text}")

    async def _summarize_reasons_via_llm(
        self,
        verdicts: List[Dict[str, str]],
    ) -> Tuple[str, float]:

        grouped: Dict[str, List[str]] = {}
        for v in verdicts:
            grouped.setdefault(v["verdict"], []).append(v["reason"])

        bullets: List[str] = []
        for tag in ("fully", "mostly", "partial", "minor", "none"):
            if tag in grouped:
                examples = grouped[tag][:2]
                bullets.extend(f"- {r}" for r in examples)

        reasons_block = "\n".join(bullets)

        prompt = (
            "You are an expert evaluator who writes crisp 1-2-sentence summaries."
            "Below are bulleted findings from an answer-relevancy check. "
            "Write a single concise explanation (max two sentences) that sums up "
            "how well the answer met the user's request, mentioning the main strengths "
            "and the biggest gap. Do not enumerate bullets, just a unified summary.\n\n"
            f"{reasons_block}\n\n"
            "Unified explanation:"
        )

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return text.strip(), cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        question = test_case.input
        answer = test_case.actual_output

        # Step 1: Infer the user's intent from the question
        intent, cost = await self._infer_user_intent(question)
        llm_cost += cost

        # Step 2: Generate statements that the answer would fully address
        statements, cost = await self._generate_statements(intent, answer)
        llm_cost += cost

        # Step 3: Generate verdicts for each statement
        verdicts, _, cost = await self._generate_verdicts(question, intent, statements)
        llm_cost += cost

        weights = [VERDICT_WEIGHTS[v["verdict"]] for v in verdicts]
        verdict_score = round(softmax_agg(weights, temperature=0.5), 4)

        # Step 4: Summarize the verdict reasons
        summary_reason, cost = await self._summarize_reasons_via_llm(verdicts)
        llm_cost += cost

        # Step 4: Count final score based on verdicts
        final_score = verdict_score
        success = final_score >= self.threshold

        # Step 5: Verbose log
        evaluation_log = {
            "input_question": question,
            "answer": answer,
            "user_intent": intent,
            "comment_user_intent": "Inferred goal of the question.",
            "statements": statements,
            "comment_statements": "Atomic facts extracted from the answer.",
            "verdicts": verdicts,
            "comment_verdicts": "Each verdict explains whether a statement is relevant to the question.",
            "verdict_score": verdict_score,
            "comment_verdict_score": "Proportion of relevant statements in the answer.",
            "final_score": final_score,
            "comment_final_score": "Score based on the proportion of relevant statements.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the score exceeds the pass threshold.",
            "final_reason": summary_reason,
            "comment_reasoning": "Compressed explanation of the key verdict rationales."
        }

        return {
            "score": final_score,
            "success": success,
            "reason": summary_reason,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log
        }
