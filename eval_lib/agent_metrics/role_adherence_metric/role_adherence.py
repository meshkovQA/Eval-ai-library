from typing import List, Dict, Tuple, Any
from textwrap import dedent
import json
import re
from math import exp

from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.llm_client import chat_complete


def extract_json_block(text: str) -> str:
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def softmax_agg(scores: List[float], temperature: float = 0.5) -> float:
    if not scores:
        return 0.0
    exp_scores = [exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    return sum(s * e / total for s, e in zip(scores, exp_scores))


def render_dialog(turns) -> str:
    return "\n".join(
        f"{i+1}. User: {t.input}\n   Assistant: {t.actual_output}"
        for i, t in enumerate(turns)
    )


ROLE_VERDICT_WEIGHTS = {
    "fully":   1.0,
    "mostly":  0.9,
    "partial": 0.7,
    "minor":   0.3,
    "none":    0.0,
}


class RoleAdherenceMetric(ConversationalMetricPattern):
    name = "roleAdherenceMetric"
    template_cls = None  # шаблон не нужен, промпт генерируем прямо

    async def _generate_verdicts(
        self, role_description: str, dialogue_text: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        prompt = dedent(f"""
        SYSTEM ROLE: {role_description}

        DIALOGUE:
        {dialogue_text}

        Task: Judge how well the chatbot stays in character throughout the conversation.

        For each assistant reply, assign a score from:
        "fully", "mostly", "partial", "minor", "none".

        Respond with JSON:
        [{{"verdict": "fully", "reason": "Remained polite and helpful"}}, ...]
        """).strip()

        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        verdicts = json.loads(extract_json_block(text))
        weights = [ROLE_VERDICT_WEIGHTS.get(
            v["verdict"], 0.0) for v in verdicts]
        score = round(softmax_agg(weights), 4)
        return verdicts, score, cost or 0.0

    async def _summarize(self, verdicts: List[Dict[str, str]]) -> Tuple[str, float]:
        bullets = "\n".join(f"- {v['reason']}" for v in verdicts[:6])
        prompt = (
            "Write a concise (max 2 sentences) summary of how well the chatbot stayed in character:\n\n"
            f"{bullets}\n\nSummary:"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return text.strip(), cost or 0.0

    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        role = test_case.chatbot_role or "No role specified"
        dialogue_text = render_dialog(test_case.turns)

        # 1. Verdicts
        verdicts, score, c = await self._generate_verdicts(role, dialogue_text)
        llm_cost += c

        # 2. Summary
        summary, c = await self._summarize(verdicts)
        llm_cost += c

        success = score >= self.threshold

        evaluation_log = {
            "dialogue": dialogue_text,
            "chatbot_role": role,
            "comment_dialogue": "Conversation evaluated for role alignment.",
            "verdicts": verdicts,
            "comment_verdicts": "LLM judgement on role adherence per turn.",
            "final_score": score,
            "comment_final_score": "Soft-max aggregation of verdict weights.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "True if final_score ≥ threshold.",
            "final_reason": summary,
            "comment_reasoning": "Short explanation of how well the chatbot stayed in character.",
        }

        return {
            "score": score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log,
        }
