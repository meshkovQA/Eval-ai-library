from typing import List, Dict, Tuple, Any
import json
import re
import numpy as np
from math import exp

from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.llm_client import chat_complete


VERDICT_WEIGHTS = {
    "fully":   1.0,
    "mostly":  0.8,
    "partial": 0.5,
    "minor":   0.2,
    "none":    0.0,
}

MAX_CRITERIA = 10
LINK_CRITERION = "The user got the link to the requested resource."


def extract_json_block(text: str) -> str:
    """
    Extracts the first JSON block from Markdown-like fenced code blocks.
    """
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def softmax_agg(scores: List[float], temperature: float = 1.1) -> float:
    if not scores:
        return 0.0
    exp_scores = [exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    return sum(s * e / total for s, e in zip(scores, exp_scores))


def render_dialog(turns) -> str:
    rows = []
    for i, t in enumerate(turns, 1):
        rows.append(f"{i}. User: {t.input}\n   Assistant: {t.actual_output}")
    return "\n".join(rows)


class TaskSuccessRateMetric(ConversationalMetricPattern):

    name = "taskSuccessRateMetric"
    template_cls = None  # prompts задаём прямо в коде

    # ------------------------------------------------------------------ #
    async def _infer_user_goal(self, dialogue: str) -> Tuple[str, float]:
        prompt = (
            "You will be shown an ENTIRE short conversation between a user and an assistant.\n"
            "Write ONE concise sentence describing the user's PRIMARY GOAL in this conversation.\n\n"
            f"CONVERSATION:\n{dialogue}\n\n"
            "User goal:"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return text.strip(), cost or 0.0

    # ------------------------------------------------------------------ #
    async def _generate_criteria(self, goal: str) -> Tuple[List[str], float]:
        few_shot = (
            "Example 1:\n"
            "User goal: Order a pizza online\n"
            "Criteria: [\n"
            "  \"The assistant provided available pizza options.\",\n"
            "  \"The user received an order confirmation number.\"\n"
            "]\n\n"
            "Example 2:\n"
            "User goal: Reset an email password\n"
            "Criteria: [\n"
            "  \"The assistant gave a working password-reset link.\",\n"
            "  \"The user confirmed they could log in.\"\n"
            "]\n\n"
        )
        prompt = (
            f"{few_shot}"
            f"Now do the same for the next case.\n\n"
            f"User goal: {goal}\n"
            f"List up to {MAX_CRITERIA} concrete SUCCESS CRITERIA that could realistically be satisfied "
            f"within a brief chat of 2–5 turns. "
            "then **add** this exact sentence: "
            f"\"{LINK_CRITERION}\".\n"
            "Each criterion must be a short, observable statement.\n"
            "Return only a JSON array of strings."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        criteria = json.loads(extract_json_block(text))

        # Optional quick criterion about receiving a link
        if LINK_CRITERION not in criteria:
            criteria.append(LINK_CRITERION)

        if len(criteria) > MAX_CRITERIA:

            criteria = (
                [LINK_CRITERION] +
                [c for c in criteria if c != LINK_CRITERION][:MAX_CRITERIA - 1]
            )

        # Truncate to MAX_CRITERIA
        criteria = criteria[:MAX_CRITERIA]
        return criteria, cost or 0.0

    # ------------------------------------------------------------------ #

    async def _generate_verdicts(
        self,
        goal: str,
        criteria: List[str],
        dialogue: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        prompt = (
            f"USER GOAL: {goal}\n\n"
            f"FULL DIALOGUE:\n{dialogue}\n\n"
            f"SUCCESS CRITERIA (as JSON array):\n{json.dumps(criteria, ensure_ascii=False)}\n\n"
            "For  **each** criterion decide how well it is satisfied at the END of the dialogue, "
            "using exactly one of the following labels: fully, mostly, partial, minor, none.\n"
            "Return JSON array with **exactly the same length and order** as the criteria list. "
            "[{\"verdict\":\"fully|mostly|partial|minor|none\",\"reason\":\"<one sentence>\"}, …]"
            "No extra text."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        verdicts = json.loads(extract_json_block(text))
        if len(verdicts) != len(criteria):
            verdicts = verdicts * len(criteria)

        weights = [VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in verdicts]
        score = round(softmax_agg(weights), 4)
        return verdicts, score, cost or 0.0

    # ------------------------------------------------------------------ #
    async def _summarize(self, verdicts: List[Dict[str, str]]) -> Tuple[str, float]:
        bullets = "\n".join(f"- {v['reason']}" for v in verdicts[:6])
        prompt = (
            "Write a concise (max 2 sentences) overall assessment of task success, "
            "based on the points:\n\n"
            f"{bullets}\n\nSummary:"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return text.strip(), cost or 0.0

    # =================== PUBLIC evaluate ============================== #
    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        first_user = test_case.turns[0].input.strip()
        dialogue_text = render_dialog(test_case.turns)

        # 1. Goal
        goal, c = await self._infer_user_goal(dialogue_text)
        llm_cost += c

        # 2. Criteria
        criteria, c = await self._generate_criteria(goal)
        llm_cost += c

        # 3. Verdicts & score
        verdicts, score, c = await self._generate_verdicts(goal, criteria, dialogue_text)
        llm_cost += c

        # 4. Summary
        summary, c = await self._summarize(verdicts)
        llm_cost += c

        success = score >= self.threshold

        # 5. Log
        evaluation_log = {
            "dialogue": dialogue_text,
            "comment_dialogue": "Full conversation used for evaluation.",
            "user_goal": goal,
            "comment_user_goal": "High-level user intention.",
            "success_criteria": criteria,
            "comment_success_criteria": "LLM-generated checklist of what needs to happen to succeed.",
            "verdicts": verdicts,
            "comment_verdicts": "LLM judgement per criterion (fully/mostly/partial/minor/none).",
            "final_score": score,
            "comment_final_score": "Soft-max aggregation of verdict weights.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "True if final_score ≥ threshold.",
            "final_reason": summary,
            "comment_reasoning": "Short human-readable explanation of the result."
        }

        return {
            "score": score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log,
        }
