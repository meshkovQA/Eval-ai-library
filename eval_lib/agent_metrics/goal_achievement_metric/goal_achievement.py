# goal_achievement.py
"""
Goal Achievement Rate Metric: Evaluates whether the user actually got what they
wanted from the conversation (outcome/satisfaction lens), not just whether a
formal task checklist was ticked off.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from eval_lib.llm_client import chat_complete
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.utils import extract_json_block


VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}


class GoalAchievementRateMetric(ConversationalMetricPattern):
    """
    Evaluates whether the user actually got what they wanted — outcome-oriented.
    Looks at satisfaction signals, unresolved requests, and whether the final
    assistant turns delivered the desired outcome.
    """

    name = "goalAchievementRateMetric"

    def __init__(
        self,
        model: str,
        threshold: float = 0.7,
        temperature: float = 0.5,
        verbose: bool = False,
        user_goal: Optional[str] = None,
    ):
        """
        Args:
            model: LLM model name
            threshold: Success threshold (0.0-1.0)
            temperature: Reserved for future multi-verdict aggregation; kept
                for surface consistency with peer metrics.
            user_goal: Optional user-provided description of what the user
                actually wanted out of the conversation. When set, skips the
                goal-inference LLM call.
        """
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature
        self.user_goal = user_goal

    # ==================== HELPERS ====================

    @staticmethod
    def _render_dialogue(turns) -> str:
        return "\n".join(
            f"{i+1}. User: {t.input}\n   Assistant: {t.actual_output}"
            for i, t in enumerate(turns)
        )

    @staticmethod
    def _prompt_label_help() -> str:
        return """Rate goal achievement (worst → best):

none    – user did not get what they wanted at all
minor   – user got a tiny fraction of what they wanted
partial – user got part of what they wanted, significant gaps remain
mostly  – user got what they wanted with minor gaps
fully   – user fully got what they wanted"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _infer_user_goal(self, dialogue: str) -> Tuple[str, float]:
        prompt = (
            "You will be shown a conversation between a user and an assistant.\n"
            "Write ONE concise sentence describing what the USER ACTUALLY WANTED "
            "to get out of this conversation (desired outcome, not the surface-level task).\n\n"
            f"CONVERSATION:\n{dialogue}\n\n"
            "What the user wanted:"
        )
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return text.strip(), cost or 0.0

    async def _extract_signals(
        self, goal: str, dialogue: str
    ) -> Tuple[Dict[str, List[str]], float]:
        prompt = (
            "You are analyzing a conversation to detect user-satisfaction signals.\n\n"
            f"USER GOAL: {goal}\n\n"
            f"DIALOGUE:\n{dialogue}\n\n"
            "Identify:\n"
            "- positive_signals: explicit thanks, confirmation the user got what they needed, "
            "enthusiastic follow-ups.\n"
            "- negative_signals: frustration, repeated clarifications, explicit dissatisfaction, "
            "giving up, expressing the answer is wrong or unhelpful.\n"
            "- unmet_requests: things the user asked for that the assistant never addressed.\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            '{"positive_signals": [string, ...], '
            '"negative_signals": [string, ...], '
            '"unmet_requests": [string, ...]}'
        )
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        try:
            data = json.loads(extract_json_block(text))
            signals = {
                "positive_signals": list(data.get("positive_signals", [])),
                "negative_signals": list(data.get("negative_signals", [])),
                "unmet_requests": list(data.get("unmet_requests", [])),
            }
            return signals, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse signals: {e}\n{text}")

    async def _generate_verdict(
        self, goal: str, dialogue: str, signals: Dict[str, List[str]]
    ) -> Tuple[Dict[str, str], float]:
        prompt = (
            f"{self._prompt_label_help()}\n\n"
            f"USER GOAL: {goal}\n\n"
            f"DIALOGUE:\n{dialogue}\n\n"
            f"OBSERVED SIGNALS:\n{json.dumps(signals, ensure_ascii=False, indent=2)}\n\n"
            "Judge the OUTCOME: did the user actually get what they wanted by the end?\n"
            "Return ONLY a JSON object:\n"
            '{"verdict": "fully|mostly|partial|minor|none", "reason": "<one sentence>"}'
        )
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        try:
            data = json.loads(extract_json_block(text))
            if "verdict" not in data:
                raise ValueError("Missing 'verdict' field")
            return data, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse verdict: {e}\n{text}")

    async def _summarize(
        self, verdict: Dict[str, str], signals: Dict[str, List[str]]
    ) -> Tuple[str, float]:
        prompt = (
            "Write a concise (max 2 sentences) assessment of whether the user achieved "
            "their goal, based on:\n\n"
            f"Verdict: {verdict.get('verdict')} — {verdict.get('reason', '')}\n"
            f"Positive signals: {signals.get('positive_signals')}\n"
            f"Negative signals: {signals.get('negative_signals')}\n"
            f"Unmet requests: {signals.get('unmet_requests')}\n\n"
            "Summary:"
        )
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return text.strip(), cost or 0.0

    # ==================== MAIN EVALUATION ====================

    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        total_cost = 0.0
        dialogue_text = self._render_dialogue(test_case.turns)

        if self.user_goal:
            goal = self.user_goal
            goal_source = "user_provided"
        else:
            goal, cost = await self._infer_user_goal(dialogue_text)
            total_cost += cost
            goal_source = "llm_inferred"

        signals, cost = await self._extract_signals(goal, dialogue_text)
        total_cost += cost

        verdict, cost = await self._generate_verdict(goal, dialogue_text, signals)
        total_cost += cost

        summary, cost = await self._summarize(verdict, signals)
        total_cost += cost

        base_weight = VERDICT_WEIGHTS.get(verdict.get("verdict", "none"), 0.0)
        penalty = 0.1 * len(signals.get("negative_signals", []))
        final_score = round(max(0.0, base_weight - penalty), 4)
        success = final_score >= self.threshold

        evaluation_log = {
            "dialogue": dialogue_text,
            "comment_dialogue": "Full conversation text used for goal achievement evaluation.",
            "number_of_turns": len(test_case.turns),
            "user_goal": goal,
            "comment_user_goal": "Desired outcome the user wanted (user-provided or LLM-inferred).",
            "user_goal_source": goal_source,
            "signals": signals,
            "comment_signals": "Satisfaction/frustration signals and unmet requests extracted from the dialogue.",
            "verdict": verdict,
            "comment_verdict": "LLM verdict on whether the user got what they wanted.",
            "base_weight": base_weight,
            "negative_signal_penalty": round(penalty, 4),
            "final_score": final_score,
            "comment_final_score": "Base verdict weight minus 0.1 * count(negative_signals), floored at 0.",
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": success,
            "final_reason": summary,
        }

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }

        self.print_result(result)
        return result
