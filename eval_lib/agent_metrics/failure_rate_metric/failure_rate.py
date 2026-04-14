# failure_rate.py
"""
Failure Rate Metric: Evaluates how the agent behaves when it does not know the
answer — does it hallucinate, stall, or honestly propose alternatives?

Higher score = better failure handling. Turns where the agent confidently had
the answer are classified as "not_applicable" and excluded from scoring.
"""
import json
from typing import Any, Dict, List, Tuple

from eval_lib.llm_client import chat_complete
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.utils import extract_json_block, score_agg


FAILURE_WEIGHTS = {
    "honest_fallback": 1.0,
    "alternative_proposed": 1.0,
    "stall": 0.4,
    "hallucination": 0.0,
}

VALID_LABELS = set(FAILURE_WEIGHTS.keys()) | {"not_applicable"}


class FailureRateMetric(ConversationalMetricPattern):
    """
    Grades assistant failure-handling behavior across a conversation.
    """

    name = "failureRateMetric"

    def __init__(
        self,
        model: str,
        threshold: float = 0.7,
        temperature: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature

    # ==================== HELPERS ====================

    @staticmethod
    def _render_dialogue(turns) -> str:
        return "\n".join(
            f"{i+1}. User: {t.input}\n   Assistant: {t.actual_output}"
            for i, t in enumerate(turns)
        )

    @staticmethod
    def _prompt_label_help() -> str:
        return """Classify each assistant turn's failure-handling behavior:

honest_fallback      – agent admits it does not know (best)
alternative_proposed – agent offers a useful alternative or next step (best)
stall                – agent gives a vague non-answer or deflects without helping
hallucination        – agent confidently fabricates information (worst)
not_applicable       – this turn had no uncertainty; the agent simply answered"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _classify_turns(
        self, dialogue_text: str
    ) -> Tuple[List[Dict[str, str]], float]:
        prompt = (
            f"{self._prompt_label_help()}\n\n"
            "For EACH assistant reply in the DIALOGUE below, classify its failure-handling "
            "behavior into exactly ONE of: honest_fallback, alternative_proposed, stall, "
            "hallucination, not_applicable.\n\n"
            f"DIALOGUE:\n{dialogue_text}\n\n"
            "Return ONLY a JSON array with one classification object per assistant reply, in order:\n"
            '[{"label": "honest_fallback|alternative_proposed|stall|hallucination|not_applicable", '
            '"reason": "<one sentence>"}, ...]'
        )
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        try:
            data = json.loads(extract_json_block(text))
            if not isinstance(data, list):
                raise ValueError("Expected JSON array of classifications")
            for item in data:
                label = item.get("label", "not_applicable")
                if label not in VALID_LABELS:
                    item["label"] = "not_applicable"
            return data, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse classifications: {e}\n{text}")

    async def _summarize(
        self, classifications: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        bullets = "\n".join(
            f"- {c.get('label')}: {c.get('reason', '')}" for c in classifications[:6]
        )
        prompt = (
            "Write a concise (max 2 sentences) assessment of how the assistant handled "
            "uncertainty across the conversation, based on:\n\n"
            f"{bullets}\n\n"
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

        classifications, cost = await self._classify_turns(dialogue_text)
        total_cost += cost

        summary, cost = await self._summarize(classifications)
        total_cost += cost

        scored_labels = [
            c.get("label") for c in classifications if c.get("label") != "not_applicable"
        ]

        if not scored_labels:
            final_score = 1.0
            no_failures_reason = (
                "No uncertainty/failure situations observed in the conversation."
            )
            effective_summary = summary or no_failures_reason
        else:
            weights = [FAILURE_WEIGHTS.get(label, 0.0) for label in scored_labels]
            final_score = round(score_agg(weights, temperature=self.temperature), 4)
            effective_summary = summary

        success = final_score >= self.threshold

        evaluation_log = {
            "dialogue": dialogue_text,
            "number_of_turns": len(test_case.turns),
            "classifications": classifications,
            "comment_classifications": (
                "Per-turn failure-handling labels. 'not_applicable' turns are excluded from scoring."
            ),
            "scored_turn_count": len(scored_labels),
            "final_score": final_score,
            "comment_final_score": (
                "Softmax aggregation over failure-handling weights. If no failure situations "
                "occurred, score defaults to 1.0 (nothing to penalize)."
            ),
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": success,
            "final_reason": effective_summary,
        }

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": effective_summary,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }

        self.print_result(result)
        return result
