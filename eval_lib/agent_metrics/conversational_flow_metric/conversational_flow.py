# conversational_flow.py
"""
Conversational Flow Rate Metric: Evaluates how natural and coherent the dialogue
is. Penalizes unnecessary clarifications, repetition, and ignoring user signals.
"""
import json
from typing import Any, Dict, List, Tuple

from eval_lib.llm_client import chat_complete
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.utils import extract_json_block, score_agg


VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}


class ConversationalFlowRateMetric(ConversationalMetricPattern):
    """
    Grades dialogue naturalness and coherence per assistant turn.
    """

    name = "conversationalFlowRateMetric"

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
        return """Rate conversational flow (worst → best):

none    – broken flow: ignores user signals, irrelevant tangents, incoherent
minor   – frequent redundant clarifications or misreads of user intent
partial – noticeable friction but still progresses
mostly  – smooth with minor awkwardness
fully   – natural, coherent, directly addresses user signals"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _generate_verdicts(
        self, dialogue_text: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        prompt = (
            f"{self._prompt_label_help()}\n\n"
            "Evaluate the DIALOGUE below. For EACH assistant reply, judge the conversational "
            "flow: does it directly address the user's signals, avoid unnecessary "
            "clarifications, maintain coherence with prior turns, and avoid irrelevant "
            "tangents or repetition of earlier content?\n\n"
            f"DIALOGUE:\n{dialogue_text}\n\n"
            "Return ONLY a JSON array with one verdict object per assistant reply, in order:\n"
            '[{"verdict": "fully|mostly|partial|minor|none", "reason": "<one sentence>"}, ...]'
        )
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        try:
            verdicts = json.loads(extract_json_block(text))
            if not isinstance(verdicts, list):
                raise ValueError("Expected JSON array of verdicts")
            weights = [
                VERDICT_WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts
            ]
            score = round(score_agg(weights, temperature=self.temperature), 4)
            return verdicts, score, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse verdicts: {e}\n{text}")

    async def _summarize_verdicts(
        self, verdicts: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        bullets = "\n".join(f"- {v.get('reason', '')}" for v in verdicts[:6])
        prompt = (
            "Write a concise (max 2 sentences) assessment of conversational flow "
            "based on these observations:\n\n"
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

        verdicts, verdict_score, cost = await self._generate_verdicts(dialogue_text)
        total_cost += cost

        summary, cost = await self._summarize_verdicts(verdicts)
        total_cost += cost

        final_score = verdict_score
        success = final_score >= self.threshold

        evaluation_log = {
            "dialogue": dialogue_text,
            "number_of_turns": len(test_case.turns),
            "verdicts": verdicts,
            "comment_verdicts": "Per-turn verdicts on conversational flow quality.",
            "verdict_weights": {
                i: VERDICT_WEIGHTS.get(v.get("verdict", "none"), 0.0)
                for i, v in enumerate(verdicts)
            },
            "final_score": final_score,
            "comment_final_score": (
                f"Softmax aggregation of per-turn verdict weights (temperature={self.temperature})."
            ),
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
