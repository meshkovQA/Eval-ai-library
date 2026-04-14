# repetitive_pattern.py
"""
Repetitive Pattern Detection Metric: Detects loops where the agent repeats the
same actions or responses instead of progressing.

Higher score = fewer repetitions = better. The verdict scale is inverted
compared to peer metrics to keep "higher = better" semantics.
"""
import json
from typing import Any, Dict, List, Tuple

from eval_lib.llm_client import chat_complete
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.utils import extract_json_block


# Inverted: "none" (no repetition) is best; "fully" (fully stuck in a loop) is worst.
REPETITION_VERDICT_WEIGHTS = {
    "none": 1.0,
    "minor": 0.9,
    "partial": 0.7,
    "mostly": 0.3,
    "fully": 0.0,
}


class RepetitivePatternDetectionMetric(ConversationalMetricPattern):
    """
    Detects whether the agent gets stuck in loops — repeating the same actions,
    phrases, or responses without progressing toward a resolution.
    """

    name = "repetitivePatternDetectionMetric"

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
        return """Rate repetition severity (best → worst):

none    – no repetition; every assistant turn adds new value
minor   – a single slightly-repeated phrase but conversation progresses
partial – noticeable repetition but still makes some progress
mostly  – frequent repetition, little progress
fully   – stuck in a loop, same action/response repeated without any progress"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _generate_verdict(
        self, dialogue_text: str
    ) -> Tuple[Dict[str, Any], float]:
        prompt = (
            f"{self._prompt_label_help()}\n\n"
            "Analyze the DIALOGUE below for REPETITIVE PATTERNS in the assistant's "
            "behavior — repeated phrases, repeated tool/action attempts, or the same "
            "response cycling without progress.\n\n"
            f"DIALOGUE:\n{dialogue_text}\n\n"
            "Return ONLY a JSON object:\n"
            '{"verdict": "none|minor|partial|mostly|fully", '
            '"repeated_segments": [string, ...], '
            '"reason": "<one sentence>"}'
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
            data.setdefault("repeated_segments", [])
            data.setdefault("reason", "")
            return data, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse verdict: {e}\n{text}")

    async def _summarize(self, verdict: Dict[str, Any]) -> Tuple[str, float]:
        segments_preview = verdict.get("repeated_segments", [])[:3]
        prompt = (
            "Write a concise (max 2 sentences) assessment of whether the agent got "
            "stuck in repetitive patterns, based on:\n\n"
            f"Verdict: {verdict.get('verdict')} — {verdict.get('reason', '')}\n"
            f"Repeated segments (sample): {segments_preview}\n\n"
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

        verdict, cost = await self._generate_verdict(dialogue_text)
        total_cost += cost

        summary, cost = await self._summarize(verdict)
        total_cost += cost

        final_score = round(
            REPETITION_VERDICT_WEIGHTS.get(verdict.get("verdict", "none"), 1.0), 4
        )
        success = final_score >= self.threshold

        evaluation_log = {
            "dialogue": dialogue_text,
            "number_of_turns": len(test_case.turns),
            "verdict": verdict,
            "comment_verdict": (
                "LLM verdict on repetition severity. Higher score = fewer repetitions."
            ),
            "repeated_segments": verdict.get("repeated_segments", []),
            "final_score": final_score,
            "comment_final_score": (
                "Inverted weight: none=1.0 (best), fully=0.0 (stuck in a loop)."
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
