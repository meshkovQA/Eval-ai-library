# repetitive_pattern.py
"""
Repetitive Pattern Detection Metric: Detects loops where the agent repeats the
same actions or responses instead of progressing.

Higher score = fewer repetitions = better. The verdict scale is inverted
compared to peer metrics to keep "higher = better" semantics.
"""
import json
from typing import Any, Dict, Tuple

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
        verbose: bool = False,
    ):
        super().__init__(model=model, threshold=threshold, verbose=verbose)

    # ==================== HELPERS ====================

    @staticmethod
    def _render_dialogue(turns) -> str:
        return "\n".join(
            f"{i+1}. User: {t.input}\n   Assistant: {t.actual_output}"
            for i, t in enumerate(turns)
        )

    @staticmethod
    def _rubric() -> str:
        return """Rate repetition severity (best → worst):

none    – no repetition: every assistant turn adds new information or makes
          new progress. Natural re-use of terminology does not count.
minor   – a single slightly-repeated phrase or one paraphrased rehash, but
          the conversation still progresses toward a resolution.
partial – noticeable repetition: the assistant repeats the same idea or
          tool attempt 2–3 times, but still eventually makes some progress.
mostly  – frequent repetition: the assistant cycles through the same answers
          or apologies with little meaningful progress.
fully   – stuck in a loop: the same action, phrase, or refusal repeats
          without ANY progress, even when the user pushes back or rephrases.

Scoring rules:
- Distinguish repetition-without-progress (bad) from consistent phrasing
  across similar user questions (neutral). Saying "Paris" twice for two
  different "capital of France?" questions is not repetition.
- A refusal loop counts as "fully" even if the refusals are politely worded
  differently each time — what matters is lack of progress.
- Weight the END of the conversation more than the beginning: if the agent
  recovers by the last turns, the verdict should reflect that."""

    @staticmethod
    def _few_shot() -> str:
        return """EXAMPLE 1 (none):
1. User: What is Python?
   Assistant: A high-level programming language created by Guido van Rossum.
2. User: And who created Ruby?
   Assistant: Yukihiro "Matz" Matsumoto created Ruby in 1995.
Verdict: {"verdict": "none", "repeated_segments": [], "reason": "Each turn delivers distinct new information."}

EXAMPLE 2 (partial):
1. User: How do I fix this bug?
   Assistant: Try clearing your cache.
2. User: Didn't help.
   Assistant: Have you tried clearing the cache?
3. User: Yes, twice.
   Assistant: Okay — check the network logs for the actual error.
Verdict: {"verdict": "partial", "reason": "Assistant repeated the cache suggestion once but eventually moved to a new approach."}

EXAMPLE 3 (fully):
1. User: Can you help me with my code?
   Assistant: I cannot help with that request.
2. User: Please, it's urgent.
   Assistant: I'm unable to help with that request.
3. User: Why not? It's just a syntax error.
   Assistant: Sorry, I cannot assist with that.
Verdict: {"verdict": "fully", "repeated_segments": ["I cannot help with that request"], "reason": "Assistant is stuck in a refusal loop and makes zero progress despite repeated user clarifications."}"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _generate_verdict(
        self, dialogue_text: str
    ) -> Tuple[Dict[str, Any], float]:
        prompt = (
            "You are an expert dialogue evaluator looking for REPETITIVE "
            "PATTERNS — places where the assistant repeats the same phrase, "
            "action, or response without making progress.\n\n"
            f"{self._rubric()}\n\n"
            f"{self._few_shot()}\n\n"
            f"DIALOGUE TO EVALUATE:\n{dialogue_text}\n\n"
            "Return ONLY a JSON object:\n"
            '{"verdict": "none|minor|partial|mostly|fully", '
            '"repeated_segments": [string, ...], '
            '"reason": "<one or two sentences>"}'
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

    # ==================== MAIN EVALUATION ====================

    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        total_cost = 0.0
        dialogue_text = self._render_dialogue(test_case.turns)

        verdict, cost = await self._generate_verdict(dialogue_text)
        total_cost += cost

        final_score = round(
            REPETITION_VERDICT_WEIGHTS.get(verdict.get("verdict", "none"), 1.0), 4
        )
        success = final_score >= self.threshold
        reason = verdict.get("reason", "")

        evaluation_log = {
            "dialogue": dialogue_text,
            "number_of_turns": len(test_case.turns),
            "verdict": verdict,
            "comment_verdict": (
                "Holistic LLM verdict on repetition severity. Higher score = "
                "fewer repetitions."
            ),
            "repeated_segments": verdict.get("repeated_segments", []),
            "final_score": final_score,
            "comment_final_score": (
                "Inverted weight: none=1.0 (best), fully=0.0 (stuck in a loop)."
            ),
            "threshold": self.threshold,
            "success": success,
            "final_reason": reason,
        }

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": reason,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }

        self.print_result(result)
        return result
