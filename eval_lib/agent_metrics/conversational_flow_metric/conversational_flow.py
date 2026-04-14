# conversational_flow.py
"""
Conversational Flow Rate Metric: Evaluates how natural and coherent the dialogue
is as a whole. Penalizes unnecessary clarifications, repetition, and ignoring
user signals.

Holistic approach: one verdict for the entire dialogue. Flow is a property of
the sequence, not independent turns — a single broken turn can derail the whole
conversation, and averaging per-turn scores would dilute that signal.
"""
import json
from typing import Any, Dict, Tuple

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


class ConversationalFlowRateMetric(ConversationalMetricPattern):
    """
    Grades overall dialogue naturalness and coherence with a single holistic
    verdict grounded in a few-shot rubric.
    """

    name = "conversationalFlowRateMetric"

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
        return """Rate the OVERALL conversational flow on this scale (worst → best):

none    – broken flow: assistant ignores user signals, goes off on irrelevant
          tangents, is incoherent with prior turns, or blocks progress entirely.
minor   – frequent redundant clarifications, misreads of user intent, awkward
          transitions that force the user to repeat themselves.
partial – noticeable friction: several missed signals or awkward transitions,
          but the conversation still progresses overall.
mostly  – smooth overall with minor awkwardness (one small redundant clarification,
          a slight tone mismatch) that doesn't meaningfully hurt progress.
fully   – natural, coherent, directly addresses user signals turn after turn,
          no unnecessary clarifications, smooth transitions, feels like a
          competent human conversation.

Scoring rules:
- Judge the dialogue AS A WHOLE, not as an average of turns. A single severely
  broken turn that derails the rest of the conversation should pull the verdict
  down even if other turns are fine.
- Weight LATER turns more than earlier ones when they diverge — the end state
  of the conversation matters most.
- Do NOT reward verbosity or politeness; reward directness and coherence."""

    @staticmethod
    def _few_shot() -> str:
        return """EXAMPLE 1 (fully):
1. User: I need to book a flight from NYC to SF next Friday.
   Assistant: Got it — NYC to SF for Friday. Morning or evening preferred?
2. User: Morning.
   Assistant: Here are three morning options: 7:00, 8:30, 10:15. Want me to hold one?
Verdict: {"verdict": "fully", "reason": "Each reply directly advances the booking task with no redundant questions."}

EXAMPLE 2 (partial):
1. User: My code crashes with a KeyError.
   Assistant: Can you tell me more about your project?
2. User: It's a Python script reading JSON.
   Assistant: What version of Python are you using?
3. User: Just help me fix the KeyError please.
   Assistant: A KeyError means the key isn't in the dict. Check with .get().
Verdict: {"verdict": "partial", "reason": "Assistant asked two unnecessary clarifications before finally addressing the actual error, creating friction."}

EXAMPLE 3 (none):
1. User: Book me a flight to Paris.
   Assistant: What's your favorite color?
2. User: Just book the flight.
   Assistant: Colors can say a lot about a person. Do you like blue?
Verdict: {"verdict": "none", "reason": "Assistant ignores the user's explicit request and derails the conversation with irrelevant tangents."}"""

    # ==================== CORE EVALUATION ====================

    async def _generate_verdict(
        self, dialogue_text: str
    ) -> Tuple[Dict[str, Any], float]:
        prompt = (
            "You are an expert dialogue evaluator. Judge the CONVERSATIONAL FLOW of "
            "the dialogue below — how natural, coherent, and progress-oriented the "
            "assistant's behavior is ACROSS THE WHOLE CONVERSATION.\n\n"
            f"{self._rubric()}\n\n"
            f"{self._few_shot()}\n\n"
            f"DIALOGUE TO EVALUATE:\n{dialogue_text}\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            '{"verdict": "fully|mostly|partial|minor|none", '
            '"key_issues": [string, ...], '
            '"reason": "<one or two sentences explaining the verdict>"}'
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
            data.setdefault("key_issues", [])
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
            VERDICT_WEIGHTS.get(verdict.get("verdict", "none"), 0.0), 4
        )
        success = final_score >= self.threshold

        evaluation_log = {
            "dialogue": dialogue_text,
            "number_of_turns": len(test_case.turns),
            "verdict": verdict,
            "comment_verdict": (
                "Holistic LLM verdict on overall conversational flow, graded "
                "against a few-shot rubric."
            ),
            "key_issues": verdict.get("key_issues", []),
            "final_score": final_score,
            "comment_final_score": (
                "Weight of the holistic verdict label "
                "(fully=1.0, mostly=0.9, partial=0.7, minor=0.3, none=0.0)."
            ),
            "threshold": self.threshold,
            "success": success,
            "final_reason": verdict.get("reason", ""),
        }

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": verdict.get("reason", ""),
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }

        self.print_result(result)
        return result
