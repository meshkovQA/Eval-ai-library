# knowledge_retention.py
"""
Knowledge Retention Metric: Evaluates how well the assistant remembers and maintains
context across a multi-turn conversation.

Score calculation: Softmax aggregation of retention verdicts
"""
import json
from typing import List, Dict, Any, Tuple
from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block


# Verdict weights for knowledge retention levels
VERDICT_WEIGHTS = {
    "fully": 1.0,      # Remembers every relevant fact
    "mostly": 0.9,     # Slight omissions, no contradiction
    "partial": 0.7,    # Several lapses but overall context kept
    "minor": 0.3,      # One small lapse or omission
    "none": 0.0        # Assistant contradicts or forgets previous facts
}


class KnowledgeRetentionMetric(ConversationalMetricPattern):
    """
    Evaluates how well an AI assistant retains and recalls information
    shared across multiple conversation turns.
    """

    name = "knowledgeRetentionMetric"

    def __init__(
        self,
        model: str,
        threshold: float = 0.7,
        temperature: float = 0.5,
        verbose: bool = False
    ):
        """
        Initialize Knowledge Retention metric.

        Args:
            model: LLM model name
            threshold: Success threshold (0.0-1.0)
            temperature: Score aggregation temperature for softmax
        """
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature

    # ==================== HELPER METHODS ====================

    @staticmethod
    def _render_dialogue(turns) -> str:
        """Convert conversation turns into readable format"""
        return "\n".join(
            f"{i+1}. User: {t.input}\n   Assistant: {t.actual_output}"
            for i, t in enumerate(turns)
        )

    @staticmethod
    def _prompt_label_help() -> str:
        """Explanation of retention verdict levels"""
        return """Rate knowledge retention (worst → best):

none   – assistant contradicts or forgets previous facts  
minor  – one small lapse or omission  
partial– several lapses but overall context kept  
mostly – slight omissions, no contradiction  
fully  – remembers every relevant fact"""

    @staticmethod
    def _prompt_few_shot() -> str:
        """Few-shot examples for verdict generation"""
        return """Example GOOD:
Conversation:
1. User: What year was Python created?
   Assistant: Python was first released in 1991.
2. User: Remind me, who created it?
   Assistant: It was created by Guido van Rossum in 1991.
Verdicts:
[{"verdict":"fully","reason":"Assistant repeated year and author correctly"}]

Example BAD:
Conversation:
1. User: I live in Spain.
   Assistant: Great! How's the weather?
2. User: Remind me later that I live in Spain.
   Assistant: Sure, I'll remind you that you live in Italy.
Verdicts:
[{"verdict":"none","reason":"Assistant contradicted the country"}]"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _generate_verdicts(
        self,
        dialogue: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        """
        Generate retention verdicts for the conversation.

        Args:
            dialogue: Formatted conversation text

        Returns:
            Tuple of (verdicts_list, aggregated_score, llm_cost)
        """
        prompt = (
            f"{self._prompt_label_help()}\n\n"
            f"{self._prompt_few_shot()}\n\n"
            "Now analyse the next conversation.\n\n"
            f"Conversation:\n{dialogue}\n\n"
            "Return ONE JSON array with 1 object in the form:\n"
            "[{\"verdict\":\"fully|mostly|partial|minor|none\",\"reason\":\"…\"}]"
        )

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        try:
            raw_json = extract_json_block(text)
            verdicts = json.loads(raw_json)

            if not isinstance(verdicts, list):
                raise ValueError("Expected JSON array of verdicts")

            # Calculate aggregated score from verdicts
            weights = [VERDICT_WEIGHTS.get(
                v.get("verdict", "none"), 0.0) for v in verdicts]
            score = round(score_agg(weights, temperature=self.temperature), 4)

            return verdicts, score, cost or 0.0

        except Exception as e:
            raise RuntimeError(f"Failed to parse verdicts: {e}\n{text}")

    async def _summarize_verdicts(
        self,
        verdicts: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        """
        Generate concise summary of retention assessment.

        Args:
            verdicts: List of verdict objects with reasons

        Returns:
            Tuple of (summary_text, llm_cost)
        """
        bullets = "\n".join(f"- {v['reason']}" for v in verdicts)

        prompt = (
            "Write a concise (max 2 sentences) summary of the assistant's knowledge retention, "
            "based on these points:\n\n"
            f"{bullets}\n\n"
            "Summary:"
        )

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        return text.strip(), cost or 0.0

    # ==================== MAIN EVALUATION ====================

    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        """
        Evaluate knowledge retention across conversation turns.

        Steps:
        1. Format dialogue into readable text
        2. Generate retention verdicts (fully/mostly/partial/minor/none)
        3. Aggregate verdicts into final score using softmax
        4. Generate summary explanation
        5. Build comprehensive evaluation log

        Args:
            test_case: Conversational test case with multiple turns

        Returns:
            Evaluation results with score, success, reason, cost, and detailed log
        """
        total_cost = 0.0

        # Step 1: Format dialogue
        dialogue_text = self._render_dialogue(test_case.turns)

        # Step 2: Generate retention verdicts
        verdicts, verdict_score, cost = await self._generate_verdicts(dialogue_text)
        total_cost += cost

        # Step 3: Generate summary explanation
        summary, cost = await self._summarize_verdicts(verdicts)
        total_cost += cost

        # Step 4: Determine success
        final_score = verdict_score
        success = final_score >= self.threshold

        # Step 5: Build evaluation log
        evaluation_log = {
            "dialogue": dialogue_text,
            "comment_dialogue": "Full conversation text used for retention evaluation.",
            "number_of_turns": len(test_case.turns),
            "comment_number_of_turns": "Total conversation turns analyzed.",
            "verdicts": verdicts,
            "comment_verdicts": "LLM-generated verdicts assessing retention level (fully/mostly/partial/minor/none).",
            "verdict_weights": {v["verdict"]: VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in verdicts},
            "comment_verdict_weights": "Numeric weights assigned to each verdict for score calculation.",
            "final_score": final_score,
            "comment_final_score": f"Weighted average of verdict scores using softmax aggregation (temperature={self.temperature}).",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the retention score meets the required threshold.",
            "final_reason": summary,
            "comment_reasoning": "Concise explanation of the assistant's knowledge retention performance."
        }

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log
        }
        self.print_result(result)

        return result
