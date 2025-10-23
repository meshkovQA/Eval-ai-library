# task_success_rate.py
"""
Task Success Rate Metric: Evaluates whether the AI assistant successfully helped
the user achieve their goal in a conversation.

Score calculation: Softmax aggregation of success criteria verdicts
"""
import json
from typing import List, Dict, Any, Tuple
from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block
import re


def _contains_links(dialogue: str) -> bool:
    """Check if dialogue contains any URLs/links"""
    url_pattern = r'https?://[^\s]+|www\.[^\s]+|\[.*?\]\(.*?\)'
    return bool(re.search(url_pattern, dialogue))


# Verdict weights for task completion levels
VERDICT_WEIGHTS = {
    "fully": 1.0,      # Criterion completely satisfied
    "mostly": 0.9,     # Criterion largely satisfied with minor gaps
    "partial": 0.7,    # Criterion partially satisfied
    "minor": 0.3,      # Criterion minimally addressed
    "none": 0.0        # Criterion not satisfied at all
}

# Configuration constants
MAX_CRITERIA = 2
LINK_CRITERION = "The user got the link to the requested resource."


class TaskSuccessRateMetric(ConversationalMetricPattern):
    """
    Evaluates whether an AI assistant successfully helped the user complete
    their intended task across a multi-turn conversation.
    """

    name = "taskSuccessRateMetric"

    def __init__(
        self,
        model: str,
        threshold: float = 0.7,
        temperature: float = 0.5,
        verbose: bool = False
    ):
        """
        Initialize Task Success Rate metric.

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
        """Explanation of task success verdict levels"""
        return """Rate task success criteria satisfaction (worst → best):

none    – criterion not satisfied at all
minor   – criterion minimally addressed
partial – criterion partially satisfied
mostly  – criterion largely satisfied with minor gaps
fully   – criterion completely satisfied"""

    @staticmethod
    def _prompt_criteria_few_shot() -> str:
        """Few-shot examples for criteria generation"""
        return """Example 1:
User goal: Order a pizza online
Criteria: [
  "The assistant provided available pizza options.",
  "The user received an order confirmation number."
]

Example 2:
User goal: Reset an email password
Criteria: [
  "The assistant gave a working password-reset link.",
  "The user confirmed they could log in."
]"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _infer_user_goal(self, dialogue: str) -> Tuple[str, float]:
        """
        Infer the user's primary goal from the conversation.

        Args:
            dialogue: Formatted conversation text

        Returns:
            Tuple of (user_goal_description, llm_cost)
        """
        prompt = (
            "You will be shown an ENTIRE conversation between a user and an assistant.\n"
            "Write ONE concise sentence describing the user's PRIMARY GOAL in this conversation.\n\n"
            f"CONVERSATION:\n{dialogue}\n\n"
            "User goal:"
        )

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        return text.strip(), cost or 0.0

    async def _generate_success_criteria(self, goal: str, dialogue: str) -> Tuple[List[str], float]:
        """
        Generate concrete success criteria for the user's goal.

        Args:
            goal: The inferred user goal
            dialogue: Full conversation text (needed to check for links)
        """
        prompt = (
            f"{self._prompt_criteria_few_shot()}\n\n"
            f"Now do the same for the next case.\n\n"
            f"User goal: {goal}\n\n"
            f"List up to {MAX_CRITERIA} concrete SUCCESS CRITERIA that could realistically be satisfied "
            f"within a brief chat of 2–5 turns.\n\n"
            "Each criterion must be a short, observable statement.\n"
            "Return only a JSON array of strings."
        )

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        try:
            raw_json = extract_json_block(text)
            criteria = json.loads(raw_json)

            if not isinstance(criteria, list):
                raise ValueError("Expected JSON array of criteria")

            # Add LINK_CRITERION only if dialogue contains links
            if _contains_links(dialogue) and LINK_CRITERION not in criteria:
                criteria.append(LINK_CRITERION)

            # Truncate to MAX_CRITERIA
            criteria = criteria[:MAX_CRITERIA]

            return criteria, cost or 0.0

        except Exception as e:
            raise RuntimeError(
                f"Failed to parse success criteria: {e}\n{text}")

    async def _generate_verdicts(
        self,
        goal: str,
        criteria: List[str],
        dialogue: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        """
        Generate verdicts for each success criterion.

        Args:
            goal: The user's goal
            criteria: List of success criteria
            dialogue: Formatted conversation text

        Returns:
            Tuple of (verdicts_list, aggregated_score, llm_cost)
        """
        prompt = (
            f"{self._prompt_label_help()}\n\n"
            f"USER GOAL: {goal}\n\n"
            f"FULL DIALOGUE:\n{dialogue}\n\n"
            f"SUCCESS CRITERIA (as JSON array):\n{json.dumps(criteria, ensure_ascii=False)}\n\n"
            "For **each** criterion, decide how well it is satisfied at the END of the dialogue.\n"
            "Use exactly one of: fully, mostly, partial, minor, none.\n\n"
            "Return JSON array with **exactly the same length and order** as the criteria list:\n"
            "[{\"verdict\":\"fully|mostly|partial|minor|none\",\"reason\":\"<one sentence>\"}, …]\n\n"
            "No extra text."
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

            # Ensure verdicts match criteria length
            if len(verdicts) != len(criteria):
                # Pad or truncate to match
                if len(verdicts) < len(criteria):
                    verdicts.extend(
                        [{"verdict": "none", "reason": "Missing evaluation"}] * (len(criteria) - len(verdicts)))
                else:
                    verdicts = verdicts[:len(criteria)]

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
        Generate concise summary of task success assessment.

        Args:
            verdicts: List of verdict objects with reasons

        Returns:
            Tuple of (summary_text, llm_cost)
        """
        # Take up to 6 most relevant verdicts for summary
        bullets = "\n".join(f"- {v['reason']}" for v in verdicts[:6])

        prompt = (
            "Write a concise (max 2 sentences) overall assessment of task success, "
            "based on these observations:\n\n"
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
        Evaluate task success rate across conversation turns.

        Steps:
        1. Format dialogue into readable text
        2. Infer user's primary goal from conversation
        3. Generate concrete success criteria for the goal
        4. Generate verdicts for each criterion (fully/mostly/partial/minor/none)
        5. Aggregate verdicts into final score using softmax
        6. Generate summary explanation
        7. Build comprehensive evaluation log

        Args:
            test_case: Conversational test case with multiple turns

        Returns:
            Evaluation results with score, success, reason, cost, and detailed log
        """
        total_cost = 0.0

        # Step 1: Format dialogue
        dialogue_text = self._render_dialogue(test_case.turns)

        # Step 2: Infer user goal
        user_goal, cost = await self._infer_user_goal(dialogue_text)
        total_cost += cost

        # Step 3: Generate success criteria
        success_criteria, cost = await self._generate_success_criteria(user_goal, dialogue_text)
        total_cost += cost

        # Step 4: Generate verdicts for each criterion
        verdicts, verdict_score, cost = await self._generate_verdicts(
            user_goal,
            success_criteria,
            dialogue_text
        )
        total_cost += cost

        # Step 5: Generate summary explanation
        summary, cost = await self._summarize_verdicts(verdicts)
        total_cost += cost

        # Step 6: Determine success
        final_score = verdict_score
        success = final_score >= self.threshold

        # Step 7: Build evaluation log
        evaluation_log = {
            "dialogue": dialogue_text,
            "comment_dialogue": "Full conversation text used for task success evaluation.",
            "number_of_turns": len(test_case.turns),
            "comment_number_of_turns": "Total conversation turns analyzed.",
            "user_goal": user_goal,
            "comment_user_goal": "LLM-inferred primary goal the user wanted to achieve.",
            "success_criteria": success_criteria,
            "comment_success_criteria": f"Auto-generated checklist of {len(success_criteria)} observable criteria for task completion.",
            "verdicts": verdicts,
            "comment_verdicts": "LLM-generated verdicts assessing each criterion (fully/mostly/partial/minor/none).",
            "verdict_weights": {i: VERDICT_WEIGHTS.get(v["verdict"], 0.0) for i, v in enumerate(verdicts)},
            "comment_verdict_weights": "Numeric weights assigned to each verdict for score calculation.",
            "final_score": final_score,
            "comment_final_score": f"Weighted average of verdict scores using softmax aggregation (temperature={self.temperature}).",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the task success score meets the required threshold.",
            "final_reason": summary,
            "comment_reasoning": "Concise explanation of the overall task completion assessment."
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
