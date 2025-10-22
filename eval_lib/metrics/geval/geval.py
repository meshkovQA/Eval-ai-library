# geval.py
"""
G-Eval: LLM-Based NLG Evaluation with Probability-Weighted Scoring
Based on: https://arxiv.org/abs/2303.16634

Core formula: score = Σ p(si) × si
Always uses probability-weighted scoring with n samples at high temperature
"""
import json
import re
from typing import Optional, Dict, Any, List
from collections import Counter
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.llm_client import chat_complete


class GEval(MetricPattern):
    name = "gEval"

    def __init__(
        self,
        model: str,
        threshold: float,
        name: Optional[str] = None,
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        n_samples: int = 20,
        sampling_temperature: float = 2.0,
    ):
        super().__init__(model=model, threshold=threshold)
        self.criteria = criteria
        self.custom_name = name
        self.evaluation_steps = evaluation_steps
        self.n_samples = n_samples
        self.sampling_temperature = sampling_temperature

    # ==================== PROMPTS ====================

    @staticmethod
    def _prompt_generate_steps(criteria: str) -> str:
        """Generate prompt for auto-creating evaluation steps from criteria"""
        return f"""Given the evaluation criteria below, generate 3-5 evaluation steps.

Evaluation Criteria:
{criteria}

**
IMPORTANT: Return ONLY JSON:
{{
  "steps": ["Step 1...", "Step 2...", "Step 3..."]
}}
**

JSON:"""

    @staticmethod
    def _prompt_evaluate(criteria: str, evaluation_steps: List[str], test_case: EvalTestCase) -> str:
        """Generate evaluation prompt (used for probability-weighted scoring)"""
        steps_text = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(evaluation_steps)])

        parts = [f"Input: {test_case.input}",
                 f"Actual Output: {test_case.actual_output}"]

        if test_case.expected_output:
            parts.append(f"Expected Output: {test_case.expected_output}")

        if test_case.retrieval_context:
            parts.append(f"Retrieval Context:\n" +
                         "\n".join(test_case.retrieval_context))

        input_block = "\n\n".join(parts)

        return f"""Evaluation Criteria:
{criteria}

Evaluation Steps:
{steps_text}

{input_block}

Assign a score from 0 to 100 based on the evaluation steps.

**
Return ONLY JSON:
{{
  "score": <number from 0 to 100>
}}
**

JSON:"""

    @staticmethod
    def _prompt_reason(criteria: str, evaluation_steps: List[str], test_case: EvalTestCase, score: float) -> str:
        """Generate prompt for explaining the score"""
        steps_text = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(evaluation_steps)])

        parts = [f"Input: {test_case.input}",
                 f"Actual Output: {test_case.actual_output}"]

        if test_case.expected_output:
            parts.append(f"Expected Output: {test_case.expected_output}")

        if test_case.retrieval_context:
            parts.append(f"Retrieval Context:\n" +
                         "\n".join(test_case.retrieval_context))

        input_block = "\n\n".join(parts)

        return f"""You assigned a score of {score:.1f}/100 for this evaluation.

Evaluation Criteria:
{criteria}

Evaluation Steps:
{steps_text}

{input_block}

Final Score: {score:.1f}/100

Explain why this score was assigned.

**
Return ONLY JSON:
{{
  "reason": "Your explanation..."
}}
**

JSON:"""

    # ==================== HELPER METHODS ====================

    def _extract_score_from_response(self, text: str) -> Optional[int]:
        """Extract integer score from LLM response"""
        text = text.strip()

        # Try pure number first
        if text.isdigit():
            score = int(text)
            if 0 <= score <= 100:
                return score

        # Try JSON parsing
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "score" in data:
                score = int(data["score"])
                if 0 <= score <= 100:
                    return score
        except:
            pass

        # Try regex patterns
        patterns = [
            r'"score"\s*:\s*(\d+)',
            r'score[:\s]+(\d+)',
            r'^\s*(\d+)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 100:
                    return score

        return None

    # ==================== CORE ALGORITHM ====================

    async def _probability_weighted_scoring(self, prompt: str) -> tuple[float, List[int], float]:
        """
        Implement probability-weighted scoring from G-Eval paper.

        Formula: score = Σ p(si) × si

        Samples n times with high temperature to estimate probability distribution,
        then calculates weighted average.

        Returns:
            (final_score, sampled_scores, total_cost)
        """
        total_cost = 0.0
        scores = []

        # Sample n times with high temperature
        for _ in range(self.n_samples):
            text, cost = await chat_complete(
                self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.sampling_temperature
            )
            total_cost += cost or 0.0

            score = self._extract_score_from_response(text)
            if score is not None:
                scores.append(score)

        if not scores:
            raise RuntimeError(
                f"Failed to extract any valid scores from {self.n_samples} samples"
            )

        # Calculate probability-weighted score: Σ p(si) × si
        score_counts = Counter(scores)

        weighted_score = 0.0
        for score_value in range(0, 101):
            count = score_counts.get(score_value, 0)
            probability = count / len(scores)  # p(si)
            weighted_score += probability * score_value  # p(si) × si

        return weighted_score, scores, total_cost

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        """
        Main evaluation method implementing G-Eval algorithm.

        Steps:
        1. Generate evaluation_steps if not provided (auto chain-of-thought)
        2. Generate evaluation prompt
        3. Apply probability-weighted scoring (n samples with high temperature)
        4. Generate reason explanation
        5. Build evaluation_log with all intermediate results
        """
        total_cost = 0.0

        # Step 1: Generate evaluation steps if not provided
        if not self.evaluation_steps:
            if not self.criteria:
                raise ValueError(
                    "Either 'criteria' or 'evaluation_steps' must be provided for G-Eval."
                )

            prompt = self._prompt_generate_steps(self.criteria)
            steps_text, step_cost = await chat_complete(
                self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            total_cost += step_cost or 0.0

            try:
                parsed = json.loads(steps_text)
                self.evaluation_steps = parsed["steps"]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse evaluation steps: {e}\n{steps_text}")

        # Step 2: Generate evaluation prompt
        eval_prompt = self._prompt_evaluate(
            self.criteria, self.evaluation_steps, test_case)

        # Step 3: Apply probability-weighted scoring
        final_score, sampled_scores, scoring_cost = await self._probability_weighted_scoring(eval_prompt)
        total_cost += scoring_cost

        # Step 4: Generate reason explanation
        reason_prompt = self._prompt_reason(
            self.criteria, self.evaluation_steps, test_case, final_score)
        reason_text, reason_cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": reason_prompt}],
            temperature=0.0
        )
        total_cost += reason_cost or 0.0

        # Parse reason
        try:
            reason_data = json.loads(reason_text)
            reason = reason_data.get("reason", reason_text)
        except:
            reason = reason_text.strip()

        success = final_score >= self.threshold

        # Step 5: Build evaluation_log
        evaluation_log = {
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "retrieval_context": test_case.retrieval_context,
            "criteria": self.criteria,
            "comment_criteria": "Evaluation criteria used for assessment.",
            "evaluation_steps": self.evaluation_steps,
            "comment_evaluation_steps": "Auto-generated or provided evaluation steps (chain-of-thought).",
            "sampled_scores": sampled_scores,
            "comment_sampled_scores": f"Individual scores from {len(sampled_scores)} samples with temperature={self.sampling_temperature}.",
            "score_distribution": dict(Counter(sampled_scores)),
            "comment_score_distribution": "Frequency distribution of sampled scores used for probability-weighted calculation.",
            "final_score": round(final_score, 2),
            "comment_final_score": "Probability-weighted score calculated as: Σ p(si) × si",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the final score meets the required threshold.",
            "final_reason": reason,
            "comment_reasoning": "LLM-generated explanation for the assigned score."
        }

        return {
            "score": round(final_score, 2),
            "success": success,
            "reason": reason,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log
        }

    @property
    def name(self):
        return self.custom_name or self.__class__.name
