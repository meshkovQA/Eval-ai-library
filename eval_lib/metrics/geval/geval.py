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
    template_cls = None

    def __init__(
        self,
        model: str,
        threshold: float,
        verbose: bool = False,
        name: Optional[str] = None,
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        n_samples: int = 20,
        sampling_temperature: float = 2.0,
    ):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.criteria = criteria
        self.custom_name = name
        self.evaluation_steps = evaluation_steps
        self.n_samples = n_samples
        self.sampling_temperature = sampling_temperature

    # ==================== PROMPTS ====================

    @staticmethod
    def _prompt_generate_steps(criteria: str) -> str:
        """Generate evaluation steps from criteria (Chain-of-Thought)"""
        return f"""Given the evaluation criteria below, generate 3-5 detailed evaluation steps.

Evaluation Criteria:
{criteria}

Generate steps that are:
1. Specific and actionable
2. Logically ordered
3. Lead to assigning a score from 0.0 to 1.0

**
Return ONLY JSON:
{{
  "steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."]
}}
**

JSON:"""

    @staticmethod
    def _prompt_evaluate(criteria: str, evaluation_steps: List[str], test_case: EvalTestCase) -> str:
        """Generate evaluation prompt with CoT steps"""
        steps_text = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(evaluation_steps)])

        parts = [
            f"User Input:\n{test_case.input}",
            f"Model Output:\n{test_case.actual_output}"
        ]

        if test_case.expected_output:
            parts.append(f"Expected Output:\n{test_case.expected_output}")

        if test_case.retrieval_context:
            parts.append(f"Context:\n" +
                         "\n".join(test_case.retrieval_context))

        input_block = "\n\n".join(parts)

        return f"""You are a strict evaluator. Use the criteria and evaluation steps below.

Evaluation Criteria:
{criteria}

Evaluation Steps:
{steps_text}

{input_block}

Based on the evaluation steps, assign a score from 0.0 to 1.0 (where 0.0 is worst and 1.0 is best).

**
Return ONLY JSON:
{{
  "score": <float between 0.0 and 1.0>
}}
**

JSON:"""

    @staticmethod
    def _prompt_reason(
        criteria: str,
        evaluation_steps: List[str],
        test_case: EvalTestCase,
        score: float
    ) -> str:
        """Generate explanation for the score"""
        steps_text = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(evaluation_steps)])

        parts = [
            f"User Input:\n{test_case.input}",
            f"Model Output:\n{test_case.actual_output}"
        ]

        if test_case.expected_output:
            parts.append(f"Expected Output:\n{test_case.expected_output}")

        if test_case.retrieval_context:
            parts.append(f"Context:\n" +
                         "\n".join(test_case.retrieval_context))

        input_block = "\n\n".join(parts)

        return f"""You assigned a score of {score:.2f} (out of 1.0) for this evaluation.

Evaluation Criteria:
{criteria}

Evaluation Steps:
{steps_text}

{input_block}

Final Score: {score:.2f}/1.0

Explain why this score was assigned, referencing specific aspects from the evaluation steps.

**
Return ONLY JSON:
{{
  "reason": "Your explanation..."
}}
**

JSON:"""

    # ==================== HELPER METHODS ====================

    def _extract_score_from_response(self, text: str) -> Optional[float]:
        """Extract float score from LLM response (0.0-1.0 range)"""
        text = text.strip()

        # Try JSON parsing first
        try:
            data = json.loads(text)
            if "score" in data:
                score = float(data["score"])
                if 0.0 <= score <= 1.0:
                    return score
        except:
            pass

        # Try regex patterns
        patterns = [
            r'"score"\s*:\s*(\d+\.?\d*)',
            r'score[:\s]+(\d+\.?\d*)',
            r'^\s*(\d+\.?\d*)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score

        return None

    # ==================== CORE ALGORITHM ====================

    async def _probability_weighted_scoring(
        self,
        prompt: str,
        n_samples: int = 20,
        temperature: float = 2.0
    ) -> tuple[float, List[float], float]:
        """
        Probability-weighted scoring: score = Σ p(si) × si
        Samples multiple times to estimate probability distribution

        Returns:
            (final_score, sampled_scores, total_cost)
        """
        total_cost = 0.0
        scores = []

        # Sample n times with high temperature
        for _ in range(n_samples):
            text, cost = await chat_complete(
                self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            total_cost += cost or 0.0

            try:
                score = self._extract_score_from_response(text)
                if score is not None:
                    scores.append(score)
            except:
                continue

        if not scores:
            raise RuntimeError(
                f"Failed to extract any valid scores from {n_samples} samples")

        # Calculate probability-weighted score: Σ p(si) × si
        # For continuous scores, we use the mean as an approximation
        weighted_score = sum(scores) / len(scores)

        return weighted_score, scores, total_cost

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        """
        Evaluate using Chain-of-Thought and Probability-Weighted Scoring.

        Algorithm:
        1. Auto-generate evaluation steps from criteria (CoT)
        2. Apply probability-weighted scoring (20 samples, temp=2.0)
        3. Generate detailed explanation
        4. Build comprehensive evaluation_log
        """
        total_cost = 0.0

        # Step 1: Auto-generate evaluation steps (Chain-of-Thought from G-Eval)
        if not self.evaluation_steps:
            if not self.criteria:
                raise ValueError(
                    "Either 'criteria' or 'evaluation_steps' must be provided for G-Eval."
                )

            steps_prompt = self._prompt_generate_steps(self.criteria)
            steps_text, step_cost = await chat_complete(
                self.model,
                messages=[{"role": "user", "content": steps_prompt}],
                temperature=0.0
            )
            total_cost += step_cost or 0.0

            try:
                parsed_steps = json.loads(steps_text)
                self.evaluation_steps = parsed_steps["steps"]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse evaluation steps: {e}\n{steps_text}")

        # Step 2: Generate evaluation prompt with CoT
        eval_prompt = self._prompt_evaluate(
            self.criteria, self.evaluation_steps, test_case)

        # Step 3: Probability-weighted scoring (20 samples from G-Eval)
        final_score, sampled_scores, scoring_cost = await self._probability_weighted_scoring(
            eval_prompt,
            n_samples=self.n_samples,
            temperature=self.sampling_temperature
        )
        total_cost += scoring_cost

        # Step 4: Generate explanation
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

        # Step 5: Build comprehensive evaluation_log
        evaluation_log = {
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "retrieval_context": test_case.retrieval_context,
            "criteria": self.criteria,
            "comment_criteria": "Custom evaluation criteria provided by user.",
            "evaluation_steps": self.evaluation_steps,
            "comment_evaluation_steps": "Auto-generated evaluation steps using Chain-of-Thought (CoT) technique from G-Eval.",
            "sampled_scores": sampled_scores,
            "comment_sampled_scores": f"Individual scores from {len(sampled_scores)} samples with temperature={self.sampling_temperature}.",
            "score_distribution": {f"{s:.2f}": sampled_scores.count(s) for s in set(sampled_scores)},
            "comment_score_distribution": "Frequency distribution of sampled scores for probability-weighted calculation.",
            "final_score": round(final_score, 4),
            "comment_final_score": "Probability-weighted score calculated as mean of sampled scores (G-Eval technique).",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the final score passes the threshold.",
            "final_reason": reason,
            "comment_reasoning": "LLM-generated explanation based on evaluation steps and criteria."
        }

        result = {
            "name": self.name,
            "score": round(final_score, 4),
            "success": success,
            "reason": reason,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }

        self.print_result(result)

        return result

    @property
    def name(self):
        return self.custom_name or self.__class__.name
