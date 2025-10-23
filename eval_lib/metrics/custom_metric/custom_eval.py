# custom_eval.py
"""
Custom Evaluation Metric with Verdict-based Scoring
Breaks down evaluation into multiple criteria with individual verdicts
"""
import json
from typing import Dict, Any, List, Tuple
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block


# Verdict weights for scoring
VERDICT_WEIGHTS = {
    "fully": 1.0,      # Criterion fully satisfied
    "mostly": 0.9,     # Criterion largely satisfied with minor gaps
    "partial": 0.7,    # Criterion partially satisfied
    "minor": 0.3,      # Criterion minimally addressed
    "none": 0.0        # Criterion not satisfied at all
}


class CustomEvalMetric(MetricPattern):
    """
    Custom evaluation metric with verdict-based scoring.
    Allows defining custom criteria and evaluates each one separately.
    """

    name = "customEval"

    def __init__(
        self,
        model: str,
        threshold: float,
        name: str,
        criteria: str,
        evaluation_steps: List[str] = None,
        temperature: float = 0.8,
        verbose: bool = False
    ):
        """
        Initialize Custom Evaluation Metric.

        Args:
            model: LLM model name
            threshold: Success threshold (0.0-1.0)
            name: Custom metric name
            criteria: High-level evaluation criteria description
            evaluation_steps: List of specific criteria to evaluate (auto-generated if None)
            temperature: Score aggregation temperature for softmax
            verbose: Enable detailed logging
        """
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.custom_name = name
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.temperature = temperature

    # ==================== PROMPTS ====================

    @staticmethod
    def _prompt_label_help() -> str:
        """Explanation of verdict levels"""
        return """Rate how well each criterion is satisfied (worst → best):

none    – criterion not satisfied at all
minor   – criterion minimally addressed
partial – criterion partially satisfied
mostly  – criterion largely satisfied with minor gaps
fully   – criterion fully satisfied"""

    @staticmethod
    def _prompt_generate_criteria(main_criteria: str) -> str:
        """Generate specific evaluation criteria from high-level description"""
        return f"""Given the high-level evaluation criteria below, generate 3-5 specific, measurable sub-criteria.

High-level Criteria:
{main_criteria}

Generate sub-criteria that are:
1. Specific and observable
2. Can be evaluated independently
3. Together cover all aspects of the main criteria

**
Return ONLY JSON:
{{
  "criteria": ["Criterion 1: ...", "Criterion 2: ...", "Criterion 3: ..."]
}}
**

JSON:"""

    @classmethod
    def _prompt_evaluate(
        cls,
        main_criteria: str,
        evaluation_steps: List[str],
        test_case: EvalTestCase
    ) -> str:
        """Generate evaluation prompt with verdict scoring"""

        # Build input block
        parts = [f"User Input:\n{test_case.input}"]
        parts.append(f"Model Output:\n{test_case.actual_output}")

        if test_case.expected_output:
            parts.append(f"Expected Output:\n{test_case.expected_output}")

        if test_case.retrieval_context:
            context_text = "\n".join(test_case.retrieval_context)
            parts.append(f"Context:\n{context_text}")

        input_block = "\n\n".join(parts)

        # Format criteria
        criteria_text = "\n".join(
            [f"{i+1}. {criterion}" for i,
                criterion in enumerate(evaluation_steps)]
        )

        return f"""{cls._prompt_label_help()}

HIGH-LEVEL CRITERIA:
{main_criteria}

SPECIFIC CRITERIA TO EVALUATE:
{criteria_text}

{input_block}

Task: For EACH criterion, decide how well it is satisfied in the Model Output.
Use exactly one of: fully, mostly, partial, minor, none.

**
Return JSON array with exactly {len(evaluation_steps)} verdicts:
[
  {{"verdict": "fully|mostly|partial|minor|none", "reason": "<one sentence>"}},
  ...
]
**

JSON:"""

    # ==================== CORE EVALUATION ====================

    async def _generate_evaluation_steps(self, main_criteria: str) -> Tuple[List[str], float]:
        """
        Auto-generate specific evaluation criteria from high-level description.

        Args:
            main_criteria: High-level evaluation criteria

        Returns:
            Tuple of (criteria_list, llm_cost)
        """
        prompt = self._prompt_generate_criteria(main_criteria)

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        try:
            raw_json = extract_json_block(text)
            data = json.loads(raw_json)
            criteria = data.get("criteria", [])

            if not isinstance(criteria, list) or len(criteria) == 0:
                raise ValueError("Expected non-empty list of criteria")

            return criteria, cost or 0.0

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate evaluation criteria: {e}\n{text}"
            )

    async def _generate_verdicts(
        self,
        main_criteria: str,
        evaluation_steps: List[str],
        test_case: EvalTestCase
    ) -> Tuple[List[Dict[str, str]], float, float]:
        """
        Generate verdicts for each evaluation criterion.

        Args:
            main_criteria: High-level criteria description
            evaluation_steps: List of specific criteria
            test_case: Test case to evaluate

        Returns:
            Tuple of (verdicts_list, aggregated_score, llm_cost)
        """
        prompt = self._prompt_evaluate(
            main_criteria, evaluation_steps, test_case)

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
            if len(verdicts) != len(evaluation_steps):
                if len(verdicts) < len(evaluation_steps):
                    # Pad with "none" verdicts
                    verdicts.extend([
                        {"verdict": "none", "reason": "Missing evaluation"}
                    ] * (len(evaluation_steps) - len(verdicts)))
                else:
                    # Truncate
                    verdicts = verdicts[:len(evaluation_steps)]

            # Calculate aggregated score
            weights = [
                VERDICT_WEIGHTS.get(v.get("verdict", "none"), 0.0)
                for v in verdicts
            ]
            score = round(score_agg(weights, temperature=self.temperature), 4)

            return verdicts, score, cost or 0.0

        except Exception as e:
            raise RuntimeError(
                f"Failed to parse verdicts: {e}\n{text}"
            )

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        """
        Evaluate using custom criteria with verdict-based scoring.

        Steps:
        1. Auto-generate specific criteria if not provided (1 LLM call)
        2. Generate verdicts for each criterion (1 LLM call)
        3. Aggregate verdicts into final score using softmax
        4. Build evaluation log

        Args:
            test_case: Test case to evaluate

        Returns:
            Evaluation results with score, success, reason, cost, and detailed log
        """
        total_cost = 0.0

        # Step 1: Generate evaluation steps if not provided
        if not self.evaluation_steps:
            evaluation_steps, cost = await self._generate_evaluation_steps(
                self.criteria
            )
            total_cost += cost
            self.evaluation_steps = evaluation_steps
        else:
            evaluation_steps = self.evaluation_steps

        # Step 2: Generate verdicts for each criterion
        verdicts, final_score, cost = await self._generate_verdicts(
            self.criteria,
            evaluation_steps,
            test_case
        )
        total_cost += cost

        # Step 3: Determine success
        success = final_score >= self.threshold

        # Step 4: Build summary reason from verdicts
        positive_verdicts = [
            v for v in verdicts
            if v.get("verdict") in ["fully", "mostly"]
        ]
        negative_verdicts = [
            v for v in verdicts
            if v.get("verdict") in ["none", "minor", "partial"]
        ]

        if len(positive_verdicts) >= len(verdicts) * 0.7:
            summary = f"Strong performance: {len(positive_verdicts)}/{len(verdicts)} criteria fully or mostly satisfied."
        elif len(negative_verdicts) >= len(verdicts) * 0.7:
            summary = f"Weak performance: {len(negative_verdicts)}/{len(verdicts)} criteria not satisfied or minimally addressed."
        else:
            summary = f"Mixed performance: {len(positive_verdicts)}/{len(verdicts)} criteria satisfied, with room for improvement."

        # Step 5: Build evaluation log
        evaluation_log = {
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "retrieval_context": test_case.retrieval_context,
            "main_criteria": self.criteria,
            "comment_main_criteria": "High-level evaluation criteria provided by user.",
            "evaluation_criteria": evaluation_steps,
            "comment_evaluation_criteria": f"Specific sub-criteria ({len(evaluation_steps)} items) used for verdict-based evaluation.",
            "verdicts": verdicts,
            "comment_verdicts": "Individual verdicts for each criterion (fully/mostly/partial/minor/none).",
            "verdict_weights": {
                i: VERDICT_WEIGHTS.get(v["verdict"], 0.0)
                for i, v in enumerate(verdicts)
            },
            "comment_verdict_weights": "Numeric weights assigned to each verdict for score calculation.",
            "final_score": final_score,
            "comment_final_score": f"Weighted average of verdict scores calculated using softmax aggregation (temperature={self.temperature}).",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the final score meets the required threshold.",
            "summary": summary,
            "comment_summary": "High-level summary of evaluation performance."
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

    @property
    def name(self):
        return f"Custom: {self.custom_name}"
