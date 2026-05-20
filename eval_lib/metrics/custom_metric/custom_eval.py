# custom_eval.py
"""
Custom Evaluation Metric with Verdict-based Scoring
Breaks down evaluation into multiple criteria with individual verdicts
"""
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block


_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


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

    # ==================== TEMPLATE SUBSTITUTION ====================

    @staticmethod
    def _build_substitution_context(test_case: EvalTestCase) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Collect available values + variable->column aliases from the test case.

        Sources (in priority order, later overrides earlier):
          - test_case._meta["dataset_row"]  (raw dataset row from connector)
          - built-in fields from EvalTestCase (input, actual_output, expected_output, retrieval_context)
          - test_case._meta["system_prompt"]

        Returns (values_by_name, variable_map). variable_map allows {{var}} in templates
        to alias dataset columns (mirrors connector substitute_template behavior).
        """
        meta = getattr(test_case, "_meta", None) or {}
        values: Dict[str, Any] = {}

        row = meta.get("dataset_row") or {}
        if isinstance(row, dict):
            values.update(row)

        values["input"] = test_case.input
        values["actual_output"] = test_case.actual_output
        if test_case.expected_output is not None:
            values["expected_output"] = test_case.expected_output
        if test_case.retrieval_context:
            values["retrieval_context"] = "\n".join(test_case.retrieval_context)

        sys_prompt = meta.get("system_prompt")
        if sys_prompt:
            values["system_prompt"] = sys_prompt

        variable_map = meta.get("template_variable_map") or {}
        if not isinstance(variable_map, dict):
            variable_map = {}

        return values, variable_map

    @staticmethod
    def _substitute(text: str, values: Dict[str, Any], variable_map: Dict[str, str]) -> str:
        """Replace {{name}} placeholders. Resolves name via variable_map alias first."""
        if not text or "{{" not in text:
            return text

        def replacer(match: "re.Match[str]") -> str:
            var_name = match.group(1)
            resolved = variable_map.get(var_name, var_name)
            value = values.get(resolved)
            if value is None and resolved != var_name:
                value = values.get(var_name)
            if value is None:
                return match.group(0)
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        return _PLACEHOLDER_RE.sub(replacer, text)

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

        # Step 0: Resolve {{placeholders}} in criteria / evaluation_steps from
        # the current test case (dataset row + system_prompt + built-in fields).
        # We work with local copies so the metric instance remains reusable across
        # rows where placeholders resolve to different values.
        values, variable_map = self._build_substitution_context(test_case)
        rendered_criteria = self._substitute(self.criteria, values, variable_map)

        # Step 1: Generate evaluation steps if not provided. We use the rendered
        # criteria for generation, and the user-provided steps (if any) are
        # rendered per-row so {{column}} works inside each step too.
        if not self.evaluation_steps:
            evaluation_steps, cost = await self._generate_evaluation_steps(
                rendered_criteria
            )
            total_cost += cost
        else:
            evaluation_steps = [
                self._substitute(step, values, variable_map)
                for step in self.evaluation_steps
            ]

        # Step 2: Generate verdicts for each criterion
        verdicts, final_score, cost = await self._generate_verdicts(
            rendered_criteria,
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
            "main_criteria": rendered_criteria,
            "comment_main_criteria": "High-level evaluation criteria provided by user (with {{placeholders}} resolved from the current test case).",
            "main_criteria_template": self.criteria,
            "comment_main_criteria_template": "Original criteria template before placeholder substitution.",
            "evaluation_criteria": evaluation_steps,
            "comment_evaluation_criteria": f"Specific sub-criteria ({len(evaluation_steps)} items) used for verdict-based evaluation (placeholders resolved per row).",
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
            "temperature": self.temperature,
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
