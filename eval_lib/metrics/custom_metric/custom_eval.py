# custom_eval.py
"""
Custom Evaluation Metric with Verdict-based Scoring.

Model:
    - The user supplies one or more `evaluation_criteria` (free-text strings).
    - Each criterion references the data it cares about via {{placeholders}}
      (e.g. "Each item in {{follow_up_questions}} is relevant to {{input}}").
    - The metric builds a single prompt with two blocks:
          DATA:      every placeholder that appears in any criterion, resolved
                     from the EvalTestCase + extra_fields + connector metadata.
          CRITERIA:  the original criterion strings, {{placeholders}} preserved.
    - The judge LLM returns one verdict per criterion; verdicts are aggregated
      into a final score via TCVA (Temperature-Controlled Verdict Aggregation).

Filtering rules (logged as warnings in evaluation_log):
    - A criterion with no {{placeholders}} at all is skipped — there is no data
      to evaluate against.
    - A criterion that references an unknown placeholder is skipped — the judge
      would otherwise see a dangling "{{x}}" with no value.
    - If every criterion gets filtered out the metric returns score=0 with a
      clear reason.
"""
import json
import re
from typing import Dict, Any, List, Tuple
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block


_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


# Verdict weights for scoring.
VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}


class CustomEvalMetric(MetricPattern):
    """Verdict-based custom evaluation against user-defined criteria.

    See module docstring for the prompt model. The metric is reusable across
    test cases — per-row data is resolved fresh every `evaluate()` call.
    """

    name = "customEval"

    def __init__(
        self,
        model: str,
        threshold: float,
        name: str,
        evaluation_criteria: List[str],
        temperature: float = 0.8,
        max_evaluation_criteria: int = 8,
        verbose: bool = False,
    ):
        """
        Args:
            model: LLM model id (e.g. "openai:gpt-4o-mini").
            threshold: Pass/fail threshold for the aggregated score (0.0-1.0).
            name: Human-readable metric name; surfaced as "Custom: <name>".
            evaluation_criteria: Non-empty list of criteria. Each item must
                reference data via {{placeholders}}; criteria without any
                placeholder are skipped at evaluate-time.
            temperature: TCVA aggregation temperature (NOT LLM sampling — judge
                calls always run at temperature=0). Low (~0.1) ≈ strict (min),
                0.5 ≈ arithmetic mean, high (~1.5) ≈ lenient (max). Default 0.8.
            max_evaluation_criteria: Hard cap on the number of criteria. Excess
                items are truncated; a warning is recorded in evaluation_log.
                Matches the convention used by faithfulness/answer_relevancy
                (default 8).
            verbose: Enable per-result console logging.
        """
        super().__init__(model=model, threshold=threshold, verbose=verbose)

        if not isinstance(evaluation_criteria, list) or len(evaluation_criteria) == 0:
            raise ValueError(
                "CustomEvalMetric requires a non-empty list of evaluation_criteria."
            )
        cleaned = [str(c).strip() for c in evaluation_criteria if str(c).strip()]
        if not cleaned:
            raise ValueError(
                "CustomEvalMetric: evaluation_criteria contained only empty/whitespace items."
            )

        self.custom_name = name
        self.evaluation_criteria = cleaned
        self.temperature = temperature
        self.max_evaluation_criteria = max(1, int(max_evaluation_criteria))

    # ==================== TEMPLATE CONTEXT ====================

    @staticmethod
    def _build_substitution_context(test_case: EvalTestCase) -> Dict[str, Any]:
        """Collect every value addressable from the test case.

        Priority (later wins):
          1. _meta["dataset_row"]  (raw dataset row from the connector)
          2. extra_fields          (user-provided ad-hoc fields)
          3. built-in EvalTestCase fields
          4. _meta["system_prompt"]

        Connector aliases (`template_variable_map`) are expanded inline so the
        caller only needs to look up by the placeholder name as written.
        """
        meta = getattr(test_case, "_meta", None) or {}
        values: Dict[str, Any] = {}

        row = meta.get("dataset_row") or {}
        if isinstance(row, dict):
            values.update(row)

        if test_case.extra_fields:
            values.update(test_case.extra_fields)

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
        if isinstance(variable_map, dict):
            # Aliases: register the alias name -> resolved value so a criterion
            # written as {{q}} works the same as {{question}} when the
            # connector maps q -> question.
            for alias, real_name in variable_map.items():
                if real_name in values and alias not in values:
                    values[alias] = values[real_name]

        return values

    @staticmethod
    def _extract_placeholders(text: str) -> List[str]:
        """Return placeholder names in order of first appearance, deduped."""
        seen: Dict[str, None] = {}
        for name in _PLACEHOLDER_RE.findall(text or ""):
            if name not in seen:
                seen[name] = None
        return list(seen.keys())

    # ==================== FILTERING ====================

    @classmethod
    def _filter_criteria(
        cls,
        criteria: List[str],
        values: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Return (kept_criteria, skipped_info).

        A criterion is kept only if it has at least one placeholder AND every
        placeholder it references is present in `values`.
        """
        kept: List[str] = []
        skipped: List[Dict[str, Any]] = []
        for c in criteria:
            placeholders = cls._extract_placeholders(c)
            if not placeholders:
                skipped.append({
                    "criterion": c,
                    "reason": "no_placeholders",
                    "detail": "Criterion contains no {{placeholders}}; nothing to evaluate against.",
                })
                continue
            missing = [p for p in placeholders if p not in values]
            if missing:
                skipped.append({
                    "criterion": c,
                    "reason": "unknown_placeholders",
                    "detail": f"Criterion references unknown placeholder(s): {missing}",
                    "missing": missing,
                })
                continue
            kept.append(c)
        return kept, skipped

    @staticmethod
    def _collect_used_placeholders(criteria: List[str]) -> List[str]:
        """Union of all placeholders mentioned across kept criteria, in order."""
        seen: Dict[str, None] = {}
        for c in criteria:
            for p in _PLACEHOLDER_RE.findall(c):
                if p not in seen:
                    seen[p] = None
        return list(seen.keys())

    # ==================== PROMPT ====================

    @staticmethod
    def _format_value(value: Any) -> str:
        """How a single DATA-block value is rendered for the judge."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    @staticmethod
    def _prompt_label_help() -> str:
        return """Rate how well each criterion is satisfied (worst → best):

none    – criterion not satisfied at all
minor   – criterion minimally addressed
partial – criterion partially satisfied
mostly  – criterion largely satisfied with minor gaps
fully   – criterion fully satisfied"""

    @classmethod
    def _build_data_block(
        cls,
        used_names: List[str],
        values: Dict[str, Any],
    ) -> str:
        """Render the DATA: block listing only placeholders actually referenced."""
        lines = []
        for name in used_names:
            rendered = cls._format_value(values[name])
            # Multi-line values get their own line for readability.
            if "\n" in rendered:
                lines.append(f"- {name}:\n{rendered}")
            else:
                lines.append(f"- {name}: {rendered}")
        return "\n".join(lines)

    @classmethod
    def _prompt_evaluate(
        cls,
        kept_criteria: List[str],
        used_names: List[str],
        values: Dict[str, Any],
    ) -> str:
        data_block = cls._build_data_block(used_names, values)
        criteria_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(kept_criteria)
        )
        return f"""{cls._prompt_label_help()}

DATA:
{data_block}

EVALUATION CRITERIA:
{criteria_text}

Task: For EACH criterion above, decide how well it is satisfied by the DATA.
The {{{{placeholders}}}} inside each criterion refer to entries in the DATA block.
Use exactly one of: fully, mostly, partial, minor, none.

**
Return JSON array with exactly {len(kept_criteria)} verdicts:
[
  {{"verdict": "fully|mostly|partial|minor|none", "reason": "<one sentence>"}},
  ...
]
**

JSON:"""

    # ==================== CORE EVALUATION ====================

    async def _generate_verdicts(
        self,
        kept_criteria: List[str],
        used_names: List[str],
        values: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], float, float]:
        prompt = self._prompt_evaluate(kept_criteria, used_names, values)

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        try:
            raw_json = extract_json_block(text)
            verdicts = json.loads(raw_json)
            if not isinstance(verdicts, list):
                raise ValueError("Expected JSON array of verdicts")
        except Exception as e:
            raise RuntimeError(f"Failed to parse verdicts: {e}\n{text}")

        # Reconcile length mismatches: pad with "none" or truncate.
        if len(verdicts) < len(kept_criteria):
            verdicts.extend(
                [{"verdict": "none", "reason": "Missing evaluation"}]
                * (len(kept_criteria) - len(verdicts))
            )
        elif len(verdicts) > len(kept_criteria):
            verdicts = verdicts[: len(kept_criteria)]

        weights = [
            VERDICT_WEIGHTS.get(v.get("verdict", "none"), 0.0)
            for v in verdicts
        ]
        score = round(score_agg(weights, temperature=self.temperature), 4)
        return verdicts, score, cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        # Step 1: resolve all addressable data for this test case.
        values = self._build_substitution_context(test_case)

        # Step 2: hard cap before any LLM work. Truncating later would still
        # cost a verdict call for the dropped criteria.
        truncation_warning = None
        criteria = list(self.evaluation_criteria)
        if len(criteria) > self.max_evaluation_criteria:
            truncation_warning = (
                f"evaluation_criteria had {len(criteria)} items; "
                f"truncated to max_evaluation_criteria={self.max_evaluation_criteria}."
            )
            criteria = criteria[: self.max_evaluation_criteria]

        # Step 3: filter out criteria that cannot be evaluated.
        kept, skipped = self._filter_criteria(criteria, values)
        used_names = self._collect_used_placeholders(kept)

        # If nothing survived, return a zero result with a clear reason rather
        # than calling the LLM with an empty criteria list.
        if not kept:
            reason = (
                "No criteria could be evaluated: "
                "all entries lacked placeholders or referenced unknown data."
            )
            evaluation_log = {
                "input_question": test_case.input,
                "actual_output": test_case.actual_output,
                "evaluation_criteria_template": list(self.evaluation_criteria),
                "kept_criteria": [],
                "skipped_criteria": skipped,
                "comment_skipped_criteria": "Each entry explains why the criterion was excluded from judging.",
                "data_used": {},
                "verdicts": [],
                "final_score": 0.0,
                "threshold": self.threshold,
                "temperature": self.temperature,
                "max_evaluation_criteria": self.max_evaluation_criteria,
                "success": False,
                "summary": reason,
            }
            if truncation_warning:
                evaluation_log["truncation_warning"] = truncation_warning
            result = {
                "name": self.name,
                "score": 0.0,
                "success": False,
                "reason": reason,
                "evaluation_cost": 0.0,
                "evaluation_log": evaluation_log,
            }
            self.print_result(result)
            return result

        # Step 4: judge.
        verdicts, final_score, cost = await self._generate_verdicts(
            kept, used_names, values
        )
        success = final_score >= self.threshold

        positive = [v for v in verdicts if v.get("verdict") in ("fully", "mostly")]
        negative = [v for v in verdicts if v.get("verdict") in ("none", "minor", "partial")]
        total = len(verdicts)
        if len(positive) >= total * 0.7:
            summary = f"Strong performance: {len(positive)}/{total} criteria fully or mostly satisfied."
        elif len(negative) >= total * 0.7:
            summary = f"Weak performance: {len(negative)}/{total} criteria not satisfied or minimally addressed."
        else:
            summary = f"Mixed performance: {len(positive)}/{total} criteria satisfied, with room for improvement."

        evaluation_log: Dict[str, Any] = {
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "evaluation_criteria_template": list(self.evaluation_criteria),
            "comment_evaluation_criteria_template": "Original user-provided criteria with {{placeholders}} preserved.",
            "kept_criteria": kept,
            "comment_kept_criteria": f"Criteria actually scored ({len(kept)} of {len(criteria)} after filtering).",
            "skipped_criteria": skipped,
            "comment_skipped_criteria": "Criteria excluded from judging, with the reason for each.",
            "data_used": {name: values[name] for name in used_names},
            "comment_data_used": "Subset of resolved values that appeared in at least one kept criterion (the DATA block shown to the judge).",
            "verdicts": verdicts,
            "comment_verdicts": "Individual verdicts per kept criterion (fully/mostly/partial/minor/none).",
            "verdict_weights": {
                i: VERDICT_WEIGHTS.get(v.get("verdict", "none"), 0.0)
                for i, v in enumerate(verdicts)
            },
            "final_score": final_score,
            "comment_final_score": f"TCVA aggregation of verdict weights (temperature={self.temperature}).",
            "threshold": self.threshold,
            "temperature": self.temperature,
            "max_evaluation_criteria": self.max_evaluation_criteria,
            "success": success,
            "summary": summary,
        }
        if truncation_warning:
            evaluation_log["truncation_warning"] = truncation_warning

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(cost, 6),
            "evaluation_log": evaluation_log,
        }
        self.print_result(result)
        return result

    @property
    def name(self):
        return f"Custom: {self.custom_name}"
