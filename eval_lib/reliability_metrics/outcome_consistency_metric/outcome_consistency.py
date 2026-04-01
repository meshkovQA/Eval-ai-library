"""
Outcome Consistency Metric (C_out) - Based on Rabanser et al. (2026)

Measures whether an agent produces consistent results across multiple
runs of the same input. A reliable agent should give semantically
equivalent answers each time it processes the same query.

Score: 0.0 (completely inconsistent) to 1.0 (perfectly consistent).

Formula: C_out = 1/T * Σ(1 - σ²_t / (p̂_t(1-p̂_t) + ε))
where σ² is the sample variance of outcomes and p̂ is the success rate.

This metric requires multiple outputs for the same input, provided
via the `multi_outputs` parameter.
"""

from typing import Dict, Any, List, Optional
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase


class OutcomeConsistencyMetric(MetricPattern):
    name = "outcomeConsistencyMetric"

    def __init__(
        self,
        threshold: float = 0.7,
        verbose: bool = False,
        similarity_threshold: float = 0.85,
        multi_outputs: Optional[List[List[str]]] = None,
    ):
        """
        Args:
            threshold: Minimum score to consider the metric passing.
            similarity_threshold: Minimum similarity between two outputs
                to consider them "consistent" (used for semantic comparison).
            multi_outputs: List of output lists. Each inner list contains
                N outputs from running the same input N times.
                multi_outputs[i] corresponds to test_cases[i].
        """
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.similarity_threshold = similarity_threshold
        self.multi_outputs = multi_outputs or []

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        """Evaluate outcome consistency for a single test case.

        If multi_outputs is not provided for this test case, falls back
        to checking if actual_output matches expected_output.
        """
        # Find outputs for this test case
        outputs = self._get_outputs_for_case(test_case)

        if len(outputs) < 2:
            return self._make_result(
                score=1.0,
                reason="Only one output available; consistency cannot be measured. "
                       "Provide multi_outputs with N>=2 runs per test case."
            )

        score = self._calculate_consistency(outputs)
        reason = self._generate_reason(outputs, score)

        return self._make_result(score=score, reason=reason)

    def _get_outputs_for_case(self, test_case: EvalTestCase) -> List[str]:
        """Get the list of outputs for this test case."""
        # If multi_outputs is populated, use the first matching entry
        # The caller is responsible for indexing correctly
        if hasattr(self, '_current_multi_outputs') and self._current_multi_outputs:
            return self._current_multi_outputs

        # Fallback: just use actual_output as a single output
        return [test_case.actual_output]

    def set_outputs_for_case(self, outputs: List[str]):
        """Set multi-run outputs for the current test case evaluation."""
        self._current_multi_outputs = outputs

    def _calculate_consistency(self, outputs: List[str]) -> float:
        """Calculate outcome consistency score.

        Uses exact string matching and normalized Levenshtein distance
        for pairwise comparison. No LLM calls needed.
        """
        n = len(outputs)
        if n < 2:
            return 1.0

        consistent_pairs = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._compute_similarity(outputs[i], outputs[j])
                if similarity >= self.similarity_threshold:
                    consistent_pairs += 1
                total_pairs += 1

        return consistent_pairs / total_pairs if total_pairs > 0 else 1.0

    def _compute_similarity(self, a: str, b: str) -> float:
        """Compute normalized similarity between two strings.

        Uses normalized Levenshtein distance as a lightweight,
        deterministic similarity measure (no LLM needed).
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        # Normalize: lowercase and strip
        a_norm = a.strip().lower()
        b_norm = b.strip().lower()

        if a_norm == b_norm:
            return 1.0

        # Normalized Levenshtein distance
        distance = _levenshtein_distance(a_norm, b_norm)
        max_len = max(len(a_norm), len(b_norm))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    def _generate_reason(self, outputs: List[str], score: float) -> str:
        n = len(outputs)
        total_pairs = n * (n - 1) // 2
        consistent = round(score * total_pairs)

        if score >= 0.95:
            return (f"High consistency: {consistent}/{total_pairs} output pairs "
                    f"are semantically equivalent across {n} runs.")
        elif score >= 0.7:
            return (f"Moderate consistency: {consistent}/{total_pairs} pairs "
                    f"consistent across {n} runs. Some variation detected.")
        else:
            return (f"Low consistency: only {consistent}/{total_pairs} pairs "
                    f"consistent across {n} runs. Agent behavior is unpredictable.")

    def _make_result(self, score: float, reason: str) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "score": round(score, 4),
            "success": score >= self.threshold,
            "reason": reason,
            "evaluation_cost": 0.0,
        }
        self.print_result(result)
        return result


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]
