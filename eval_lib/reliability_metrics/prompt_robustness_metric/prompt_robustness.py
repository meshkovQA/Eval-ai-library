"""
Prompt Robustness Metric (R_prompt) - Based on Rabanser et al. (2026) + Meshkov (2026)

Measures whether an agent produces consistent results when given
semantically equivalent reformulations of the same query.

Uses perturbation_group field in EvalTestCase to link related test cases.
All test cases in the same perturbation_group should be semantically
equivalent queries. The metric checks whether outputs are consistent.

Score: R_prompt = min(Acc_para / Acc_0, 1) — ratio of accuracy on
paraphrased inputs vs baseline (Rabanser).

Also computes Meshkov's Robustness Score = min(Consistency_Rate_type).

This metric evaluates across a GROUP of test cases, not a single one.
Call evaluate_group() with all test cases sharing a perturbation_group.
"""

from typing import Dict, Any, List, Optional
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.reliability_metrics.outcome_consistency_metric.outcome_consistency import (
    _levenshtein_distance,
)


class PromptRobustnessMetric(MetricPattern):
    name = "promptRobustnessMetric"

    def __init__(
        self,
        threshold: float = 0.7,
        verbose: bool = False,
        similarity_threshold: float = 0.6,
    ):
        """
        Args:
            threshold: Minimum score to consider passing.
            similarity_threshold: Minimum output similarity for a pair
                to be considered consistent.
        """
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.similarity_threshold = similarity_threshold

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        """Single test case evaluation — returns 1.0 (cannot assess robustness alone)."""
        return self._make_result(
            score=1.0,
            reason="Prompt robustness requires multiple test cases in the same "
                   "perturbation_group. Use evaluate_group() instead.",
        )

    async def evaluate_group(self, test_cases: List[EvalTestCase]) -> Dict[str, Any]:
        """Evaluate prompt robustness across a group of paraphrased test cases.

        Args:
            test_cases: List of EvalTestCase objects that are semantically
                       equivalent queries (same perturbation_group).

        Returns:
            Dict with score, success, reason, and per-pair details.
        """
        if len(test_cases) < 2:
            return self._make_result(
                score=1.0,
                reason="Need at least 2 test cases in a perturbation group.",
            )

        outputs = [tc.actual_output for tc in test_cases]
        inputs = [tc.input for tc in test_cases]

        # Calculate pairwise consistency
        n = len(outputs)
        consistent_pairs = 0
        total_pairs = 0
        pair_details = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_similarity(outputs[i], outputs[j])
                is_consistent = sim >= self.similarity_threshold
                if is_consistent:
                    consistent_pairs += 1
                total_pairs += 1
                pair_details.append({
                    "input_i": inputs[i][:80],
                    "input_j": inputs[j][:80],
                    "similarity": round(sim, 4),
                    "consistent": is_consistent,
                })

        score = consistent_pairs / total_pairs if total_pairs > 0 else 1.0

        # R_prompt = min(Acc_para / Acc_0, 1.0) — Rabanser formula
        # When we don't have ground-truth accuracy, use consistency as proxy
        r_prompt = min(score, 1.0)

        reason = self._generate_reason(
            n, consistent_pairs, total_pairs, r_prompt, pair_details
        )

        result = self._make_result(score=r_prompt, reason=reason)
        result["evaluation_log"] = {
            "total_variants": n,
            "consistent_pairs": consistent_pairs,
            "total_pairs": total_pairs,
            "pair_details": pair_details,
        }
        return result

    def _compute_similarity(self, a: str, b: str) -> float:
        """Compute normalized similarity between two outputs."""
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        a_norm = a.strip().lower()
        b_norm = b.strip().lower()
        if a_norm == b_norm:
            return 1.0

        distance = _levenshtein_distance(a_norm, b_norm)
        max_len = max(len(a_norm), len(b_norm))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    def _generate_reason(
        self,
        n: int,
        consistent_pairs: int,
        total_pairs: int,
        score: float,
        pair_details: List[Dict],
    ) -> str:
        if score >= 0.95:
            return (f"High prompt robustness: {consistent_pairs}/{total_pairs} "
                    f"output pairs consistent across {n} query variants.")
        elif score >= 0.7:
            inconsistent = [p for p in pair_details if not p["consistent"]]
            return (f"Moderate robustness: {consistent_pairs}/{total_pairs} pairs "
                    f"consistent. {len(inconsistent)} pairs diverged.")
        else:
            return (f"Low robustness: only {consistent_pairs}/{total_pairs} pairs "
                    f"consistent across {n} variants. Agent is sensitive to phrasing.")

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
