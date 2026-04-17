"""
Calibration Metric - Based on Rabanser et al. (2026)

Measures whether an agent's self-reported confidence matches its actual
success rate. A well-calibrated agent that claims 80% confidence should
succeed roughly 80% of the time.

Three sub-metrics:
- P_cal (ECE): Expected Calibration Error — 1 - Σ(n_b/N)|ȳ_b - c̄_b|
- P_AUROC: Discrimination — can the agent tell when it's right vs wrong?
- P_brier: Brier score — 1 - 1/T·Σ(c_i - y_i)²

This is a SUITE-LEVEL metric: it evaluates across multiple test cases.
Call evaluate_suite() with a list of (confidence, success) pairs.

Requires agent_confidence field in EvalTestCase.
"""

from typing import Dict, Any, List, Tuple, Optional
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase


class CalibrationMetric(MetricPattern):
    name = "calibrationMetric"

    def __init__(
        self,
        threshold: float = 0.7,
        verbose: bool = False,
        n_bins: int = 10,
    ):
        """
        Args:
            threshold: Minimum Brier score to pass.
            n_bins: Number of bins for calibration (ECE) computation.
        """
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.n_bins = n_bins

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        """Single test case — cannot compute calibration. Use evaluate_suite()."""
        return self._make_result(
            score=1.0,
            reason="Calibration requires multiple test cases. "
                   "Use evaluate_suite() with confidence/success pairs.",
            details={},
        )

    async def evaluate_suite(
        self,
        results: List[Tuple[float, bool]],
    ) -> Dict[str, Any]:
        """Evaluate calibration across a suite of test results.

        Args:
            results: List of (confidence, success) tuples.
                confidence: float in [0, 1] — agent's self-reported confidence.
                success: bool — whether the agent succeeded on that task.

        Returns:
            Dict with P_cal, P_AUROC, P_brier scores and details.
        """
        if len(results) < 5:
            return self._make_result(
                score=1.0,
                reason=f"Only {len(results)} results; need >= 5 for calibration.",
                details={},
            )

        confidences = [r[0] for r in results]
        outcomes = [1.0 if r[1] else 0.0 for r in results]

        p_cal = self._compute_ece(confidences, outcomes)
        p_auroc = self._compute_auroc(confidences, outcomes)
        p_brier = self._compute_brier(confidences, outcomes)

        # Use Brier as the primary score (proper scoring rule)
        score = p_brier

        details = {
            "p_cal": round(p_cal, 4),
            "p_auroc": round(p_auroc, 4),
            "p_brier": round(p_brier, 4),
            "n_samples": len(results),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
            "success_rate": round(sum(outcomes) / len(outcomes), 4),
        }

        reason = self._generate_reason(details, score)
        return self._make_result(score=score, reason=reason, details=details)

    def _compute_ece(
        self, confidences: List[float], outcomes: List[float]
    ) -> float:
        """Expected Calibration Error: P_cal = 1 - ECE."""
        n = len(confidences)
        bin_boundaries = [i / self.n_bins for i in range(self.n_bins + 1)]
        ece = 0.0

        for b in range(self.n_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
            indices = [
                i for i in range(n)
                if lo <= confidences[i] < hi or (b == self.n_bins - 1 and confidences[i] == hi)
            ]
            if not indices:
                continue
            n_b = len(indices)
            avg_conf = sum(confidences[i] for i in indices) / n_b
            avg_acc = sum(outcomes[i] for i in indices) / n_b
            ece += (n_b / n) * abs(avg_acc - avg_conf)

        return 1.0 - ece

    def _compute_auroc(
        self, confidences: List[float], outcomes: List[float]
    ) -> float:
        """Discrimination: AUROC between success/failure confidence distributions."""
        successes = [c for c, o in zip(confidences, outcomes) if o == 1.0]
        failures = [c for c, o in zip(confidences, outcomes) if o == 0.0]

        if not successes or not failures:
            return 0.5  # Cannot discriminate

        # Wilcoxon-Mann-Whitney statistic
        concordant = 0
        total = 0
        for s in successes:
            for f in failures:
                total += 1
                if s > f:
                    concordant += 1
                elif s == f:
                    concordant += 0.5

        return concordant / total if total > 0 else 0.5

    def _compute_brier(
        self, confidences: List[float], outcomes: List[float]
    ) -> float:
        """Brier score: P_brier = 1 - 1/T·Σ(c_i - y_i)²."""
        n = len(confidences)
        mse = sum((c - y) ** 2 for c, y in zip(confidences, outcomes)) / n
        return 1.0 - mse

    def _generate_reason(self, details: Dict, score: float) -> str:
        p_cal = details["p_cal"]
        p_auroc = details["p_auroc"]
        avg_conf = details["avg_confidence"]
        success_rate = details["success_rate"]

        parts = []
        if avg_conf > success_rate + 0.15:
            parts.append("overconfident")
        elif avg_conf < success_rate - 0.15:
            parts.append("underconfident")

        if p_auroc < 0.6:
            parts.append("poor discrimination (can't tell right from wrong)")
        elif p_auroc >= 0.8:
            parts.append("good discrimination")

        cal_str = f"Calibration: {p_cal:.2f}, Discrimination: {p_auroc:.2f}, Brier: {score:.2f}"

        if parts:
            return f"Agent is {', '.join(parts)}. {cal_str}"
        return f"Agent is well-calibrated. {cal_str}"

    def _make_result(
        self, score: float, reason: str, details: Dict
    ) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "score": round(score, 4),
            "success": score >= self.threshold,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": details,
        }
        self.print_result(result)
        return result
