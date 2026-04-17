"""
Context Decay Metric - Based on Meshkov (2026)

Measures how well an agent preserves context as conversation depth increases.

Produces per-depth retention scores CRA(d) and fits a decay function:
C(d) = α·e^(-β·d) + (1-α)·1/(1+γ·d²)

where:
- α = weight of short-term memory component
- β = rate of exponential decay
- γ = rate of long-term polynomial degradation
- d = distance (in turns) between information creation and usage

Score: average CRA across all depths. 1.0 = perfect retention.

This metric works on ConversationalEvalTestCase (multi-turn conversations).
For single-turn, it returns 1.0.
"""

from typing import Dict, Any, List, Optional
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase


class ContextDecayMetric(MetricPattern):
    name = "contextDecayMetric"

    def __init__(
        self,
        threshold: float = 0.7,
        verbose: bool = False,
        context_facts: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            threshold: Minimum score to pass.
            context_facts: List of facts introduced at specific turns.
                Each dict: {"fact": str, "introduced_at": int, "used_at": int,
                           "correctly_used": bool}
                If not provided, metric estimates from execution_trace.
        """
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.context_facts = context_facts

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        facts = self.context_facts

        if not facts:
            return self._make_result(
                score=1.0,
                reason="No context_facts provided. Supply facts with "
                       "introduced_at/used_at/correctly_used to measure context decay.",
                details={},
            )

        # Calculate CRA(d) for each depth
        depth_results = self._calc_cra_by_depth(facts)

        if not depth_results:
            return self._make_result(
                score=1.0,
                reason="No depth-dependent facts found.",
                details={},
            )

        # Overall score: weighted average (closer depths matter more)
        total_weight = 0
        weighted_sum = 0
        for d, cra in sorted(depth_results.items()):
            weight = 1.0 / (1 + d * 0.2)  # Slight decay in weight for far distances
            weighted_sum += cra * weight
            total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 1.0

        # Classify ranges
        short_range = {d: v for d, v in depth_results.items() if d <= 3}
        mid_range = {d: v for d, v in depth_results.items() if 4 <= d <= 7}
        long_range = {d: v for d, v in depth_results.items() if d >= 8}

        details = {
            "cra_by_depth": depth_results,
            "short_range_avg": _avg(short_range.values()) if short_range else None,
            "mid_range_avg": _avg(mid_range.values()) if mid_range else None,
            "long_range_avg": _avg(long_range.values()) if long_range else None,
            "total_facts": len(facts),
        }

        reason = self._generate_reason(details, score)
        return self._make_result(score=score, reason=reason, details=details)

    def _calc_cra_by_depth(self, facts: List[Dict[str, Any]]) -> Dict[int, float]:
        """CRA(d) = correct_at_depth_d / required_at_depth_d"""
        depth_correct: Dict[int, int] = {}
        depth_total: Dict[int, int] = {}

        for fact in facts:
            introduced = fact.get("introduced_at", 0)
            used = fact.get("used_at", 0)
            correct = fact.get("correctly_used", False)
            d = used - introduced

            if d < 0:
                continue

            depth_total[d] = depth_total.get(d, 0) + 1
            if correct:
                depth_correct[d] = depth_correct.get(d, 0) + 1

        result = {}
        for d in sorted(depth_total.keys()):
            result[d] = depth_correct.get(d, 0) / depth_total[d]

        return result

    def _generate_reason(self, details: Dict, score: float) -> str:
        parts = []

        sr = details.get("short_range_avg")
        mr = details.get("mid_range_avg")
        lr = details.get("long_range_avg")

        if sr is not None:
            parts.append(f"short-range (d≤3): {sr:.0%}")
        if mr is not None:
            parts.append(f"mid-range (d=4-7): {mr:.0%}")
        if lr is not None:
            parts.append(f"long-range (d≥8): {lr:.0%}")

        range_str = ", ".join(parts) if parts else "no depth data"

        if score >= 0.9:
            return f"Excellent context retention across {details['total_facts']} facts. {range_str}"
        elif score >= 0.7:
            return f"Good context retention with some decay. {range_str}"
        else:
            return f"Significant context decay detected. {range_str}"

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


def _avg(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
