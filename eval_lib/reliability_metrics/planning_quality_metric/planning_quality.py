"""
Planning Quality Metric - Based on Meshkov (2026)

Evaluates whether an agent's task decomposition and planning is
appropriate for the given task complexity.

Combines:
- Granularity Score: 1 - |agent_steps - ref_steps| / max(...)
- Efficiency Ratio: min_necessary_actions / actual_actions
- Planning Overhead: planning_time / total_time

Score: 0.0 (terrible planning) to 1.0 (optimal planning).

Requires execution_trace and optionally planning_steps in EvalTestCase.
"""

from typing import Dict, Any, List, Optional
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase


class PlanningQualityMetric(MetricPattern):
    name = "planningQualityMetric"

    def __init__(
        self,
        threshold: float = 0.6,
        verbose: bool = False,
        expected_steps: Optional[int] = None,
        expected_tool_count: Optional[int] = None,
    ):
        """
        Args:
            threshold: Minimum score to pass.
            expected_steps: Reference number of steps for this task type.
                If None, uses heuristic based on task complexity.
            expected_tool_count: Expected number of unique tools needed.
        """
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.expected_steps = expected_steps
        self.expected_tool_count = expected_tool_count

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        trace = test_case.execution_trace
        if not trace:
            return self._make_result(
                score=1.0,
                reason="No execution trace available for planning analysis.",
                details={},
            )

        # Count action types
        tool_calls = [s for s in trace if _get_type(s) == "tool_call"]
        reasoning_steps = [s for s in trace if _get_type(s) in ("reasoning", "agent_step")]
        all_steps = [s for s in trace if _get_type(s) in ("tool_call", "agent_step", "reasoning", "retrieval")]

        actual_steps = len(all_steps)
        actual_tools = len(tool_calls)
        reasoning_count = len(reasoning_steps)

        # Calculate sub-metrics
        granularity = self._calc_granularity(actual_steps)
        efficiency = self._calc_efficiency(tool_calls)
        overhead = self._calc_planning_overhead(trace, reasoning_steps)

        # Composite score: weighted average
        score = 0.4 * granularity + 0.4 * efficiency + 0.2 * (1.0 - overhead)
        score = max(0.0, min(1.0, score))

        details = {
            "actual_steps": actual_steps,
            "actual_tool_calls": actual_tools,
            "reasoning_steps": reasoning_count,
            "granularity_score": round(granularity, 4),
            "efficiency_ratio": round(efficiency, 4),
            "planning_overhead": round(overhead, 4),
        }

        reason = self._generate_reason(details, score)
        return self._make_result(score=score, reason=reason, details=details)

    def _calc_granularity(self, actual_steps: int) -> float:
        """Granularity Score = 1 - |agent_steps - ref_steps| / max(...)"""
        if self.expected_steps is None:
            return 1.0  # No reference → assume OK
        if self.expected_steps == 0 and actual_steps == 0:
            return 1.0
        max_val = max(actual_steps, self.expected_steps, 1)
        return 1.0 - abs(actual_steps - self.expected_steps) / max_val

    def _calc_efficiency(self, tool_calls: List[Any]) -> float:
        """Efficiency: penalize redundant tool calls (same tool+input)."""
        if not tool_calls:
            return 1.0

        seen = set()
        unique = 0
        for tc in tool_calls:
            name = _get_name(tc)
            inp = str(_get_input(tc))
            key = f"{name}:{inp}"
            if key not in seen:
                seen.add(key)
                unique += 1

        return unique / len(tool_calls) if tool_calls else 1.0

    def _calc_planning_overhead(
        self, trace: List[Any], reasoning_steps: List[Any]
    ) -> float:
        """Planning Overhead = reasoning_time / total_time."""
        total_ms = sum(_get_duration(s) for s in trace)
        reasoning_ms = sum(_get_duration(s) for s in reasoning_steps)

        if total_ms <= 0:
            # Fallback: use step counts
            total_count = len(trace)
            reasoning_count = len(reasoning_steps)
            return reasoning_count / total_count if total_count > 0 else 0.0

        return reasoning_ms / total_ms

    def _generate_reason(self, details: Dict, score: float) -> str:
        parts = []
        g = details["granularity_score"]
        e = details["efficiency_ratio"]
        o = details["planning_overhead"]

        if g < 0.6:
            diff = details["actual_steps"] - (self.expected_steps or 0)
            if diff > 0:
                parts.append(f"Over-decomposed: {diff} extra steps")
            else:
                parts.append(f"Under-decomposed: {-diff} missing steps")

        if e < 0.7:
            redundant = details["actual_tool_calls"] - round(e * details["actual_tool_calls"])
            parts.append(f"~{redundant} redundant tool calls")

        if o > 0.5:
            parts.append(f"High planning overhead ({o:.0%} of execution time)")

        if not parts:
            return f"Planning quality is good. {details['actual_steps']} steps, efficiency {e:.0%}."

        return f"Planning issues: {'; '.join(parts)}. Score: {score:.2f}"

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


def _get_type(step: Any) -> str:
    if isinstance(step, dict):
        return step.get("type", "")
    return getattr(step, "type", "")


def _get_name(step: Any) -> str:
    if isinstance(step, dict):
        return step.get("name", "")
    return getattr(step, "name", "")


def _get_input(step: Any) -> Any:
    if isinstance(step, dict):
        return step.get("input", "")
    return getattr(step, "input", "")


def _get_duration(step: Any) -> float:
    if isinstance(step, dict):
        return step.get("duration_ms", 0) or 0
    return getattr(step, "duration_ms", 0) or 0
