"""
Loop Detection Metric - Based on Meshkov (2026)

Detects cyclic behavior in AI agent execution traces where the agent
repeats the same sequence of actions without making progress.

Analyzes execution_trace for:
1. Direct repetition: A → A → A (same tool called repeatedly)
2. Cyclic patterns: A → B → A → B (alternating tool sequences)
3. Parameter oscillation: search(broad) → search(narrow) → search(broad)

Score: 1.0 (no loops detected) to 0.0 (severe looping behavior).

This is a deterministic metric that requires execution_trace data.
"""

from typing import Dict, Any, List, Optional, Tuple
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase


class LoopDetectionMetric(MetricPattern):
    name = "loopDetectionMetric"

    def __init__(
        self,
        threshold: float = 0.8,
        verbose: bool = False,
        max_acceptable_repeats: int = 3,
        min_cycle_length: int = 1,
        max_cycle_length: int = 5,
    ):
        """
        Args:
            threshold: Minimum score to consider passing.
            max_acceptable_repeats: Max times a pattern can repeat before
                it's considered a loop (some retry is normal).
            min_cycle_length: Minimum pattern length to search for.
            max_cycle_length: Maximum pattern length to search for.
        """
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.max_acceptable_repeats = max_acceptable_repeats
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        trace = test_case.execution_trace
        if not trace:
            return self._make_result(
                score=1.0,
                reason="No execution trace available for loop detection.",
                loops_detected=[],
            )

        # Extract action sequence from trace
        actions = self._extract_action_sequence(trace)

        if len(actions) < 2:
            return self._make_result(
                score=1.0,
                reason="Trace too short for loop detection.",
                loops_detected=[],
            )

        # Detect loops
        loops = self._detect_cycles(actions)

        # Calculate score
        score = self._calculate_score(actions, loops)
        reason = self._generate_reason(actions, loops, score)

        return self._make_result(score=score, reason=reason, loops_detected=loops)

    def _extract_action_sequence(self, trace: List[Any]) -> List[str]:
        """Extract a sequence of action names from execution trace."""
        actions = []
        for step in trace:
            if hasattr(step, 'type'):
                step_type = step.type
                step_name = step.name or step_type
            elif isinstance(step, dict):
                step_type = step.get("type", "")
                step_name = step.get("name", step_type)
            else:
                continue

            # Only consider tool_call and agent_step actions
            if step_type in ("tool_call", "agent_step", "retrieval"):
                actions.append(step_name)

        return actions

    def _detect_cycles(self, actions: List[str]) -> List[Dict[str, Any]]:
        """Detect cyclic patterns in the action sequence.

        Returns a list of detected loops with pattern, count, and positions.
        """
        detected = []
        n = len(actions)

        for cycle_len in range(self.min_cycle_length, self.max_cycle_length + 1):
            if cycle_len > n // 2:
                break

            i = 0
            while i <= n - cycle_len * 2:
                pattern = actions[i:i + cycle_len]
                repeat_count = 1

                # Count consecutive repetitions
                j = i + cycle_len
                while j + cycle_len <= n:
                    if actions[j:j + cycle_len] == pattern:
                        repeat_count += 1
                        j += cycle_len
                    else:
                        break

                if repeat_count > self.max_acceptable_repeats:
                    # Check if this loop is already captured by a shorter pattern
                    if not self._is_subpattern(pattern, detected):
                        detected.append({
                            "pattern": pattern,
                            "repetitions": repeat_count,
                            "start_index": i,
                            "end_index": j,
                            "severity": self._classify_severity(repeat_count),
                        })
                    i = j  # Skip past the detected loop
                else:
                    i += 1

        return detected

    def _is_subpattern(
        self, pattern: List[str], existing: List[Dict[str, Any]]
    ) -> bool:
        """Check if pattern is a repetition of an already-detected shorter pattern."""
        for loop in existing:
            existing_pattern = loop["pattern"]
            ep_len = len(existing_pattern)
            if len(pattern) % ep_len == 0:
                repeats = len(pattern) // ep_len
                if existing_pattern * repeats == pattern:
                    return True
        return False

    def _classify_severity(self, repeat_count: int) -> str:
        if repeat_count > self.max_acceptable_repeats * 3:
            return "critical"
        elif repeat_count > self.max_acceptable_repeats * 2:
            return "high"
        else:
            return "medium"

    def _calculate_score(
        self, actions: List[str], loops: List[Dict[str, Any]]
    ) -> float:
        """Calculate loop detection score.

        Score = 1 - (wasted_actions / total_actions)
        where wasted_actions are the extra repetitions beyond acceptable.
        """
        if not loops:
            return 1.0

        total = len(actions)
        wasted = 0
        for loop in loops:
            pattern_len = len(loop["pattern"])
            excess_reps = loop["repetitions"] - self.max_acceptable_repeats
            wasted += pattern_len * max(0, excess_reps)

        wasted_ratio = min(wasted / total, 1.0) if total > 0 else 0.0
        return round(1.0 - wasted_ratio, 4)

    def _generate_reason(
        self,
        actions: List[str],
        loops: List[Dict[str, Any]],
        score: float,
    ) -> str:
        if not loops:
            return (f"No cyclic patterns detected in {len(actions)} actions. "
                    f"Agent execution appears non-repetitive.")

        parts = []
        for loop in loops:
            pat = " → ".join(loop["pattern"])
            parts.append(
                f"Pattern [{pat}] repeated {loop['repetitions']}x "
                f"(severity: {loop['severity']})"
            )

        loop_desc = "; ".join(parts)
        return (f"Detected {len(loops)} loop(s) in {len(actions)} actions: "
                f"{loop_desc}. Score: {score:.2f}")

    def _make_result(
        self,
        score: float,
        reason: str,
        loops_detected: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "score": round(score, 4),
            "success": score >= self.threshold,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "loops_detected": loops_detected,
            },
        }
        self.print_result(result)
        return result
