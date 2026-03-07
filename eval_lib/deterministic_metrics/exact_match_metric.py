"""
Exact Match Metric: Compare actual_output with expected_output.
Score: 1.0 if match, 0.0 otherwise.
"""
from __future__ import annotations
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class ExactMatchMetric(MetricPattern):
    name = "exactMatchMetric"

    def __init__(self, threshold: float = 0.5, case_sensitive: bool = True,
                 strip_whitespace: bool = True, verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        actual = tc.actual_output or ""
        expected = tc.expected_output or ""

        if self.strip_whitespace:
            actual = actual.strip()
            expected = expected.strip()

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        matched = actual == expected
        score = 1.0 if matched else 0.0
        success = score >= self.threshold
        reason = "Exact match." if matched else "Output does not match expected."

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "matched": matched,
                "case_sensitive": self.case_sensitive,
                "strip_whitespace": self.strip_whitespace,
                "actual_length": len(tc.actual_output or ""),
                "expected_length": len(tc.expected_output or ""),
            }
        }
        self.print_result(result)
        return result
