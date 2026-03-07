"""
Non-Empty Metric: Check that output is not empty or whitespace-only.
Score: 1.0 if non-empty, 0.0 otherwise.
"""
from __future__ import annotations
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class NonEmptyMetric(MetricPattern):
    name = "nonEmptyMetric"

    def __init__(self, threshold: float = 0.5, verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = tc.actual_output or ""
        non_empty = len(text.strip()) > 0

        score = 1.0 if non_empty else 0.0
        success = score >= self.threshold
        reason = "Output is non-empty." if non_empty else "Output is empty or whitespace-only."

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "non_empty": non_empty,
                "raw_length": len(text),
                "stripped_length": len(text.strip()),
            }
        }
        self.print_result(result)
        return result
