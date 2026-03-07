"""
Ends With Metric: Check if output ends with a given suffix.
Score: 1.0 if ends with suffix, 0.0 otherwise.
"""
from __future__ import annotations
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class EndsWithMetric(MetricPattern):
    name = "endsWithMetric"

    def __init__(self, threshold: float = 0.5, suffix: str = "", case_sensitive: bool = True,
                 verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.suffix = suffix
        self.case_sensitive = case_sensitive

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = tc.actual_output or ""
        if self.case_sensitive:
            matched = text.endswith(self.suffix)
        else:
            matched = text.lower().endswith(self.suffix.lower())

        score = 1.0 if matched else 0.0
        success = score >= self.threshold
        reason = f"Output {'ends' if matched else 'does not end'} with '{self.suffix}'"

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "suffix": self.suffix,
                "case_sensitive": self.case_sensitive,
                "matched": matched,
            }
        }
        self.print_result(result)
        return result
