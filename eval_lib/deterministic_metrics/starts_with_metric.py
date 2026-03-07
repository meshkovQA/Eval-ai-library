"""
Starts With Metric: Check if output starts with a given prefix.
Score: 1.0 if starts with prefix, 0.0 otherwise.
"""
from __future__ import annotations
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class StartsWithMetric(MetricPattern):
    name = "startsWithMetric"

    def __init__(self, threshold: float = 0.5, prefix: str = "", case_sensitive: bool = True,
                 verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.prefix = prefix
        self.case_sensitive = case_sensitive

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = tc.actual_output or ""
        if self.case_sensitive:
            matched = text.startswith(self.prefix)
        else:
            matched = text.lower().startswith(self.prefix.lower())

        score = 1.0 if matched else 0.0
        success = score >= self.threshold
        reason = f"Output {'starts' if matched else 'does not start'} with '{self.prefix}'"

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "prefix": self.prefix,
                "case_sensitive": self.case_sensitive,
                "matched": matched,
            }
        }
        self.print_result(result)
        return result
