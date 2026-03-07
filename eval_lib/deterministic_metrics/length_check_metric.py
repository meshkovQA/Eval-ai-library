"""
Length Check Metric: Check output length in chars or words.
Score: 1.0 if within range, 0.0 otherwise.
"""
from __future__ import annotations
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class LengthCheckMetric(MetricPattern):
    name = "lengthCheckMetric"

    def __init__(self, threshold: float = 0.5, min_length: int = 0, max_length: int = 10000,
                 unit: str = "chars", verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = tc.actual_output or ""

        if self.unit == "words":
            length = len(text.split())
        else:
            length = len(text)

        in_range = self.min_length <= length <= self.max_length
        score = 1.0 if in_range else 0.0
        success = score >= self.threshold
        reason = (f"Length {length} {self.unit} is {'within' if in_range else 'outside'} "
                  f"range [{self.min_length}, {self.max_length}]")

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "length": length,
                "unit": self.unit,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "in_range": in_range,
            }
        }
        self.print_result(result)
        return result
