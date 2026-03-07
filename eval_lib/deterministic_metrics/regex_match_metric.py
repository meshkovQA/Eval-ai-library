"""
Regex Match Metric: Check if actual_output matches a regex pattern.
Score: 1.0 if match found, 0.0 otherwise.
"""
from __future__ import annotations
import re
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class RegexMatchMetric(MetricPattern):
    name = "regexMatchMetric"

    def __init__(self, threshold: float = 0.5, pattern: str = "", full_match: bool = False, verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.pattern = pattern
        self.full_match = full_match

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = tc.actual_output or ""
        matched = False
        match_info = None

        try:
            if self.full_match:
                m = re.fullmatch(self.pattern, text)
            else:
                m = re.search(self.pattern, text)
            matched = m is not None
            match_info = m.group(0) if m else None
        except re.error as e:
            match_info = f"Invalid regex: {e}"

        score = 1.0 if matched else 0.0
        success = score >= self.threshold
        reason = f"Pattern {'matched' if matched else 'not matched'}: {self.pattern}"

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "pattern": self.pattern,
                "full_match": self.full_match,
                "matched": matched,
                "match_value": match_info,
                "output_length": len(text),
            }
        }
        self.print_result(result)
        return result
