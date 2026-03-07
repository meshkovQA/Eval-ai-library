"""
Contains Metric: Check presence/absence of keywords in output.
Modes: any (at least one), all (proportional score), none (none present).
"""
from __future__ import annotations
from typing import Dict, Any, List
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class ContainsMetric(MetricPattern):
    name = "containsMetric"

    def __init__(self, threshold: float = 0.5, keywords: List[str] = None,
                 mode: str = "any", case_sensitive: bool = False, verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.keywords = keywords or []
        self.mode = mode
        self.case_sensitive = case_sensitive

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = tc.actual_output or ""
        check_text = text if self.case_sensitive else text.lower()

        found = []
        not_found = []
        for kw in self.keywords:
            check_kw = kw if self.case_sensitive else kw.lower()
            if check_kw in check_text:
                found.append(kw)
            else:
                not_found.append(kw)

        total = len(self.keywords)
        if total == 0:
            score = 1.0
        elif self.mode == "any":
            score = 1.0 if len(found) > 0 else 0.0
        elif self.mode == "all":
            score = len(found) / total
        elif self.mode == "none":
            score = 1.0 if len(found) == 0 else 0.0
        else:
            score = 0.0

        success = score >= self.threshold
        reason = f"Found {len(found)}/{total} keywords (mode={self.mode})"

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "mode": self.mode,
                "case_sensitive": self.case_sensitive,
                "found": found,
                "not_found": not_found,
                "score": score,
            }
        }
        self.print_result(result)
        return result
