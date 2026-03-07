"""
Format Check Metric: Validate output format (email, url, phone, date).
Score: 1.0 if valid format, 0.0 otherwise.
"""
from __future__ import annotations
import re
from urllib.parse import urlparse
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern

FORMAT_PATTERNS = {
    "email": re.compile(
        r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    ),
    "phone": re.compile(
        r"^[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]*$"
    ),
    "date": re.compile(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"
    ),
}


def _is_valid_url(text: str) -> bool:
    try:
        r = urlparse(text.strip())
        return all([r.scheme, r.netloc])
    except Exception:
        return False


class FormatCheckMetric(MetricPattern):
    name = "formatCheckMetric"

    def __init__(self, threshold: float = 0.5, format_type: str = "email", verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.format_type = format_type

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = (tc.actual_output or "").strip()
        valid = False

        if self.format_type == "url":
            valid = _is_valid_url(text)
        elif self.format_type in FORMAT_PATTERNS:
            valid = bool(FORMAT_PATTERNS[self.format_type].match(text))
        else:
            valid = False

        score = 1.0 if valid else 0.0
        success = score >= self.threshold
        reason = f"Format '{self.format_type}' is {'valid' if valid else 'invalid'}."

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "format_type": self.format_type,
                "valid": valid,
                "checked_text": text[:100],
            }
        }
        self.print_result(result)
        return result
