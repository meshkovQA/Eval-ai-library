"""
Language Detection Metric: Check if output is in the expected language.
Score: 1.0 if language matches, 0.0 otherwise.
"""
from __future__ import annotations
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class LanguageDetectionMetric(MetricPattern):
    name = "languageDetectionMetric"

    def __init__(self, threshold: float = 0.5, expected_language: str = "en", verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.expected_language = expected_language

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
        except ImportError:
            raise ImportError(
                "langdetect is required for LanguageDetectionMetric. "
                "Install with: pip install eval-ai-library[deterministic] or pip install langdetect"
            )

        text = tc.actual_output or ""
        detected = None
        matched = False

        try:
            detected = detect(text)
            matched = detected == self.expected_language
        except Exception:
            detected = "unknown"
            matched = False

        score = 1.0 if matched else 0.0
        success = score >= self.threshold
        reason = f"Detected language: {detected}, expected: {self.expected_language}"

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "detected_language": detected,
                "expected_language": self.expected_language,
                "matched": matched,
            }
        }
        self.print_result(result)
        return result
