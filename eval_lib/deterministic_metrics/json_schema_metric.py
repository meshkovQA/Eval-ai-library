"""
JSON Schema Metric: Validate actual_output against a JSON schema.
Score: 1.0 if valid, 0.0 otherwise.
"""
from __future__ import annotations
import json
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


class JsonSchemaMetric(MetricPattern):
    name = "jsonSchemaMetric"

    def __init__(self, threshold: float = 0.5, schema: dict = None, verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.schema = schema or {}

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        text = tc.actual_output or ""
        valid = False
        error_msg = None

        try:
            import jsonschema
        except ImportError:
            raise ImportError(
                "jsonschema is required for JsonSchemaMetric. "
                "Install with: pip install eval-ai-library[deterministic] or pip install jsonschema"
            )

        try:
            data = json.loads(text)
            jsonschema.validate(instance=data, schema=self.schema)
            valid = True
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {e}"
        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
        except jsonschema.SchemaError as e:
            error_msg = f"Invalid schema: {e.message}"

        score = 1.0 if valid else 0.0
        success = score >= self.threshold
        reason = "JSON schema valid." if valid else f"JSON schema invalid: {error_msg}"

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "valid": valid,
                "error": error_msg,
                "schema_keys": list(self.schema.keys()) if self.schema else [],
            }
        }
        self.print_result(result)
        return result
