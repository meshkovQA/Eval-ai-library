# geval.py
import json
from typing import Optional, Dict, Any
from eval_lib.metric_pattern import MetricPattern
from .geval_template import GEvalTemplate
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.llm_client import chat_complete, model_pricing


class GEval(MetricPattern):
    name = "gEval"
    template_cls = GEvalTemplate

    def __init__(
        self,
        model: str,
        threshold: float,
        name: Optional[str] = None,
        criteria: Optional[str] = None,
        evaluation_steps: Optional[str] = None,
    ):
        super().__init__(model=model, threshold=threshold)
        self.criteria = criteria
        self.custom_name = name
        self.evaluation_steps = evaluation_steps  # может быть задано заранее

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        total_cost = 0.0
        # 1) IF evaluation_steps is not provided, generate them using the criteria
        if not self.evaluation_steps:
            if not self.criteria:
                raise ValueError(
                    "Either 'criteria' or 'evaluation_steps' must be provided for GEval.")
            # Generate evaluation steps using the criteria
            prompt = self.template.generate_evaluation_steps(
                criteria=self.criteria
            )
            steps_text, step_cost = await chat_complete(
                self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            if step_cost:
                total_cost += step_cost

            try:
                self.evaluation_steps = json.loads(steps_text)["steps"]
            except Exception as e:
                raise RuntimeError(f"Invalid steps JSON: {e}\n{steps_text}")

        # 2) Generate evaluation results

        result_prompt = self.template.generate_evaluation_results(
            evaluation_steps=self.evaluation_steps,
            test_case=test_case
        )

        result_text, result_cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": result_prompt}],
            temperature=0.0
        )

        if result_cost:
            total_cost += result_cost

        try:
            parsed = json.loads(result_text)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse evaluation result JSON: {e}\n{result_text}")

        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")
        success = score >= self.threshold

        return {
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": round(total_cost, 6)
        }

    @property
    def name(self):
        return self.custom_name or self.__class__.name
