# metric_pattern.py
import json
from typing import Type, Dict, Any, Union, Optional

from eval_lib.testcases_schema import EvalTestCase, ConversationalEvalTestCase
from eval_lib.llm_client import chat_complete


class MetricPattern:
    """
    Base class for metrics that use a pattern-based approach to evaluation.
    This class is designed to be subclassed for specific metrics.
    """
    name: str               # name of the metric

    def __init__(self, model: str, threshold: float):
        self.model = model  # OpenAI model name
        self.threshold = threshold

    async def evaluate(
        self,
        test_case: Union[EvalTestCase]
    ) -> Dict[str, Any]:

        # 1) Generate prompt
        prompt = self.template.generate_prompt(
            test_case=test_case,
            threshold=self.threshold
        )

        # 2) Make API call to OpenAI
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        # 3) Parse the response
        try:
            data = json.loads(text)
        except Exception as e:
            raise RuntimeError(
                f"Cannot parse JSON from model response: {e}\n{text}")

        score = float(data.get("score", 0.0))
        reason = data.get("reason")

        # Check if the score is above the threshold
        success = score >= self.threshold

        # 4) Calculate the cost of the evaluation
        evaluation_cost = cost

        # 4) Get the score
        return {
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": evaluation_cost,
        }


class ConversationalMetricPattern:
    """
    Base class for conversational metrics (evaluating full dialogues).
    Used for metrics like RoleAdherence, DialogueCoherence, etc.
    """
    name: str
    template_cls: Type  # used to generate the prompt

    def __init__(self, model: str, threshold: float):
        self.model = model
        self.threshold = threshold
        if self.template_cls:  # if a template class is provided
            self.template = self.template_cls()
        else:
            self.template = None
        self.chatbot_role: Optional[str] = None

    async def evaluate(
        self,
        test_case: ConversationalEvalTestCase
    ) -> Dict[str, Any]:

        # 1. Generate prompt
        if hasattr(self.template, "generate_prompt"):
            try:
                prompt = self.template.generate_prompt(
                    test_case=test_case,
                    threshold=self.threshold,
                    chatbot_role=self.chatbot_role  # pass the role
                )
            except TypeError:
                # fallback: if the template doesn't accept chatbot_role
                prompt = self.template.generate_prompt(
                    test_case=test_case,
                    threshold=self.threshold,
                    temperature=0.0  # default temperature
                )
        else:
            raise RuntimeError("Template is missing method generate_prompt")

        # 2. Call OpenAI API
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        # 3. Parse the response
        try:
            data = json.loads(text)
        except Exception as e:
            raise RuntimeError(
                f"Cannot parse JSON from model response: {e}\n{text}"
            )

        score = float(data.get("score", 0.0))
        reason = data.get("reason")
        success = score >= self.threshold

        # 4. Calculate the cost of the evaluation
        evaluation_cost = cost

        return {
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": evaluation_cost,
        }
