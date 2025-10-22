# custom_eval.py
from typing import Dict, Any, Tuple
import json
import re
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.llm_client import chat_complete


def extract_json_block(text: str) -> str:
    """
    Extracts the first JSON block from Markdown-like fenced code blocks.
    """
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    json_match = re.search(r"({.*?})", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    return text.strip()


class CustomEvalMetric(MetricPattern):
    template_cls = None

    def __init__(self, model: str, threshold: float, name: str, criteria: str):
        super().__init__(model=model, threshold=threshold)
        self.custom_name = name
        self.criteria = criteria

    async def _run_eval_prompt(self, test_case: EvalTestCase) -> Tuple[float, str, float]:
        prompt = (
            "You are a strict evaluator. Use the following evaluation criteria to judge the quality of the model's answer.\n\n"
            f"Criteria:\n{self.criteria}\n\n"
            f"User Input:\n{test_case.input}\n\n"
            f"Model Output:\n{test_case.actual_output}\n\n"
            f"Expected Output (optional):\n{test_case.expected_output or 'N/A'}\n\n"
            f"Context (optional):\n{test_case.retrieval_context or 'N/A'}\n\n"
            "Respond ONLY with JSON like this:\n"
            '{ "score": 0–10, "reason": "<explanation>" }'
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        try:
            parsed = json.loads(extract_json_block(text))
            score = float(parsed.get("score", 0)) / 10  # normalize 0–1
            reason = parsed.get("reason", "")
            return score, reason, cost or 0.0
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse custom metric response: {e}\n{text}")

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0

        score, reason, cost = await self._run_eval_prompt(test_case)
        llm_cost += cost

        success = score >= self.threshold

        evaluation_log = {
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "retrieval_context": test_case.retrieval_context,
            "criteria": self.criteria,
            "comment_criteria": "Custom rules for evaluation.",
            "final_score": score,
            "comment_final_score": "Score from LLM based on custom criteria.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the score passes the custom threshold.",
            "final_reason": reason,
            "comment_reasoning": "LLM explanation based on custom evaluation."
        }

        return {
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log,
        }

    @property
    def name(self):
        return f"Custom: {self.custom_name}"
