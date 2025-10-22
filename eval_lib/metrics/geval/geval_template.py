from textwrap import dedent
from eval_lib.testcases_schema import EvalTestCase


class GEvalTemplate:
    @staticmethod
    def generate_evaluation_steps(criteria: str) -> str:
        return f"""Given the evaluation criteria below, generate 3-4 concise evaluation steps that define how to assess the quality of an actual response to an input. Make the steps specific, actionable, and logically ordered.

Evaluation Criteria:
{criteria}

**
IMPORTANT: Please return only JSON in the following format:
{{
  "steps": ["...", "...", "..."]
}}
No explanations, only the JSON.
**

JSON:
"""

    @staticmethod
    def generate_evaluation_results(
        evaluation_steps: str,
        test_case: EvalTestCase
    ) -> str:
        text_parts = [
            f"Input: {test_case.input}",
            f"Actual Output: {test_case.actual_output}"
        ]
        if test_case.retrieval_context:
            joined_ctx = "\n".join(test_case.retrieval_context)
            text_parts.append(f"Retrieval Context:\n{joined_ctx}")

        eval_block = "\n".join(text_parts)

        return f"""Using the evaluation steps below, assign a `score` (from 0 to 10) and a `reason` explaining your decision. The score must reflect how well the actual output fulfills the input, and aligns with the context and/or expectations if provided.

Evaluation Steps:
{evaluation_steps}

{eval_block}

**
IMPORTANT: Only return a JSON in this format:
{{
  "score": <float>,
  "reason": "<your explanation>"
}}
Do NOT mention the evaluation steps or the score directly in the reason.
**

JSON:
"""

    @staticmethod
    def generate_strict_evaluation_results(
        evaluation_steps: str,
        test_case: EvalTestCase
    ) -> str:
        text_parts = [
            f"Input: {test_case.input}",
            f"Actual Output: {test_case.actual_output}"
        ]
        if test_case.retrieval_context:
            joined_ctx = "\n".join(test_case.retrieval_context)
            text_parts.append(f"Retrieval Context:\n{joined_ctx}")

        eval_block = "\n".join(text_parts)

        return f"""Using the evaluation steps below, determine whether the actual output meets the criteria strictly. Return a JSON with:
- `score`: 1 if the output meets the criteria, 0 otherwise
- `reason`: a short explanation why

Evaluation Steps:
{evaluation_steps}

{eval_block}

**
IMPORTANT: Only return a JSON in this format:
{{
  "score": <0 or 1>,
  "reason": "<your explanation>"
}}
Do NOT mention the evaluation steps or score explicitly in the reason.
**

JSON:
"""
