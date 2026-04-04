"""
DeepEval evaluation module.
Runs DeepEval FaithfulnessMetric (ternary verdicts).

Patches:
  1. max_completion_tokens=4096 on openai parse calls (prevents 16K blow-up)
  2. Removes LengthFinishReasonError from retryable exceptions (no infinite retry)
"""
from typing import List

MAX_COMPLETION_TOKENS = 4096

_patched = False


def _apply_patches():
    global _patched
    if _patched:
        return
    _patched = True

    import openai
    import deepeval.models.llms.openai_model as dm

    # Remove LengthFinishReasonError from retryable — no point retrying
    dm.retryable_exceptions = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
    )

    # Patch openai parse calls to inject max_completion_tokens
    from openai.resources.chat.completions import completions as oai_comp

    _orig_parse = oai_comp.Completions.parse
    _orig_a_parse = oai_comp.AsyncCompletions.parse

    def patched_parse(self, *args, **kwargs):
        if "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        return _orig_parse(self, *args, **kwargs)

    async def patched_a_parse(self, *args, **kwargs):
        if "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        return await _orig_a_parse(self, *args, **kwargs)

    oai_comp.Completions.parse = patched_parse
    oai_comp.AsyncCompletions.parse = patched_a_parse


_apply_patches()


def evaluate_single_deepeval(question: str, answer: str, context: List[str]) -> float:
    """
    Evaluate one sample with DeepEval Faithfulness.
    Returns score (0-1) or -1.0 on failure.
    """
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric

        tc = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context if context else ["No context."],
        )
        metric = FaithfulnessMetric(
            threshold=0.0,
            model="gpt-4.1-mini",
            include_reason=False,
            async_mode=True,
            truths_extraction_limit=15,
        )
        metric.measure(tc)
        return float(metric.score) if metric.score is not None else -1.0
    except Exception as e:
        print(f"    [DeepEval error] {e}")
        return -1.0
