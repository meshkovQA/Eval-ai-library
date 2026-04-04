"""
RAGAS evaluation module.
Runs RAGAS faithfulness metric (binary verdicts).
Requires: ragas>=0.3, langchain-openai
"""
import asyncio
import os
from typing import List

from experiments.config import EVAL_MODEL

# Map common model names to langchain-openai compatible names
_RAGAS_LLM = None


def _get_ragas_llm():
    global _RAGAS_LLM
    if _RAGAS_LLM is None:
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        _RAGAS_LLM = LangchainLLMWrapper(ChatOpenAI(model=EVAL_MODEL))
    return _RAGAS_LLM


def evaluate_single_ragas(question: str, answer: str, context: List[str]) -> float:
    """
    Evaluate one sample with RAGAS Faithfulness.
    Returns score (0-1) or -1.0 on failure.
    """
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=context,
        )
        scorer = Faithfulness(llm=_get_ragas_llm())

        loop = asyncio.new_event_loop()
        try:
            score = loop.run_until_complete(scorer.single_turn_ascore(sample))
        finally:
            loop.close()

        return float(score) if score is not None else -1.0
    except Exception as e:
        print(f"    [RAGAS error] {e}")
        return -1.0
