"""
TCVA evaluation module.
Runs eval-ai-library FaithfulnessMetric / AnswerRelevancyMetric at multiple temperatures.
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_lib import EvalTestCase, FaithfulnessMetric, AnswerRelevancyMetric
from experiments.config import EVAL_MODEL, TEMPERATURES, VERDICT_WEIGHTS


async def evaluate_single_tcva(
    question: str,
    answer: str,
    context: List[str],
    metric_type: str,
    temperatures: List[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate one sample with TCVA at multiple temperatures.

    Returns: {
        "verdicts": [...],
        "scores": {"T0.2": float, "T0.3": float, ...},
        "ablation_arithmetic": float,
    }
    """
    temperatures = temperatures or TEMPERATURES

    tc = EvalTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context if context else None,
    )

    # Run once at T=0.5 to get verdicts (verdicts are the same regardless of T)
    if metric_type == "faithfulness":
        metric = FaithfulnessMetric(model=EVAL_MODEL, threshold=0.0,
                                     temperature=0.5, verbose=False)
    else:
        metric = AnswerRelevancyMetric(model=EVAL_MODEL, threshold=0.0,
                                        temperature=0.5, verbose=False)

    result = await metric.evaluate(tc)
    verdicts = result.get("evaluation_log", {}).get("verdicts", [])

    # Extract verdict weights
    weights = [VERDICT_WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]

    # Now compute TCVA score at each temperature using score_agg directly
    # (avoids re-calling LLM for each temperature)
    from eval_lib.utils import score_agg

    scores = {}
    for T in temperatures:
        if weights:
            scores[f"T{T}"] = score_agg(weights, temperature=T)
        else:
            scores[f"T{T}"] = 0.0

    # Ablation: 5-level + arithmetic mean (T=0.5 -> p=1 -> arithmetic mean)
    ablation_arith = score_agg(weights, temperature=0.5) if weights else 0.0

    return {
        "verdicts": verdicts,
        "weights": weights,
        "scores": scores,
        "ablation_arithmetic": ablation_arith,
        "base_score": result["score"],
    }
