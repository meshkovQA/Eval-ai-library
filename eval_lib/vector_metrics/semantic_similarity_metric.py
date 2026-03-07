"""
Semantic Similarity Metric: Cosine similarity between actual_output and expected_output embeddings.
Score: cosine similarity 0.0-1.0.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.vector_metrics.embedding_provider import get_embedding_provider


class SemanticSimilarityMetric(MetricPattern):
    name = "semanticSimilarityMetric"

    def __init__(self, threshold: float = 0.7, embedding_provider: str = "openai",
                 model_name: str = None, verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.provider = get_embedding_provider(embedding_provider, model_name)

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        actual = tc.actual_output or ""
        expected = tc.expected_output or ""

        embeddings, cost = await self.provider.embed([actual, expected])
        emb_actual = np.array(embeddings[0])
        emb_expected = np.array(embeddings[1])

        norm_a = np.linalg.norm(emb_actual)
        norm_b = np.linalg.norm(emb_expected)

        if norm_a == 0 or norm_b == 0:
            score = 0.0
        else:
            score = float(np.dot(emb_actual, emb_expected) / (norm_a * norm_b))
            score = max(0.0, min(1.0, score))

        success = score >= self.threshold
        reason = f"Semantic similarity: {score:.4f} (threshold: {self.threshold})"

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": cost,
            "evaluation_log": {
                "cosine_similarity": score,
                "embedding_cost": cost,
                "actual_length": len(actual),
                "expected_length": len(expected),
            }
        }
        self.print_result(result)
        return result
