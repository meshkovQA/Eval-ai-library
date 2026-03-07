"""
Reference Match Metric: Similarity of actual_output against multiple reference texts.
Score: aggregated (max or mean) cosine similarity 0.0-1.0.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.vector_metrics.embedding_provider import get_embedding_provider


class ReferenceMatchMetric(MetricPattern):
    name = "referenceMatchMetric"

    def __init__(self, threshold: float = 0.7, references: List[str] = None,
                 aggregation: str = "max", embedding_provider: str = "openai",
                 model_name: str = None, verbose: bool = False):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.references = references or []
        self.aggregation = aggregation
        self.provider = get_embedding_provider(embedding_provider, model_name)

    async def evaluate(self, tc: EvalTestCase) -> Dict[str, Any]:
        actual = tc.actual_output or ""

        if not self.references:
            result = {
                "name": self.name,
                "score": 0.0,
                "success": False,
                "reason": "No reference texts provided.",
                "evaluation_cost": 0.0,
                "evaluation_log": {"error": "No references"},
            }
            self.print_result(result)
            return result

        texts = [actual] + self.references
        embeddings, cost = await self.provider.embed(texts)

        emb_actual = np.array(embeddings[0])
        norm_a = np.linalg.norm(emb_actual)

        similarities = []
        for i, ref_emb in enumerate(embeddings[1:]):
            emb_ref = np.array(ref_emb)
            norm_r = np.linalg.norm(emb_ref)
            if norm_a == 0 or norm_r == 0:
                sim = 0.0
            else:
                sim = float(np.dot(emb_actual, emb_ref) / (norm_a * norm_r))
                sim = max(0.0, min(1.0, sim))
            similarities.append(sim)

        if self.aggregation == "max":
            score = max(similarities)
        else:
            score = float(np.mean(similarities))

        success = score >= self.threshold
        reason = (f"Reference match ({self.aggregation}): {score:.4f} "
                  f"across {len(self.references)} references")

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": cost,
            "evaluation_log": {
                "aggregation": self.aggregation,
                "similarities": [round(s, 4) for s in similarities],
                "num_references": len(self.references),
                "embedding_cost": cost,
            }
        }
        self.print_result(result)
        return result
