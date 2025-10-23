"""
Utility functions for metrics evaluation
"""
import re
import json
from typing import List
import math


"""
Utility functions for metrics evaluation
"""


def _map_temperature_to_p(
    temperature: float,
    t_min: float = 0.1,
    t_max: float = 2.0,
    p_min: float = -8.0,
    p_max: float = 8.0
) -> float:
    """Map temperature value to power exponent (p) for the generalized mean."""
    t = max(t_min, min(t_max, temperature))
    alpha = (t - t_min) / (t_max - t_min)
    return p_min + alpha * (p_max - p_min)


def score_agg(
    scores: List[float],
    temperature: float = 1.0,
    penalty: float = 0.1,
    eps_for_neg_p: float = 1e-9
) -> float:
    """
    Aggregate verdict scores with temperature-controlled strictness.

    Uses the **power mean (generalized mean)** model:
      - Low temperature → strict (negative power → closer to min)
      - High temperature → lenient (positive power → closer to max)

    Additionally, applies a penalty for "none" verdicts (0.0 scores).

    Args:
        scores: List of scores (values between 0.0 and 1.0)
        temperature: Strictness control (0.1 = very strict, 2.0 = very lenient)
        penalty: Penalty factor for "none" verdicts (default: 0.1)
        eps_for_neg_p: Small epsilon to avoid division by zero when p < 0

    Returns:
        Aggregated score between 0.0 and 1.0
    """
    if not scores:
        return 0.0

    # Map temperature to the power exponent p
    p = _map_temperature_to_p(temperature)

    # Handle zero scores for negative powers to avoid infinity
    if p < 0:
        base = [(s if s > 0.0 else eps_for_neg_p) for s in scores]
    else:
        base = scores

    # Compute power mean:
    # M_p = ( (1/n) * Σ(s_i^p) )^(1/p)
    if abs(p) < 1e-12:
        # Limit as p → 0 → geometric mean
        logs = [math.log(s if s > 0 else eps_for_neg_p) for s in base]
        agg = math.exp(sum(logs) / len(logs))
    else:
        s_pow = [s ** p for s in base]
        mean_pow = sum(s_pow) / len(s_pow)
        agg = mean_pow ** (1.0 / p)

    # Apply penalty for "none" verdicts
    none_count = sum(1 for s in scores if s == 0.0)
    penalty_factor = max(0.0, 1 - penalty * none_count)

    return round(agg * penalty_factor, 4)


def extract_json_block(text: str) -> str:
    """
    Extract JSON from LLM response that may contain markdown code blocks.

    This function handles various formats:
    - Markdown JSON code blocks: ```json ... ```
    - Plain JSON objects/arrays
    - JSON embedded in text

    Args:
        text: Raw text from LLM that may contain JSON

    Returns:
        Extracted JSON string

    Raises:
        No exception - returns original text if no JSON found

    Example:
        >>> text = '```json\\n{"score": 0.8}\\n```'
        >>> extract_json_block(text)
        '{"score": 0.8}'
    """
    # Try to extract from markdown code blocks
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to parse as direct JSON
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    # Try to find JSON object/array pattern
    json_match = re.search(r"({.*?}|\[.*?\])", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # Return as-is if nothing found
    return text.strip()
