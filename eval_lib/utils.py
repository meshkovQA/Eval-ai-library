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
    t_max: float = 1.0,
    p_min: float = -8.0,
    p_max: float = 12.25,  # chosen so that t=0.5 -> p=1
) -> float:
    """
    Map temperature in [t_min, t_max] linearly to power exponent p, with:
      t=0.1 -> p=-8 (very strict)
      t=0.5 -> p=+1 (arithmetic mean)
      t=1.0 -> p=+12.25 (very lenient)
    """
    t = max(t_min, min(t_max, temperature))
    alpha = (t - t_min) / (t_max - t_min)  # in [0,1]
    return p_min + alpha * (p_max - p_min)


def score_agg(
    scores: List[float],
    temperature: float = 0.5,
    penalty: float = 0.1,
    eps_for_neg_p: float = 1e-9
) -> float:
    """
    Aggregate verdict scores with temperature-controlled strictness via power mean.

    - Low temperature (~0.1): strict (p negative) -> close to min
    - Medium temperature (=0.5): balanced (p=1) -> arithmetic mean
    - High temperature (=1.0): lenient (large positive p) -> close to max

    Applies a penalty for "none" verdicts (0.0) only.
    """
    if not scores:
        return 0.0

    p = _map_temperature_to_p(temperature)

    # For negative p, clamp zeros to a small epsilon to avoid 0**p blowing up
    base = [(s if s > 0.0 else eps_for_neg_p)
            for s in scores] if p < 0 else scores

    # Power mean: M_p = ( (Σ s_i^p) / n )^(1/p)
    if abs(p) < 1e-12:
        # Limit p -> 0 is geometric mean
        logs = [math.log(s if s > 0 else eps_for_neg_p) for s in base]
        agg = math.exp(sum(logs) / len(logs))
    else:
        mean_pow = sum(s ** p for s in base) / len(base)
        agg = mean_pow ** (1.0 / p)

    # Apply penalty for "none" verdicts only
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
