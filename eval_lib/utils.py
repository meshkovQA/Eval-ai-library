"""
Utility functions for metrics evaluation
"""
import re
import json
from typing import List
from math import exp


"""
Utility functions for metrics evaluation
"""


def score_agg(
    scores: List[float],
    temperature: float = 0.5,
    penalty: float = 0.1
) -> float:
    """
    Compute a temperature-weighted aggregate of scores with penalty for "none" verdicts.

    This function applies temperature-based weighting using power function (score^temperature)
    where higher temperature makes the metric more lenient by exponentially suppressing low scores.

    Args:
        scores: List of scores (0.0 to 1.0) to aggregate
        temperature: Controls strictness of aggregation (0.1 to 3.0)
            - Lower (0.1-0.5): **STRICT** - All scores matter, low scores heavily penalize
            - Medium (0.6-1.2): **BALANCED** - Moderate weighting (default: 1.0 = arithmetic mean)
            - Higher (1.5-3.0): **LENIENT** - High scores dominate, aggressively ignores low scores
        penalty: Penalty factor for "none" verdicts (default 0.1)
            - Applied only to scores == 0.0 (verdict: "none")

    Returns:
        Aggregated score between 0.0 and 1.0

    """
    if not scores:
        return 0.0

    # Temperature-based weighting using power function: score^temperature
    # - temperature < 1.0: Favors low scores (strict)
    # - temperature = 1.0: Arithmetic mean (balanced)
    # - temperature > 1.0: Suppresses low scores (lenient)
    weighted_scores = [s ** (1/temperature) for s in scores]
    weighted_score = sum(weighted_scores) / len(weighted_scores)

    # Apply penalty ONLY for "none" verdicts (0.0)
    none_count = sum(1 for s in scores if s == 0.0)
    penalty_factor = max(0.0, 1 - penalty * none_count)

    return round(weighted_score * penalty_factor, 4)


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
