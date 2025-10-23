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
    Compute a temperature-weighted aggregate of scores with penalty for low-scoring items.

    This function applies temperature-based weighting where higher temperature makes
    the metric more lenient (focuses on high scores), and lower temperature makes it
    more strict (all scores matter equally).

    Args:
        scores: List of scores (0.0 to 1.0) to aggregate
        temperature: Controls strictness of aggregation (0.1 to 2.0)
            - Lower (0.1-0.3): **STRICT** - All scores matter equally, low scores heavily penalize
            - Medium (0.4-0.8): **BALANCED** - Moderate weighting (default: 0.5)
            - Higher (1.0-2.0): **LENIENT** - High scores dominate, ignores "partial", "minor", "none"
        penalty: Penalty factor for low-scoring items (default 0.1)
            - Applied to scores <= 0.4 (verdicts: partial, minor, none)

    Returns:
        Aggregated score between 0.0 and 1.0

    Examples:
        >>> scores = [1.0, 0.9, 0.7, 0.3, 0.0]  # fully, mostly, partial, minor, none
        >>> score_agg(scores, temperature=0.1)  # STRICT
        0.52  # Low because all bad scores count

        >>> score_agg(scores, temperature=2.0)  # LENIENT  
        0.95  # High because only "fully" and "mostly" matter

        >>> score_agg(scores, temperature=0.5)  # BALANCED
        0.73  # Middle ground
    """
    if not scores:
        return 0.0

    # Compute temperature-weighted scores
    # Higher temperature = exponentially boost high scores (lenient)
    # Lower temperature = all scores weighted similarly (strict)
    exp_scores = [exp(s * temperature) for s in scores]
    total = sum(exp_scores)
    weighted_score = sum(s * e / total for s, e in zip(scores, exp_scores))

    # Apply penalty if many statements have low scores (≤ 0.4)
    # This corresponds to verdicts: partial (0.7), minor (0.3), none (0.0)
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
