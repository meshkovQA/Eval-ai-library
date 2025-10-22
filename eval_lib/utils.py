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
    Compute a softmax-weighted aggregate of scores with penalty for low-scoring items.

    This function applies softmax weighting (higher scores get more weight) and then
    applies a penalty proportional to the number of low-scoring items.

    Args:
        scores: List of scores (0.0 to 1.0) to aggregate
        temperature: Controls strictness of aggregation
            - Lower (0.1-0.3): Strict - high scores dominate
            - Medium (0.4-0.6): Balanced - default behavior
            - Higher (0.8-1.5): Lenient - closer to arithmetic mean
        penalty: Penalty factor for low-scoring items (default 0.1)
            - Applied to scores <= 0.4

    Returns:
        Aggregated score between 0.0 and 1.0

    Example:
        >>> scores = [1.0, 0.9, 0.7, 0.3, 0.0]
        >>> score_agg(scores, temperature=0.5)
        0.73
    """
    if not scores:
        return 0.0

    # Compute softmax weights
    exp_scores = [exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    softmax_score = sum(s * e / total for s, e in zip(scores, exp_scores))

    # Apply penalty if many statements have low scores (≤ 0.4)
    irrelevant = sum(1 for s in scores if s <= 0.4)
    penalty_factor = max(0.0, 1 - penalty * irrelevant)

    return round(softmax_score * penalty_factor, 4)


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
