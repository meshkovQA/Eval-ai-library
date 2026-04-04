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

    Penalty uses proportional none-fraction with temperature-adaptive alpha:
      penalty_factor = (1 - none_frac) ^ alpha
    where alpha = 1.5 at low T (strict) and 0.5 at high T (lenient).
    This avoids double-punishing "none" verdicts that already pull
    the power mean down.
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

    # Proportional penalty: (1 - none_frac) ^ alpha
    # alpha depends on temperature: strict T → higher alpha (harsher), lenient T → lower alpha (softer)
    n = len(scores)
    none_frac = sum(1 for s in scores if s == 0.0) / n if n > 0 else 0.0
    alpha = 1.5 - temperature  # T=0.1 → alpha=1.4 (strict), T=0.5 → alpha=1.0, T=1.0 → alpha=0.5 (lenient)
    penalty_factor = (1.0 - none_frac) ** alpha

    return round(agg * penalty_factor, 4)


def _try_fix_escaped_quotes(text: str) -> str:
    """Fix invalid backslash-escaped quotes outside JSON strings.

    Some models (e.g. GPT-4.1) produce output like:
        "key": \\"value with spaces\\"
    instead of the valid:
        "key": "value with spaces"
    This function detects and repairs that pattern.
    """
    # Pattern: `: \"...\"` where \" should be plain " (value-level escaping)
    # We fix by replacing \" that appears right after : (with optional whitespace)
    # and before , or } or ] with plain "
    fixed = re.sub(r':\s*\\"', ': "', text)
    fixed = re.sub(r'\\"(\s*[,\}\]])', r'"\1', fixed)
    if fixed != text:
        try:
            json.loads(fixed)
            return fixed
        except Exception:
            pass
    return text


def _try_fix_truncated_json(text: str) -> str:
    """Attempt to fix truncated JSON by closing open brackets/braces."""
    open_brackets = 0
    open_braces = 0
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1
        elif ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1

    if open_braces > 0 or open_brackets > 0:
        # Remove trailing incomplete key-value pair or element
        # Find last complete element (ends with }, ], number, string, true/false/null)
        stripped = text.rstrip()
        # Remove trailing comma if present
        if stripped.endswith(','):
            stripped = stripped[:-1]
        # Close remaining brackets/braces
        result = stripped + '}' * open_braces + ']' * open_brackets
        try:
            json.loads(result)
            return result
        except Exception:
            pass

    return text


def extract_json_block(text: str) -> str:
    """
    Extract JSON from LLM response that may contain markdown code blocks.

    This function handles various formats:
    - Markdown JSON code blocks: ```json ... ```
    - Plain JSON objects/arrays
    - JSON embedded in text
    - Truncated JSON (attempts to close open brackets)

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
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            fixed = _try_fix_truncated_json(candidate)
            try:
                json.loads(fixed)
                return fixed
            except Exception:
                pass

    # Try to parse as direct JSON
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    # Try fixing escaped quotes (common GPT-4.1 issue: \"value\" instead of "value")
    fixed_quotes = _try_fix_escaped_quotes(text)
    if fixed_quotes != text:
        return fixed_quotes

    # Try to find JSON object/array pattern (greedy to capture full nested structures)
    json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            # Try fixing escaped quotes in the candidate
            fixed = _try_fix_escaped_quotes(candidate)
            if fixed != candidate:
                return fixed

    # Try to find start of JSON and fix truncation
    json_start = re.search(r"[\[{]", text)
    if json_start:
        candidate = text[json_start.start():].strip()
        fixed = _try_fix_truncated_json(candidate)
        try:
            json.loads(fixed)
            return fixed
        except Exception:
            pass

    # Return as-is if nothing found
    return text.strip()
