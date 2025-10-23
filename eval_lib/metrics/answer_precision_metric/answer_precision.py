'''
Answer Precision Metric: Evaluates how precisely a model's answer matches the expected answer 
using multiple text similarity components and weighted aggregation.

Score Components:
- Exact Match: 1.0 if actual == expected else 0.0
- Normalized Match: 1.0 if normalized(actual) == normalized(expected) else 0.0
- Character Similarity: Ratio of matching characters after normalization.   
- Token Precision: Proportion of expected tokens present in actual answer.
- Numeric Agreement: Proportion of expected numeric values correctly represented in actual answer.


'''

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern


# -------------------------------
# Helpers to normalize and compare text
# -------------------------------

def _normalize_text_basic(text: str) -> str:
    """Lowercase, strip, collapse whitespace; also strip markdown links/URLs."""
    if text is None:
        return ""
    # [label](https://...) -> label
    text = re.sub(r"\[([^\]]+)\]\(\s*https?://[^)]+\s*\)", r"\1", text)
    # bare URLs
    text = re.sub(r"https?://\S+", "", text)
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")


def _normalize_for_tokens(text: str) -> str:
    text = _normalize_text_basic(text)
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _power_mean_score(components: Dict[str, float], weights: Dict[str, float], p: float = 0.3) -> float:
    """
    Weighted power mean:
        M_p = ( sum_i alpha_i * s_i^p )^(1/p)
    alpha_i = w_i / sum(w_i). Recommended 0 < p < 1.
    """
    keys = [k for k in components.keys() if k in weights]
    if not keys:
        return 0.0

    raw = {k: max(0.0, float(weights[k])) for k in keys}
    total = sum(raw.values())
    if total <= 0:
        alpha = {k: 1.0 / len(keys) for k in keys}
    else:
        alpha = {k: raw[k] / total for k in keys}

    acc = 0.0
    for k in keys:
        s = max(0.0, min(1.0, float(components[k])))
        acc += alpha[k] * (s ** p)
    score = acc ** (1.0 / p)
    return max(0.0, min(1.0, score))


STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "to", "of", "in", "on", "for", "with", "as", "by", "at",
    "from", "that", "this", "it", "is", "are", "was", "were",
}


def _line_word_diffs(actual: str, expected: str, drop_stopwords: bool = True) -> List[Dict[str, Any]]:
    """
    Post-line diagnostics: words added/removed per line (human-readable).
    """
    a_lines = (actual or "").splitlines()
    e_lines = (expected or "").splitlines()
    n = max(len(a_lines), len(e_lines))
    diffs: List[Dict[str, Any]] = []

    def words(s: str) -> List[str]:
        return _tokenize(s, drop_stopwords=drop_stopwords)

    for i in range(n):
        e_line = e_lines[i] if i < len(e_lines) else ""
        a_line = a_lines[i] if i < len(a_lines) else ""
        if _normalize_text_basic(e_line) == _normalize_text_basic(a_line):
            continue
        e_set = set(words(e_line))
        a_set = set(words(a_line))
        removed = sorted(list(e_set - a_set))
        added = sorted(list(a_set - e_set))
        if removed or added:
            diffs.append({
                "line_no": i + 1,  # 1-based
                "expected": e_line,
                "actual": a_line,
                "removed": removed,
                "added": added,
            })
    return diffs


def _tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    text = _normalize_for_tokens(text)
    tokens = [t for t in text.split(" ") if t]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_numbers(text: str) -> List[float]:
    nums: List[float] = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            pass
    return nums


def _token_sets(a: str, e: str) -> Tuple[Set[str], Set[str]]:
    a_tokens = set(_tokenize(a))
    e_tokens = set(_tokenize(e))
    return a_tokens, e_tokens


def _token_overlap_coefficient(a: str, e: str) -> float:
    """
    Overlap coefficient: |A ∩ E| / min(|A|, |E|).
    """
    A, E = _token_sets(a, e)
    if not A and not E:
        return 1.0
    if not A or not E:
        return 0.0
    inter = len(A & E)
    return inter / min(len(A), len(E))


@dataclass
class PrecisionConfig:
    token_stopwords: Set[str] = field(default_factory=lambda: set(STOPWORDS))
    numeric_tolerance_abs: float = 0.0
    numeric_tolerance_rel: float = 0.0
    require_expected_present: bool = True
    weights: Optional[Dict[str, float]] = None
    power_p: float = 0.3

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "contains": 0.48,
                "char_ratio": 0.32,
                "token_precision": 0.15,
                "numeric": 0.35,
            }


class AnswerPrecisionMetric(MetricPattern):
    name = "answerPrecisionMetric"

    def __init__(self, model: str, threshold: float = 0.8, verbose: bool = False, config: Optional[PrecisionConfig] = None):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.config = config or PrecisionConfig()

    # --- core similarity components ---
    def _exact_match(self, a: str, e: str) -> float:
        return 1.0 if (a or "") == (e or "") else 0.0

    def _normalized_match(self, a: str, e: str) -> float:
        return 1.0 if _normalize_text_basic(a) == _normalize_text_basic(e) else 0.0

    def _char_similarity(self, a: str, e: str) -> float:
        a_norm = _normalize_text_basic(a)
        e_norm = _normalize_text_basic(e)
        if not a_norm and not e_norm:
            return 1.0
        return SequenceMatcher(a=a_norm, b=e_norm).ratio()

    def _token_precision(self, a: str, e: str) -> Tuple[float, Dict[str, Any]]:
        a_tokens = set(_tokenize(a))
        e_tokens = set(_tokenize(e))
        if not a_tokens:
            return 1.0 if not e_tokens else 0.0, {
                "actual_tokens": [],
                "expected_tokens": sorted(list(e_tokens)),
                "true_positive": [],
                "false_positive": [],
                "false_negative": sorted(list(e_tokens)),
            }
        tp = sorted(list(a_tokens & e_tokens))
        fp = sorted(list(a_tokens - e_tokens))
        fn = sorted(list(e_tokens - a_tokens))
        precision = len(tp) / (len(tp) + len(fp)
                               ) if (len(tp) + len(fp)) > 0 else 0.0
        return precision, {
            "actual_tokens": sorted(list(a_tokens)),
            "expected_tokens": sorted(list(e_tokens)),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    def _numeric_agreement(self, a: str, e: str) -> Tuple[float, Dict[str, Any]]:
        a_nums = _extract_numbers(a)
        e_nums = _extract_numbers(e)
        if not e_nums and not a_nums:
            return 1.0, {"actual_numbers": [], "expected_numbers": [], "matches": [], "mismatches": []}
        if not e_nums and a_nums:
            return 0.0, {"actual_numbers": a_nums, "expected_numbers": [], "matches": [], "mismatches": a_nums}
        tol_abs = self.config.numeric_tolerance_abs
        tol_rel = self.config.numeric_tolerance_rel
        used_idx: Set[int] = set()
        matches: List[Tuple[float, float]] = []
        mismatches: List[Tuple[float, float, float]] = []
        for exp_v in e_nums:
            best_i = None
            best_err = math.inf
            for i, act_v in enumerate(a_nums):
                if i in used_idx:
                    continue
                err = abs(act_v - exp_v)
                if err < best_err:
                    best_err = err
                    best_i = i
            if best_i is None:
                mismatches.append((exp_v, float("nan"), float("inf")))
                continue
            act_v = a_nums[best_i]
            used_idx.add(best_i)
            rel_err = abs(act_v - exp_v) / (abs(exp_v) + 1e-12)
            within = (best_err <= tol_abs) or (rel_err <= tol_rel)
            if within:
                matches.append((exp_v, act_v))
            else:
                mismatches.append((exp_v, act_v, rel_err))
        score = len(matches) / \
            len(e_nums) if e_nums else (1.0 if not a_nums else 0.0)
        detail = {
            "actual_numbers": a_nums,
            "expected_numbers": e_nums,
            "matches": matches,
            "mismatches": mismatches,
            "tolerance_abs": tol_abs,
            "tolerance_rel": tol_rel,
        }
        return score, detail

    # --- evaluation entrypoint ---
    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        if self.config.require_expected_present and not test_case.expected_output:
            return {
                "score": 0.0,
                "success": False,
                "reason": "No expected output provided.",
                "evaluation_cost": 0.0,
                "evaluation_log": {"note": "expected_output is required for AnswerPrecisionMetric"},
            }

        w = self.config.weights

        exact = self._exact_match(actual, expected)
        normalized = self._normalized_match(actual, expected)
        contains = _token_overlap_coefficient(actual, expected)
        char_ratio = self._char_similarity(actual, expected)
        token_prec, token_detail = self._token_precision(actual, expected)
        numeric_score, numeric_detail = self._numeric_agreement(
            actual, expected)

        # ------- Diagnostics / Issues -------
        missing_terms = token_detail.get("false_negative", [])
        extra_terms = token_detail.get("false_positive", [])

        numeric_mismatches_raw = numeric_detail.get("mismatches", [])
        line_diffs = _line_word_diffs(actual, expected, drop_stopwords=True)

        # Human-readable diagnostics
        num_msgs: List[str] = []
        normalized_numeric_mismatches: List[Dict[str, Any]] = []
        for m in numeric_mismatches_raw:
            if isinstance(m, (list, tuple)) and len(m) >= 2:
                exp = m[0]
                act = m[1]
                rel = float(m[2]) if (isinstance(m, (list, tuple))
                                      and len(m) > 2) else None
                num_msgs.append(f"expected {exp}, got {act}" + (
                    f" (rel_err={rel:.4g})" if rel is not None and rel != float('inf') else ""))
                normalized_numeric_mismatches.append(
                    {"expected": exp, "actual": act, "rel_err": rel})
            else:
                # unexpected format
                num_msgs.append(f"unexpected number {m}")
                normalized_numeric_mismatches.append(
                    {"expected": None, "actual": m, "rel_err": None})

        if num_msgs:
            numbers_summary = "numeric mismatches: " + "; ".join(num_msgs)
        else:
            numbers_summary = "Numbers: all matched."

        missing_summary = f"Missing terms: {missing_terms}" if missing_terms else "Missing terms: none."
        extra_summary = f"Extra terms: {extra_terms}" if extra_terms else "Extra terms: none."
        diagnostics = f"{missing_summary} {extra_summary} {numbers_summary}"

        # --- Aggregation via power mean ---
        component_scores = {
            "contains": contains,
            "char_ratio": char_ratio,
            "token_precision": token_prec,
            "numeric": numeric_score,
        }
        base_score = _power_mean_score(
            component_scores, w, p=self.config.power_p)

        # Penalties
        fp = token_detail["false_positive"]
        heavy_penalty = 0.0
        if len(fp) >= 5 and token_prec < 0.7:
            heavy_penalty = 0.05

        final_score = base_score * (1.0 - heavy_penalty)
        final_score = max(0.0, min(1.0, final_score))
        success = final_score >= self.threshold

        # Human-readable reason
        reason_bits: List[str] = []
        if exact == 1.0:
            reason_bits.append("exact match")
        elif normalized == 1.0:
            reason_bits.append("normalized match")
        else:
            reason_bits.append(f"char similarity {char_ratio:.2f}")
            reason_bits.append(f"token precision {token_prec:.2f}")
        if numeric_detail["expected_numbers"] or numeric_detail["actual_numbers"]:
            reason_bits.append(f"numeric agreement {numeric_score:.2f}")
        if contains == 1.0:
            reason_bits.append("full containment")
        elif contains > 0:
            reason_bits.append(f"containment {contains:.2f}")
        if heavy_penalty > 0:
            reason_bits.append("penalized for many extra tokens")
        reason_bits.append(
            f"power-mean aggregation (p={self.config.power_p:.2f}); {missing_summary} {extra_summary}")
        reason = ", ".join(reason_bits)  # <<< фикс: формируем reason

        evaluation_log = {
            "actual": actual,
            "expected": expected,
            "components": {
                "exact": exact,
                "normalized": normalized,
                "contains": contains,
                "char_ratio": char_ratio,
                "token_precision": token_prec,
                "numeric": numeric_score,
                "weights": w,
                "heavy_penalty": heavy_penalty,
            },
            "token_detail": token_detail,
            "numeric_detail": numeric_detail,
            "issues": {
                "missing_terms": missing_terms,
                "extra_terms": extra_terms,
                "numeric_mismatches": normalized_numeric_mismatches,
                "line_diffs": line_diffs,
            },
            "threshold": self.threshold,
            "config": {
                "numeric_tolerance_abs": self.config.numeric_tolerance_abs,
                "numeric_tolerance_rel": self.config.numeric_tolerance_rel,
                "require_expected_present": self.config.require_expected_present,
                "stopwords_count": len(self.config.token_stopwords),
            },
        }

        result = {
            "name": self.name,
            "score": round(final_score, 4),
            "success": success,
            "reason": reason,
            "diagnostics": diagnostics,
            "evaluation_cost": 0.0,
            "evaluation_log": evaluation_log,
        }

        self.print_result(result)

        return result
