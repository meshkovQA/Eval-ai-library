"""Shared helpers for the HuggingFace sequence-classification detectors.

Three security metrics run a local ``AutoModelForSequenceClassification``.
They used to hard-code two assumptions about the model head that are not
true in general:

* that the logits form a *distribution* (softmax) — false for multi-label
  heads such as ``unitary/toxic-bert`` (``problem_type ==
  "multi_label_classification"``), which are independent sigmoids;
* that index ``1`` is the "harmful" class — false whenever the label order
  differs, and meaningless for a multi-label head (index 1 of toxic-bert is
  ``severe_toxic``, not "toxic").

Both facts live in ``model.config`` (``problem_type``, ``id2label``); these
helpers read them instead of guessing. Everything here is plain Python so it
can be unit-tested without torch or a downloaded model.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

# Label names that denote the *absence* of harm. Matched case-insensitively
# after normalising ``-``/`` `` to ``_``.
SAFE_LABEL_NAMES = frozenset({
    "safe", "benign", "neutral", "clean", "ok", "normal", "legit", "legitimate",
    "non_toxic", "not_toxic", "nontoxic", "no", "negative", "label_0",
})

# Label names that denote harm, for the case where no safe label is found.
HARMFUL_LABEL_NAMES = frozenset({
    "toxic", "harmful", "unsafe", "hate", "hateful", "offensive", "abusive",
    "jailbreak", "injection", "attack", "malicious", "unsafe_content",
    "yes", "positive", "label_1",
})


def normalise_label(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def id2label_of(config: Any) -> Dict[int, str]:
    """``config.id2label`` as ``{int: str}``, tolerant of string keys."""
    raw = getattr(config, "id2label", None)
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return out


def is_multi_label(config: Any) -> bool:
    """True when the head is independent sigmoids rather than a softmax.

    ``problem_type`` is the authoritative signal; a head with more than two
    labels and no declared problem type is treated as multi-label only when
    its labels look like a toxicity taxonomy (no "safe"-style label present),
    since a genuine multi-class head always has a safe/neutral class.
    """
    problem_type = getattr(config, "problem_type", None)
    if problem_type == "multi_label_classification":
        return True
    if problem_type in ("single_label_classification", "regression"):
        return False
    labels = [normalise_label(v) for v in id2label_of(config).values()]
    if len(labels) > 2 and not any(label in SAFE_LABEL_NAMES for label in labels):
        return True
    return False


def harmful_label_index(id2label: Dict[int, str], default: int = 1) -> Tuple[int, str]:
    """Index of the "harmful" class for a single-label head.

    Resolution order: the one label that is *not* a safe-name (binary heads);
    a label whose name is a known harmful-name; otherwise ``default``
    (index ``1`` when the head has ≥2 labels, else ``0``). The second item
    says how the index was found — ``"safe_complement"``, ``"harmful_name"``
    or ``"default"`` — so callers can log when they are guessing.
    """
    if not id2label:
        return default, "default"

    names = {i: normalise_label(n) for i, n in id2label.items()}

    if len(names) == 2:
        safe = [i for i, n in names.items() if n in SAFE_LABEL_NAMES]
        if len(safe) == 1:
            return next(i for i in names if i not in safe), "safe_complement"

    for i, n in names.items():
        if n in HARMFUL_LABEL_NAMES:
            return i, "harmful_name"

    if len(names) >= 2 and default in names:
        return default, "default"
    return min(names), "default"


def probabilities_from_logits(logits: Any, multi_label: bool) -> List[float]:
    """Sigmoid (multi-label) or softmax (single-label) over one row of logits.

    ``logits`` is a torch tensor of shape ``(1, num_labels)`` or a plain
    sequence; the result is a plain list of floats.
    """
    if hasattr(logits, "shape"):
        row = logits[0] if len(getattr(logits, "shape", ())) == 2 else logits
    else:
        row = logits
        # A plain nested sequence ([[...]]) — unwrap the single batch row.
        if isinstance(row, (list, tuple)) and row and isinstance(row[0], (list, tuple)):
            row = row[0]
    try:
        import torch  # local import — the helper must import without torch

        if isinstance(row, torch.Tensor):
            if multi_label:
                return torch.sigmoid(row).detach().cpu().tolist()
            return torch.softmax(row, dim=-1).detach().cpu().tolist()
    except ImportError:
        pass

    import math

    values = [float(v) for v in row]
    if multi_label:
        return [1.0 / (1.0 + math.exp(-v)) for v in values]
    peak = max(values) if values else 0.0
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def label_scores(probabilities: Iterable[float], id2label: Dict[int, str]) -> Dict[str, float]:
    """``{label_name: probability}`` keyed by the model's own (normalised) names."""
    return {
        normalise_label(id2label.get(i, f"label_{i}")): float(p)
        for i, p in enumerate(probabilities)
    }
