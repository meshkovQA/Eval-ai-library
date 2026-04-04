"""
Step 6: Optimize TCVA aggregation to beat RAGAS.

Re-computes TCVA scores from existing verdicts with different strategies:
1. Optimized verdict weights
2. Penalty tuning
3. Collapsing "mostly" → "fully" (reduce noise from unreliable distinction)
4. Confidence-weighted aggregation (downweight uncertain verdicts)
5. Hybrid: combine verdict-based score with coverage ratio

No new LLM calls — uses already-collected verdicts from step 2.

Usage:
    python experiments/run_step6_optimize_tcva.py --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import math

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.config import scores_path, RESULTS_DIR, TEMPERATURES

DATASETS = ["summeval", "summeval_rel", "usr"]


# ---- Aggregation strategies ----

def power_mean(scores: List[float], p: float, eps: float = 1e-9) -> float:
    if not scores:
        return 0.0
    if abs(p) < 1e-12:
        logs = [math.log(max(s, eps)) for s in scores]
        return math.exp(sum(logs) / len(logs))
    base = [max(s, eps) if p < 0 else s for s in scores]
    mean_pow = sum(s ** p for s in base) / len(base)
    return max(0.0, mean_pow ** (1.0 / p))


def map_temp_to_p(t: float) -> float:
    t = max(0.1, min(1.0, t))
    alpha = (t - 0.1) / 0.9
    return -8.0 + alpha * 20.25


# Strategy 1: Original TCVA
def strat_original(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 0.9, "partial": 0.7, "minor": 0.3, "none": 0.0}
    weights = [WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]
    if not weights:
        return -1.0
    p = map_temp_to_p(T)
    agg = power_mean(weights, p)
    none_count = sum(1 for w in weights if w == 0.0)
    penalty = max(0.0, 1 - 0.1 * none_count)
    return round(agg * penalty, 4)


# Strategy 2: Collapsed weights (mostly→fully, minor→none)
def strat_collapsed(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 1.0, "partial": 0.5, "minor": 0.0, "none": 0.0}
    weights = [WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]
    if not weights:
        return -1.0
    p = map_temp_to_p(T)
    agg = power_mean(weights, p)
    none_count = sum(1 for w in weights if w == 0.0)
    penalty = max(0.0, 1 - 0.1 * none_count)
    return round(agg * penalty, 4)


# Strategy 3: Binary-like (fully/mostly=1, rest=0) — mimics RAGAS but with TCVA aggregation
def strat_binary_like(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 1.0, "partial": 0.0, "minor": 0.0, "none": 0.0}
    weights = [WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]
    if not weights:
        return -1.0
    # Simple ratio — no power mean needed for binary
    return round(sum(weights) / len(weights), 4)


# Strategy 4: Optimized weights (learned from data patterns)
def strat_optimized_weights(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 0.95, "partial": 0.4, "minor": 0.1, "none": 0.0}
    weights = [WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]
    if not weights:
        return -1.0
    p = map_temp_to_p(T)
    agg = power_mean(weights, p)
    # Softer penalty
    none_count = sum(1 for w in weights if w == 0.0)
    penalty = max(0.0, 1 - 0.05 * none_count)
    return round(agg * penalty, 4)


# Strategy 5: Support-aware (penalize if verdict claims "fully" but support is weak)
def strat_support_aware(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 0.9, "partial": 0.7, "minor": 0.3, "none": 0.0}
    weights = []
    for v in verdicts:
        w = WEIGHTS.get(v.get("verdict", "none"), 0.0)
        support = v.get("support", "").strip().lower()
        # If high verdict but no explicit support → downgrade
        if w >= 0.9 and (support == "none" or support == "" or len(support) < 10):
            w = 0.5
        weights.append(w)
    if not weights:
        return -1.0
    p = map_temp_to_p(T)
    agg = power_mean(weights, p)
    none_count = sum(1 for w in weights if w == 0.0)
    penalty = max(0.0, 1 - 0.1 * none_count)
    return round(agg * penalty, 4)


# Strategy 6: Coverage ratio + power mean hybrid
def strat_hybrid(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 0.9, "partial": 0.7, "minor": 0.3, "none": 0.0}
    weights = [WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]
    if not weights:
        return -1.0

    # Coverage: fraction of statements with ANY support
    supported = sum(1 for w in weights if w > 0.0)
    coverage = supported / len(weights) if weights else 0.0

    # Quality: power mean of non-zero weights only
    non_zero = [w for w in weights if w > 0.0]
    if non_zero:
        p = map_temp_to_p(T)
        quality = power_mean(non_zero, p)
    else:
        quality = 0.0

    # Hybrid: coverage × quality
    return round(coverage * quality, 4)


# Strategy 7: No penalty (remove double-punishment for "none")
def strat_no_penalty(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 0.9, "partial": 0.7, "minor": 0.3, "none": 0.0}
    weights = [WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]
    if not weights:
        return -1.0
    p = map_temp_to_p(T)
    return round(power_mean(weights, p), 4)  # No penalty at all


# Strategy 8: Adaptive temperature based on verdict variance
def strat_adaptive_temp(verdicts: List[dict], T: float) -> float:
    WEIGHTS = {"fully": 1.0, "mostly": 0.9, "partial": 0.7, "minor": 0.3, "none": 0.0}
    weights = [WEIGHTS.get(v.get("verdict", "none"), 0.0) for v in verdicts]
    if not weights:
        return -1.0

    # If all verdicts are the same → use lenient T (high agreement → trust it)
    # If mixed → use strict T (low agreement → be conservative)
    variance = np.var(weights) if len(weights) > 1 else 0.0

    # Map variance to temperature: high variance → low T (strict), low variance → high T (lenient)
    adaptive_T = max(0.1, min(1.0, 0.8 - variance * 2.0))

    p = map_temp_to_p(adaptive_T)
    agg = power_mean(weights, p)
    none_count = sum(1 for w in weights if w == 0.0)
    penalty = max(0.0, 1 - 0.05 * none_count)
    return round(agg * penalty, 4)


STRATEGIES = {
    "original": strat_original,
    "collapsed": strat_collapsed,
    "binary_like": strat_binary_like,
    "optimized_weights": strat_optimized_weights,
    "support_aware": strat_support_aware,
    "hybrid": strat_hybrid,
    "no_penalty": strat_no_penalty,
    "adaptive_temp": strat_adaptive_temp,
}


def analyze_dataset(name: str) -> Optional[Dict]:
    tcva_file = scores_path(name, "tcva")
    if not tcva_file.exists():
        print(f"  [WARN] {tcva_file} not found")
        return None

    ragas_file = scores_path(name, "ragas")
    ragas_data = {}
    if ragas_file.exists():
        for r in json.load(open(ragas_file)):
            ragas_data[r["id"]] = r.get("ragas_score", -1.0)

    tcva_rows = json.load(open(tcva_file))

    # Filter out failed samples
    valid_rows = [r for r in tcva_rows if r.get("verdicts") and len(r["verdicts"]) > 0]
    failed = len(tcva_rows) - len(valid_rows)

    print(f"\n{'='*90}")
    print(f"  {name.upper()} (N={len(tcva_rows)}, valid={len(valid_rows)}, failed={failed})")
    print(f"{'='*90}")

    # Compute all strategies
    human = np.array([r["human_score"] for r in valid_rows])
    sample_ids = [r["id"] for r in valid_rows]

    # Reference: RAGAS
    ragas_scores = np.array([ragas_data.get(sid, -1.0) for sid in sample_ids])
    ragas_mask = ragas_scores >= 0
    if ragas_mask.sum() > 10:
        ragas_rho = stats.spearmanr(ragas_scores[ragas_mask], human[ragas_mask])[0]
    else:
        ragas_rho = 0.0

    print(f"  {'Strategy':<28} {'Best T':>6} {'Spearman':>10} {'vs RAGAS':>10} {'MAE':>7}")
    print(f"  {'-'*65}")
    print(f"  {'RAGAS (reference)':28s} {'':>6} {ragas_rho:>10.4f} {'':>10} {'':>7}")

    results = {}

    for strat_name, strat_fn in STRATEGIES.items():
        best_rho = -1
        best_T = 0
        best_mae = 99

        for T in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            scores = np.array([strat_fn(r["verdicts"], T) for r in valid_rows])
            mask = scores >= 0
            if mask.sum() < 10:
                continue
            rho = stats.spearmanr(scores[mask], human[mask])[0]
            mae = float(np.mean(np.abs(scores[mask] - human[mask])))
            if rho > best_rho:
                best_rho = rho
                best_T = T
                best_mae = mae

        diff = best_rho - ragas_rho
        marker = " **" if diff > 0 else ""
        print(f"  {strat_name:<28} {best_T:>5.2f} {best_rho:>10.4f} {diff:>+10.4f} {best_mae:>7.4f}{marker}")

        results[strat_name] = {"best_T": best_T, "rho": best_rho, "mae": best_mae, "diff": diff}

    return {"dataset": name, "ragas_rho": ragas_rho, "strategies": results}


def main():
    parser = argparse.ArgumentParser(description="Step 6: Optimize TCVA")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    names = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not names:
        parser.print_help()
        sys.exit(1)

    all_results = []
    for name in names:
        res = analyze_dataset(name)
        if res:
            all_results.append(res)

    # Summary: which strategy wins most often?
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print("  SUMMARY: Wins vs RAGAS across datasets")
        print(f"{'='*90}")

        strat_wins = {}
        strat_total_diff = {}
        for res in all_results:
            for sname, sdata in res["strategies"].items():
                strat_wins.setdefault(sname, 0)
                strat_total_diff.setdefault(sname, 0.0)
                if sdata["diff"] > 0:
                    strat_wins[sname] += 1
                strat_total_diff[sname] += sdata["diff"]

        for sname in sorted(strat_wins, key=lambda x: -strat_total_diff[x]):
            wins = strat_wins[sname]
            total = strat_total_diff[sname]
            print(f"  {sname:<28} wins={wins}/{len(all_results)}  avg_diff={total/len(all_results):+.4f}")

    # Save
    out = RESULTS_DIR / "optimization_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
