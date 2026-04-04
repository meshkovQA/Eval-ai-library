"""
Step 9: Ablation study — measure contribution of each TCVA component.

Configurations (all reuse stored verdicts, NO new LLM calls):
  A. Full TCVA          — 5-level verdicts + power mean + None-penalty
  B. TCVA w/o penalty   — 5-level verdicts + power mean, NO None-penalty
  C. 5-level + arith    — 5-level verdicts + arithmetic mean (T=0.5 fixed) + None-penalty
  D. Binary + power mean — map verdicts to {1.0, 0.0} + power mean + penalty

Config D simulates binary verdicts by collapsing: fully/mostly → 1.0, rest → 0.0

Usage:
    python experiments/run_step9_ablation.py
"""
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import scores_path, RESULTS_DIR, VERDICT_WEIGHTS

DATASETS = ["summeval", "summeval_rel", "usr"]
BEST_TEMPS = {"summeval": 0.9, "summeval_rel": 0.5, "usr": 0.9}


# ── Score aggregation variants ───────────────────────────────────────────

def _map_temperature_to_p(temperature: float) -> float:
    t = max(0.1, min(1.0, temperature))
    return -8.0 + ((t - 0.1) / 0.9) * 20.25


def agg_full_tcva(weights: List[float], temperature: float) -> float:
    """Config A: full TCVA (power mean + penalty)."""
    if not weights:
        return 0.0
    p = _map_temperature_to_p(temperature)
    eps = 1e-9
    base = [(w if w > 0 else eps) for w in weights] if p < 0 else list(weights)

    if abs(p) < 1e-12:
        logs = [math.log(max(s, eps)) for s in base]
        agg = math.exp(sum(logs) / len(logs))
    else:
        mean_pow = sum(s ** p for s in base) / len(base)
        agg = mean_pow ** (1.0 / p)

    n = len(weights)
    none_frac = sum(1 for w in weights if w == 0.0) / n
    alpha = 1.5 - temperature
    penalty_factor = (1.0 - none_frac) ** alpha
    return round(agg * penalty_factor, 4)


def agg_no_penalty(weights: List[float], temperature: float) -> float:
    """Config B: power mean WITHOUT None-penalty."""
    if not weights:
        return 0.0
    p = _map_temperature_to_p(temperature)
    eps = 1e-9
    base = [(w if w > 0 else eps) for w in weights] if p < 0 else list(weights)

    if abs(p) < 1e-12:
        logs = [math.log(max(s, eps)) for s in base]
        agg = math.exp(sum(logs) / len(logs))
    else:
        mean_pow = sum(s ** p for s in base) / len(base)
        agg = mean_pow ** (1.0 / p)

    return round(agg, 4)


def agg_arithmetic_with_penalty(weights: List[float], temperature: float) -> float:
    """Config C: arithmetic mean (p=1 fixed) + None-penalty."""
    if not weights:
        return 0.0
    agg = sum(weights) / len(weights)

    n = len(weights)
    none_frac = sum(1 for w in weights if w == 0.0) / n
    alpha = 1.5 - temperature
    penalty_factor = (1.0 - none_frac) ** alpha
    return round(agg * penalty_factor, 4)


def binarize_weights(verdicts: List[Dict]) -> List[float]:
    """Config D: collapse verdicts to binary — fully/mostly → 1.0, rest → 0.0."""
    binary = []
    for v in verdicts:
        vname = v["verdict"]
        if vname in ("fully", "mostly"):
            binary.append(1.0)
        else:
            binary.append(0.0)
    return binary


# ── Main ─────────────────────────────────────────────────────────────────

def run_ablation() -> Dict:
    print("=" * 86)
    print("  ABLATION STUDY")
    print("=" * 86)

    all_results = {}

    for ds in DATASETS:
        tcva_file = scores_path(ds, "tcva")
        if not tcva_file.exists():
            print(f"  [{ds}] not found, skipping")
            continue

        with open(tcva_file) as f:
            data = json.load(f)

        samples = [r for r in data if r.get("verdicts")]
        human = np.array([r["human_score"] for r in samples])
        T = BEST_TEMPS[ds]

        configs = {}

        # A: Full TCVA
        scores_a = np.array([
            agg_full_tcva(
                [VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in r["verdicts"]],
                T
            ) for r in samples
        ])
        configs["A. Full TCVA"] = scores_a

        # B: No penalty
        scores_b = np.array([
            agg_no_penalty(
                [VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in r["verdicts"]],
                T
            ) for r in samples
        ])
        configs["B. No penalty"] = scores_b

        # C: Arithmetic mean (+ penalty)
        scores_c = np.array([
            agg_arithmetic_with_penalty(
                [VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in r["verdicts"]],
                T
            ) for r in samples
        ])
        configs["C. Arithmetic"] = scores_c

        # D: Binary verdicts + power mean + penalty
        scores_d = np.array([
            agg_full_tcva(binarize_weights(r["verdicts"]), T)
            for r in samples
        ])
        configs["D. Binary"] = scores_d

        print(f"\n  Dataset: {ds.upper()} (N={len(samples)}, T={T})")
        print(f"  {'Config':<22} {'Spearman rho':>14} {'p-value':>12} {'MAE':>8}")
        print(f"  {'-' * 58}")

        ds_results = {}
        for label, scores in configs.items():
            rho, p_val = stats.spearmanr(scores, human)
            mae = float(np.mean(np.abs(scores - human)))
            ds_results[label] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": float(p_val),
                "mae": round(mae, 4),
            }
            print(f"  {label:<22} {rho:>14.4f} {p_val:>12.2e} {mae:>8.4f}")

        # Deltas vs full TCVA
        base_rho = ds_results["A. Full TCVA"]["spearman_rho"]
        print(f"\n  Deltas vs Full TCVA:")
        for label, res in ds_results.items():
            if label == "A. Full TCVA":
                continue
            delta = res["spearman_rho"] - base_rho
            print(f"    {label:<22} delta = {delta:+.4f}")

        all_results[ds] = ds_results

    return all_results


def main():
    print("\nRunning Step 9: Ablation Study")
    print("(No LLM calls — recomputation from stored verdicts)\n")

    results = run_ablation()

    out_path = RESULTS_DIR / "step9_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    # Summary
    print("\n" + "=" * 86)
    print("  SUMMARY: Component contributions")
    print("=" * 86)
    for ds, res in results.items():
        base = res["A. Full TCVA"]["spearman_rho"]
        print(f"\n  {ds.upper()} (Full TCVA rho = {base}):")
        for label, vals in res.items():
            if label == "A. Full TCVA":
                continue
            delta = vals["spearman_rho"] - base
            component = {
                "B. No penalty": "None-penalty",
                "C. Arithmetic": "Power mean (vs arithmetic)",
                "D. Binary": "5-level verdicts (vs binary)",
            }.get(label, label)
            direction = "hurts" if delta < -0.005 else "helps" if delta > 0.005 else "neutral"
            print(f"    Remove {component:<35} delta = {delta:+.4f}  [{direction}]")


if __name__ == "__main__":
    main()
