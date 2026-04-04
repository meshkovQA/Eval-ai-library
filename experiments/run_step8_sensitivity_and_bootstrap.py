"""
Step 8: Sensitivity analysis for verdict weights + Bootstrap confidence intervals.

Both analyses reuse stored verdicts — NO additional LLM calls required.

Part 1 — Weight Sensitivity:
  Recompute TCVA scores with 4 different weight schemes to show robustness.

Part 2 — Bootstrap CI:
  Compute 95% confidence intervals for Spearman's rho via 10k bootstrap resamples.
  Includes paired bootstrap test (TCVA vs RAGAS).

Usage:
    python experiments/run_step8_sensitivity_and_bootstrap.py
"""
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import scores_path, RESULTS_DIR, TEMPERATURES

DATASETS = ["summeval", "summeval_rel", "usr"]

# ── Weight schemes for sensitivity analysis ──────────────────────────────

WEIGHT_SCHEMES = {
    "Default":      {"fully": 1.0, "mostly": 0.9,  "partial": 0.7, "minor": 0.3,  "none": 0.0},
    "Linear":       {"fully": 1.0, "mostly": 0.75, "partial": 0.5, "minor": 0.25, "none": 0.0},
    "Aggressive":   {"fully": 1.0, "mostly": 0.95, "partial": 0.8, "minor": 0.1,  "none": 0.0},
    "Conservative": {"fully": 1.0, "mostly": 0.8,  "partial": 0.5, "minor": 0.2,  "none": 0.0},
}


# ── Reusable score_agg with custom weights ───────────────────────────────

def _map_temperature_to_p(temperature: float) -> float:
    t = max(0.1, min(1.0, temperature))
    return -8.0 + ((t - 0.1) / 0.9) * 20.25


def score_agg_custom(weights: List[float], temperature: float, eps: float = 1e-9) -> float:
    """score_agg that accepts pre-mapped weights (same logic as eval_lib/utils.py)."""
    if not weights:
        return 0.0

    p = _map_temperature_to_p(temperature)
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


def remap_weights(verdicts: List[Dict], scheme: Dict[str, float]) -> List[float]:
    """Map verdict strings to weights using a given scheme."""
    return [scheme.get(v["verdict"], 0.0) for v in verdicts]


# ── Part 1: Weight Sensitivity Analysis ──────────────────────────────────

def run_sensitivity_analysis() -> Dict:
    """Recompute TCVA at best T for each weight scheme; report Spearman rho."""
    print("=" * 86)
    print("  PART 1: WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 86)

    # Best temperatures per dataset from the paper
    best_temps = {"summeval": 0.9, "summeval_rel": 0.5, "usr": 0.9}

    all_results = {}

    for ds in DATASETS:
        tcva_file = scores_path(ds, "tcva")
        if not tcva_file.exists():
            print(f"  [{ds}] TCVA scores not found, skipping")
            continue

        with open(tcva_file) as f:
            data = json.load(f)

        # Filter samples with verdicts
        samples = [r for r in data if r.get("verdicts")]
        human = np.array([r["human_score"] for r in samples])
        T = best_temps.get(ds, 0.5)

        print(f"\n  Dataset: {ds.upper()} (N={len(samples)}, T={T})")
        print(f"  {'Scheme':<16} {'Spearman rho':>14} {'p-value':>12} {'MAE':>8}")
        print(f"  {'-' * 52}")

        ds_results = {}

        for scheme_name, scheme in WEIGHT_SCHEMES.items():
            scores = []
            for r in samples:
                w = remap_weights(r["verdicts"], scheme)
                scores.append(score_agg_custom(w, T))

            scores_arr = np.array(scores)
            rho, p_val = stats.spearmanr(scores_arr, human)
            mae = float(np.mean(np.abs(scores_arr - human)))

            ds_results[scheme_name] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": float(p_val),
                "mae": round(mae, 4),
            }

            print(f"  {scheme_name:<16} {rho:>14.4f} {p_val:>12.2e} {mae:>8.4f}")

        # Compute variation across schemes
        rhos = [v["spearman_rho"] for v in ds_results.values()]
        variation = max(rhos) - min(rhos)
        print(f"\n  Rho variation across schemes: {variation:.4f}")
        ds_results["_variation"] = round(variation, 4)

        all_results[ds] = ds_results

    return all_results


# ── Part 2: Bootstrap Confidence Intervals ───────────────────────────────

def bootstrap_spearman(
    method_scores: np.ndarray,
    human_scores: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (rho, lower_ci, upper_ci) via bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(method_scores)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rhos[i], _ = stats.spearmanr(method_scores[idx], human_scores[idx])
    lower = np.percentile(rhos, (1 - ci) / 2 * 100)
    upper = np.percentile(rhos, (1 + ci) / 2 * 100)
    return float(np.median(rhos)), float(lower), float(upper)


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    human: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Test H0: rho_a == rho_b. Returns (delta_rho, p_value)."""
    rng = np.random.RandomState(seed)
    n = len(human)
    observed_a, _ = stats.spearmanr(scores_a, human)
    observed_b, _ = stats.spearmanr(scores_b, human)
    observed_delta = observed_a - observed_b

    count = 0
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ra, _ = stats.spearmanr(scores_a[idx], human[idx])
        rb, _ = stats.spearmanr(scores_b[idx], human[idx])
        if (rb - ra) >= observed_delta:
            count += 1

    p_value = count / n_boot
    return float(observed_delta), float(p_value)


def run_bootstrap_ci() -> Dict:
    """Compute bootstrap CI for all methods and paired tests TCVA vs RAGAS."""
    print("\n" + "=" * 86)
    print("  PART 2: BOOTSTRAP CONFIDENCE INTERVALS (95%, 10k resamples)")
    print("=" * 86)

    best_tcva_keys = {"summeval": "T0.9", "summeval_rel": "T0.5", "usr": "T0.9"}

    all_results = {}

    for ds in DATASETS:
        # Load all scores
        merged = {}

        tcva_file = scores_path(ds, "tcva")
        if tcva_file.exists():
            with open(tcva_file) as f:
                for row in json.load(f):
                    sid = row["id"]
                    merged.setdefault(sid, {"human": row["human_score"]})
                    for key, val in row.get("tcva_scores", {}).items():
                        merged[sid][f"tcva_{key}"] = val

        ragas_file = scores_path(ds, "ragas")
        if ragas_file.exists():
            with open(ragas_file) as f:
                for row in json.load(f):
                    sid = row["id"]
                    merged.setdefault(sid, {"human": row["human_score"]})
                    merged[sid]["ragas"] = row.get("ragas_score", -1.0)

        de_file = scores_path(ds, "deepeval")
        if de_file.exists():
            with open(de_file) as f:
                for row in json.load(f):
                    sid = row["id"]
                    merged.setdefault(sid, {"human": row["human_score"]})
                    merged[sid]["deepeval"] = row.get("deepeval_score", -1.0)

        if not merged:
            continue

        sample_ids = list(merged.keys())
        human = np.array([merged[sid]["human"] for sid in sample_ids])

        best_key = f"tcva_{best_tcva_keys[ds]}"
        methods = {
            f"TCVA (best)": best_key,
            "RAGAS": "ragas",
            "DeepEval": "deepeval",
        }

        print(f"\n  Dataset: {ds.upper()} (N={len(sample_ids)})")
        print(f"  {'Method':<18} {'rho':>7} {'95% CI':>20} {'CI width':>10}")
        print(f"  {'-' * 57}")

        ds_results = {}
        method_arrays = {}

        for label, key in methods.items():
            scores = np.array([merged[sid].get(key, -1.0) for sid in sample_ids])
            mask = scores >= 0.0
            if mask.sum() < 10:
                print(f"  {label:<18}  insufficient data")
                continue

            h = human[mask]
            s = scores[mask]
            method_arrays[label] = (s, h, mask)

            rho_med, lower, upper = bootstrap_spearman(s, h)
            point_rho, _ = stats.spearmanr(s, h)

            ds_results[label] = {
                "spearman_rho": round(float(point_rho), 4),
                "bootstrap_median": round(rho_med, 4),
                "ci_lower": round(lower, 4),
                "ci_upper": round(upper, 4),
                "ci_width": round(upper - lower, 4),
                "n": int(mask.sum()),
            }

            print(f"  {label:<18} {point_rho:>7.4f} [{lower:>7.4f}, {upper:>7.4f}] {upper - lower:>10.4f}")

        # Paired bootstrap: TCVA vs RAGAS
        if "TCVA (best)" in method_arrays and "RAGAS" in method_arrays:
            # Use only overlapping valid samples
            tcva_s, tcva_h, tcva_mask = method_arrays["TCVA (best)"]
            ragas_s, ragas_h, ragas_mask = method_arrays["RAGAS"]

            # Find common valid samples
            both_mask = np.array([
                merged[sid].get(best_key, -1.0) >= 0 and merged[sid].get("ragas", -1.0) >= 0
                for sid in sample_ids
            ])
            if both_mask.sum() >= 10:
                h_common = human[both_mask]
                t_common = np.array([merged[sid][best_key] for sid, m in zip(sample_ids, both_mask) if m])
                r_common = np.array([merged[sid]["ragas"] for sid, m in zip(sample_ids, both_mask) if m])

                delta, p_val = paired_bootstrap_test(t_common, r_common, h_common)
                ds_results["_paired_tcva_vs_ragas"] = {
                    "delta_rho": round(delta, 4),
                    "p_value": round(p_val, 4),
                    "significant_at_005": p_val < 0.05,
                    "n_common": int(both_mask.sum()),
                }
                sig = "YES" if p_val < 0.05 else "NO"
                print(f"\n  Paired test TCVA vs RAGAS: delta={delta:+.4f}, p={p_val:.4f} (significant: {sig})")

        all_results[ds] = ds_results

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("\nRunning Step 8: Sensitivity Analysis + Bootstrap CI")
    print("(No LLM calls — recomputation from stored verdicts)\n")

    sensitivity = run_sensitivity_analysis()
    bootstrap = run_bootstrap_ci()

    # Save combined results
    output = {
        "sensitivity_analysis": sensitivity,
        "bootstrap_ci": bootstrap,
    }

    out_path = RESULTS_DIR / "step8_sensitivity_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    # Summary
    print("\n" + "=" * 86)
    print("  SUMMARY")
    print("=" * 86)

    print("\n  Weight sensitivity (rho variation across 4 schemes):")
    for ds, res in sensitivity.items():
        var = res.get("_variation", "N/A")
        robust = "ROBUST" if isinstance(var, float) and var < 0.03 else "CHECK"
        print(f"    {ds:<16} variation = {var}  [{robust}]")

    print("\n  Bootstrap paired tests (TCVA vs RAGAS):")
    for ds, res in bootstrap.items():
        paired = res.get("_paired_tcva_vs_ragas", {})
        if paired:
            d = paired["delta_rho"]
            p = paired["p_value"]
            print(f"    {ds:<16} delta = {d:+.4f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
