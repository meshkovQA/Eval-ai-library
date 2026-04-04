"""
Step 5: Compute correlations and generate results tables.
Reads all *_scores.json files and produces:
  - Correlation table (Spearman, Kendall, MAE) per dataset
  - Aggregate correlation table
  - Best temperature per dataset
  - LaTeX-ready table output

Usage:
    python experiments/run_step5_analyze.py --dataset ragtruth
    python experiments/run_step5_analyze.py --all
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import scores_path, RESULTS_DIR, TEMPERATURES

DATASETS = ["summeval", "summeval_rel", "usr"]


def load_scores(name: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load and merge all method scores for a dataset.
    Returns: {sample_id: {"human": float, "tcva_T0.2": float, ..., "ragas": float, "deepeval": float}}
    """
    merged = {}

    # TCVA scores
    tcva_file = scores_path(name, "tcva")
    if tcva_file.exists():
        with open(tcva_file) as f:
            for row in json.load(f):
                sid = row["id"]
                merged.setdefault(sid, {"human": row["human_score"]})
                for key, val in row.get("tcva_scores", {}).items():
                    merged[sid][f"tcva_{key}"] = val
                merged[sid]["ablation_5level_arith"] = row.get("ablation_arithmetic", -1.0)
    else:
        print(f"  [WARN] TCVA scores not found: {tcva_file}")

    # RAGAS scores
    ragas_file = scores_path(name, "ragas")
    if ragas_file.exists():
        with open(ragas_file) as f:
            for row in json.load(f):
                sid = row["id"]
                merged.setdefault(sid, {"human": row["human_score"]})
                merged[sid]["ragas"] = row.get("ragas_score", -1.0)
    else:
        print(f"  [WARN] RAGAS scores not found: {ragas_file}")

    # DeepEval scores
    de_file = scores_path(name, "deepeval")
    if de_file.exists():
        with open(de_file) as f:
            for row in json.load(f):
                sid = row["id"]
                merged.setdefault(sid, {"human": row["human_score"]})
                merged[sid]["deepeval"] = row.get("deepeval_score", -1.0)
    else:
        print(f"  [WARN] DeepEval scores not found: {de_file}")

    if not merged:
        return None
    return merged


def compute_correlation(human: np.ndarray, method: np.ndarray) -> Dict[str, Any]:
    """Compute Spearman rho, Kendall tau, MAE, and p-values."""
    # Filter out failed evaluations
    mask = method >= 0.0
    if mask.sum() < 10:
        return {"n": int(mask.sum()), "insufficient": True}

    h = human[mask]
    m = method[mask]

    rho, rho_p = stats.spearmanr(m, h)
    tau, tau_p = stats.kendalltau(m, h)
    mae = float(np.mean(np.abs(m - h)))

    return {
        "n": int(mask.sum()),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rho_p),
        "kendall_tau": round(float(tau), 4),
        "kendall_p": float(tau_p),
        "mae": round(mae, 4),
    }


def analyze_dataset(name: str) -> Optional[Dict]:
    """Analyze one dataset and print results."""
    merged = load_scores(name)
    if not merged:
        print(f"[{name}] No scores found. Run steps 2-4 first.")
        return None

    sample_ids = list(merged.keys())
    human = np.array([merged[sid]["human"] for sid in sample_ids])

    # Define methods to compare
    method_keys = {}
    for T in TEMPERATURES:
        method_keys[f"TCVA (T={T})"] = f"tcva_T{T}"
    method_keys["5-level + Arith. Mean"] = "ablation_5level_arith"
    method_keys["RAGAS (binary)"] = "ragas"
    method_keys["DeepEval (ternary)"] = "deepeval"

    print(f"\n{'='*86}")
    print(f"  RESULTS: {name.upper()} (N={len(sample_ids)})")
    print(f"{'='*86}")
    print(f"  {'Method':<28} {'Spearman':>10} {'p-val':>11} {'Kendall':>10} {'p-val':>11} {'MAE':>7} {'n':>5}")
    print(f"  {'-'*82}")

    results = {}
    best_rho = -1
    best_method = ""

    for label, key in method_keys.items():
        scores = np.array([merged[sid].get(key, -1.0) for sid in sample_ids])
        corr = compute_correlation(human, scores)
        results[label] = corr

        if corr.get("insufficient"):
            print(f"  {label:<28}  insufficient data (n={corr['n']})")
        else:
            rho = corr["spearman_rho"]
            if rho > best_rho:
                best_rho = rho
                best_method = label
            print(f"  {label:<28} {rho:>10.4f} {corr['spearman_p']:>10.2e} "
                  f"{corr['kendall_tau']:>10.4f} {corr['kendall_p']:>10.2e} "
                  f"{corr['mae']:>7.4f} {corr['n']:>5}")

    print(f"\n  Best method: {best_method} (rho={best_rho:.4f})")

    return {"dataset": name, "n_samples": len(sample_ids), "correlations": results,
            "best_method": best_method, "best_rho": best_rho}


def print_latex_table(all_results: List[Dict]):
    """Generate LaTeX table for the paper."""
    print(f"\n{'='*86}")
    print("  LATEX TABLE")
    print(f"{'='*86}")

    print(r"""
\begin{table}[h]
\caption{Correlation with human judgments across datasets (Spearman's $\rho$ / Kendall's $\tau$)}
\label{tab:correlation-results}
\centering
\begin{tabular}{@{}l""" + "c" * len(all_results) + r"""@{}}
\toprule
\textbf{Method}""")

    # Column headers
    for res in all_results:
        ds = res["dataset"].capitalize()
        n = res["n_samples"]
        print(f" & \\textbf{{{ds}}} ($n$={n})", end="")
    print(r" \\")
    print(r"\midrule")

    # Collect all method names
    method_names = []
    for res in all_results:
        for m in res["correlations"]:
            if m not in method_names:
                method_names.append(m)

    for method in method_names:
        row = f"{method}"
        for res in all_results:
            corr = res["correlations"].get(method, {})
            if corr.get("insufficient"):
                row += " & ---"
            else:
                rho = corr.get("spearman_rho", 0)
                tau = corr.get("kendall_tau", 0)
                # Bold the best
                is_best = (method == res["best_method"])
                if is_best:
                    row += f" & \\textbf{{{rho:.3f}}} / \\textbf{{{tau:.3f}}}"
                else:
                    row += f" & {rho:.3f} / {tau:.3f}"
        row += r" \\"
        print(row)

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # MAE table
    print(r"""
\begin{table}[h]
\caption{Mean Absolute Error (MAE) compared to human judgments}
\label{tab:mae-results}
\centering
\begin{tabular}{@{}l""" + "c" * len(all_results) + r"""@{}}
\toprule
\textbf{Method}""")
    for res in all_results:
        ds = res["dataset"].capitalize()
        print(f" & \\textbf{{{ds}}}", end="")
    print(r" \\")
    print(r"\midrule")

    for method in method_names:
        row = f"{method}"
        for res in all_results:
            corr = res["correlations"].get(method, {})
            mae = corr.get("mae")
            if mae is not None:
                row += f" & {mae:.3f}"
            else:
                row += " & ---"
        row += r" \\"
        print(row)

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    parser = argparse.ArgumentParser(description="Step 5: Analyze results")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
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

    # Save analysis
    if all_results:
        out = RESULTS_DIR / "analysis_summary.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAnalysis saved -> {out}")

    # LaTeX output
    if args.latex and len(all_results) > 0:
        print_latex_table(all_results)


if __name__ == "__main__":
    main()
