"""
Generate fig7 (weight sensitivity) and fig8 (bootstrap CI) for the TCVA paper.

Run: python figures/generate_fig7_fig8.py
Outputs: figures/fig7.eps, fig7.png, fig8.eps, fig8.png
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'serif'

OUT_DIR = "figures"

with open("experiments/results/step8_sensitivity_bootstrap.json") as f:
    results = json.load(f)

sens = results["sensitivity_analysis"]
boot = results["bootstrap_ci"]


# ============================================================
# Fig 7: Weight sensitivity — grouped bar chart
# ============================================================

def fig7_weight_sensitivity():
    datasets = ["summeval", "summeval_rel", "usr"]
    ds_labels = ["SummEval\n(Faithfulness)", "SummEval\n(Relevancy)", "USR\n(Dialogue)"]
    schemes = ["Default", "Linear", "Aggressive", "Conservative"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    hatches = ["", "//", "\\\\", "xx"]

    x = np.arange(len(datasets))
    n = len(schemes)
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, scheme in enumerate(schemes):
        vals = [sens[ds][scheme]["spearman_rho"] for ds in datasets]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=scheme, color=colors[i],
                      hatch=hatches[i], edgecolor='white', linewidth=0.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Add variation annotations
    for j, ds in enumerate(datasets):
        var = sens[ds]["_variation"]
        ymax = max(sens[ds][s]["spearman_rho"] for s in schemes) + 0.04
        ax.text(x[j], ymax, f'$\\Delta$ = {var:.4f}', ha='center', fontsize=8.5,
                fontstyle='italic', color='#555')

    ax.set_ylabel("Spearman's $\\rho$")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc='upper right', fontsize=9, title="Weight scheme", title_fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title("Weight sensitivity analysis: Spearman's $\\rho$ across weight schemes")

    # Adjust y-axis to give room for annotations
    all_rhos = [sens[ds][s]["spearman_rho"] for ds in datasets for s in schemes]
    ax.set_ylim([0, max(all_rhos) + 0.08])

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig7.eps", format='eps', bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/fig7.png", dpi=300, bbox_inches='tight')
    print("Saved fig7.eps + fig7.png")
    plt.close()


# ============================================================
# Fig 8: Bootstrap CI — forest plot
# ============================================================

def fig8_bootstrap_ci():
    datasets = ["summeval", "summeval_rel", "usr"]
    ds_labels = ["SummEval (Faithfulness)", "SummEval (Relevancy)", "USR (Dialogue)"]
    methods = ["TCVA (best)", "RAGAS", "DeepEval"]
    colors = {"TCVA (best)": "#2196F3", "RAGAS": "#FF9800", "DeepEval": "#4CAF50"}
    markers = {"TCVA (best)": "o", "RAGAS": "s", "DeepEval": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, ds, ds_label in zip(axes, datasets, ds_labels):
        ds_boot = boot[ds]

        y_positions = []
        y_labels = []

        for i, method in enumerate(methods):
            if method not in ds_boot or "ci_lower" not in ds_boot[method]:
                continue

            info = ds_boot[method]
            rho = info["spearman_rho"]
            lo = info["ci_lower"]
            hi = info["ci_upper"]
            y = len(methods) - 1 - i

            ax.errorbar(rho, y, xerr=[[rho - lo], [hi - rho]],
                        fmt=markers[method], color=colors[method], markersize=8,
                        capsize=5, capthick=1.5, linewidth=1.5, zorder=3,
                        label=method if ds == "summeval" else None)

            ax.text(hi + 0.02, y, f'{rho:.3f} [{lo:.3f}, {hi:.3f}]',
                    va='center', fontsize=8)

            y_positions.append(y)
            y_labels.append(method)

        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Spearman's $\\rho$")
        ax.set_title(ds_label, fontsize=10)
        ax.grid(axis='x', alpha=0.3, zorder=0)
        ax.set_xlim([-0.3, 0.95])

        # Add paired test annotation if available
        paired = ds_boot.get("_paired_tcva_vs_ragas", {})
        if paired:
            p_val = paired["p_value"]
            sig = "*" if paired["significant_at_005"] else "n.s."
            delta = paired["delta_rho"]
            ax.text(0.02, 0.02,
                    f'$\\Delta\\rho$ = {delta:+.3f} ({sig}, p={p_val:.3f})',
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              edgecolor='gray', alpha=0.8))

    axes[0].legend(loc='lower left', fontsize=9)

    fig.suptitle("Bootstrap 95% confidence intervals for Spearman's $\\rho$", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8.eps", format='eps', bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/fig8.png", dpi=300, bbox_inches='tight')
    print("Saved fig8.eps + fig8.png")
    plt.close()


if __name__ == "__main__":
    fig7_weight_sensitivity()
    fig8_bootstrap_ci()
    print("\nFigures 7 and 8 generated!")
