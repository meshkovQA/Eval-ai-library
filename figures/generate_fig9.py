"""
Generate fig9 (ablation study) for the TCVA paper.

Run: python figures/generate_fig9.py
Outputs: figures/fig9.eps, fig9.png
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'serif'

OUT_DIR = "figures"

with open("experiments/results/step9_ablation.json") as f:
    results = json.load(f)


def fig9_ablation():
    datasets = ["summeval", "summeval_rel", "usr"]
    ds_labels = ["SummEval\n(Faithfulness)", "SummEval\n(Relevancy)", "USR\n(Dialogue)"]

    configs = ["A. Full TCVA", "B. No penalty", "C. Arithmetic", "D. Binary"]
    short_labels = ["Full TCVA", "w/o Penalty", "Arithmetic\nMean", "Binary\nVerdicts"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    x = np.arange(len(datasets))
    n = len(configs)
    width = 0.19

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (cfg, short, color) in enumerate(zip(configs, short_labels, colors)):
        vals = [results[ds][cfg]["spearman_rho"] for ds in datasets]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=short, color=color,
                      edgecolor='white', linewidth=0.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.008,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Delta annotations — show biggest drops
    annotations = [
        # (dataset_idx, config, component_removed, y_offset)
        (0, "B. No penalty", "Penalty", -0.057),
        (1, "D. Binary", "5-level", -0.244),
    ]
    for ds_idx, cfg, component, delta in annotations:
        cfg_idx = configs.index(cfg)
        bar_x = x[ds_idx] + (cfg_idx - (n - 1) / 2) * width
        val = results[datasets[ds_idx]][cfg]["spearman_rho"]
        ax.annotate(f'{delta:+.3f}',
                    xy=(bar_x, val), xytext=(bar_x, val - 0.06),
                    fontsize=8, color='red', fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    ax.set_ylabel("Spearman's $\\rho$")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc='upper right', fontsize=9, title="Configuration", title_fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title("Ablation study: contribution of each TCVA component")

    all_vals = [results[ds][cfg]["spearman_rho"] for ds in datasets for cfg in configs]
    ax.set_ylim([0, max(all_vals) + 0.09])

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig9.eps", format='eps', bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/fig9.png", dpi=300, bbox_inches='tight')
    print("Saved fig9.eps + fig9.png")
    plt.close()


if __name__ == "__main__":
    fig9_ablation()
    print("\nFigure 9 generated!")
