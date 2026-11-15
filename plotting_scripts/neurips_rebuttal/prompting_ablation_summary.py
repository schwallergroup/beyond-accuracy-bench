"""Plot paired four-step minus standard effects on reasoning and accuracy.

Reasoning uses matched iteration-2--4 traces, with target-model self-judgments excluded.
Accuracy uses the shared-system iteration-5 estimates from paired_prompt_effects.py.
Intervals are exploratory, unadjusted 95% percentile bootstrap intervals.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from plotting_scripts.neurips.paper_model_metadata import LABELS, MODEL_COLORS, DISPLAY_LABELS
from plotting_scripts.neurips_rebuttal.step_instruction_standard_vs_step_scatter import (
    paired_trace_scores, cluster_ci,
)
from plotting_scripts.neurips_rebuttal.step_instruction_reasoning_accuracy_correlation import STEP_RUNS
from plotting_scripts.styling import apply_paper_rcparams

OUTPUT_DIR = SCRIPT_DIR / "step_instruction_judge_analysis"
OUTPUT_PNG = OUTPUT_DIR / "prompting_ablation_paired_changes.png"
OUTPUT_CSV = OUTPUT_DIR / "prompting_ablation_paired_changes.csv"
METRICS = (
    ("predictive_sharpness", "(a) Predictive Reasoning", "Δ PR score (Iterations 2–4)", 1),
    ("feature_to_action_faithfulness", "(b) Discriminative Experiment Design", "Δ DED score (Iterations 2–4)", 1),
    ("Accuracy", "(c) Identification Accuracy", "Δ Accuracy (Iteration 5, pp)", 100),
)


def build_summary():
    paired = paired_trace_scores()
    rows = []
    for (model, metric), group in paired.groupby(["model", "rubric_key"], sort=True):
        if metric not in {item[0] for item in METRICS[:2]}:
            continue
        group = group.copy()
        group["difference"] = group["score_step"] - group["score_standard"]
        values = group["difference"].to_numpy(float)
        low, high = cluster_ci(group, "difference")
        rows.append(dict(model=model, metric=metric, paired_change=values.mean(),
                         change_ci_low=low, change_ci_high=high, n_paired=len(values),
                         n_systems=group["sample_id"].nunique()))
    accuracy = pd.read_csv(OUTPUT_DIR / "nine_model_paired_prompt_effects.csv")
    accuracy = accuracy[accuracy["metric"] == "Accuracy"]
    return pd.concat([pd.DataFrame(rows), accuracy], ignore_index=True)


def plot(summary):
    available = {model for model, _ in STEP_RUNS}
    models = [label.removesuffix(" with Mechanism Info") for label in LABELS
              if label.removesuffix(" with Mechanism Info") in available]
    apply_paper_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(182 / 25.4, 88 / 25.4), sharey=True)
    for ax, (metric, title, xlabel, scale) in zip(axes, METRICS):
        panel = summary[summary["metric"] == metric].set_index("model").loc[models]
        ax.axvline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
        for y, (model, row) in enumerate(panel.iterrows()):
            mean, low, high = row[["paired_change", "change_ci_low", "change_ci_high"]] * scale
            ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o",
                        color=MODEL_COLORS[f"{model} with Mechanism Info"], markersize=4,
                        markeredgecolor="0.2", markeredgewidth=0.4,
                        elinewidth=1, capsize=2)
        ax.set_title(title, fontsize=7.5, pad=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(axis="y", color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.12)
    # PR and DED share the rubric scale, so use identical limits.
    limits = [ax.get_xlim() for ax in axes[:2]]
    for ax in axes[:2]:
        ax.set_xlim(min(limit[0] for limit in limits), max(limit[1] for limit in limits))
    axes[2].set_xlim(-20, 20)
    axes[2].set_xticks([-20, -10, 0, 10, 20])
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([DISPLAY_LABELS[f"{model} with Mechanism Info"] for model in models])
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.3)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = build_summary()
    summary.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    plot(summary)
    print(summary[["model", "metric", "paired_change", "change_ci_low", "change_ci_high", "n_paired"]].to_string(index=False))
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
