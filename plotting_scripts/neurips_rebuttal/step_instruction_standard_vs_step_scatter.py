"""Compare standard-prompt and four-step ablation results in matched scatterplots."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting_scripts.neurips.archived_experiment_data import (  # noqa: E402
    discover_experiments,
    select_preferred_experiments,
)
from plotting_scripts.neurips.judge_score_accuracy_scatter import _bootstrap_ci  # noqa: E402
from plotting_scripts.neurips.paper_model_metadata import MODEL_COLORS, DISPLAY_LABELS  # noqa: E402
from plotting_scripts.neurips_rebuttal.judge_filtering import canonical_model_id  # noqa: E402
from plotting_scripts.neurips_rebuttal.mechanism_info_accuracy_comparison import (  # noqa: E402
    gather_step5_accuracy,
)
from plotting_scripts.neurips_rebuttal.step_instruction_reasoning_accuracy_correlation import (  # noqa: E402
    JUDGE_INPUTS as STEP_JUDGE_INPUTS,
    STEP_RUNS,
)
from plotting_scripts.styling import apply_paper_rcparams  # noqa: E402


STANDARD_JUDGE_INPUTS = {
    "gpt-5.4": (
        PROJECT_ROOT / "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "claude-opus-4.7": (
        PROJECT_ROOT / "bench/judge/runs/standard_iter3/results/claude47/judge_results.csv",
    ),
}
OUTPUT_DIR = SCRIPT_DIR / "step_instruction_judge_analysis"
OUTPUT_PNG = OUTPUT_DIR / "standard_vs_step_instruction_accuracy_ded_pr_scatter.png"
OUTPUT_CSV = OUTPUT_DIR / "standard_vs_step_instruction_accuracy_ded_pr_scatter_data.csv"
METRICS = (
    ("accuracy", "Task Accuracy at Iteration 5", (0.34, 0.66)),
    ("feature_to_action_faithfulness", "Discriminative Experiment Design", (1.0, 2.03)),
    ("predictive_sharpness", "Predictive Reasoning", (0.0, 2.03)),
)
PLOT_METRICS = METRICS[1:]


def _trace_scores(judge_inputs: dict[str, tuple[Path, ...]], prompt_suffix: str) -> pd.DataFrame:
    frames = []
    for judge_id, paths in judge_inputs.items():
        for path in paths:
            frame = pd.read_csv(path)
            frame["model"] = frame["model_label"].str.replace(prompt_suffix, "", regex=True, case=False)
            frame["judge_id"] = judge_id
            frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    scores["score"] = pd.to_numeric(scores["score"], errors="raise")
    eligible = scores[
        scores.apply(lambda row: row["judge_id"] != canonical_model_id(row["model"]), axis=1)
    ]
    return eligible.groupby(["model", "sample_id", "rubric_key"], as_index=False)["score"].mean()


def _summary_score(trace_scores: pd.DataFrame, model: str, rubric_key: str) -> tuple[float, float, float, int]:
    values = trace_scores.loc[
        (trace_scores["model"] == model) & (trace_scores["rubric_key"] == rubric_key), "score"
    ].to_numpy(dtype=float)
    low, high = _bootstrap_ci(values)
    return values.mean(), low, high, len(values)


def build_plot_data() -> pd.DataFrame:
    standard_scores = _trace_scores(STANDARD_JUDGE_INPUTS, r"\s+with\s+mechanism\s+info$")
    step_scores = _trace_scores(STEP_JUDGE_INPUTS, r"\s+step\s+instruction$")
    experiments = discover_experiments(str(PROJECT_ROOT / "results"))
    standard_labels = [f"{model} with Mechanism Info" for model, _ in STEP_RUNS]
    selected_standard = {
        exp.label: exp for exp in select_preferred_experiments(experiments, standard_labels)
    }
    by_source = {experiment.source_tar: experiment for experiment in experiments}
    try:
        rows = []
        for model_index, (model, step_source) in enumerate(STEP_RUNS):
            standard_accuracy = gather_step5_accuracy(
                selected_standard[f"{model} with Mechanism Info"]
            )["accuracy"].to_numpy(dtype=float)
            step_accuracy = gather_step5_accuracy(by_source[step_source])["accuracy"].to_numpy(dtype=float)
            for metric_key, _, _ in METRICS:
                if metric_key == "accuracy":
                    standard_values, step_values = standard_accuracy, step_accuracy
                else:
                    standard_values = standard_scores.loc[
                        (standard_scores["model"] == model)
                        & (standard_scores["rubric_key"] == metric_key), "score"
                    ].to_numpy(dtype=float)
                    step_values = step_scores.loc[
                        (step_scores["model"] == model)
                        & (step_scores["rubric_key"] == metric_key), "score"
                    ].to_numpy(dtype=float)
                standard_low, standard_high = _bootstrap_ci(standard_values)
                step_low, step_high = _bootstrap_ci(step_values)
                rows.append(
                    {
                        "model": model,
                        "metric_key": metric_key,
                        "standard_mean": standard_values.mean(),
                        "standard_ci_low": standard_low,
                        "standard_ci_high": standard_high,
                        "step_instruction_mean": step_values.mean(),
                        "step_instruction_ci_low": step_low,
                        "step_instruction_ci_high": step_high,
                        "n_standard": len(standard_values),
                        "n_step_instruction": len(step_values),
                        "color": MODEL_COLORS[f"{model} with Mechanism Info"],
                    }
                )
    finally:
        for experiment in experiments:
            experiment.close()
    return pd.DataFrame(rows)


def plot(plot_df: pd.DataFrame) -> None:
    apply_paper_rcparams()
    fig, axs = plt.subplots(1, len(PLOT_METRICS), figsize=(160 / 25.4, 92 / 25.4), sharex=False, sharey=False)
    for ax, (metric_key, label, limits) in zip(axs, PLOT_METRICS):
        panel = plot_df[plot_df["metric_key"] == metric_key]
        x = panel["standard_mean"].to_numpy(dtype=float)
        y = panel["step_instruction_mean"].to_numpy(dtype=float)
        xerr = np.vstack([x - panel["standard_ci_low"], panel["standard_ci_high"] - x])
        yerr = np.vstack([y - panel["step_instruction_ci_low"], panel["step_instruction_ci_high"] - y])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.25", elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=1)
        ax.scatter(x, y, c=panel["color"].tolist(), s=52, edgecolors="black", linewidths=0.55, zorder=3)
        ax.plot(limits, limits, color="0.35", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_xlabel(f"Standard Prompt\n{label}", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10)
    axs[0].set_ylabel("Four-Step Instruction", fontsize=12)
    from matplotlib.lines import Line2D
    models = plot_df["model"].drop_duplicates().tolist()
    handles = [Line2D([], [], marker="o", linestyle="none",
                      color=MODEL_COLORS[f"{model} with Mechanism Info"],
                      markeredgecolor="black", markersize=5,
                      label=DISPLAY_LABELS[f"{model} with Mechanism Info"])
               for model in models]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.52, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_df = build_plot_data()
    plot_df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    plot(plot_df)
    print(f"Saved figure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
