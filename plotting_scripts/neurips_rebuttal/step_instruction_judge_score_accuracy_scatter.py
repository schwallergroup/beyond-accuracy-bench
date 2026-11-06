"""Plot PR/DED reasoning scores against iteration-5 accuracy for step ablations."""

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

from plotting_scripts.neurips.judge_score_accuracy_scatter import (  # noqa: E402
    AXIS_LABEL_FONTSIZE,
    DESIGN_KEY,
    PREDICTIVE_KEY,
    RUBRIC_LABELS,
    TICK_LABEL_FONTSIZE,
    _bootstrap_ci,
)
from plotting_scripts.neurips.paper_model_metadata import MODEL_COLORS, DISPLAY_LABELS  # noqa: E402
from plotting_scripts.neurips.archived_experiment_data import discover_experiments  # noqa: E402
from plotting_scripts.neurips_rebuttal.judge_filtering import canonical_model_id  # noqa: E402
from plotting_scripts.neurips_rebuttal.mechanism_info_accuracy_comparison import (  # noqa: E402
    gather_step5_accuracy,
)
from plotting_scripts.neurips_rebuttal.step_instruction_reasoning_accuracy_correlation import (  # noqa: E402
    JUDGE_INPUTS,
    STEP_RUNS,
)
from plotting_scripts.styling import apply_paper_rcparams  # noqa: E402


OUTPUT_DIR = SCRIPT_DIR / "step_instruction_judge_analysis"
OUTPUT_PNG = OUTPUT_DIR / "iteration3_pr_ded_vs_iteration5_accuracy_scatter.png"
OUTPUT_CSV = OUTPUT_DIR / "iteration3_pr_ded_vs_iteration5_accuracy_scatter_data.csv"
RUBRIC_ORDER = (DESIGN_KEY, PREDICTIVE_KEY)
MODEL_ORDER = tuple(model for model, _ in STEP_RUNS)
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 0


def _reasoning_trace_scores() -> pd.DataFrame:
    frames = []
    for judge_id, paths in JUDGE_INPUTS.items():
        for path in paths:
            frame = pd.read_csv(path)
            frame["model"] = frame["model_label"].str.replace(
                r"\s+step\s+instruction$", "", regex=True, case=False
            )
            frame["judge_id"] = judge_id
            frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    scores["score"] = pd.to_numeric(scores["score"], errors="raise")
    return scores[
        scores.apply(
            lambda row: row["judge_id"] != canonical_model_id(row["model"]), axis=1
        )
    ].groupby(["model", "sample_id", "rubric_key"], as_index=False)["score"].mean()


def build_plot_dataframe() -> pd.DataFrame:
    scores = _reasoning_trace_scores()
    experiments = discover_experiments(str(PROJECT_ROOT / "results"))
    try:
        by_source = {experiment.source_tar: experiment for experiment in experiments}
        rows = []
        for index, (model, source) in enumerate(STEP_RUNS):
            accuracy = gather_step5_accuracy(by_source[source])["accuracy"].to_numpy(dtype=float)
            for rubric_key in RUBRIC_ORDER:
                trace_scores = scores.loc[
                    (scores["model"] == model) & (scores["rubric_key"] == rubric_key), "score"
                ].to_numpy(dtype=float)
                rows.append(
                    {
                        "model": model,
                        "rubric_key": rubric_key,
                        "mean_score": trace_scores.mean(),
                        "score_ci_low": _bootstrap_ci(trace_scores)[0],
                        "score_ci_high": _bootstrap_ci(trace_scores)[1],
                        "n_traces": len(trace_scores),
                        "accuracy": accuracy.mean(),
                        "accuracy_ci_low": _bootstrap_ci(accuracy)[0],
                        "accuracy_ci_high": _bootstrap_ci(accuracy)[1],
                        "n_accuracy": len(accuracy),
                        "color": MODEL_COLORS[f"{model} with Mechanism Info"],
                    }
                )
    finally:
        for experiment in experiments:
            experiment.close()
    return pd.DataFrame(rows)


def _bootstrap_regression_band(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return OLS prediction and paired-model bootstrap 95% confidence band."""
    slope, intercept = np.polyfit(x, y, deg=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    predictions = []
    for indices in rng.integers(0, len(x), size=(BOOTSTRAP_SAMPLES, len(x))):
        sampled_x, sampled_y = x[indices], y[indices]
        if np.unique(sampled_x).size >= 2:
            sampled_slope, sampled_intercept = np.polyfit(sampled_x, sampled_y, deg=1)
            predictions.append(sampled_slope * x_grid + sampled_intercept)
    bands = np.asarray(predictions)
    low, high = np.percentile(bands, [2.5, 97.5], axis=0)
    return slope * x_grid + intercept, low, high


def plot_scatter(plot_df: pd.DataFrame) -> None:
    apply_paper_rcparams()
    fig, axs = plt.subplots(1, 2, figsize=(160 / 25.4, 92 / 25.4), sharey=True)
    for ax, rubric_key in zip(axs, RUBRIC_ORDER):
        panel = plot_df[plot_df["rubric_key"] == rubric_key]
        x = panel["mean_score"].to_numpy(dtype=float)
        y = panel["accuracy"].to_numpy(dtype=float)
        lower, upper = (1.0, 2.03) if rubric_key == DESIGN_KEY else (0.5, 2.03)
        x_grid = np.linspace(lower, upper, 200)
        fit, band_low, band_high = _bootstrap_regression_band(x, y, x_grid)
        ax.fill_between(x_grid, band_low, band_high, color="0.3", alpha=0.18, linewidth=0, zorder=0)
        ax.plot(x_grid, fit, color="0.15", linewidth=1.2, zorder=1)
        xerr = np.vstack([x - panel["score_ci_low"], panel["score_ci_high"] - x])
        yerr = np.vstack([y - panel["accuracy_ci_low"], panel["accuracy_ci_high"] - y])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.25", elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=1)
        ax.scatter(x, y, c=panel["color"].tolist(), s=52, edgecolors="black", linewidths=0.55, zorder=3)
        ax.set_xlabel(RUBRIC_LABELS[rubric_key].replace("\n", " "), fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(lower, upper)
        ax.set_xticks(np.arange(1.0, 2.01, 0.25) if rubric_key == DESIGN_KEY else np.arange(0.5, 2.01, 0.5))
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("Accuracy at Iteration 5", fontsize=AXIS_LABEL_FONTSIZE)
    axs[0].set_ylim(0.34, 0.66)
    axs[0].set_yticks(np.arange(0.35, 0.66, 0.1))
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
    plot_df = build_plot_dataframe()
    plot_df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    plot_scatter(plot_df)
    print(f"Saved figure: {OUTPUT_PNG}")
    print(f"Saved data: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
