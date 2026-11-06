"""Camera-ready figures for pooled iteration-2--4 reasoning scores.

Every score pools 40 M*_0/M*_1 traces from each of iterations 2, 3, and 4
(120 trace scores per model and rubric).  Target-model self-judgments are
removed before combining GPT-5.4 and Claude Opus 4.7 judge scores.
"""

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

from plotting_scripts.neurips.judge_rubric_scores_by_model import (  # noqa: E402
    MODEL_DISPLAY,
    RUBRIC_LABELS,
    RUBRIC_ORDER,
    _bootstrap_ci,
)
from plotting_scripts.neurips.paper_model_metadata import LABELS, MODEL_COLORS  # noqa: E402
from plotting_scripts.neurips_rebuttal.iteration_reasoning_score_analysis import (  # noqa: E402
    build_accuracy_iteration5,
    build_iteration_reasoning_per_sample_scores,
    spearman_with_bootstrap_ci,
)
from plotting_scripts.styling import apply_paper_rcparams  # noqa: E402


OUTPUT_DIR = SCRIPT_DIR / "imgs_for_paper"
ANALYSIS_DIR = SCRIPT_DIR / "iteration_analysis"
BAR_PNG = OUTPUT_DIR / "camera_ready_reasoning_scores_iterations_2_to_4.png"
SCATTER_PNG = OUTPUT_DIR / "camera_ready_ded_pr_vs_accuracy_iterations_2_to_4.png"
SCORES_CSV = ANALYSIS_DIR / "camera_ready_reasoning_scores_iterations_2_to_4.csv"
CORRELATION_CSV = ANALYSIS_DIR / "camera_ready_ded_pr_vs_accuracy_iterations_2_to_4.csv"
BAR_RUBRIC_ORDER = (
    "hypothesis_space_alignment",
    "mechanistic_feature_extraction",
    "feature_to_action_faithfulness",
    "predictive_sharpness",
)
SCATTER_KEYS = ("feature_to_action_faithfulness", "predictive_sharpness")
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 0


def _model_color_map() -> dict[str, str]:
    return {
        label.removesuffix(" with Mechanism Info"): color
        for label, color in MODEL_COLORS.items()
        if label.endswith(" with Mechanism Info")
    }


def build_score_summary() -> pd.DataFrame:
    """Pool 120 self-judge-excluded trace scores for each model/rubric."""
    per_sample, shared_ids = build_iteration_reasoning_per_sample_scores()
    summary_rows = []
    for (model_name, rubric_key), group in per_sample.groupby(["model_name", "rubric_key"], sort=False):
        values = group["reasoning_score"]
        # Preserve the three repeated observations from each independent system.
        system_means = group.groupby("sample_id")["reasoning_score"].mean()
        ci_low, ci_high = _bootstrap_ci(system_means)
        summary_rows.append(
            {
                "model_name": model_name,
                "rubric_key": rubric_key,
                "rubric_label": RUBRIC_LABELS[rubric_key].replace("\n", " "),
                "reasoning_score": values.mean(),
                "score_ci_low": ci_low,
                "score_ci_high": ci_high,
                "n_reasoning_samples": len(values),
                "n_independent_systems": len(system_means),
                "n_iterations": group["iteration"].nunique(),
                "n_shared_sample_ids_per_iteration": len(shared_ids),
            }
        )
    return pd.DataFrame(summary_rows)


def _ordered_models(accuracy: pd.DataFrame) -> list[str]:
    label_order = [label.removesuffix(" with Mechanism Info") for label in LABELS]
    available = set(accuracy["model_name"])
    return [model for model in label_order if model in available]


def plot_bar(summary: pd.DataFrame, model_order: list[str], *, output_path=None, colors=None, display=None) -> None:
    apply_paper_rcparams()
    colors = _model_color_map() if colors is None else colors
    display = MODEL_DISPLAY if display is None else display
    fig, ax = plt.subplots(figsize=(270 / 25.4, 92 / 25.4))
    x = np.arange(len(BAR_RUBRIC_ORDER), dtype=float)
    width = min(0.12, 0.78 / len(model_order))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2) * width
    for offset, model_name in zip(offsets, model_order):
        model = summary[summary["model_name"] == model_name].set_index("rubric_key").loc[list(BAR_RUBRIC_ORDER)]
        means = model["reasoning_score"].to_numpy(float)
        low = means - model["score_ci_low"].to_numpy(float)
        high = model["score_ci_high"].to_numpy(float) - means
        ax.bar(x + offset, means, width=width * 0.92, color=colors[model_name],
               edgecolor="black", linewidth=0.6, label=display[model_name])
        ax.errorbar(x + offset, means, yerr=np.vstack([low, high]), fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=1.8, capthick=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([RUBRIC_LABELS[key] for key in BAR_RUBRIC_ORDER], fontsize=12)
    ax.set_ylabel("Model-aggregated reasoning score", fontsize=12)
    ax.set_ylim(-0.05, 2.1)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(BAR_PNG if output_path is None else output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_regression_band(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """OLS line and paired-model bootstrap 95% confidence band."""
    slope, intercept = np.polyfit(x, y, deg=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lines = []
    for indices in rng.integers(0, len(x), size=(BOOTSTRAP_SAMPLES, len(x))):
        sampled_x, sampled_y = x[indices], y[indices]
        if np.unique(sampled_x).size < 2:
            continue
        sampled_slope, sampled_intercept = np.polyfit(sampled_x, sampled_y, deg=1)
        lines.append(sampled_slope * x_grid + sampled_intercept)
    band = np.asarray(lines)
    low, high = np.percentile(band, [2.5, 97.5], axis=0)
    return slope * x_grid + intercept, low, high, float(slope), float(intercept)


def plot_scatter(summary: pd.DataFrame, accuracy: pd.DataFrame, model_order: list[str], *, output_path=None, colors=None, custom=False) -> pd.DataFrame:
    apply_paper_rcparams()
    colors = _model_color_map() if colors is None else colors
    merged = summary.merge(accuracy, on="model_name", validate="many_to_one")
    fig, axs = plt.subplots(1, 2, figsize=(160 / 25.4, 72 / 25.4), sharey=True)
    band_rows = []
    for ax, rubric_key in zip(axs, SCATTER_KEYS):
        panel = merged[merged["rubric_key"] == rubric_key].set_index("model_name").loc[model_order].reset_index()
        x = panel["reasoning_score"].to_numpy(float)
        y = panel["task_accuracy"].to_numpy(float)
        lower, upper = (-0.05, 2.05) if custom else ((1.0, 2.03) if rubric_key == SCATTER_KEYS[0] else (0.5, 2.03))
        x_grid = np.linspace(lower, upper, 200)
        can_fit = not custom or (len(x) >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2)
        if can_fit:
            line, band_low, band_high, slope, intercept = _bootstrap_regression_band(x, y, x_grid)
            ax.fill_between(x_grid, band_low, band_high, color="0.3", alpha=0.18, linewidth=0)
            ax.plot(x_grid, line, color="0.15", linewidth=1.2, zorder=2)
        else:
            ax.text(0.03, 0.97, "Too few distinct methods for correlation", transform=ax.transAxes,
                    va="top", fontsize=7)
        xerr = np.vstack([x - panel["score_ci_low"], panel["score_ci_high"] - x])
        yerr = np.vstack([y - panel["accuracy_ci_low"], panel["accuracy_ci_high"] - y])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.25", elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=3)
        ax.scatter(x, y, c=[colors[model] for model in panel["model_name"]], s=52,
                   edgecolors="black", linewidths=0.55, zorder=4)
        ax.set_xlabel(RUBRIC_LABELS[rubric_key].replace("\n", " "), fontsize=12)
        ax.set_xlim(lower, upper)
        if custom:
            ax.set_xticks(np.arange(0, 2.01, 0.5))
            for model_name, px, py in zip(panel["model_name"], x, y):
                ax.annotate(model_name, (px, py), xytext=(4, 4), textcoords="offset points", fontsize=6)
        elif rubric_key == SCATTER_KEYS[0]:
            ax.set_xticks(np.arange(1.0, 2.01, 0.25))
        else:
            ax.set_xticks(np.arange(0.5, 2.01, 0.5))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        if not can_fit:
            continue
        # Match the canonical model order used by the report tables so finite
        # Monte Carlo bootstrap draws reproduce their confidence intervals.
        correlation_panel = panel.sort_values("model_name")
        rho, pvalue, ci_low, ci_high, n_models = spearman_with_bootstrap_ci(
            correlation_panel["reasoning_score"], correlation_panel["task_accuracy"]
        )
        for grid_x, fit, low, high in zip(x_grid, line, band_low, band_high):
            band_rows.append(
                {
                    "rubric_key": rubric_key, "x": grid_x, "ols_prediction": fit,
                    "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                    "ols_slope": slope, "ols_intercept": intercept, "spearman_rho": rho,
                    "spearman_ci_low": ci_low, "spearman_ci_high": ci_high,
                    "spearman_pvalue": pvalue, "n_models": n_models,
                }
            )
    axs[0].set_ylabel("Task Accuracy at Iteration 5", fontsize=12)
    axs[0].set_ylim((-0.05, 1.05) if custom else (0.34, 0.66))
    axs[0].set_yticks(np.arange(0, 1.01, 0.2) if custom else np.arange(0.35, 0.66, 0.1))
    fig.tight_layout()
    fig.savefig(SCATTER_PNG if output_path is None else output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(band_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ANALYSIS_DIR.mkdir(exist_ok=True)
    summary = build_score_summary()
    accuracy = build_accuracy_iteration5()
    model_order = _ordered_models(accuracy)
    summary.to_csv(SCORES_CSV, index=False, float_format="%.6f")
    plot_bar(summary, model_order)
    bands = plot_scatter(summary, accuracy, model_order)
    correlation = bands.groupby("rubric_key", as_index=False).first()[
        ["rubric_key", "n_models", "spearman_rho", "spearman_ci_low", "spearman_ci_high", "spearman_pvalue"]
    ]
    correlation.to_csv(CORRELATION_CSV, index=False, float_format="%.6f")
    print(f"Saved bar plot: {BAR_PNG}")
    print(f"Saved scatter plot: {SCATTER_PNG}")
    print(correlation.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
