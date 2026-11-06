import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from plotting_scripts.styling import apply_paper_rcparams
from plotting_scripts.neurips.archived_experiment_data import (
    PREFERRED_SOURCES_BY_LABEL,
    build_metric_dataframe,
    discover_experiments,
    filter_df_to_preferred_sources,
    select_preferred_experiments,
)

BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 0
MARKERS_BY_LABEL = {
    "Maximin LHS": "s",
    "Bayes": "^",
    "EIG Reference": "D",
}


def _bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(values)
    for idx in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[idx] = sample.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _mean_and_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(array.mean())
    ci_low, ci_high = _bootstrap_ci(array)
    return mean, ci_low, ci_high


def _plot_metric_panel(ax, df: pd.DataFrame, metric_name: str, experiments, ylabel: str, ylim=None):
    metric_df = df[df["metric"] == metric_name]
    for exp in experiments:
        subset = metric_df[(metric_df["method"] == exp.label) & (metric_df["source_tar"] == exp.source_tar)]
        if subset.empty:
            continue
        summary_rows = []
        for step, step_df in subset.groupby("step", sort=True):
            mean, ci_low, ci_high = _mean_and_ci(step_df["value"])
            summary_rows.append({"step": step, "mean": mean, "ci_low": ci_low, "ci_high": ci_high})
        summary = pd.DataFrame(summary_rows).dropna(subset=["mean"])
        if summary.empty:
            continue
        xs = summary["step"].to_numpy(dtype=float)
        means = summary["mean"].to_numpy(dtype=float)
        lower = means - summary["ci_low"].to_numpy(dtype=float)
        upper = summary["ci_high"].to_numpy(dtype=float) - means
        ax.plot(
            xs,
            means,
            color=exp.color,
            linestyle=exp.linestyle,
            marker=MARKERS_BY_LABEL.get(exp.label, "o"),
            markersize=5,
            label=exp.label,
        )
        ax.errorbar(
            xs,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="black" if metric_name == "Occupancy Entropy" else exp.color,
            elinewidth=1.0 if metric_name == "Occupancy Entropy" else 0.8,
            capsize=3.0 if metric_name == "Occupancy Entropy" else 2.0,
            capthick=1.0 if metric_name == "Occupancy Entropy" else 0.8,
            zorder=4,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if metric_name in {"Proposal Entropy", "Adaptivity"}:
        ax.set_xticks(range(2, 6))
    else:
        ax.set_xticks(range(1, 6))
    if ylim is not None:
        ax.set_ylim(*ylim)
    legend = ax.get_legend()
    if legend:
        legend.set_title("")


def _plot_aggregate_bars(ax, df: pd.DataFrame, experiments, metric_name: str, title: str, short_labels: dict[str, str]):
    selected_sources = {exp.label: exp.source_tar for exp in experiments}
    if metric_name in {
        "Accuracy from NLL",
        "Mechanism Entropy",
        "Occupancy Entropy",
        "S-cat Corner Mass",
    }:
        subset = df[
            (df["metric"] == metric_name)
            & (df["step"] == 5)
            & (df.apply(lambda row: selected_sources.get(row["method"]) == row["source_tar"], axis=1))
        ]
    else:
        subset = df[
            (df["metric"] == metric_name)
            & (df["step"] == 0)
            & (df.apply(lambda row: selected_sources.get(row["method"]) == row["source_tar"], axis=1))
        ]
    labels = [exp.label for exp in experiments]
    values = []
    ci_lows = []
    ci_highs = []
    for exp in experiments:
        exp_subset = subset[(subset["method"] == exp.label) & (subset["source_tar"] == exp.source_tar)]
        mean, ci_low, ci_high = _mean_and_ci(exp_subset["value"])
        values.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
    bars = ax.bar(
        [short_labels.get(label, label) for label in labels],
        values,
        color=[exp.color for exp in experiments],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, label in zip(bars, labels):
        if label == "Maximin LHS":
            bar.set_hatch("")
    for bar, value, ci_low, ci_high in zip(bars, values, ci_lows, ci_highs):
        if not all(np.isfinite([value, ci_low, ci_high])):
            continue
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2,
            value,
            yerr=np.array([[value - ci_low], [ci_high - value]]),
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
        )
    ax.set_title(title)
    ylabel_by_metric = {
        "Occupancy Entropy": "Occupancy Entropy",
        "S-cat Corner Mass": "S-cat Corner Mass",
    }
    ax.set_ylabel(ylabel_by_metric.get(metric_name, "Value"))
    if metric_name == "Accuracy from NLL":
        ax.set_ylim(0, 0.8)
    elif metric_name == "Mechanism Entropy":
        ax.set_ylim(0, 1.0)
    elif metric_name == "Occupancy Entropy":
        ax.set_ylim(0, 0.6)
    elif metric_name == "S-cat Corner Mass":
        ax.set_ylim(0, 0.3)
    elif metric_name == "Aggregate Adaptivity":
        ax.set_ylim(0, 0.4)
    else:
        ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", rotation=20, labelsize=7)


def main():
    results_dir = os.path.join(project_root, "results")
    output_dir = os.path.join(script_dir, "imgs_for_paper")
    os.makedirs(output_dir, exist_ok=True)

    gemini_vs_lhs_labels = ["Gemini 3.1 Pro with Mechanism Info", "Maximin LHS", "Bayes", "EIG Reference"]

    experiments = discover_experiments(results_dir)
    if not experiments:
        raise RuntimeError("No matching tar or directory experiments found for ground_truths_200_samples_v2.")
    experiments = select_preferred_experiments(experiments, gemini_vs_lhs_labels)
    if not experiments:
        raise RuntimeError("No preferred-source experiments found for ground_truths_200_samples_v2.")

    df = build_metric_dataframe(experiments)
    df = filter_df_to_preferred_sources(df, gemini_vs_lhs_labels)

    apply_paper_rcparams()

    gemini_vs_lhs_selected = select_preferred_experiments(experiments, gemini_vs_lhs_labels)
    gemini_vs_lhs_by_label = {exp.label: exp for exp in gemini_vs_lhs_selected}
    gemini_vs_lhs_exps = [gemini_vs_lhs_by_label[label] for label in gemini_vs_lhs_labels if label in gemini_vs_lhs_by_label]
    gemini_vs_lhs_short = {
        "Gemini 3.1 Pro with Mechanism Info": "Gemini 3.1 Pro",
        "Maximin LHS": "Maximin LHS",
        "Bayes": "Bayes",
        "EIG Reference": "EIG Reference",
    }

    fig1, axs1 = plt.subplots(2, 2, figsize=[182 / 25.4, 105 / 25.4])
    axs1 = axs1.flatten()
    _plot_metric_panel(axs1[0], df, "Accuracy from NLL", gemini_vs_lhs_exps, "Accuracy", ylim=(0, 0.8))
    _plot_metric_panel(axs1[1], df, "Mechanism Entropy", gemini_vs_lhs_exps, "Mechanism Entropy", ylim=(0.2, 1.0))
    for ax, metric_name, ylabel, ylim in zip(
        axs1[2:],
        ["Occupancy Entropy", "S-cat Corner Mass"],
        ["Occupancy Entropy", "S-cat Corner Mass"],
        [(0, 0.6), (0, 0.3)],
    ):
        _plot_metric_panel(ax, df, metric_name, gemini_vs_lhs_exps, ylabel, ylim=ylim)
    for ax in axs1[:2]:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    handles, labels = axs1[0].get_legend_handles_labels()
    if handles:
        display_labels = [gemini_vs_lhs_short.get(label, label) for label in labels]
        axs1[0].legend(handles, display_labels, loc="upper left", frameon=False, fontsize=7)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "baseline_accuracy_entropy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    for exp in experiments:
        exp.close()


if __name__ == "__main__":
    main()
