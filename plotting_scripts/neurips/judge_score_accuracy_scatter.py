import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from plotting_scripts.neurips.paper_model_metadata import (  # noqa: E402
    DISPLAY_LABELS,
    LABELS,
    MODEL_COLOR_SEQUENCE,
)
from plotting_scripts.neurips.judge_rubric_scores_by_model import (  # noqa: E402
    DEFAULT_INPUTS,
    RUBRIC_LABELS,
    RUBRIC_ORDER,
    _accuracy_sorted_model_order,
    _build_summary_dataframe,
    _load_judge_csvs,
)
from plotting_scripts.neurips.archived_experiment_data import (  # noqa: E402
    build_metric_dataframe,
    discover_experiments,
    select_preferred_experiments,
)
from plotting_scripts.styling import apply_paper_rcparams  # noqa: E402


OUTPUT_DIR = os.path.join(script_dir, "imgs_for_paper")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "judge_score_accuracy_scatter.png")

BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 0
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
JUDGE_SCORE_XLIMS = {
    "hypothesis_space_alignment": (1.0, 2.03),
    "mechanistic_feature_extraction": (1.0, 2.03),
    "predictive_sharpness": (0.0, 2.03),
    "feature_to_action_faithfulness": (1.2, 2.03),
}
JUDGE_SCORE_XTICKS = {
    "hypothesis_space_alignment": np.arange(1.0, 2.01, 0.25),
    "mechanistic_feature_extraction": np.arange(1.0, 2.01, 0.25),
    "predictive_sharpness": np.arange(0.0, 2.01, 0.5),
    "feature_to_action_faithfulness": np.arange(1.25, 2.01, 0.25),
}
PREDICTIVE_KEY = "predictive_sharpness"
DESIGN_KEY = "feature_to_action_faithfulness"
SCATTER_RUBRIC_ORDER = [
    PREDICTIVE_KEY,
    DESIGN_KEY,
]


def _strip_mechanism_suffix(label: str) -> str:
    return label.replace(" with Mechanism Info", "").replace(" without Mechanism Info", "")


def _model_order_from_judge_summary(judge_df: pd.DataFrame) -> list[str]:
    seen = []
    for model_name in judge_df["model_name"].tolist():
        if model_name not in seen:
            seen.append(model_name)
    return seen


def _bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = sample.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def build_plot_dataframe() -> pd.DataFrame:
    frames = _load_judge_csvs(DEFAULT_INPUTS)
    model_order = [model_name for model_name in _accuracy_sorted_model_order() if model_name in DEFAULT_INPUTS]
    judge_df, _ = _build_summary_dataframe(frames, [("Model comparison", model_order)])
    judge_wide = (
        judge_df[judge_df["rubric_key"].isin(RUBRIC_ORDER)]
        .pivot(index="model_name", columns="rubric_key", values=["mean_score", "ci_low", "ci_high", "n_samples"])
        .reset_index()
    )
    judge_wide.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in judge_wide.columns.to_flat_index()
    ]

    experiments = discover_experiments(os.path.join(project_root, "results"))
    selected = select_preferred_experiments(experiments, LABELS)
    try:
        metric_df = build_metric_dataframe(selected)
        accuracy_step5_df = (
            metric_df[(metric_df["metric"] == "Accuracy from NLL") & (metric_df["step"] == 5)]
            .groupby(["method", "source_tar"], as_index=False)
            .agg(
                accuracy=("value", "mean"),
                accuracy_ci_low=("value", lambda values: _bootstrap_ci(values.to_numpy(dtype=float))[0]),
                accuracy_ci_high=("value", lambda values: _bootstrap_ci(values.to_numpy(dtype=float))[1]),
                accuracy_n=("value", "size"),
            )
        )
    finally:
        for exp in selected:
            exp.close()

    accuracy_step5_df["model_name"] = accuracy_step5_df["method"].map(_strip_mechanism_suffix)
    plot_df = judge_wide.merge(accuracy_step5_df, on="model_name", how="inner")
    order = _model_order_from_judge_summary(judge_df)
    plot_df["model_order"] = plot_df["model_name"].map({name: idx for idx, name in enumerate(order)})
    plot_df = plot_df.sort_values("model_order").reset_index(drop=True)
    plot_df["display_label"] = plot_df["method"].map(DISPLAY_LABELS).fillna(plot_df["model_name"])
    plot_df["color"] = MODEL_COLOR_SEQUENCE[: len(plot_df)]
    return plot_df


def plot_scatter(plot_df: pd.DataFrame) -> None:
    apply_paper_rcparams()
    fig, axs = plt.subplots(1, len(SCATTER_RUBRIC_ORDER), figsize=(160 / 25.4, 72 / 25.4), sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for ax, rubric_key in zip(axs, SCATTER_RUBRIC_ORDER):
        x = plot_df[f"mean_score_{rubric_key}"].to_numpy(dtype=float)
        y = plot_df["accuracy"].to_numpy(dtype=float)
        xerr = np.vstack(
            [
                x - plot_df[f"ci_low_{rubric_key}"].to_numpy(dtype=float),
                plot_df[f"ci_high_{rubric_key}"].to_numpy(dtype=float) - x,
            ]
        )
        yerr = np.vstack(
            [
                y - plot_df["accuracy_ci_low"].to_numpy(dtype=float),
                plot_df["accuracy_ci_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="0.25",
            elinewidth=0.7,
            capsize=1.8,
            capthick=0.7,
            zorder=1,
        )
        ax.scatter(
            x,
            y,
            c=plot_df["color"].tolist(),
            s=52,
            edgecolors="black",
            linewidths=0.55,
            zorder=3,
        )
        ax.set_xlabel(RUBRIC_LABELS[rubric_key].replace("\n", " "), fontsize=AXIS_LABEL_FONTSIZE)
        ax.margins(x=0)
        ax.set_xticks(JUDGE_SCORE_XTICKS[rubric_key])
        ax.set_xlim(*JUDGE_SCORE_XLIMS[rubric_key])
        ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    axs[0].set_ylabel("Accuracy at Iteration 5", fontsize=AXIS_LABEL_FONTSIZE)
    axs[0].set_ylim(0.34, 0.66)
    axs[0].set_yticks(np.arange(0.35, 0.66, 0.1))
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_df = build_plot_dataframe()
    plot_scatter(plot_df)
    print(f"Saved figure: {OUTPUT_PNG}")
    print(plot_df[["display_label", "accuracy", "accuracy_n"]])


if __name__ == "__main__":
    main()
