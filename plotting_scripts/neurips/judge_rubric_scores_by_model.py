import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from plotting_scripts.neurips.paper_model_metadata import (
    DISPLAY_LABELS as OVERLAY_DISPLAY_LABELS,
    LABELS as OVERLAY_LABELS,
    MODEL_COLOR_SEQUENCE as OVERLAY_MODEL_COLOR_SEQUENCE,
)
from plotting_scripts.neurips.archived_experiment_data import (
    build_metric_dataframe,
    discover_experiments,
    select_preferred_experiments,
)
from plotting_scripts.styling import apply_paper_rcparams


RUBRIC_ORDER = [
    "hypothesis_space_alignment",
    "mechanistic_feature_extraction",
    "predictive_sharpness",
    "feature_to_action_faithfulness",
]

RUBRIC_LABELS = {
    "hypothesis_space_alignment": "Belief\nUpdate",
    "mechanistic_feature_extraction": "Hypothesis\nDiscrimination",
    "predictive_sharpness": "Predictive\nReasoning",
    "feature_to_action_faithfulness": "Discriminative\nExperiment Design",
}

MODEL_DISPLAY = {
    "Gemini 3.1 Pro": "Gemini 3.1 Pro",
    "Gemini 2.5 Pro": "Gemini 2.5 Pro",
    "GPT-5.4": "GPT-5.4",
    "GPT-4o": "GPT-4o",
    "Claude Opus 4.7": "Claude 4.7 Opus",
    "Claude Haiku 4.5": "Claude 4.5 Haiku",
    "DeepSeek V3.2": "DeepSeek V3.2",
    "Mistral Large 3": "Mistral Large 3",
    "Llama 4 Maverick": "Llama 4 Maverick",
    "Qwen3-32B": "Qwen3-32B",
}


def _strip_mechanism_suffix(label: str) -> str:
    return label.replace(" with Mechanism Info", "").replace(" without Mechanism Info", "")


DEFAULT_INPUTS = {
    "Gemini 3.1 Pro": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "Gemini 2.5 Pro": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "GPT-5.4": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "GPT-4o": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "Claude Opus 4.7": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "Claude Haiku 4.5": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "DeepSeek V3.2": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "Mistral Large 3": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "Llama 4 Maverick": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
    "Qwen3-32B": os.path.join(
        project_root,
        "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    ),
}

BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 0
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 10


def _load_judge_csvs(input_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    csv_cache: dict[str, pd.DataFrame] = {}
    for model_name, csv_path in input_map.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing judge result CSV for {model_name}: {csv_path}")
        if csv_path not in csv_cache:
            csv_cache[csv_path] = pd.read_csv(csv_path)
        frame = csv_cache[csv_path].copy()
        if "model_label" in frame.columns:
            frame["_normalized_model_name"] = frame["model_label"].map(_strip_mechanism_suffix)
            frame = frame[frame["_normalized_model_name"] == model_name].copy()
            frame = frame.drop(columns=["_normalized_model_name"])
            if frame.empty:
                raise RuntimeError(f"No rows found for model {model_name} in {csv_path}")
        frame["model_name"] = model_name
        frames[model_name] = frame
    return frames


def _bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
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


def _summarize_group(group_name: str, model_names: list[str], frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    common_sample_ids = sorted(
        set.intersection(*(set(frames[name]["sample_id"].unique()) for name in model_names))
    )
    if not common_sample_ids:
        raise RuntimeError(f"No shared sample IDs found for group: {group_name}")

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        frame = frames[model_name]
        subset = frame[frame["sample_id"].isin(common_sample_ids)].copy()
        for rubric_key in RUBRIC_ORDER:
            rubric_scores = (
                subset[subset["rubric_key"] == rubric_key]
                .sort_values("sample_id")["score"]
                .to_numpy(dtype=float)
            )
            mean_score = float(rubric_scores.mean())
            ci_low, ci_high = _bootstrap_ci(rubric_scores)
            rows.append(
                {
                    "group_name": group_name,
                    "model_name": model_name,
                    "rubric_key": rubric_key,
                    "mean_score": mean_score,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_samples": len(rubric_scores),
                }
            )
    return pd.DataFrame(rows), common_sample_ids


def _build_summary_dataframe(
    frames: dict[str, pd.DataFrame],
    model_groups: list[tuple[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    intersection_rows = []
    for group_name, model_names in model_groups:
        group_df, common_sample_ids = _summarize_group(group_name, model_names, frames)
        summary_frames.append(group_df)
        for sample_id in common_sample_ids:
            intersection_rows.append(
                {
                    "group_name": group_name,
                    "sample_id": sample_id,
                }
            )
    return pd.concat(summary_frames, ignore_index=True), pd.DataFrame(intersection_rows)


def _accuracy_sorted_model_order() -> list[str]:
    results_dir = os.path.join(project_root, "results")
    experiments = discover_experiments(results_dir)
    selected = select_preferred_experiments(experiments, OVERLAY_LABELS)
    selected_by_label = {exp.label: exp for exp in selected}
    selected = [selected_by_label[label] for label in OVERLAY_LABELS if label in selected_by_label]
    if not selected:
        raise RuntimeError("No experiments found for accuracy-based model ordering.")

    df = build_metric_dataframe(selected)
    metric_df = df[(df["metric"] == "Accuracy from NLL") & (df["step"] == 5)]
    rows = []
    for exp in selected:
        subset = metric_df[(metric_df["method"] == exp.label) & (metric_df["source_tar"] == exp.source_tar)]
        rows.append(
            {
                "model_name": _strip_mechanism_suffix(exp.label),
                "display_label": OVERLAY_DISPLAY_LABELS[exp.label],
                "accuracy": float(subset["value"].mean()),
            }
        )
    for exp in selected:
        exp.close()

    rows.sort(key=lambda row: (-row["accuracy"], row["display_label"]))
    return [row["model_name"] for row in rows]


def _colors_for_model_order(model_names: list[str]) -> dict[str, str]:
    if len(model_names) > len(OVERLAY_MODEL_COLOR_SEQUENCE):
        raise RuntimeError("Not enough overlay colors for judge model order.")
    return dict(zip(model_names, OVERLAY_MODEL_COLOR_SEQUENCE))


def _plot_group(
    ax,
    summary_df: pd.DataFrame,
    group_name: str,
    model_names: list[str],
    model_colors: dict[str, str],
) -> None:
    group_df = summary_df[summary_df["group_name"] == group_name].copy()
    x = np.arange(len(RUBRIC_ORDER), dtype=float)
    width = min(0.12, 0.78 / max(len(model_names), 1))
    offsets = (np.arange(len(model_names)) - (len(model_names) - 1) / 2.0) * width

    for offset, model_name in zip(offsets, model_names):
        model_df = group_df[group_df["model_name"] == model_name].set_index("rubric_key").loc[RUBRIC_ORDER]
        xs = x + offset
        means = model_df["mean_score"].to_numpy(dtype=float)
        lower = means - model_df["ci_low"].to_numpy(dtype=float)
        upper = model_df["ci_high"].to_numpy(dtype=float) - means
        ax.bar(
            xs,
            means,
            color=model_colors[model_name],
            width=width * 0.92,
            edgecolor="black",
            linewidth=0.6,
            label=MODEL_DISPLAY[model_name],
        )
        ax.errorbar(
            xs,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([RUBRIC_LABELS[key] for key in RUBRIC_ORDER])
    ax.tick_params(axis="x", labelsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(-0.05, 2.1)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    output_dir = os.path.join(script_dir, "imgs_for_paper")
    os.makedirs(output_dir, exist_ok=True)

    apply_paper_rcparams()

    frames = _load_judge_csvs(DEFAULT_INPUTS)
    model_order = [model_name for model_name in _accuracy_sorted_model_order() if model_name in DEFAULT_INPUTS]
    model_colors = _colors_for_model_order(model_order)
    model_groups = [("Model comparison", model_order)]
    summary_df, intersection_df = _build_summary_dataframe(frames, model_groups)

    figure_path = os.path.join(output_dir, "judge_rubric_scores_by_model.png")

    if intersection_df.empty:
        raise RuntimeError("No shared judge samples found for model comparison.")

    fig, axs = plt.subplots(1, len(model_groups), figsize=(270 / 25.4, 92 / 25.4), sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    for ax, (group_name, model_names) in zip(axs, model_groups):
        _plot_group(ax, summary_df, group_name, model_names, model_colors)

    axs[0].set_ylabel("Judge score", fontsize=AXIS_LABEL_FONTSIZE)
    legend_map = {}
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_map[label] = handle
    if legend_map:
        legend_labels = [MODEL_DISPLAY[model_name] for model_name in model_order]
        legend_handles = [legend_map[label] for label in legend_labels if label in legend_map]
        legend_labels = [label for label in legend_labels if label in legend_map]
        row_major = []
        split = int(np.ceil(len(legend_labels) / 2))
        for idx in range(split):
            row_major.append(idx)
            if idx + split < len(legend_labels):
                row_major.append(idx + split)
        fig.legend(
            [legend_handles[idx] for idx in row_major],
            [legend_labels[idx] for idx in row_major],
            loc="upper center",
            ncol=split,
            frameon=False,
            bbox_to_anchor=(0.5, 1.1),
            fontsize=LEGEND_FONTSIZE,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
