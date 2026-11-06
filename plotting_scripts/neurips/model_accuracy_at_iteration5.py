import os
import sys
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from plotting_scripts.neurips.archived_experiment_data import (  # noqa: E402
    _latest_valid_fit_items,
    discover_experiments,
    select_preferred_experiments,
)
from plotting_scripts.neurips.paper_model_metadata import (  # noqa: E402
    DISPLAY_LABELS,
    LABELS,
    MODEL_COLORS,
)
from plotting_scripts.styling import apply_paper_rcparams  # noqa: E402


OUTPUT_DIR = os.path.join(script_dir, "imgs_for_paper")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "model_accuracy_at_iteration5.png")
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 0
BASELINE_LABELS = ["Maximin LHS", "Bayes"]
PLOT_LABELS = LABELS + BASELINE_LABELS
DISPLAY_LABEL_OVERRIDES = {
    "Maximin LHS": "Maximin LHS (baseline)",
    "Bayes": "Bayes (baseline)",
}


def _stable_seed(*parts) -> int:
    return zlib.crc32("::".join(str(part) for part in parts).encode("utf-8"))


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot[idx] = sample.mean()
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return mean, float(lower), float(upper)


def gather_step5_accuracy(exp) -> list[dict[str, object]]:
    rows = []
    for sample_id in exp.sample_ids:
        gt = sample_id.split("_")[0]
        exp_dir = exp.sample_exp_dir(sample_id)
        vals, fit_step = _latest_valid_fit_items(exp, exp_dir, 5)
        if not vals:
            continue

        rank = next((idx + 1 for idx, (name, _) in enumerate(vals) if name == gt), len(vals) + 1)
        rows.append(
            {
                "method": exp.label,
                "display_label": DISPLAY_LABELS[exp.label],
                "source_tar": exp.source_tar,
                "sample_id": sample_id,
                "fit_step": fit_step,
                "accuracy": 1.0 if rank == 1 else 0.0,
            }
        )
    return rows


def build_plot_dataframe() -> pd.DataFrame:
    experiments = discover_experiments(os.path.join(project_root, "results"))
    selected = select_preferred_experiments(experiments, PLOT_LABELS)
    try:
        selected_by_label = {exp.label: exp for exp in selected}
        missing = [label for label in PLOT_LABELS if label not in selected_by_label]
        if missing:
            raise RuntimeError(f"Missing preferred experiments for labels: {missing}")

        rows = []
        for label in PLOT_LABELS:
            rows.extend(gather_step5_accuracy(selected_by_label[label]))
    finally:
        for exp in selected:
            exp.close()

    df = pd.DataFrame(rows)
    summary_rows = []
    for label in PLOT_LABELS:
        subset = df[df["method"] == label]
        mean, lower, upper = _bootstrap_mean_ci(
            subset["accuracy"].to_numpy(dtype=float),
            seed=_stable_seed("iteration5_accuracy", label),
        )
        summary_rows.append(
            {
                "method": label,
                "display_label": DISPLAY_LABEL_OVERRIDES.get(label, DISPLAY_LABELS[label]),
                "mean": mean,
                "ci_low": lower,
                "ci_high": upper,
                "n": len(subset),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df.sort_values(["mean", "display_label"], ascending=[False, True]).reset_index(drop=True)


def plot_accuracy_bar(summary_df: pd.DataFrame) -> None:
    apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(98 / 25.4, 88 / 25.4))

    positions = np.arange(len(summary_df), dtype=float)
    means = summary_df["mean"].to_numpy(dtype=float)
    lower_err = means - summary_df["ci_low"].to_numpy(dtype=float)
    upper_err = summary_df["ci_high"].to_numpy(dtype=float) - means
    labels = summary_df["method"].tolist()

    ax.barh(
        positions,
        means,
        height=0.58,
        color=[MODEL_COLORS[label] for label in labels],
        edgecolor="black",
        linewidth=0.75,
        xerr=np.vstack([lower_err, upper_err]),
        ecolor="black",
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    for y, mean, upper in zip(positions, means, summary_df["ci_high"].to_numpy(dtype=float)):
        ax.text(upper + 0.015, y, f"{mean:.2f}", va="center", ha="left", fontsize=7)

    ax.set_yticks(positions)
    ax.set_yticklabels(summary_df["display_label"].tolist(), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy at Iteration 5", fontsize=8)
    ax.set_xlim(0.0, 0.75)
    ax.set_xticks(np.arange(0.0, 0.71, 0.1))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_df = build_plot_dataframe()
    plot_accuracy_bar(summary_df)
    print(f"Saved figure: {OUTPUT_PNG}")
    print(summary_df[["display_label", "mean", "ci_low", "ci_high", "n"]])


if __name__ == "__main__":
    main()
