"""Compare iteration-5 accuracy with and without mechanism information."""

from __future__ import annotations

import os
import sys

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
from plotting_scripts.neurips.model_accuracy_at_iteration5 import (  # noqa: E402
    _bootstrap_mean_ci,
    _stable_seed,
)


OUTPUT_DIR = os.path.join(script_dir, "mechanism_info_ablation")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "iteration5_accuracy_with_vs_without_mechanism.csv")
MODEL_SOURCES = {
    "Gemini 3.1 Pro": "20260728_123558.tar",
    "Gemini 2.5 Pro": "20260728_172602.tar",
    "GPT-4o": "20260728_223705.tar",
    "Claude Haiku 4.5": "20260728_230109.tar",
}
BOOTSTRAP_SAMPLES = 4000


def gather_step5_accuracy(experiment) -> pd.DataFrame:
    """Reuse the paper's step-5 rank-accuracy logic without display-label needs."""
    rows = []
    for sample_id in experiment.sample_ids:
        ground_truth = sample_id.split("_")[0]
        fit_items, _ = _latest_valid_fit_items(experiment, experiment.sample_exp_dir(sample_id), 5)
        if not fit_items:
            continue
        rank = next(
            (index + 1 for index, (model_name, _) in enumerate(fit_items) if model_name == ground_truth),
            len(fit_items) + 1,
        )
        rows.append({"sample_id": sample_id, "accuracy": float(rank == 1)})
    return pd.DataFrame(rows, columns=["sample_id", "accuracy"])


def _bootstrap_independent_difference_ci(
    with_values: np.ndarray, without_values: np.ndarray, seed: int
) -> tuple[float, float, float]:
    """Bootstrap the difference of independent run-level means."""
    with_values = np.asarray(with_values, dtype=float)
    without_values = np.asarray(without_values, dtype=float)
    rng = np.random.default_rng(seed)
    differences = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        differences[index] = (
            rng.choice(with_values, size=len(with_values), replace=True).mean()
            - rng.choice(without_values, size=len(without_values), replace=True).mean()
        )
    mean = float(with_values.mean() - without_values.mean())
    low, high = np.percentile(differences, [2.5, 97.5])
    return mean, float(low), float(high)


def build_comparison_table() -> pd.DataFrame:
    experiments = discover_experiments(os.path.join(project_root, "results"))
    with_info_labels = [f"{model_name} with Mechanism Info" for model_name in MODEL_SOURCES]
    selected_with_info = select_preferred_experiments(experiments, with_info_labels)
    selected_by_source = {experiment.source_tar: experiment for experiment in experiments}
    try:
        rows = []
        for model_name, without_source in MODEL_SOURCES.items():
            with_label = f"{model_name} with Mechanism Info"
            without_label = f"{model_name} without Mechanism Info"
            with_experiment = next((exp for exp in selected_with_info if exp.label == with_label), None)
            without_experiment = selected_by_source.get(without_source)
            if with_experiment is None:
                raise RuntimeError(f"Missing preferred mechanism-info experiment: {with_label}")
            if without_experiment is None or without_experiment.label != without_label:
                raise RuntimeError(f"Missing mechanism-hidden experiment: {without_source}")

            with_accuracy = gather_step5_accuracy(with_experiment)
            without_accuracy = gather_step5_accuracy(without_experiment)
            if without_accuracy.empty:
                rows.append(
                    {
                        "model": model_name,
                        "n_with_mechanism": len(with_accuracy),
                        "n_without_mechanism": 0,
                        "n_shared_samples": 0,
                        "accuracy_with_mechanism": np.nan,
                        "with_mechanism_ci_low": np.nan,
                        "with_mechanism_ci_high": np.nan,
                        "accuracy_without_mechanism": np.nan,
                        "without_mechanism_ci_low": np.nan,
                        "without_mechanism_ci_high": np.nan,
                        "with_minus_without": np.nan,
                        "difference_ci_low": np.nan,
                        "difference_ci_high": np.nan,
                        "status": "No valid iteration-5 fitting result in mechanism-hidden archive",
                        "with_source": with_experiment.source_tar,
                        "without_source": without_experiment.source_tar,
                    }
                )
                continue
            paired = with_accuracy.merge(without_accuracy, on="sample_id", suffixes=("_with", "_without"))

            with_mean, with_low, with_high = _bootstrap_mean_ci(
                with_accuracy["accuracy"].to_numpy(dtype=float),
                seed=_stable_seed("mechanism_info", model_name, "with"),
            )
            without_mean, without_low, without_high = _bootstrap_mean_ci(
                without_accuracy["accuracy"].to_numpy(dtype=float),
                seed=_stable_seed("mechanism_info", model_name, "without"),
            )
            difference_mean, difference_low, difference_high = _bootstrap_independent_difference_ci(
                with_accuracy["accuracy"].to_numpy(dtype=float),
                without_accuracy["accuracy"].to_numpy(dtype=float),
                seed=_stable_seed("mechanism_info", model_name, "difference"),
            )
            rows.append(
                {
                        "model": model_name,
                    "n_with_mechanism": len(with_accuracy),
                    "n_without_mechanism": len(without_accuracy),
                    "n_shared_samples": len(paired),
                    "accuracy_with_mechanism": with_mean,
                    "with_mechanism_ci_low": with_low,
                    "with_mechanism_ci_high": with_high,
                    "accuracy_without_mechanism": without_mean,
                    "without_mechanism_ci_low": without_low,
                    "without_mechanism_ci_high": without_high,
                    "with_minus_without": difference_mean,
                    "difference_ci_low": difference_low,
                    "difference_ci_high": difference_high,
                    "status": "complete",
                    "with_source": with_experiment.source_tar,
                    "without_source": without_experiment.source_tar,
                }
            )
    finally:
        for experiment in experiments:
            experiment.close()
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    table = build_comparison_table()
    table.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved comparison: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
