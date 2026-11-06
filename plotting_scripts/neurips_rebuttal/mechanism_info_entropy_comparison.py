"""Compare iteration-5 proposal-space metrics with and without mechanism info."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from plotting_scripts.neurips.archived_experiment_data import (  # noqa: E402
    build_metric_dataframe,
    discover_experiments,
    select_preferred_experiments,
)
from plotting_scripts.neurips.model_accuracy_at_iteration5 import (  # noqa: E402
    _bootstrap_mean_ci,
    _stable_seed,
)
from plotting_scripts.neurips_rebuttal.mechanism_info_accuracy_comparison import (  # noqa: E402
    _bootstrap_independent_difference_ci,
)


OUTPUT_DIR = os.path.join(script_dir, "mechanism_info_ablation")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "iteration5_entropy_with_vs_without_mechanism.csv")
MODEL_SOURCES = {
    "Gemini 3.1 Pro": "20260728_123558.tar",
    "Gemini 2.5 Pro": "20260728_172602.tar",
}
METRICS = ("Occupancy Entropy", "S-cat Corner Mass")


def _metric_summary(values: pd.Series, seed: int) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    return _bootstrap_mean_ci(array, seed=seed)


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
            if with_experiment is None or without_experiment is None:
                raise RuntimeError(f"Missing experiment pair for {model_name}")
            if without_experiment.label != without_label:
                raise RuntimeError(f"Unexpected label for {without_source}: {without_experiment.label}")

            metric_df = build_metric_dataframe([with_experiment, without_experiment])
            for metric_name in METRICS:
                with_values = metric_df[
                    (metric_df["method"] == with_experiment.label)
                    & (metric_df["source_tar"] == with_experiment.source_tar)
                    & (metric_df["metric"] == metric_name)
                    & (metric_df["step"] == 5)
                ]["value"]
                without_values = metric_df[
                    (metric_df["method"] == without_experiment.label)
                    & (metric_df["source_tar"] == without_experiment.source_tar)
                    & (metric_df["metric"] == metric_name)
                    & (metric_df["step"] == 5)
                ]["value"]
                with_mean, with_low, with_high = _metric_summary(
                    with_values, _stable_seed("mechanism_info_metric", model_name, metric_name, "with")
                )
                without_mean, without_low, without_high = _metric_summary(
                    without_values, _stable_seed("mechanism_info_metric", model_name, metric_name, "without")
                )
                difference_mean, difference_low, difference_high = _bootstrap_independent_difference_ci(
                    with_values.to_numpy(dtype=float),
                    without_values.to_numpy(dtype=float),
                    _stable_seed("mechanism_info_metric", model_name, metric_name, "difference"),
                )
                rows.append(
                    {
                        "model": model_name,
                        "metric": metric_name,
                        "n_with_mechanism": len(with_values),
                        "n_without_mechanism": len(without_values),
                        "with_mechanism": with_mean,
                        "with_ci_low": with_low,
                        "with_ci_high": with_high,
                        "without_mechanism": without_mean,
                        "without_ci_low": without_low,
                        "without_ci_high": without_high,
                        "with_minus_without": difference_mean,
                        "difference_ci_low": difference_low,
                        "difference_ci_high": difference_high,
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
