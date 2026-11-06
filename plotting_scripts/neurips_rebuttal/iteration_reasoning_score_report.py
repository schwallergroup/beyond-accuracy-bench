"""Report iteration-wise reasoning-score stability and accuracy correlations."""

from __future__ import annotations

import os
import sys
import pandas as pd


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from plotting_scripts.neurips.judge_rubric_scores_by_model import RUBRIC_LABELS  # noqa: E402
from plotting_scripts.neurips_rebuttal.iteration_reasoning_score_analysis import (  # noqa: E402
    ITERATIONS,
    build_accuracy_iteration5,
    build_iteration_reasoning_scores,
    spearman_with_bootstrap_ci,
)


OUTPUT_DIR = os.path.join(script_dir, "iteration_analysis")
def build_report_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    scores, shared_sample_ids = build_iteration_reasoning_scores()
    accuracy = build_accuracy_iteration5()
    score_table = scores.merge(accuracy, on="model_name", how="inner")
    score_table["rubric_label"] = score_table["rubric_key"].map(RUBRIC_LABELS).str.replace("\n", " ")

    accuracy_rows = []
    for rubric_key, rubric_scores in score_table.groupby("rubric_key", sort=False):
        for iteration in ITERATIONS:
            iteration_scores = rubric_scores[rubric_scores["iteration"] == iteration]
            rho, p_value, ci_low, ci_high, n_models = spearman_with_bootstrap_ci(
                iteration_scores["reasoning_score"], iteration_scores["task_accuracy"]
            )
            accuracy_rows.append(
                {
                    "iteration": iteration,
                    "rubric_key": rubric_key,
                    "reasoning_score": RUBRIC_LABELS[rubric_key].replace("\n", " "),
                    "n_models": n_models,
                    "spearman_rho_with_task_accuracy": rho,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "two_sided_p_value": p_value,
                }
            )
    return score_table, pd.DataFrame(accuracy_rows), shared_sample_ids


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    score_table, accuracy_table, shared_sample_ids = build_report_tables()
    score_path = os.path.join(OUTPUT_DIR, "reasoning_scores_by_iteration.csv")
    accuracy_path = os.path.join(OUTPUT_DIR, "iteration_score_accuracy_spearman.csv")
    score_table.to_csv(score_path, index=False, float_format="%.6f")
    accuracy_table.to_csv(accuracy_path, index=False, float_format="%.6f")
    print(f"Shared complete samples ({len(shared_sample_ids)}): {', '.join(shared_sample_ids)}")
    print(accuracy_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved tables: {score_path}, {accuracy_path}")


if __name__ == "__main__":
    main()
