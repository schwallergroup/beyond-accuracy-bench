"""Relate four-step reasoning scores to iteration-5 task accuracy.

Uses the nine step-instruction ablations judged on M*_0/M*_1 at iteration 3.
The target model is never scored by an identical judge model: GPT-5.4 targets
retain Claude Opus 4.7 scores only, Claude Opus 4.7 targets retain GPT-5.4
scores only, and all other targets average both judges.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plotting_scripts.neurips_rebuttal.judge_filtering import (  # noqa: E402
    canonical_model_id,
)
from plotting_scripts.neurips_rebuttal.mechanism_info_accuracy_comparison import (  # noqa: E402
    gather_step5_accuracy,
)
from plotting_scripts.neurips_rebuttal.iteration_reasoning_score_analysis import (  # noqa: E402
    spearman_with_bootstrap_ci,
)
from plotting_scripts.neurips.archived_experiment_data import discover_experiments  # noqa: E402


STEP_RUNS = (
    ("Claude Haiku 4.5", "20260919_112018.tar"),
    ("Claude Opus 4.7", "20260919_160526.tar"),
    ("Gemini 2.5 Pro", "20260730_071419.tar"),
    ("Gemini 3.1 Pro", "20260729_111220.tar"),
    ("GPT-5.4", "20260730_121845.tar"),
    ("GPT-4o", "20260729_160817.tar"),
    ("Qwen3-32B", "20260730_171350.tar"),
    ("DeepSeek V3.2", "20260729_205512.tar"),
    ("Llama 4 Maverick", "20260919_204951.tar"),
)
JUDGE_INPUTS = {
    "gpt-5.4": (
        PROJECT_ROOT / "bench/judge/runs/four_step_iter3/results/gpt54/judge_results.csv",
    ),
    "claude-opus-4.7": (
        PROJECT_ROOT / "bench/judge/runs/four_step_iter3/results/claude47/judge_results.csv",
    ),
}
OUTPUT_DIR = SCRIPT_DIR / "step_instruction_judge_analysis"
MODEL_CSV = OUTPUT_DIR / "iteration3_reasoning_vs_iteration5_accuracy.csv"
CORRELATION_CSV = OUTPUT_DIR / "iteration3_pr_ded_vs_iteration5_accuracy_spearman.csv"


def build_model_table() -> pd.DataFrame:
    judged = []
    for judge_id, paths in JUDGE_INPUTS.items():
        for path in paths:
            frame = pd.read_csv(path)
            observed = {canonical_model_id(value) for value in frame["judge_model"].unique()}
            if observed != {judge_id}:
                raise RuntimeError(f"Unexpected judge identity in {path}: {observed}")
            frame["judge_id"] = judge_id
            frame["model"] = frame["model_label"].str.replace(
                r"\s+step\s+instruction$", "", regex=True, case=False
            )
            judged.append(frame)
    scores = pd.concat(judged, ignore_index=True)
    scores["score"] = pd.to_numeric(scores["score"], errors="raise")
    eligible = scores[
        scores.apply(
            lambda row: row["judge_id"] != canonical_model_id(row["model"]), axis=1
        )
    ]
    reasoning = (
        eligible.groupby(["model", "rubric_key"], as_index=False)
        .agg(reasoning_score=("score", "mean"), n_trace_scores=("score", "size"))
        .pivot(index="model", columns="rubric_key", values="reasoning_score")
        .reset_index()
    )
    reasoning.columns.name = None
    experiments = discover_experiments(str(PROJECT_ROOT / "results"))
    try:
        by_source = {experiment.source_tar: experiment for experiment in experiments}
        accuracy_rows = []
        for model, source in STEP_RUNS:
            accuracy = gather_step5_accuracy(by_source[source])
            accuracy_rows.append(
                {"model": model, "accuracy_iteration5": accuracy["accuracy"].mean(), "n_accuracy": len(accuracy)}
            )
    finally:
        for experiment in experiments:
            experiment.close()
    return pd.DataFrame(accuracy_rows).merge(reasoning, on="model", validate="one_to_one")


def build_correlation_table(model_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rubric_key in ("predictive_sharpness", "feature_to_action_faithfulness"):
        rho, pvalue, ci_low, ci_high, n_models = spearman_with_bootstrap_ci(
            model_table[rubric_key], model_table["accuracy_iteration5"]
        )
        rows.append(
            {
                "rubric_key": rubric_key,
                "n_models": n_models,
                "spearman_rho": rho,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "two_sided_pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_table = build_model_table()
    correlations = build_correlation_table(model_table)
    model_table.to_csv(MODEL_CSV, index=False, float_format="%.6f")
    correlations.to_csv(CORRELATION_CSV, index=False, float_format="%.6f")
    print(model_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(correlations.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
