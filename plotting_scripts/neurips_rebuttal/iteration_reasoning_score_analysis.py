"""Reusable data preparation for iteration-wise self-judge-controlled analyses."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from plotting_scripts.neurips.archived_experiment_data import (
    build_metric_dataframe,
    discover_experiments,
    select_preferred_experiments,
)
from plotting_scripts.neurips.judge_rubric_scores_by_model import _strip_mechanism_suffix
from plotting_scripts.neurips.paper_model_metadata import LABELS
from plotting_scripts.neurips_rebuttal.judge_filtering import canonical_model_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ITERATIONS = (2, 3, 4)
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 0
JUDGE_INPUTS = {
    "gpt-5.4": PROJECT_ROOT
    / "bench/judge/runs/standard_iteration2_3_4/results/gpt54/judge_results.csv",
    "claude-opus-4.7": PROJECT_ROOT
    / "bench/judge/runs/standard_iteration2_3_4/results/claude47/judge_results.csv",
}


def _bootstrap_ci(values: pd.Series, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(array, size=len(array), replace=True).mean() for _ in range(n_boot)])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def spearman_with_bootstrap_ci(
    x: pd.Series, y: pd.Series, n_boot: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float, float, int]:
    """Spearman's rho and percentile CI from paired-model resampling."""
    paired = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    result = spearmanr(paired["x"], paired["y"])
    rng = np.random.default_rng(seed)
    values = paired.to_numpy(dtype=float)
    # Kept as a single vectorized batch to match scipy's average-rank
    # Spearman calculation while avoiding 4,000 Python/scipy calls.
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    sampled_x = values[indices, 0]
    sampled_y = values[indices, 1]
    ranked_x = rankdata(sampled_x, axis=1)
    ranked_y = rankdata(sampled_y, axis=1)
    centered_x = ranked_x - ranked_x.mean(axis=1, keepdims=True)
    centered_y = ranked_y - ranked_y.mean(axis=1, keepdims=True)
    denominator = np.sqrt((centered_x**2).sum(axis=1) * (centered_y**2).sum(axis=1))
    bootstrap_rhos = np.divide(
        (centered_x * centered_y).sum(axis=1),
        denominator,
        out=np.full(n_boot, np.nan),
        where=denominator != 0,
    )
    bootstrap_rhos = bootstrap_rhos[np.isfinite(bootstrap_rhos)]
    ci_low, ci_high = np.percentile(bootstrap_rhos, [2.5, 97.5])
    return float(result.statistic), float(result.pvalue), float(ci_low), float(ci_high), len(paired)


def _load_judge_scores() -> pd.DataFrame:
    frames = []
    for judge_id, path in JUDGE_INPUTS.items():
        frame = pd.read_csv(path)
        observed = {canonical_model_id(value) for value in frame["judge_model"].dropna().astype(str)}
        if observed != {judge_id}:
            raise RuntimeError(f"Expected judge {judge_id!r} in {path}, found {sorted(observed)}")
        frame = frame[frame["iteration"].isin(ITERATIONS)].copy()
        frame["model_name"] = frame["model_label"].map(_strip_mechanism_suffix)
        frame["judge_id"] = judge_id
        frames.append(frame[["sample_id", "iteration", "model_name", "rubric_key", "score", "judge_id"]])
    return pd.concat(frames, ignore_index=True)


def _complete_shared_sample_ids(scores: pd.DataFrame) -> list[str]:
    """Return samples complete for every model/rubric/judge at every iteration."""
    # Use the full intended grid, rather than the observed grid: otherwise a
    # rubric cell missing in every iteration would silently be treated as valid.
    expected_cells = (
        scores["model_name"].nunique()
        * scores["rubric_key"].nunique()
        * len(JUDGE_INPUTS)
    )
    complete_by_iteration = []
    for iteration in ITERATIONS:
        counts = scores[scores["iteration"] == iteration].groupby("sample_id").size()
        complete_by_iteration.append(set(counts[counts == expected_cells].index))
    common = set.intersection(*complete_by_iteration)
    if not common:
        raise RuntimeError("No complete samples are shared across all requested iterations.")
    return sorted(common)


def build_iteration_reasoning_per_sample_scores() -> tuple[pd.DataFrame, list[str]]:
    """Return self-judge-excluded scores, one row per iteration/model/rubric/trace.

    Non-self targets are averaged over GPT-5.4 and Claude Opus 4.7. GPT-5.4
    targets use Claude only, and Claude Opus 4.7 targets use GPT only.
    """
    scores = _load_judge_scores()
    shared_sample_ids = _complete_shared_sample_ids(scores)
    scores = scores[scores["sample_id"].isin(shared_sample_ids)].copy()
    target_ids = scores["model_name"].map(canonical_model_id)
    eligible = scores[scores["judge_id"] != target_ids].copy()
    per_sample = (
        eligible.groupby(["iteration", "model_name", "rubric_key", "sample_id"], as_index=False)
        .agg(reasoning_score=("score", "mean"), n_judges=("judge_id", "nunique"))
    )
    return per_sample, shared_sample_ids


def build_iteration_reasoning_scores() -> tuple[pd.DataFrame, list[str]]:
    """Mean self-judge-excluded score for every iteration/model/rubric cell."""
    per_sample, shared_sample_ids = build_iteration_reasoning_per_sample_scores()
    summary = (
        per_sample.groupby(["iteration", "model_name", "rubric_key"], as_index=False)
        .agg(reasoning_score=("reasoning_score", "mean"), n_shared_samples=("sample_id", "nunique"))
    )
    return summary, shared_sample_ids


def build_accuracy_iteration5() -> pd.DataFrame:
    """Return the mean task accuracy at iteration 5 for each evaluation model."""
    experiments = discover_experiments(os.path.join(PROJECT_ROOT, "results"))
    selected = select_preferred_experiments(experiments, LABELS)
    try:
        metric_df = build_metric_dataframe(selected)
        accuracy = (
            metric_df[(metric_df["metric"] == "Accuracy from NLL") & (metric_df["step"] == 5)]
            .groupby(["method", "source_tar"], as_index=False)
            .agg(
                task_accuracy=("value", "mean"),
                accuracy_ci_low=("value", lambda values: _bootstrap_ci(values)[0]),
                accuracy_ci_high=("value", lambda values: _bootstrap_ci(values)[1]),
                accuracy_n=("value", "size"),
            )
        )
    finally:
        for experiment in selected:
            experiment.close()
    accuracy["model_name"] = accuracy["method"].map(_strip_mechanism_suffix)
    return accuracy[["model_name", "task_accuracy", "accuracy_ci_low", "accuracy_ci_high", "accuracy_n"]]
