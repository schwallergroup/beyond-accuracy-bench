"""Utilities for removing judge self-preference cases from rebuttal plots."""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Mapping

import pandas as pd


# Canonical model IDs deliberately use provider-neutral spelling so that labels
# such as ``GPT-5.4 with Mechanism Info`` and API IDs such as ``gpt-5.4`` match.
MODEL_PROVIDERS = {
    "gemini-3.1-pro": "Google",
    "gemini-2.5-pro": "Google",
    "gpt-5.4": "OpenAI",
    "gpt-4o": "OpenAI",
    "claude-opus-4.7": "Anthropic",
    "claude-haiku-4.5": "Anthropic",
    "deepseek-v3.2": "DeepSeek",
    "mistral-large-3": "Mistral AI",
    "llama-4-maverick": "Meta",
    "qwen3-32b": "Alibaba",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMBINED_JUDGE_INPUTS = {
    "gpt-5.4": PROJECT_ROOT
    / "bench/judge/runs/standard_iter3/results/gpt54/judge_results.csv",
    "claude-opus-4.7": PROJECT_ROOT
    / "bench/judge/runs/standard_iter3/results/claude47/judge_results.csv",
}


def canonical_model_id(model_name: str) -> str:
    """Return a comparable ID from a display label or an API model name."""
    normalized = re.sub(r"\s+(with|without)\s+mechanism\s+info$", "", model_name, flags=re.I)
    normalized = re.sub(r"\s+step\s+instruction$", "", normalized, flags=re.I)
    normalized = normalized.lower().replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    return {
        # The API ID reverses the display label's family/version order.
        "claude-4.7-opus": "claude-opus-4.7",
    }.get(normalized, normalized)


def excluded_models(frames: Mapping[str, pd.DataFrame], policy: str) -> set[str]:
    """Find target models to omit for a self-preference exclusion policy."""
    if policy not in {"same_model", "same_company"}:
        raise ValueError(f"Unknown exclusion policy: {policy}")

    judge_ids = {
        canonical_model_id(judge)
        for frame in frames.values()
        for judge in frame["judge_model"].dropna().astype(str).unique()
    }
    if not judge_ids:
        raise RuntimeError("No judge_model values found in judge results.")

    if policy == "same_model":
        return {model_name for model_name in frames if canonical_model_id(model_name) in judge_ids}

    judge_providers = {MODEL_PROVIDERS.get(judge_id) for judge_id in judge_ids}
    unknown_judges = judge_ids - MODEL_PROVIDERS.keys()
    if unknown_judges:
        raise KeyError(f"Missing provider metadata for judge model(s): {sorted(unknown_judges)}")

    return {
        model_name
        for model_name in frames
        if MODEL_PROVIDERS.get(canonical_model_id(model_name)) in judge_providers
    }


def filter_frames(
    frames: Mapping[str, pd.DataFrame], policy: str
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """Return the retained target-model frames and their excluded counterparts."""
    excluded = excluded_models(frames, policy)
    retained = {name: frame for name, frame in frames.items() if name not in excluded}
    if not retained:
        raise RuntimeError(f"Policy {policy!r} excluded every evaluation model.")
    return retained, excluded


def build_debiased_frames(policy: str) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Average the eligible GPT-5.4 and Claude Opus 4.7 judgments per sample.

    For a target whose evaluator is excluded by ``policy``, the remaining
    evaluator's score is retained as-is. All other targets use the mean of both
    judge scores. The returned frames contain one score per sample and rubric,
    ready for the existing bootstrap and plotting helpers.
    """
    if policy not in {"same_model", "same_company"}:
        raise ValueError(f"Unknown exclusion policy: {policy}")

    return build_debiased_frames_from_inputs(COMBINED_JUDGE_INPUTS, policy)


def build_debiased_frames_from_inputs(
    judge_inputs: Mapping[str, Path], policy: str
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Apply the rebuttal's judge-exclusion policy to arbitrary judge CSVs.

    ``judge_inputs`` maps each canonical judge ID to a CSV containing only that
    judge's results. This is useful for follow-up ablations whose target labels
    are not part of the historical combined evaluation.
    """
    if policy not in {"same_model", "same_company"}:
        raise ValueError(f"Unknown exclusion policy: {policy}")

    judged_frames = []
    for expected_judge, input_path in judge_inputs.items():
        frame = pd.read_csv(input_path)
        observed_judges = {canonical_model_id(value) for value in frame["judge_model"].dropna().astype(str)}
        if observed_judges != {expected_judge}:
            raise RuntimeError(
                f"Expected only {expected_judge!r} in {input_path}, found {sorted(observed_judges)}"
            )
        frame["score"] = pd.to_numeric(frame["score"], errors="raise")
        frame["model_name"] = frame["model_label"].map(
            lambda label: re.sub(r"\s+(with|without)\s+mechanism\s+info$", "", label, flags=re.I)
        )
        frame["judge_id"] = expected_judge
        judged_frames.append(frame[["sample_id", "rubric_key", "score", "model_name", "judge_id"]])

    combined = pd.concat(judged_frames, ignore_index=True)
    frames: dict[str, pd.DataFrame] = {}
    judges_used: dict[str, list[str]] = {}
    for model_name, target_frame in combined.groupby("model_name", sort=False):
        target_id = canonical_model_id(model_name)
        if policy == "same_model":
            eligible = target_frame[target_frame["judge_id"] != target_id].copy()
        else:
            target_provider = MODEL_PROVIDERS.get(target_id)
            if target_provider is None:
                raise KeyError(f"Missing provider metadata for target model: {model_name}")
            eligible = target_frame[
                target_frame["judge_id"].map(MODEL_PROVIDERS.__getitem__) != target_provider
            ].copy()
        if eligible.empty:
            raise RuntimeError(f"No eligible judge scores for {model_name} under {policy}")
        judges_used[model_name] = sorted(eligible["judge_id"].unique())
        frames[model_name] = (
            eligible.groupby(["sample_id", "rubric_key"], as_index=False)["score"].mean()
        )
    return frames, judges_used


def build_unfiltered_averaged_frames() -> dict[str, pd.DataFrame]:
    """Average the historical GPT-5.4 and Claude Opus scores with no exclusion."""
    judged_frames = []
    for expected_judge, input_path in COMBINED_JUDGE_INPUTS.items():
        frame = pd.read_csv(input_path)
        observed_judges = {canonical_model_id(value) for value in frame["judge_model"].dropna().astype(str)}
        if observed_judges != {expected_judge}:
            raise RuntimeError(
                f"Expected only {expected_judge!r} in {input_path}, found {sorted(observed_judges)}"
            )
        frame["model_name"] = frame["model_label"].map(
            lambda label: re.sub(r"\s+(with|without)\s+mechanism\s+info$", "", label, flags=re.I)
        )
        judged_frames.append(frame[["sample_id", "rubric_key", "score", "model_name"]])

    combined = pd.concat(judged_frames, ignore_index=True)
    return {
        model_name: target_frame.groupby(["sample_id", "rubric_key"], as_index=False)["score"].mean()
        for model_name, target_frame in combined.groupby("model_name", sort=False)
    }
