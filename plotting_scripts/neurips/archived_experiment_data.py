import json
import os
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bench.config import ALL_MODEL_NAMES
from plotting_scripts.proposal_metrics import (
    PROPOSAL_BINS,
    compute_boundary_mass,
    compute_occupancy_entropy,
    proposal_bin,
)
from plotting_scripts.styling import PAPER_COLORS


NORMALIZED_ENTROPY_DENOM = np.log(len(ALL_MODEL_NAMES))
PROPOSAL_ENTROPY_DENOM = np.log(int(np.prod(list(PROPOSAL_BINS.values()))))
PREFERRED_SOURCES_BY_LABEL = {
    "Claude Haiku 4.5 with Mechanism Info": "20260417_131634.tar",
    "Claude Haiku 4.5 without Mechanism Info": "20260417_180602.tar",
    "Claude Opus 4.7 with Mechanism Info": "20260429_merged_opus_complete.tar",
    "DeepSeek V3.2 with Mechanism Info": "20260421_045049.tar",
    "Gemini 3.1 Pro with Mechanism Info": "20260417_031934.tar",
    "Gemini 3.1 Pro without Mechanism Info": "20260417_082355.tar",
    "Gemini 2.5 Pro with Mechanism Info": "20260420_035848.tar",
    "Gemini 2.5 Pro without Mechanism Info": "20260415_021958.tar",
    "GPT-5.4 with Mechanism Info": "20260420_090522.tar",
    "GPT-4o with Mechanism Info": "20260427_130523.tar",
    "GPT-4o without Mechanism Info": "20260415_061445.tar",
    "Llama 4 Maverick with Mechanism Info": "20260420_192950.tar",
    "Mistral Large 3 with Mechanism Info": "20260421_001004.tar",
    "Qwen3-32B with Mechanism Info": "20260420_140933.tar",
    "Qwen3-32B without Mechanism Info": "20260415_095315.tar",
    "Maximin LHS": "20260421_105526.tar",
    "Random": "20260418_131836.tar",
    "Bayes": "20260416_152044.tar",
    "EIG Reference": "20260429_111756.tar",
}

def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    probs = np.array([c / total for c in counts.values() if c > 0], dtype=float)
    return float((-(probs * np.log(probs)).sum()) / PROPOSAL_ENTROPY_DENOM)


def _extract_proposal_dict(proposal_blob: dict[str, Any] | None) -> dict[str, float] | None:
    if proposal_blob is None:
        return None
    if isinstance(proposal_blob, dict) and all(key in proposal_blob for key in ("S", "P", "cat")):
        return proposal_blob
    proposal = proposal_blob.get("proposal") if isinstance(proposal_blob, dict) else None
    if isinstance(proposal, dict) and all(key in proposal for key in ("S", "P", "cat")):
        return proposal
    return None


def matches_preferred_source(label: str, source_name: str) -> bool:
    preferred = PREFERRED_SOURCES_BY_LABEL.get(label)
    if preferred is None:
        return True
    return source_name == preferred


def select_preferred_experiments(
    experiments: list["ExperimentBase"],
    labels: set[str] | list[str],
) -> list["ExperimentBase"]:
    label_set = set(labels)
    return [
        exp for exp in experiments
        if exp.label in label_set and matches_preferred_source(exp.label, exp.source_tar)
    ]


def filter_df_to_preferred_sources(
    df: pd.DataFrame,
    labels: set[str] | list[str],
) -> pd.DataFrame:
    label_set = set(labels)
    return df[
        df["method"].isin(label_set)
        & df.apply(lambda row: matches_preferred_source(row["method"], row["source_tar"]), axis=1)
    ]


@dataclass
class ExperimentBase:
    source_path: str
    proposer_name: str
    label: str
    color: str
    linestyle: str
    sample_ids: list[str]

    def read_json(self, relative_path: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def sample_exp_dir(self, sample_id: str) -> str:
        return f"{sample_id}/{self.proposer_name}"

    @property
    def source_tar(self) -> str:
        return os.path.basename(self.source_path)

    def close(self) -> None:
        return None


@dataclass
class TarExperiment(ExperimentBase):
    root_prefix: str
    tar: tarfile.TarFile
    members: set[str]

    def read_json(self, relative_path: str) -> dict[str, Any] | None:
        full_path = f"{self.root_prefix}/{relative_path}"
        if full_path not in self.members:
            return None
        extracted = self.tar.extractfile(full_path)
        if extracted is None:
            return None
        return json.load(extracted)

    def close(self) -> None:
        self.tar.close()


@dataclass
class DirectoryExperiment(ExperimentBase):
    root_dir: str

    def read_json(self, relative_path: str) -> dict[str, Any] | None:
        full_path = os.path.join(self.root_dir, relative_path)
        if not os.path.exists(full_path):
            return None
        with open(full_path, "r") as f:
            return json.load(f)


def _agent_label(model: str, variant: str) -> str:
    mech_suffix = "with Mechanism Info" if variant == "standard" else "without Mechanism Info"
    if model == "gemini-2.5-pro":
        prefix = "Gemini 2.5 Pro"
    elif model == "gemini-3.1-pro":
        prefix = "Gemini 3.1 Pro"
    elif model == "claude-4-5-haiku":
        prefix = "Claude Haiku 4.5"
    elif model == "claude-4.7-opus":
        prefix = "Claude Opus 4.7"
    elif model == "gpt-5.4":
        prefix = "GPT-5.4"
    elif model == "gpt-4o":
        prefix = "GPT-4o"
    elif model == "llama-4-maverick":
        prefix = "Llama 4 Maverick"
    elif model == "mistral-large-3":
        prefix = "Mistral Large 3"
    elif model == "qwen3-32b":
        prefix = "Qwen3-32B"
    elif model == "deepseek-v3.2":
        prefix = "DeepSeek V3.2"
    else:
        prefix = model
    return f"{prefix} {mech_suffix}"


def discover_experiments(results_dir: str) -> list[ExperimentBase]:
    experiments: list[ExperimentBase] = []
    style_map = {
        "Maximin LHS": {"color": "#B5B5B5", "linestyle": "--"},
        "Random": {"color": "#4D4D4D", "linestyle": "-"},
        "Bayes": {"color": "#D1D1D1", "linestyle": ":"},
        "EIG Reference": {"color": "#6F7F8F", "linestyle": "-."},
        "Claude Haiku 4.5 with Mechanism Info": {"color": "#2F8F83", "linestyle": "-"},
        "Claude Haiku 4.5 without Mechanism Info": {"color": "#2F8F83", "linestyle": "--"},
        "Claude Opus 4.7 with Mechanism Info": {"color": "#1F5B9A", "linestyle": "-"},
        "Claude Opus 4.7 without Mechanism Info": {"color": "#1F5B9A", "linestyle": "--"},
        "DeepSeek V3.2 with Mechanism Info": {"color": "#4C78A8", "linestyle": "-"},
        "DeepSeek V3.2 without Mechanism Info": {"color": "#4C78A8", "linestyle": "--"},
        "Gemini 3.1 Pro with Mechanism Info": {"color": PAPER_COLORS[0], "linestyle": "-"},
        "Gemini 3.1 Pro without Mechanism Info": {"color": PAPER_COLORS[0], "linestyle": "--"},
        "Gemini 2.5 Pro with Mechanism Info": {"color": PAPER_COLORS[0], "linestyle": "-"},
        "Gemini 2.5 Pro without Mechanism Info": {"color": PAPER_COLORS[0], "linestyle": "--"},
        "GPT-5.4 with Mechanism Info": {"color": "#B24745", "linestyle": "-"},
        "GPT-5.4 without Mechanism Info": {"color": "#B24745", "linestyle": "--"},
        "GPT-4o with Mechanism Info": {"color": PAPER_COLORS[2], "linestyle": "-"},
        "GPT-4o without Mechanism Info": {"color": PAPER_COLORS[2], "linestyle": "--"},
        "Llama 4 Maverick with Mechanism Info": {"color": "#8E6C8A", "linestyle": "-"},
        "Llama 4 Maverick without Mechanism Info": {"color": "#8E6C8A", "linestyle": "--"},
        "Mistral Large 3 with Mechanism Info": {"color": "#A86D1B", "linestyle": "-"},
        "Qwen3-32B with Mechanism Info": {"color": PAPER_COLORS[4], "linestyle": "-"},
        "Qwen3-32B without Mechanism Info": {"color": PAPER_COLORS[4], "linestyle": "--"},
    }

    def discover_legacy_directory_experiments(source_path: str) -> list[DirectoryExperiment]:
        sample_ids = sorted(
            entry
            for entry in os.listdir(source_path)
            if entry.startswith("M") and "_" in entry and os.path.isdir(os.path.join(source_path, entry))
        )
        if not sample_ids:
            return []

        proposer_dirs = sorted(
            {
                proposer_dir
                for sample_id in sample_ids
                for proposer_dir in os.listdir(os.path.join(source_path, sample_id))
                if os.path.isdir(os.path.join(source_path, sample_id, proposer_dir))
            }
        )
        legacy_label_map = {
            "RandomProposer": ("Random", "RandomProposer"),
            "MaximinLHSProposer": ("Maximin LHS", "MaximinLHSProposer"),
            "BayesProposer": ("Bayes", "BayesProposer"),
        }
        legacy_experiments: list[DirectoryExperiment] = []
        for proposer_dir in proposer_dirs:
            if proposer_dir not in legacy_label_map:
                continue
            label, canonical_proposer_name = legacy_label_map[proposer_dir]
            legacy_experiments.append(
                DirectoryExperiment(
                    source_path=source_path,
                    root_dir=source_path,
                    proposer_name=canonical_proposer_name,
                    label=label,
                    color=style_map[label]["color"],
                    linestyle=style_map[label]["linestyle"],
                    sample_ids=sample_ids,
                )
            )
        return legacy_experiments

    for filename in sorted(os.listdir(results_dir)):
        source_path = os.path.join(results_dir, filename)
        is_tar = filename.endswith(".tar")
        is_dir = os.path.isdir(source_path)
        if not is_tar and not is_dir:
            continue

        tar = None
        names: list[str] = []
        root_prefix = ""
        if is_tar:
            tar = tarfile.open(source_path, "r")
            names = tar.getnames()
            run_config_candidates = [n for n in names if n.endswith("run_config.json")]
            if not run_config_candidates:
                tar.close()
                continue
            run_config_member = run_config_candidates[0]
            extracted = tar.extractfile(run_config_member)
            if extracted is None:
                tar.close()
                continue
            run_config = json.load(extracted)
            root_prefix = os.path.dirname(run_config_member)
        else:
            run_config_path = os.path.join(source_path, "run_config.json")
            if not os.path.exists(run_config_path):
                experiments.extend(discover_legacy_directory_experiments(source_path))
                continue
            with open(run_config_path, "r") as f:
                run_config = json.load(f)

        run_args = run_config["RUN_SETTINGS"]["run_args"]
        gt_dir = run_args.get("gt_dir")
        if gt_dir is not None and gt_dir != "ground_truths_200_samples_v2":
            if tar is not None:
                tar.close()
            continue

        if "proposer" not in run_args:
            if tar is not None:
                tar.close()
                continue
            experiments.extend(discover_legacy_directory_experiments(source_path))
            continue

        proposer = run_args["proposer"]
        if gt_dir is None and proposer == "agent":
            if tar is not None:
                tar.close()
            continue
        if proposer == "maximin-lhs":
            label = "Maximin LHS"
            proposer_name = "MaximinLHSProposer"
        elif proposer == "random":
            label = "Random"
            proposer_name = "RandomProposer"
        elif proposer == "bayes":
            label = "Bayes"
            proposer_name = "BayesProposer"
        elif proposer == "map-eig":
            label = "EIG Reference"
            proposer_name = "MapEIGProposer"
        elif proposer == "agent":
            model = run_args["proposer_model"]
            if model not in {
                "gemini-2.5-pro",
                "gemini-3.1-pro",
                "gpt-5.4",
                "gpt-4o",
                "llama-4-maverick",
                "mistral-large-3",
                "qwen3-32b",
                "deepseek-v3.2",
                "claude-4-5-haiku",
                "claude-4.7-opus",
            }:
                if tar is not None:
                    tar.close()
                continue
            variant = run_args.get("agent_prompt_variant", "standard")
            label = _agent_label(model, variant)
            proposer_name = f"AgentProposer_{model}"
        else:
            if tar is not None:
                tar.close()
            continue

        if label not in style_map:
            if tar is not None:
                tar.close()
            continue

        if is_tar:
            prefix_with_slash = root_prefix + "/"
            sample_ids = sorted(
                {
                    name[len(prefix_with_slash):].split("/")[0]
                    for name in names
                    if name.startswith(prefix_with_slash)
                    and name != f"{root_prefix}/run_config.json"
                    and len(name[len(prefix_with_slash):].split("/")) > 1
                    and name[len(prefix_with_slash):].split("/")[0].startswith("M")
                }
            )
            experiments.append(
                TarExperiment(
                    source_path=source_path,
                    root_prefix=root_prefix,
                    proposer_name=proposer_name,
                    label=label,
                    color=style_map[label]["color"],
                    linestyle=style_map[label]["linestyle"],
                    tar=tar,
                    members=set(names),
                    sample_ids=sample_ids,
                )
            )
        else:
            sample_ids = sorted(
                entry
                for entry in os.listdir(source_path)
                if entry.startswith("M") and os.path.isdir(os.path.join(source_path, entry))
            )
            experiments.append(
                DirectoryExperiment(
                    source_path=source_path,
                    root_dir=source_path,
                    proposer_name=proposer_name,
                    label=label,
                    color=style_map[label]["color"],
                    linestyle=style_map[label]["linestyle"],
                    sample_ids=sample_ids,
                )
            )

    return experiments


def _top_valid_items(data: dict[str, Any]):
    vals = [(k, v) for k, v in data.items() if isinstance(v, (int, float))]
    vals.sort(key=lambda kv: kv[1])
    return vals


def _latest_valid_fit_items(exp: ExperimentBase, exp_dir: str, step: int):
    """Return latest valid fitting items up to step, carrying belief forward on all-null fits."""
    for candidate_step in range(step, 0, -1):
        fit = exp.read_json(f"{exp_dir}/step_{candidate_step:03d}_fitting_results.json")
        if fit is None:
            continue
        vals = _top_valid_items(fit)
        if vals:
            return vals, candidate_step
    return [], None


def gather_stepwise_accuracy_entropy(exp: ExperimentBase) -> list[dict[str, Any]]:
    rows = []
    for sample_id in exp.sample_ids:
        gt = sample_id.split("_")[0]
        exp_dir = exp.sample_exp_dir(sample_id)
        for step in range(1, 6):
            vals, fit_step = _latest_valid_fit_items(exp, exp_dir, step)
            if not vals:
                continue
            rank = next((i + 1 for i, (name, _) in enumerate(vals) if name == gt), len(vals) + 1)
            accuracy = 1.0 if rank == 1 else 0.0
            nlls = np.array([v for _, v in vals], dtype=float)
            probs = np.exp(-(nlls - np.min(nlls)))
            probs = probs / probs.sum()
            entropy = float(-np.sum([p * np.log(p) for p in probs if p > 0]) / NORMALIZED_ENTROPY_DENOM)
            rows.append(
                {
                    "method": exp.label,
                    "source_tar": exp.source_tar,
                    "sample_id": sample_id,
                    "fit_step": fit_step,
                    "step": step,
                    "metric": "Accuracy from NLL",
                    "value": accuracy,
                }
            )
            rows.append(
                {
                    "method": exp.label,
                    "source_tar": exp.source_tar,
                    "sample_id": sample_id,
                    "fit_step": fit_step,
                    "step": step,
                    "metric": "Mechanism Entropy",
                    "value": entropy,
                }
            )
    return rows


def gather_occupancy_entropy(exp: ExperimentBase) -> list[dict[str, Any]]:
    rows = []
    for sample_id in exp.sample_ids:
        cumulative = []
        exp_dir = exp.sample_exp_dir(sample_id)
        for step in range(1, 6):
            proposal_blob = exp.read_json(f"{exp_dir}/step_{step:03d}_proposal.json")
            if proposal_blob is None:
                continue
            proposal = _extract_proposal_dict(proposal_blob)
            if proposal is None:
                continue
            cumulative.append(proposal)
            rows.append(
                {
                    "method": exp.label,
                    "source_tar": exp.source_tar,
                    "sample_id": sample_id,
                    "step": step,
                    "metric": "Occupancy Entropy",
                    "value": compute_occupancy_entropy(cumulative),
                }
            )
            rows.append(
                {
                    "method": exp.label,
                    "source_tar": exp.source_tar,
                    "sample_id": sample_id,
                    "step": step,
                    "metric": "S-cat Corner Mass",
                    "value": compute_boundary_mass(cumulative),
                }
            )
    return rows


def compute_step_adaptivity(exp: ExperimentBase, step: int):
    step_bins = []
    step_beliefs = []
    for sample_id in exp.sample_ids:
        exp_dir = exp.sample_exp_dir(sample_id)
        proposal_blob = exp.read_json(f"{exp_dir}/step_{step:03d}_proposal.json")
        if proposal_blob is None:
            continue
        proposal = _extract_proposal_dict(proposal_blob)
        if proposal is None:
            continue
        vals, _ = _latest_valid_fit_items(exp, exp_dir, step - 1)
        if not vals:
            continue
        step_bins.append(proposal_bin(proposal))
        step_beliefs.append(vals[0][0])
    if not step_bins:
        return None
    h_a = entropy_from_counts(Counter(step_bins))
    by_b = defaultdict(list)
    for a_bin, belief in zip(step_bins, step_beliefs):
        by_b[belief].append(a_bin)
    total = len(step_bins)
    h_a_given_b = 0.0
    for _, a_bins in by_b.items():
        h_a_given_b += (len(a_bins) / total) * entropy_from_counts(Counter(a_bins))
    return {"Proposal Entropy": h_a, "Adaptivity": h_a - h_a_given_b, "count": total}


def gather_proposal_entropy_and_adaptivity(exp: ExperimentBase) -> list[dict[str, Any]]:
    rows = []
    step_results = []
    for step in range(2, 6):
        result = compute_step_adaptivity(exp, step)
        if result is None:
            continue
        step_results.append((step, result))
        for metric in ["Proposal Entropy", "Adaptivity"]:
            rows.append({"method": exp.label, "source_tar": exp.source_tar, "step": step, "metric": metric, "value": result[metric]})
    total_count = sum(result["count"] for _, result in step_results)
    if total_count > 0:
        weighted_proposal_entropy = sum(
            result["Proposal Entropy"] * result["count"] for _, result in step_results
        ) / total_count
        weighted_adaptivity = sum(
            result["Adaptivity"] * result["count"] for _, result in step_results
        ) / total_count
        rows.append(
            {
                "method": exp.label,
                "source_tar": exp.source_tar,
                "step": 0,
                "metric": "Aggregate Proposal Entropy",
                "value": weighted_proposal_entropy,
            }
        )
        rows.append(
            {
                "method": exp.label,
                "source_tar": exp.source_tar,
                "step": 0,
                "metric": "Aggregate Adaptivity",
                "value": weighted_adaptivity,
            }
        )
    return rows


def build_metric_dataframe(experiments: list[ExperimentBase]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        rows.extend(gather_stepwise_accuracy_entropy(exp))
        rows.extend(gather_occupancy_entropy(exp))
        rows.extend(gather_proposal_entropy_and_adaptivity(exp))
    return pd.DataFrame(rows)
