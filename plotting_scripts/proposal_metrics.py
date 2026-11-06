import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd


script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from bench.config import RANDOM_PROPOSER_CONFIG
from bench.evaluation import MultiExperimentEvaluator


EPSILON = 1e-9
# Shared discretization for action-space metrics.
# S and cat use equal-width bins in normalized log space.
# P uses a coarse raw-space threshold to distinguish clear product loading
# from weak or absent loading without privileging the exact P=0 choice.
PROPOSAL_BINS = {"S": 3, "cat": 3, "P": 2}
PRODUCT_THRESHOLD = 0.1
BOUNDARY_TAU = 0.15


def _transformed_bounds(name: str, epsilon: float = EPSILON) -> tuple[float, float]:
    if name == "S":
        min_val, max_val = RANDOM_PROPOSER_CONFIG["substrate_conc_range"]
    elif name == "cat":
        min_val, max_val = RANDOM_PROPOSER_CONFIG["catalyst_conc_range"]
    elif name == "P":
        min_val, max_val = RANDOM_PROPOSER_CONFIG["product_conc_range"]
    else:
        raise ValueError(f"Unknown proposal dimension: {name}")

    if min_val > 0:
        return float(np.log10(min_val)), float(np.log10(max_val))
    return float(np.log10(epsilon)), float(np.log10(max_val + epsilon))


def _transform_value(name: str, value: float, epsilon: float = EPSILON) -> float:
    if name in {"S", "cat"}:
        return float(np.log10(max(epsilon, float(value))))
    if name == "P":
        return float(np.log10(max(epsilon, float(value) + epsilon)))
    raise ValueError(f"Unknown proposal dimension: {name}")


def _normalized_coordinate(name: str, value: float, epsilon: float = EPSILON) -> float:
    lower, upper = _transformed_bounds(name, epsilon=epsilon)
    transformed = _transform_value(name, value, epsilon=epsilon)
    if upper <= lower:
        return 0.0
    normalized = (transformed - lower) / (upper - lower)
    return float(np.clip(normalized, 0.0, 1.0))


def _dimension_bin_index(name: str, value: float, epsilon: float = EPSILON) -> int:
    if name == "P":
        return 0 if float(value) < PRODUCT_THRESHOLD else 1

    n_bins = PROPOSAL_BINS[name]
    u = _normalized_coordinate(name, value, epsilon=epsilon)
    return min(int(np.floor(u * n_bins)), n_bins - 1)


def _proposal_bin_index(proposal: dict[str, float], epsilon: float = EPSILON) -> int:
    dims = list(PROPOSAL_BINS.keys())
    bases = [PROPOSAL_BINS[d] for d in dims]
    indices: list[int] = []

    for dim in dims:
        indices.append(_dimension_bin_index(dim, proposal[dim], epsilon=epsilon))

    flat_idx = 0
    for idx, base in zip(indices, bases):
        flat_idx = flat_idx * base + idx
    return flat_idx


def proposal_bin(proposal: dict[str, float], epsilon: float = EPSILON) -> tuple[int, ...]:
    dims = list(PROPOSAL_BINS.keys())
    indices: list[int] = []
    for dim in dims:
        indices.append(_dimension_bin_index(dim, proposal[dim], epsilon=epsilon))
    return tuple(indices)


def compute_occupancy_entropy(
    proposals: Iterable[dict[str, float]],
    epsilon: float = EPSILON,
) -> float:
    proposals = list(proposals)
    if not proposals:
        return np.nan

    total_bins = int(np.prod(list(PROPOSAL_BINS.values())))
    counts = np.zeros(total_bins, dtype=float)
    for proposal in proposals:
        counts[_proposal_bin_index(proposal, epsilon=epsilon)] += 1.0

    probabilities = counts / counts.sum()
    positive = probabilities > 0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    # Normalize by the fixed discretized action space, not by the number of
    # proposals so far. Otherwise early iterations can reach 1.0 simply by
    # occupying all bins currently reachable by the short trajectory.
    if total_bins <= 1:
        return 0.0
    return float(entropy / np.log(total_bins))


def compute_boundary_mass(
    proposals: Iterable[dict[str, float]],
    tau: float = BOUNDARY_TAU,
    epsilon: float = EPSILON,
) -> float:
    proposals = list(proposals)
    if not proposals:
        return np.nan

    boundary_hits = []
    for proposal in proposals:
        is_boundary = True
        for dim in PROPOSAL_BINS:
            if dim == "P":
                continue
            u = _normalized_coordinate(dim, proposal[dim], epsilon=epsilon)
            if min(u, 1.0 - u) > tau:
                is_boundary = False
                break
        boundary_hits.append(float(is_boundary))

    return float(np.mean(boundary_hits))


def build_proposal_metric_dataframe(
    evaluator: MultiExperimentEvaluator,
    tag: str,
    proposer_name: str,
    method_label: str,
    max_step: int = 5,
) -> pd.DataFrame:
    loaded_proposals = evaluator._load_proposals_by_tag(tag)
    proposer_results = loaded_proposals.get(proposer_name, {})
    rows: list[dict[str, object]] = []

    for ground_truth, steps in proposer_results.items():
        cumulative_proposals: list[dict[str, float]] = []
        for step in range(1, max_step + 1):
            proposal_data = steps.get(step)
            if proposal_data is None:
                continue

            proposal = proposal_data.get("proposal")
            if not isinstance(proposal, dict):
                continue

            if not all(key in proposal for key in ("S", "P", "cat")):
                continue

            cumulative_proposals.append(proposal)

            rows.append(
                {
                    "tag": tag,
                    "proposer": proposer_name,
                    "ground_truth": ground_truth,
                    "step": step,
                    "method": method_label,
                    "metric": "Occupancy Entropy",
                    "value": compute_occupancy_entropy(cumulative_proposals),
                }
            )
            rows.append(
                {
                    "tag": tag,
                    "proposer": proposer_name,
                    "ground_truth": ground_truth,
                    "step": step,
                    "method": method_label,
                    "metric": "S-cat Corner Mass",
                    "value": compute_boundary_mass(cumulative_proposals),
                }
            )

    return pd.DataFrame(rows)
