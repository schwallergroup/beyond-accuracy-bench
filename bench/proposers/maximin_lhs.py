import json
from typing import Any, Optional

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from .base import Proposer, History
from bench.config import RANDOM_PROPOSER_CONFIG


def generate_maximin_lhs_design(
    num_points: int,
    base_seed: int,
    config: dict[str, list[float] | int],
    num_trials: int = 128,
    epsilon: float = 1e-9,
) -> list[dict[str, float]]:
    def scale_unit_to_log_uniform(u: float, min_val: float, max_val: float) -> float:
        if min_val < 0 or max_val < 0:
            raise ValueError("Concentration bounds cannot be negative.")
        if max_val < min_val:
            raise ValueError("max_val must be greater than or equal to min_val.")
        if max_val == 0:
            return 0.0

        if min_val > 0:
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            return float(10 ** (log_min + u * (log_max - log_min)))

        log_min = np.log10(epsilon)
        log_max = np.log10(max_val + epsilon)
        return max(0.0, float(10 ** (log_min + u * (log_max - log_min)) - epsilon))

    def to_log_feature(point: dict[str, float]) -> np.ndarray:
        return np.array(
            [
                np.log10(max(epsilon, float(point["S"]))),
                np.log10(max(epsilon, float(point["P"]))),
                np.log10(max(epsilon, float(point["cat"]))),
            ],
            dtype=float,
        )

    def unit_sample_to_proposal(sample: np.ndarray) -> dict[str, float]:
        s_min, s_max = config["substrate_conc_range"]
        p_min, p_max = config["product_conc_range"]
        cat_min, cat_max = config["catalyst_conc_range"]

        return {
            "S": scale_unit_to_log_uniform(float(sample[0]), s_min, s_max),
            "P": scale_unit_to_log_uniform(float(sample[1]), p_min, p_max),
            "cat": scale_unit_to_log_uniform(float(sample[2]), cat_min, cat_max),
        }

    best_points: list[dict[str, float]] | None = None
    best_min_dist = -np.inf

    for trial in range(num_trials):
        sampler = qmc.LatinHypercube(d=3, seed=base_seed + trial)
        unit_samples = sampler.random(n=num_points)
        proposals = [unit_sample_to_proposal(row) for row in unit_samples]

        feature_matrix = np.vstack([to_log_feature(p) for p in proposals])
        if len(feature_matrix) < 2:
            min_dist = np.inf
        else:
            min_dist = float(np.min(pdist(feature_matrix, metric="euclidean")))

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = proposals

    if best_points is None:
        raise RuntimeError("Failed to generate maximin-LHS design points.")

    return best_points


class MaximinLHSProposer(Proposer):
    """
    Non-adaptive baseline proposer.
    Generates a reproducible maximin-LHS design (default 5 points in 3D)
    for each ground-truth sample and returns one point per step.

    Each sample receives a run-level assignment id, which is used to derive a
    sample-specific fixed non-adaptive design without depending on mechanism
    identity or feedback.
    """

    ASSIGNMENT_MULTIPLIER = 1_000_003

    def __init__(
        self,
        num_points: int = 5,
        num_trials: int = 128,
        epsilon: float | None = None,
        assignment_id: int = 0,
    ):
        self.config = RANDOM_PROPOSER_CONFIG
        self.num_points = num_points
        self.num_trials = num_trials
        self.epsilon = float(
            self.config.get("product_sampling_epsilon", 1e-9)
            if epsilon is None
            else epsilon
        )
        self.seed = int(self.config["seed"])
        self.assignment_id = int(assignment_id)
        self._design_cache: dict[str, list[dict[str, float]]] = {}

    def _seed_for_ground_truth(self, ground_truth_name: str) -> int:
        return self.seed + self.ASSIGNMENT_MULTIPLIER * self.assignment_id

    def _design_for_ground_truth(self, ground_truth_name: str) -> list[dict[str, float]]:
        cache_key = f"assignment_{self.assignment_id}"
        if cache_key not in self._design_cache:
            self._design_cache[cache_key] = generate_maximin_lhs_design(
                num_points=self.num_points,
                base_seed=self._seed_for_ground_truth(ground_truth_name),
                config=self.config,
                num_trials=self.num_trials,
                epsilon=self.epsilon,
            )
        return self._design_cache[cache_key]

    def propose(
        self,
        history: History,
        proposer_name: str,
        ground_truth_name: str,
        step: int,
        num_steps: int,
        feedback_mode: str = "nll",
        cumulative_data: Optional[dict[str, Any]] = None,
        previous_representative_fits: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if step < 1 or step > self.num_points:
            return {
                "proposal": None,
                "raw_response": (
                    f"MaximinLHSProposer supports steps 1..{self.num_points}, "
                    f"but got step={step}."
                ),
            }

        design_points = self._design_for_ground_truth(ground_truth_name)
        proposal = design_points[step - 1]
        raw_response = (
            f"A fixed maximin-LHS proposal (assignment {self.assignment_id}, step {step}/{self.num_points}) has been selected.\n"
            f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
        )

        return {
            "proposal": proposal,
            "raw_response": raw_response,
        }
