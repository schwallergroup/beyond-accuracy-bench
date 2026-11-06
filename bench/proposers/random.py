import numpy as np
from typing import Any, Optional
from .base import Proposer, History
from bench.config import RANDOM_PROPOSER_CONFIG
import json # Import json for formatting raw_response

class RandomProposer(Proposer):
    """
    A proposer that suggests a random experiment by generating initial concentrations
    for substrate, product, and catalyst from a log-uniform distribution.
    This proposer ignores the history.
    """
    ASSIGNMENT_MULTIPLIER = 1_000_003

    def __init__(self, assignment_id: int = 0):
        """
        Initializes the RandomProposer.
        """
        self.config = RANDOM_PROPOSER_CONFIG
        self.assignment_id = int(assignment_id)
        self.num_points = 5
        self.product_sampling_epsilon = float(
            self.config.get("product_sampling_epsilon", 1e-9)
        )
        self._proposal_cache: dict[str, list[dict[str, float]]] = {}

    def _sample_log_uniform(
        self,
        rng: np.random.Generator,
        min_val: float,
        max_val: float,
        epsilon: float = 1e-9,
    ) -> float:
        """
        Samples from a log-uniform distribution, handling cases where the lower bound is 0.
        """
        if min_val < 0 or max_val < 0:
            raise ValueError("Concentration bounds cannot be negative.")
        if max_val < min_val:
            raise ValueError("max_val must be greater than or equal to min_val.")
        if max_val == 0:
            return 0.0

        if min_val > 0:
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            return float(10 ** rng.uniform(log_min, log_max))
        else:  # min_val is 0
            # Sample from a log-uniform-like distribution on [0, max_val]
            # by sampling on [epsilon, max_val + epsilon] and shifting.
            log_min = np.log10(epsilon)
            log_max = np.log10(max_val + epsilon)
            
            sampled_log = rng.uniform(log_min, log_max)
            val = 10 ** sampled_log
            
            # Subtract epsilon to shift the range and clamp at 0 for precision.
            return max(0, val - epsilon)

    def _seed_for_assignment(self) -> int:
        return int(self.config["seed"]) + self.ASSIGNMENT_MULTIPLIER * self.assignment_id

    def _proposal_sequence(self) -> list[dict[str, float]]:
        cache_key = f"assignment_{self.assignment_id}"
        if cache_key in self._proposal_cache:
            return self._proposal_cache[cache_key]

        seed = self._seed_for_assignment() & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        s_min, s_max = self.config["substrate_conc_range"]
        p_min, p_max = self.config["product_conc_range"]
        cat_min, cat_max = self.config["catalyst_conc_range"]
        sequence: list[dict[str, float]] = []
        for _ in range(self.num_points):
            sequence.append({
                "S": self._sample_log_uniform(rng, s_min, s_max),
                "P": self._sample_log_uniform(
                    rng,
                    p_min,
                    p_max,
                    epsilon=self.product_sampling_epsilon,
                ),
                "cat": self._sample_log_uniform(rng, cat_min, cat_max),
            })
        self._proposal_cache[cache_key] = sequence
        return sequence

    def propose(
        self,
        history: History,
        proposer_name: str,
        ground_truth_name: str,
        step: int,
        num_steps: int,
        feedback_mode: str = 'nll',
        cumulative_data: Optional[dict[str, Any]] = None,
        previous_representative_fits: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Returns a cached random proposal sequence for the current assignment id.
        The sequence is generated once per assignment and then replayed across
        steps without using feedback or ground-truth identity.
        """
        if step < 1 or step > self.num_points:
            return {
                "proposal": None,
                "raw_response": (
                    f"RandomProposer supports steps 1..{self.num_points}, "
                    f"but got step={step}."
                ),
            }

        proposal = self._proposal_sequence()[step - 1]

        raw_response = (
            f"A fixed random proposal (assignment {self.assignment_id}, step {step}/{self.num_points}) has been selected.\n"
            f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
        )

        return {
            "proposal": proposal,
            "raw_response": raw_response
        }
