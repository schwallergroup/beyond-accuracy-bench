import json
from typing import Any, Optional

from .base import Proposer, History


HEURISTIC_SEQUENCE: list[dict[str, float]] = [
    {"S": 1.0, "P": 0.0, "cat": 0.01},
    {"S": 2.0, "P": 0.0, "cat": 0.01},
    {"S": 1.0, "P": 0.0, "cat": 0.05},
    {"S": 1.0, "P": 0.2, "cat": 0.01},
    {"S": 0.5, "P": 0.2, "cat": 0.05},
]


class SimpleHeuristicProposer(Proposer):
    """
    Fixed non-adaptive proposer that follows the representative step-wise heuristic
    observed from GPT-4o without mechanism information.
    """

    def __init__(self, sequence: Optional[list[dict[str, float]]] = None):
        self.sequence = sequence or HEURISTIC_SEQUENCE

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
        if step < 1 or step > len(self.sequence):
            return {
                "proposal": None,
                "raw_response": (
                    f"SimpleHeuristicProposer supports steps 1..{len(self.sequence)}, "
                    f"but got step={step}."
                ),
            }

        proposal = self.sequence[step - 1]
        raw_response = (
            "SimpleHeuristicProposer selected the fixed heuristic proposal.\n"
            f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
        )
        return {
            "proposal": proposal,
            "raw_response": raw_response,
        }
