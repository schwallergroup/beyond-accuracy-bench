from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

# Type alias for conversation history
# Each entry is a tuple of (proposal, result)
History = List[Dict[str, str]]

class Proposer(ABC):
    """
    Abstract base class for all proposers.
    Defines the common interface for suggesting the next experiment.
    """
    @abstractmethod
    def propose(
        self,
        history: History,
        proposer_name: str,
        ground_truth_name: str,
        step: int,
        num_steps: int,
        feedback_mode: str = 'nll',
        cumulative_data: Optional[Dict[str, Any]] = None,
        previous_representative_fits: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Propose the next experiment based on the history of previous experiments.

        Args:
            history: A list of dictionaries representing the conversation history.
            proposer_name: The name of the proposer (for logging/metadata).
            ground_truth_name: The name of the ground truth (for logging/metadata).
            step: The current experiment step number.
            num_steps: The total number of steps in the experiment.
            feedback_mode: The type of feedback provided to the agent ('nll', 'raw_data', 'both').
            cumulative_data: The cumulative experimental data available before the
                current proposal step. Adaptive proposers may use this to refit
                surrogate models or score candidate conditions.
            previous_representative_fits: The latest completed fitting results,
                normalized for condition scorers. Adaptive proposers can use this
                to avoid refitting the same pre-step cumulative data.

        Returns:
            The next proposal for an experiment.
        """
        pass
