import hashlib
import json
from typing import Any, Dict, Optional

from bench.analysis.condition_scorer import ConditionScorer
from bench.config import RANDOM_PROPOSER_CONFIG
from bench.proposers.maximin_lhs import generate_maximin_lhs_design

from .base import History, Proposer


class DistanceProposer(Proposer):
    """
    Adaptive baseline that maximizes normalized mean pairwise trajectory
    distance across fitted candidate mechanisms.
    """

    def __init__(
        self,
        model_names: list[str],
        candidate_count: int = 512,
        initial_design_points: int = 5,
        candidate_num_trials: int = 16,
        initial_num_trials: int = 128,
        parallel_fitting: bool = True,
    ):
        self.config = RANDOM_PROPOSER_CONFIG
        self.model_names = model_names
        self.candidate_count = candidate_count
        self.initial_design_points = initial_design_points
        self.candidate_num_trials = candidate_num_trials
        self.initial_num_trials = initial_num_trials
        self.parallel_fitting = parallel_fitting
        self.scorer = ConditionScorer(
            model_names=self.model_names,
            parallel_fitting=self.parallel_fitting,
        )
        self._previous_fit_results: Dict[str, Dict[str, Any]] = {}

    def propose(
        self,
        history: History,
        proposer_name: str,
        ground_truth_name: str,
        step: int,
        num_steps: int,
        feedback_mode: str = "nll",
        cumulative_data: Optional[Dict[str, Any]] = None,
        previous_representative_fits: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if step == 1:
            proposal = self._initial_design_point(ground_truth_name)
            return {
                "proposal": proposal,
                "raw_response": (
                    "DistanceProposer selected the first fixed maximin-LHS design point.\n"
                    f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
                ),
            }

        if not cumulative_data and not previous_representative_fits:
            return {
                "proposal": None,
                "raw_response": "DistanceProposer requires cumulative_data or previous_representative_fits from step 2 onward.",
            }

        if previous_representative_fits:
            representative_fits = previous_representative_fits
        else:
            representative_fits = self.scorer.compute_representative_fits(
                cumulative_data=cumulative_data,
                previous_results=self._previous_fit_results,
            )
        self._previous_fit_results = {
            model_name: fit_data
            for model_name, fit_data in representative_fits.items()
            if "best_params" in fit_data
        }

        candidate_points = generate_maximin_lhs_design(
            num_points=self.candidate_count,
            base_seed=self._seed_for_step(ground_truth_name, step),
            config=self.config,
            num_trials=self.candidate_num_trials,
        )

        best_proposal = None
        best_score = None
        for candidate in candidate_points:
            score_data = self.scorer.score_condition(representative_fits, candidate)
            current_score = score_data["mean_pairwise_distance"]
            if best_score is None or current_score > best_score["mean_pairwise_distance"]:
                best_proposal = candidate
                best_score = score_data

        if best_proposal is None or best_score is None:
            return {
                "proposal": None,
                "raw_response": "DistanceProposer failed to find a valid candidate condition.",
            }

        return {
            "proposal": best_proposal,
            "raw_response": (
                "DistanceProposer selected the candidate with the highest normalized mean pairwise trajectory distance.\n"
                f"Best score: {best_score['mean_pairwise_distance']:.6f}\n"
                f"<suggestion>\n{json.dumps(best_proposal, indent=2)}\n</suggestion>"
            ),
            "representative_fits": representative_fits,
            "condition_score": best_score,
        }

    def _initial_design_point(self, ground_truth_name: str) -> Dict[str, float]:
        design = generate_maximin_lhs_design(
            num_points=self.initial_design_points,
            base_seed=self._seed_for_step(ground_truth_name, 1),
            config=self.config,
            num_trials=self.initial_num_trials,
        )
        return design[0]

    def _seed_for_step(self, ground_truth_name: str, step: int) -> int:
        model_hash = int(hashlib.md5(ground_truth_name.encode()).hexdigest(), 16)
        return (int(self.config["seed"]) + model_hash + step) & 0xFFFFFFFF


class MapEIGProposer(DistanceProposer):
    """
    Adaptive baseline that maximizes a fixed-parameter MAP approximation to
    expected information gain over the candidate mechanisms.
    """

    def __init__(
        self,
        model_names: list[str],
        candidate_count: int = 100,
        initial_design_points: int = 5,
        candidate_num_trials: int = 16,
        initial_num_trials: int = 128,
        parallel_fitting: bool = True,
        likelihood_sigma: float = 0.05,
    ):
        super().__init__(
            model_names=model_names,
            candidate_count=candidate_count,
            initial_design_points=initial_design_points,
            candidate_num_trials=candidate_num_trials,
            initial_num_trials=initial_num_trials,
            parallel_fitting=parallel_fitting,
        )
        self.likelihood_sigma = likelihood_sigma

    def propose(
        self,
        history: History,
        proposer_name: str,
        ground_truth_name: str,
        step: int,
        num_steps: int,
        feedback_mode: str = "nll",
        cumulative_data: Optional[Dict[str, Any]] = None,
        previous_representative_fits: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if step == 1:
            proposal = self._initial_design_point(ground_truth_name)
            return {
                "proposal": proposal,
                "raw_response": (
                    "MapEIGProposer selected the first fixed maximin-LHS design point.\n"
                    f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
                ),
            }

        if not cumulative_data and not previous_representative_fits:
            return {
                "proposal": None,
                "raw_response": "MapEIGProposer requires cumulative_data or previous_representative_fits from step 2 onward.",
            }

        if previous_representative_fits:
            representative_fits = previous_representative_fits
        else:
            representative_fits = self.scorer.compute_representative_fits(
                cumulative_data=cumulative_data,
                previous_results=self._previous_fit_results,
            )
        self._previous_fit_results = {
            model_name: fit_data
            for model_name, fit_data in representative_fits.items()
            if "best_params" in fit_data
        }

        candidate_points = generate_maximin_lhs_design(
            num_points=self.candidate_count,
            base_seed=self._seed_for_step(ground_truth_name, step),
            config=self.config,
            num_trials=self.candidate_num_trials,
        )

        best_proposal = None
        best_score = None
        for candidate in candidate_points:
            score_data = self.scorer.score_map_eig_condition(
                representative_fits=representative_fits,
                proposal=candidate,
                sigma=self.likelihood_sigma,
            )
            current_score = score_data["map_eig"]
            if best_score is None or current_score > best_score["map_eig"]:
                best_proposal = candidate
                best_score = score_data

        if best_proposal is None or best_score is None:
            return {
                "proposal": None,
                "raw_response": "MapEIGProposer failed to find a valid candidate condition.",
            }

        return {
            "proposal": best_proposal,
            "raw_response": (
                "MapEIGProposer selected the candidate with the highest fixed-parameter MAP-EIG score.\n"
                f"Best MAP-EIG: {best_score['map_eig']:.6f}\n"
                f"Prior entropy: {best_score['prior_entropy']:.6f}\n"
                f"Expected posterior entropy: {best_score['expected_posterior_entropy']:.6f}\n"
                f"<suggestion>\n{json.dumps(best_proposal, indent=2)}\n</suggestion>"
            ),
            "representative_fits": representative_fits,
            "condition_score": best_score,
        }
