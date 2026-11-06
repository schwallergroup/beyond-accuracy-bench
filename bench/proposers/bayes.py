import ast
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .base import Proposer, History
from bench.config import RANDOM_PROPOSER_CONFIG


class BayesProposer(Proposer):
    """
    Adaptive baseline proposer using Bayesian optimization over experimental
    conditions to minimize entropy derived from model NLLs.

    Entropy definition (nats):
        p_i = softmax(-NLL_i), H = -sum_i p_i log(p_i)
    """

    def __init__(self):
        self.config = RANDOM_PROPOSER_CONFIG
        self.num_initial_lhs = 2
        self.num_candidates = 512
        self.min_history_for_gp = 2
        self.exploration_beta = 1.0
        self.epsilon = float(self.config.get("product_sampling_epsilon", 1e-9))

    def _sample_log_uniform(
        self, min_val: float, max_val: float, rng: np.random.Generator
    ) -> float:
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

        log_min = np.log10(self.epsilon)
        log_max = np.log10(max_val + self.epsilon)
        sampled = float(10 ** rng.uniform(log_min, log_max))
        return max(0.0, sampled - self.epsilon)

    def _sample_candidate(
        self, rng: np.random.Generator
    ) -> Dict[str, float]:
        s_min, s_max = self.config["substrate_conc_range"]
        p_min, p_max = self.config["product_conc_range"]
        cat_min, cat_max = self.config["catalyst_conc_range"]

        return {
            "S": self._sample_log_uniform(s_min, s_max, rng),
            "P": self._sample_log_uniform(p_min, p_max, rng),
            "cat": self._sample_log_uniform(cat_min, cat_max, rng),
        }

    def _scale_unit_to_log_uniform(self, u: float, min_val: float, max_val: float) -> float:
        if min_val > 0:
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            return float(10 ** (log_min + u * (log_max - log_min)))
        # Handles min=0 bounds by epsilon-shifted log-uniform mapping.
        log_min = np.log10(self.epsilon)
        log_max = np.log10(max_val + self.epsilon)
        return max(0.0, float(10 ** (log_min + u * (log_max - log_min)) - self.epsilon))

    def _lhs_initial_proposal(self, ground_truth_name: str, step: int) -> Dict[str, float]:
        model_hash = int(hashlib.md5(ground_truth_name.encode()).hexdigest(), 16)
        lhs_seed = (self.config["seed"] + model_hash) & 0xFFFFFFFF
        sampler = qmc.LatinHypercube(d=3, seed=lhs_seed)
        unit_samples = sampler.random(n=self.num_initial_lhs)
        idx = max(0, min(self.num_initial_lhs - 1, step - 1))
        u_s, u_p, u_cat = unit_samples[idx]

        s_min, s_max = self.config["substrate_conc_range"]
        p_min, p_max = self.config["product_conc_range"]
        cat_min, cat_max = self.config["catalyst_conc_range"]

        return {
            "S": self._scale_unit_to_log_uniform(u_s, s_min, s_max),
            "P": self._scale_unit_to_log_uniform(u_p, p_min, p_max),
            "cat": self._scale_unit_to_log_uniform(u_cat, cat_min, cat_max),
        }

    def _proposal_to_features(self, proposal: Dict[str, float]) -> np.ndarray:
        # Log-transform for better GP behavior across orders of magnitude.
        return np.array(
            [
                np.log10(max(self.epsilon, float(proposal["S"]))),
                np.log10(max(self.epsilon, float(proposal["P"]))),
                np.log10(max(self.epsilon, float(proposal["cat"]))),
            ],
            dtype=float,
        )

    def _extract_proposal_from_message(self, content: str) -> Dict[str, float] | None:
        match = re.search(
            r"initial conditions:\s*(\{.*?\})", content, re.DOTALL
        )
        if not match:
            return None

        try:
            parsed = ast.literal_eval(match.group(1))
            if not isinstance(parsed, dict):
                return None
            if not all(key in parsed for key in ("S", "P", "cat")):
                return None
            return {
                "S": float(parsed["S"]),
                "P": float(parsed["P"]),
                "cat": float(parsed["cat"]),
            }
        except (SyntaxError, ValueError, TypeError):
            return None

    def _extract_nll_dict_from_message(self, content: str) -> Dict[str, float] | None:
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if not match:
            return None

        try:
            raw = json.loads(match.group(1))
            if not isinstance(raw, dict):
                return None

            parsed: Dict[str, float] = {}
            for model_name, value in raw.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    parsed[model_name] = float(value)
                elif isinstance(value, str):
                    try:
                        num = float(value)
                        if np.isfinite(num):
                            parsed[model_name] = num
                    except ValueError:
                        continue
            return parsed if parsed else None
        except json.JSONDecodeError:
            return None

    def _entropy_from_nll(self, nll_values: Dict[str, float]) -> float | None:
        if len(nll_values) < 2:
            return None

        nll_array = np.array(list(nll_values.values()), dtype=float)
        log_probs = -nll_array - logsumexp(-nll_array)
        probs = np.exp(log_probs)
        return float(-np.sum(probs * log_probs))

    def _extract_observations(self, history: History) -> Tuple[np.ndarray, np.ndarray]:
        x_rows: List[np.ndarray] = []
        y_rows: List[float] = []

        for message in history:
            if message.get("role") != "user":
                continue

            content = message.get("content", "")
            proposal = self._extract_proposal_from_message(content)
            nll_dict = self._extract_nll_dict_from_message(content)
            if proposal is None or nll_dict is None:
                continue

            entropy = self._entropy_from_nll(nll_dict)
            if entropy is None or not np.isfinite(entropy):
                continue

            x_rows.append(self._proposal_to_features(proposal))
            y_rows.append(entropy)

        if not x_rows:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

        return np.vstack(x_rows), np.array(y_rows, dtype=float)

    def _random_proposal(
        self, rng: np.random.Generator, reason: str
    ) -> Dict[str, Any]:
        proposal = self._sample_candidate(rng)
        raw_response = (
            f"BayesProposer fallback to random ({reason}).\n"
            f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
        )
        return {"proposal": proposal, "raw_response": raw_response}

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
        model_hash = int(hashlib.md5(ground_truth_name.encode()).hexdigest(), 16)
        seed = (self.config["seed"] + model_hash + step) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        if feedback_mode != "nll":
            return self._random_proposal(rng, "feedback_mode_is_not_nll")

        if step <= self.num_initial_lhs:
            proposal = self._lhs_initial_proposal(ground_truth_name, step)
            raw_response = (
                f"BayesProposer using LHS initialization ({step}/{self.num_initial_lhs}).\n"
                f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
            )
            return {"proposal": proposal, "raw_response": raw_response}

        X_obs, y_obs = self._extract_observations(history)
        if len(y_obs) < self.min_history_for_gp:
            return self._random_proposal(rng, "insufficient_history_for_gp")

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed,
        )

        try:
            gp.fit(X_obs, y_obs)
        except Exception:
            return self._random_proposal(rng, "gp_fit_failed")

        candidates = [self._sample_candidate(rng) for _ in range(self.num_candidates)]
        X_cand = np.vstack([self._proposal_to_features(c) for c in candidates])

        mu, sigma = gp.predict(X_cand, return_std=True)
        acquisition = mu - self.exploration_beta * sigma
        best_idx = int(np.argmin(acquisition))

        proposal = candidates[best_idx]
        raw_response = (
            "BayesProposer selected a proposal via GP-LCB to minimize expected entropy.\n"
            f"Observed entropy mean={float(np.mean(y_obs)):.4f}, best={float(np.min(y_obs)):.4f}.\n"
            f"LCB params: beta={self.exploration_beta}, candidates={self.num_candidates}.\n"
            f"Predicted entropy(mu)={float(mu[best_idx]):.4f}, sigma={float(sigma[best_idx]):.4f}.\n"
            f"<suggestion>\n{json.dumps(proposal, indent=2)}\n</suggestion>"
        )

        return {
            "proposal": proposal,
            "raw_response": raw_response,
        }
