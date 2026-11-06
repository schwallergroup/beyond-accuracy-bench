import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp

from bench.config import GROUND_TRUTH_GENERATOR_CONFIG, KINETIC_ORACLE_DATA_CONFIG
from bench.fitters.cmaes_fitter import run_cmaes_for_multiple_models
from bench.fitters.model_definition import MODEL_INFO


class ConditionScorer:
    """
    Computes a trajectory-separability score for a proposed experiment
    condition by:
    1. fitting each candidate mechanism to the current cumulative data,
    2. forward simulating each fitted mechanism under the proposed condition,
    3. computing normalized mean pairwise L2 distances across trajectories.
    """

    def __init__(
        self,
        model_names: List[str],
        oracle_config: Optional[Dict[str, Any]] = None,
        parallel_fitting: bool = True,
        epsilon: float = 1e-9,
    ):
        self.model_names = model_names
        self.parallel_fitting = parallel_fitting
        self.epsilon = epsilon
        self.oracle_config = copy.deepcopy(oracle_config or KINETIC_ORACLE_DATA_CONFIG)
        self.max_t = float(GROUND_TRUTH_GENERATOR_CONFIG["max_t"])
        self.num_points = int(self.oracle_config["num_points"])
        self.inhibitor_conc_factor = float(
            self.oracle_config.get(
                "inhibitor_conc_factor",
                GROUND_TRUTH_GENERATOR_CONFIG.get("inhibitor_conc_factor", 1.0),
            )
        )
        self.s0_ref = float(GROUND_TRUTH_GENERATOR_CONFIG.get("s0_ref", 1.0))
        self.dense_num_steps = int(
            self.oracle_config.get("copasi_config", {}).get("dense_num_steps", 5000)
        )

    def compute_representative_fits(
        self,
        cumulative_data: Dict[str, Any],
        previous_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        fitting_results = run_cmaes_for_multiple_models(
            model_names=self.model_names,
            data_sample=cumulative_data,
            previous_results=previous_results,
            parallel=self.parallel_fitting,
        )

        representative_fits: Dict[str, Dict[str, Any]] = {}
        for model_name, result in fitting_results:
            if not isinstance(result, dict) or "error" in result:
                representative_fits[model_name] = {"error": result.get("error") if isinstance(result, dict) else "Unknown fitting error"}
                continue

            representative_fits[model_name] = {
                "neg_log_likelihood": result.get("neg_log_likelihood"),
                "param_names": result.get("param_names", []),
                "best_params": result.get("best_params", []),
                "best_k_values": result.get("best_k_values", []),
            }

        return representative_fits

    def simulate_representative_fits(
        self,
        representative_fits: Dict[str, Dict[str, Any]],
        proposal: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        simulated: Dict[str, Dict[str, Any]] = {}
        time_grid = self._build_time_grid()

        for model_name, fit_data in representative_fits.items():
            if "error" in fit_data:
                simulated[model_name] = {"error": fit_data["error"]}
                continue

            model_info = MODEL_INFO[model_name]
            best_params = np.array(fit_data.get("best_params", []), dtype=float)
            param_names = fit_data.get("param_names", [])
            k_indices = [i for i, name in enumerate(param_names) if ".k" in name]
            if len(k_indices) == 0:
                simulated[model_name] = {"error": "No fitted kinetic parameters found."}
                continue

            k_values = tuple((10 ** best_params[k_indices]).tolist())
            y0 = self._build_initial_conditions_vector(model_info["species_order"], proposal)
            try:
                sol = solve_ivp(
                    fun=model_info["ode_func"],
                    t_span=(time_grid[0], time_grid[-1]),
                    y0=y0,
                    t_eval=time_grid,
                    args=k_values,
                    method="LSODA",
                    rtol=1e-4,
                    atol=1e-8,
                )
            except Exception as exc:
                simulated[model_name] = {"error": f"Simulation failed: {exc}"}
                continue

            if not sol.success:
                simulated[model_name] = {"error": "Simulation did not converge."}
                continue

            species_order = model_info["species_order"]
            try:
                s_idx = species_order.index("S")
                p_idx = species_order.index("P")
            except ValueError as exc:
                simulated[model_name] = {"error": f"Missing required species in model: {exc}"}
                continue

            simulated[model_name] = {
                "time_data": sol.t.tolist(),
                "substrate_data": sol.y[s_idx].tolist(),
                "product_data": sol.y[p_idx].tolist(),
            }

        return simulated

    def score_condition(
        self,
        representative_fits: Dict[str, Dict[str, Any]],
        proposal: Dict[str, float],
    ) -> Dict[str, Any]:
        simulated = self.simulate_representative_fits(representative_fits, proposal)
        valid_models = {
            model_name: run_data
            for model_name, run_data in simulated.items()
            if "error" not in run_data
        }

        if len(valid_models) < 2:
            return {
                "proposal": proposal,
                "mean_pairwise_distance": 0.0,
                "min_pairwise_distance": 0.0,
                "num_valid_models": len(valid_models),
                "pairwise_distances": [],
                "normalization": {"substrate": "S/S0", "product": "P/(S0+P0)"},
                "simulations": simulated,
            }

        pairwise_distances: List[Dict[str, Any]] = []
        model_names = sorted(valid_models.keys(), key=lambda x: int(x[1:]))
        for idx, model_a in enumerate(model_names):
            for model_b in model_names[idx + 1 :]:
                distance = self._compute_normalized_pairwise_distance(
                    valid_models[model_a],
                    valid_models[model_b],
                    proposal,
                )
                pairwise_distances.append(
                    {"model_a": model_a, "model_b": model_b, "distance": distance}
                )

        distances = np.array([item["distance"] for item in pairwise_distances], dtype=float)
        return {
            "proposal": proposal,
            "mean_pairwise_distance": float(np.mean(distances)),
            "min_pairwise_distance": float(np.min(distances)),
            "num_valid_models": len(valid_models),
            "pairwise_distances": pairwise_distances,
            "normalization": {"substrate": "S/S0", "product": "P/(S0+P0)"},
            "simulations": simulated,
        }

    def score_map_eig_condition(
        self,
        representative_fits: Dict[str, Dict[str, Any]],
        proposal: Dict[str, float],
        sigma: float = 0.05,
    ) -> Dict[str, Any]:
        dense_simulations = self.simulate_representative_fits_dense(
            representative_fits=representative_fits,
            proposal=proposal,
        )
        valid_models = self._valid_map_eig_models(
            representative_fits=representative_fits,
            dense_simulations=dense_simulations,
        )

        if len(valid_models) < 2:
            return {
                "proposal": proposal,
                "map_eig": 0.0,
                "prior_entropy": 0.0,
                "expected_posterior_entropy": 0.0,
                "num_valid_models": len(valid_models),
                "model_names": valid_models,
                "prior_weights": {},
                "source_entropies": [],
                "log_likelihood_matrix": [],
                "sigma": sigma,
                "time_grid": "oracle_like_source_specific_p95_logspace",
                "simulation_errors": self._simulation_errors(dense_simulations),
            }

        model_names = sorted(valid_models, key=lambda x: int(x[1:]))
        prior_log_probs = self._prior_log_probs_from_fits(
            representative_fits=representative_fits,
            model_names=model_names,
        )
        prior_probs = np.exp(prior_log_probs)
        prior_entropy = self._entropy(prior_probs)

        sigma = max(float(sigma), self.epsilon)
        log_likelihood_matrix = np.zeros((len(model_names), len(model_names)), dtype=float)
        source_entropies: List[Dict[str, Any]] = []

        for source_idx, source_model in enumerate(model_names):
            source_grid = self._oracle_like_sparse_time_grid(dense_simulations[source_model])
            source_s, source_p = self._interpolate_sp(dense_simulations[source_model], source_grid)

            for eval_idx, eval_model in enumerate(model_names):
                eval_s, eval_p = self._interpolate_sp(dense_simulations[eval_model], source_grid)
                sse = float(np.sum((source_s - eval_s) ** 2) + np.sum((source_p - eval_p) ** 2))
                log_likelihood_matrix[source_idx, eval_idx] = -0.5 * sse / (sigma ** 2)

            posterior_log_probs = prior_log_probs + log_likelihood_matrix[source_idx]
            posterior_log_probs = posterior_log_probs - logsumexp(posterior_log_probs)
            posterior_probs = np.exp(posterior_log_probs)
            posterior_entropy = self._entropy(posterior_probs)
            source_entropies.append(
                {
                    "source_model": source_model,
                    "source_prior_weight": float(prior_probs[source_idx]),
                    "posterior_entropy": posterior_entropy,
                    "effective_t_max": float(source_grid[-1]),
                }
            )

        posterior_entropies = np.array(
            [item["posterior_entropy"] for item in source_entropies],
            dtype=float,
        )
        expected_posterior_entropy = float(np.sum(prior_probs * posterior_entropies))
        map_eig = float(prior_entropy - expected_posterior_entropy)

        return {
            "proposal": proposal,
            "map_eig": map_eig,
            "prior_entropy": prior_entropy,
            "expected_posterior_entropy": expected_posterior_entropy,
            "num_valid_models": len(model_names),
            "model_names": model_names,
            "prior_weights": {
                model_name: float(prior_probs[idx])
                for idx, model_name in enumerate(model_names)
            },
            "source_entropies": source_entropies,
            "log_likelihood_matrix": log_likelihood_matrix.tolist(),
            "sigma": sigma,
            "time_grid": "oracle_like_source_specific_p95_logspace",
            "simulation_errors": self._simulation_errors(dense_simulations),
        }

    def simulate_representative_fits_dense(
        self,
        representative_fits: Dict[str, Dict[str, Any]],
        proposal: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        simulated: Dict[str, Dict[str, Any]] = {}
        time_grid = self._build_dense_time_grid()

        for model_name, fit_data in representative_fits.items():
            if "error" in fit_data:
                simulated[model_name] = {"error": fit_data["error"]}
                continue

            model_info = MODEL_INFO[model_name]
            best_params = np.array(fit_data.get("best_params", []), dtype=float)
            param_names = fit_data.get("param_names", [])
            k_indices = [i for i, name in enumerate(param_names) if ".k" in name]
            if len(k_indices) == 0:
                simulated[model_name] = {"error": "No fitted kinetic parameters found."}
                continue

            k_values = tuple((10 ** best_params[k_indices]).tolist())
            y0 = self._build_initial_conditions_vector(model_info["species_order"], proposal)
            try:
                sol = solve_ivp(
                    fun=model_info["ode_func"],
                    t_span=(time_grid[0], time_grid[-1]),
                    y0=y0,
                    t_eval=time_grid,
                    args=k_values,
                    method="LSODA",
                    rtol=1e-4,
                    atol=1e-8,
                )
            except Exception as exc:
                simulated[model_name] = {"error": f"Simulation failed: {exc}"}
                continue

            if not sol.success:
                simulated[model_name] = {"error": "Simulation did not converge."}
                continue

            species_order = model_info["species_order"]
            try:
                s_idx = species_order.index("S")
                p_idx = species_order.index("P")
            except ValueError as exc:
                simulated[model_name] = {"error": f"Missing required species in model: {exc}"}
                continue

            simulated[model_name] = {
                "time_data": sol.t.tolist(),
                "substrate_data": sol.y[s_idx].tolist(),
                "product_data": sol.y[p_idx].tolist(),
            }

        return simulated

    def compute_representative_fits_and_score(
        self,
        cumulative_data: Dict[str, Any],
        proposal: Dict[str, float],
        previous_results: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        representative_fits = self.compute_representative_fits(
            cumulative_data=cumulative_data,
            previous_results=previous_results,
        )
        score = self.score_condition(representative_fits, proposal)
        return representative_fits, score

    def _build_time_grid(self) -> np.ndarray:
        if self.num_points <= 1:
            return np.array([0.0], dtype=float)
        t_start = max(self.max_t * 1e-6, self.epsilon)
        return np.logspace(np.log10(t_start), np.log10(self.max_t), self.num_points)

    def _build_dense_time_grid(self) -> np.ndarray:
        num_steps = max(self.dense_num_steps, 1)
        return np.linspace(0.0, self.max_t, num_steps + 1)

    def _build_initial_conditions_vector(
        self,
        species_order: List[str],
        proposal: Dict[str, float],
    ) -> np.ndarray:
        y0 = np.zeros(len(species_order), dtype=float)
        proposal_with_optional_species = proposal.copy()
        if "inhibitor" in species_order and "inhibitor" not in proposal_with_optional_species:
            proposal_with_optional_species["inhibitor"] = self.s0_ref * self.inhibitor_conc_factor

        for idx, species_name in enumerate(species_order):
            y0[idx] = float(proposal_with_optional_species.get(species_name, 0.0))
        return y0

    def _compute_normalized_pairwise_distance(
        self,
        run_a: Dict[str, Any],
        run_b: Dict[str, Any],
        proposal: Dict[str, float],
    ) -> float:
        s_scale = max(float(proposal["S"]), self.epsilon)
        p_scale = max(float(proposal["S"]) + float(proposal["P"]), self.epsilon)

        s_a = np.array(run_a["substrate_data"], dtype=float) / s_scale
        s_b = np.array(run_b["substrate_data"], dtype=float) / s_scale
        p_a = np.array(run_a["product_data"], dtype=float) / p_scale
        p_b = np.array(run_b["product_data"], dtype=float) / p_scale

        s_l2 = np.sqrt(np.mean((s_a - s_b) ** 2))
        p_l2 = np.sqrt(np.mean((p_a - p_b) ** 2))
        return float(np.sqrt(s_l2 ** 2 + p_l2 ** 2))

    def _valid_map_eig_models(
        self,
        representative_fits: Dict[str, Dict[str, Any]],
        dense_simulations: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        valid_models: List[str] = []
        for model_name, fit_data in representative_fits.items():
            nll = fit_data.get("neg_log_likelihood")
            if not isinstance(nll, (int, float)) or not np.isfinite(float(nll)):
                continue
            if "best_params" not in fit_data:
                continue
            run_data = dense_simulations.get(model_name, {})
            if "error" in run_data:
                continue
            valid_models.append(model_name)
        return valid_models

    def _prior_log_probs_from_fits(
        self,
        representative_fits: Dict[str, Dict[str, Any]],
        model_names: List[str],
    ) -> np.ndarray:
        nlls = np.array(
            [float(representative_fits[model_name]["neg_log_likelihood"]) for model_name in model_names],
            dtype=float,
        )
        log_probs = -nlls
        return log_probs - logsumexp(log_probs)

    def _oracle_like_sparse_time_grid(self, run_data: Dict[str, Any]) -> np.ndarray:
        dense_t = np.array(run_data["time_data"], dtype=float)
        dense_p = np.array(run_data["product_data"], dtype=float)

        final_p_val = dense_p[-1]
        if final_p_val < 1e-12:
            effective_t_max = float(self.max_t)
        else:
            above_idx = np.where(dense_p >= final_p_val * 0.95)[0]
            effective_t_max = float(dense_t[above_idx[0]]) if len(above_idx) > 0 else float(self.max_t)

        if self.num_points <= 1 or effective_t_max <= 0:
            return np.array([0.0], dtype=float)

        t_start = max(effective_t_max * 1e-6, np.finfo(float).eps)
        return np.logspace(np.log10(t_start), np.log10(effective_t_max), self.num_points)

    def _interpolate_sp(
        self,
        run_data: Dict[str, Any],
        time_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dense_t = np.array(run_data["time_data"], dtype=float)
        dense_s = np.array(run_data["substrate_data"], dtype=float)
        dense_p = np.array(run_data["product_data"], dtype=float)
        s_values = np.interp(time_grid, dense_t, dense_s)
        p_values = np.interp(time_grid, dense_t, dense_p)
        return s_values, p_values

    def _entropy(self, probs: np.ndarray) -> float:
        safe_probs = np.clip(probs, self.epsilon, 1.0)
        return float(-np.sum(safe_probs * np.log(safe_probs)))

    def _simulation_errors(
        self,
        simulations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        return {
            model_name: str(run_data["error"])
            for model_name, run_data in simulations.items()
            if "error" in run_data
        }
