"""
GroundTruthGenerator: Generates and saves a kinetic model's ground truth (k_vector).

This module is responsible for finding a valid k_vector for a given model that 
satisfies certain conditions and then saving this k_vector along with
minimal necessary metadata to a file (e.g., JSON).

This component serves as the setup phase for a benchmark, creating the
'ground truth' that the KineticOracle will later use to simulate experiments.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
from scipy.stats import qmc

try:
    import basico
except ImportError:
    basico = None

from .condition_checker import check_k_vector_conditions


class BasicoUnavailableError(ImportError):
    """Raised when ``basico`` (and therefore COPASI) cannot be imported."""


@dataclass(slots=True)
class _TimecourseSolutionAdapter:
    """
    Lightweight wrapper that mimics the subset of SciPy's ``OdeResult`` the
    condition checker expects.  ``sol.y`` is shaped ``(num_species, num_points)``
    and ``sol.t`` is a 1-D time vector.
    """

    t: np.ndarray
    y: np.ndarray


class GroundTruthGenerator:
    """
    Generates and saves the ground truth kinetic constants (k_vector) for a given
    kinetic model, including necessary metadata.
    """

    def __init__(self, model_name: str, config: Dict):
        if basico is None:
            raise BasicoUnavailableError(
                "basico is required for GroundTruthGenerator. "
                "Install it with `pip install basico` and ensure COPASI is "
                "available on your system."
            )

        self.model_name = model_name
        
        # Unpack configuration from the dictionary
        self.s0_ref = config["s0_ref"]
        self.max_t = config["max_t"]
        self.k_sampling_params = config["k_sampling_params"]
        self.inhibitor_conc_factor = config["inhibitor_conc_factor"]
        self.use_condition_checker = config["use_condition_checker"]
        self.make_distinguishable = config["make_distinguishable"]
        self.model_dir = config["model_dir"]
        self.output_dir = config["output_dir"]
        self.num_ground_truths = int(config.get("num_ground_truths", 1))
        self.validation_sampling = config["validation_sampling"]

        # Load the COPASI model from the .cps file
        model_path = os.path.join(self.model_dir, f"{self.model_name}.cps")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"COPASI model file not found: {model_path}")
        basico.load_model(model_path)

        # Get species and reaction parameter info directly from basico
        all_species_df = basico.get_species()
        self.species_list = sorted(list(all_species_df.index.tolist()))
        self.species_map = {name: idx for idx, name in enumerate(self.species_list)}

        reactions_df = basico.get_reactions()
        self._reaction_ids = reactions_df.index.tolist()
        self._reaction_parameter_keys: List[str] = []
        for reaction_id in self._reaction_ids:
            param_df = basico.get_reaction_parameters(reaction_id)
            if not param_df.empty:
                param_key = param_df.index[0]
                self._reaction_parameter_keys.append(param_key)
            else:
                print(f"Warning: No kinetic parameter found for reaction {reaction_id}. ")

        # COPASI integration tuning (helps avoid LSODA MXSTEP failures)
        copasi_config = config["copasi_config"]
        self.copasi_dense_num_steps: int = copasi_config["dense_num_steps"]
        self.copasi_max_internal_steps: int = copasi_config["max_internal_steps"]
        self.copasi_abs_tol: float = copasi_config["abs_tol"]
        self.copasi_rel_tol: float = copasi_config["rel_tol"]
        
        # Will store the found k_vector
        self.k_vector: Optional[np.ndarray] = None
        self.last_validation_summary: Optional[Dict[str, float | int | bool]] = None


    def find_valid_parameters(
        self,
        initial_conditions_list: Optional[Sequence[Dict[str, float]]] = None,
    ) -> np.ndarray:
        """
        Draws kinetic constants until a set satisfies the condition checker.
        """
        if initial_conditions_list is None:
            initial_conditions_list = self._sample_validation_conditions()

        attempts = 0
        best_passed_conditions = 0

        while True:
            attempts += 1
            k_vector = self._sample_k_vector(self.k_sampling_params)
            
            if self.make_distinguishable:
                log10_k_vector = np.log10(k_vector)
                if not (np.max(log10_k_vector) - np.min(log10_k_vector) > 3):
                    continue

            self._apply_k_vector(k_vector)

            passed_conditions = 0
            required_passes = max(
                1,
                int(np.ceil(len(initial_conditions_list) * self.validation_sampling["min_pass_fraction"])),
            )

            for y0 in initial_conditions_list:
                sol = self._simulate_dense(y0, max_t=self.max_t)
                if sol is None:
                    continue

                effective_t_max = self._determine_effective_t_max(sol)
                sol_effective = self._truncate_solution_at_time(sol, effective_t_max)

                if "P" not in self.species_map:
                    print("Warning: Species 'P' not found in model. Cannot check conversion.")
                    break
                
                final_p = sol_effective.y[self.species_map["P"], -1]
                s0 = y0.get("S", 0.0)
                p0 = y0.get("P", 0.0)
                total_sp = s0 + p0
                conversion_ok = total_sp > 0 and 0.5 * total_sp <= final_p <= 1.0 * total_sp
                if not conversion_ok:
                    continue

                if self.use_condition_checker:
                    conditions_ok = check_k_vector_conditions(
                        sol_check=sol_effective,
                        model_name=self.model_name,
                        species_map=self.species_map,
                    )
                    if not conditions_ok:
                        continue

                passed_conditions += 1
                if passed_conditions >= required_passes:
                    self.k_vector = k_vector
                    self.last_validation_summary = {
                        "attempts_until_success": attempts,
                        "num_validation_conditions": len(initial_conditions_list),
                        "required_passes": required_passes,
                        "passed_conditions": passed_conditions,
                        "passed_fraction": passed_conditions / len(initial_conditions_list),
                        "best_passed_conditions_before_success": best_passed_conditions,
                        "used_condition_checker": self.use_condition_checker,
                    }
                    return k_vector

            best_passed_conditions = max(best_passed_conditions, passed_conditions)

            if not self.use_condition_checker and passed_conditions >= required_passes:
                self.k_vector = k_vector
                self.last_validation_summary = {
                    "attempts_until_success": attempts,
                    "num_validation_conditions": len(initial_conditions_list),
                    "required_passes": required_passes,
                    "passed_conditions": passed_conditions,
                    "passed_fraction": passed_conditions / len(initial_conditions_list),
                    "best_passed_conditions_before_success": best_passed_conditions,
                    "used_condition_checker": self.use_condition_checker,
                }
                return k_vector
    
    def generate_and_save_ground_truths(self) -> List[Dict[str, float | int | bool]]:
        """
        Generates and saves multiple ground truth kinetic constants (k_vectors) for
        the given model.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        run_summaries: List[Dict[str, float | int | bool]] = []
        for i in range(self.num_ground_truths):
            self.k_vector = None  # Ensure a new k_vector is found
            self.last_validation_summary = None
            self.find_valid_parameters()

            if self.k_vector is None:
                raise RuntimeError(f"Failed to find a valid k_vector for {self.model_name} (run {i+1}).")

            data_to_save = {
                "model_name": self.model_name,
                "k_vector": self.k_vector.tolist(),
                "species_list": self.species_list,
                "species_map": self.species_map,
                "reaction_parameter_keys": self._reaction_parameter_keys,
                "config": {
                    "s0_ref": self.s0_ref,
                    "max_t": self.max_t,
                }
            }
            if self.last_validation_summary is not None:
                data_to_save["validation_summary"] = self.last_validation_summary
                run_summaries.append(dict(self.last_validation_summary))
            
            output_filename = f"{self.model_name}_ground_truth_{i}.json"
            output_filepath = os.path.join(self.output_dir, output_filename)
            
            with open(output_filepath, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Ground truth #{i+1} for model {self.model_name} saved to {output_filepath}")

        return run_summaries


    # ------------------------------------------------------------------ #
    # COPASI integration helpers
    # ------------------------------------------------------------------ #
    def _apply_initial_conditions(self, y0: Dict[str, float]) -> None:
        """
        Pushes initial concentrations into the COPASI model.
        """
        for species_name in self.species_list:
            conc = y0.get(species_name, 0.0)
            basico.set_species(species_name, initial_concentration=float(conc))

    def _apply_k_vector(self, k_vector: np.ndarray) -> None:
        """
        Writes sampled kinetic constants into the COPASI reactions.
        """
        for value, param_key in zip(
            k_vector, self._reaction_parameter_keys
        ):
            basico.set_value(param_key, float(value))

    def _simulate_dense(self, y0: Dict[str, float], max_t: int) -> Optional[_TimecourseSolutionAdapter]:
        """
        Runs a dense COPASI time course for condition checking.
        """
        try:
            self._apply_initial_conditions(y0)
            ts = basico.run_time_course(
                duration=max_t,
                num_steps=self.copasi_dense_num_steps,
                output_event=True,
                max_steps=self.copasi_max_internal_steps,
                a_tol=self.copasi_abs_tol,
                r_tol=self.copasi_rel_tol,
            )
        except Exception:
            return None

        return self._timeseries_to_solution(ts)

    def _determine_effective_t_max(self, sol: _TimecourseSolutionAdapter) -> float:
        """
        Use the same effective time horizon as the oracle: the first time when
        product reaches 95% of its final simulated value.
        """
        final_p_val = sol.y[self.species_map["P"], -1]
        if final_p_val < 1e-12:
            return float(self.max_t)

        p_trace = sol.y[self.species_map["P"]]
        above_idx = np.where(p_trace >= final_p_val * 0.95)[0]
        return float(sol.t[above_idx[0]]) if len(above_idx) > 0 else float(self.max_t)

    def _truncate_solution_at_time(
        self, sol: _TimecourseSolutionAdapter, t_max_effective: float
    ) -> _TimecourseSolutionAdapter:
        """
        Restrict validation to the oracle-style effective reaction window rather
        than the fixed simulation cutoff.
        """
        keep_mask = sol.t <= t_max_effective
        if not np.any(keep_mask):
            keep_mask = np.zeros_like(sol.t, dtype=bool)
            keep_mask[0] = True
        return _TimecourseSolutionAdapter(t=sol.t[keep_mask], y=sol.y[:, keep_mask])

    def _timeseries_to_solution(self, timeseries) -> Optional[_TimecourseSolutionAdapter]:
        """
        Converts a basico time-course DataFrame into the SciPy-like adapter.
        """
        if timeseries is None:
            return None

        if "Time" in timeseries.columns:
            time_vector = timeseries["Time"].to_numpy(dtype=float)
            species_frame = timeseries.drop(columns=["Time"])
        else:
            time_vector = timeseries.index.to_numpy(dtype=float)
            species_frame = timeseries

        species_data = []
        for species in self.species_list:
            col_name = self._resolve_species_column(species_frame.columns, species)
            species_data.append(species_frame[col_name].to_numpy(dtype=float))

        return _TimecourseSolutionAdapter(
            t=time_vector,
            y=np.vstack(species_data),
        )

    def _resolve_species_column(self, columns: Sequence[str], species: str) -> str:
        """
        Maps species names to DataFrame columns produced by COPASI.
        """
        direct_match = species in columns
        if direct_match:
            return species

        raise KeyError(f"Species {species} not present in COPASI time-course columns: {list(columns)}")

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def _sample_k_vector(self, params: Optional[dict]) -> np.ndarray:
        """
        Samples kinetic constants using a log-uniform distribution.
        """
        num_ks = len(self._reaction_parameter_keys)
        params = params or {}

        log_min = params["log_min"]
        log_max = params["log_max"]
        log_k = np.random.uniform(log_min, log_max, size=num_ks)
        return 10 ** log_k

    def _sample_log_uniform_from_unit(self, u: float, min_val: float, max_val: float, epsilon: float = 1e-9) -> float:
        """
        Maps a unit-interval sample to a log-uniform value, handling zero lower bounds
        via epsilon shifting in the same way as RandomProposer.
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
            return float(10 ** (log_min + u * (log_max - log_min)))

        log_min = np.log10(epsilon)
        log_max = np.log10(max_val + epsilon)
        val = 10 ** (log_min + u * (log_max - log_min))
        return float(max(0.0, val - epsilon))

    def _sample_validation_conditions(self) -> List[Dict[str, float]]:
        """
        Samples validation conditions from the same action space used by the random proposer.
        """
        config = self.validation_sampling
        num_conditions = int(config["num_conditions"])
        epsilon = float(config["product_sampling_epsilon"])
        s_min, s_max = config["substrate_conc_range"]
        p_min, p_max = config["product_conc_range"]
        cat_min, cat_max = config["catalyst_conc_range"]

        sampler = qmc.LatinHypercube(d=3)
        unit_samples = sampler.random(n=num_conditions)

        conditions: List[Dict[str, float]] = []
        for u_s, u_p, u_cat in unit_samples:
            y0: Dict[str, float] = {
                "S": self._sample_log_uniform_from_unit(u_s, s_min, s_max),
                "cat": self._sample_log_uniform_from_unit(u_cat, cat_min, cat_max),
            }
            if "P" in self.species_map:
                y0["P"] = self._sample_log_uniform_from_unit(u_p, p_min, p_max, epsilon=epsilon)
            if "inhibitor" in self.species_map:
                y0["inhibitor"] = self.s0_ref * self.inhibitor_conc_factor
            conditions.append(y0)

        return conditions
