"""
KineticOracle: Evaluates proposed kinetic constants against a ground truth.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Sequence

import numpy as np
try:
    import basico
except ImportError:
    basico = None

from .base import Oracle


@dataclass(slots=True)
class _TimecourseSolutionAdapter:
    """
    Lightweight wrapper that mimics the subset of SciPy's OdeResult.
    """
    t: np.ndarray
    y: np.ndarray


class KineticOracle(Oracle):
    """
    An Oracle that simulates a real experiment. It takes proposed experimental
    conditions (initial concentrations) and returns the resulting time-course
    data, which is generated using the secret, ground truth k_vector.
    """

    def __init__(self, ground_truth_path: str, config: Dict):
        """
        Initializes the KineticOracle.

        Args:
            ground_truth_path: Path to the ground truth JSON file.
            config: Configuration dictionary (e.g., KINETIC_ORACLE_DATA_CONFIG).
        """
        if basico is None:
            raise ImportError("basico is required for KineticOracle.")

        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        self.model_name = ground_truth["model_name"]
        self.true_k_vector = np.array(ground_truth["k_vector"])
        self._reaction_parameter_keys = ground_truth["reaction_parameter_keys"]
        self.max_t = ground_truth["config"]["max_t"]
        self.s0_ref = ground_truth["config"].get("s0_ref", 1.0)
        
        # Unpack oracle config
        self.num_points = config["num_points"]
        self.time_sampling_method = config["time_sampling_method"]
        self.noise_level = config["noise_level"] 
        self.inhibitor_conc_factor = config.get("inhibitor_conc_factor", 1.0)
        model_dir = config["model_dir"]
        
        # Find and load the COPASI model
        model_path = os.path.join(model_dir, f"{self.model_name}.cps")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find COPASI model file: {model_path}")
            
        basico.load_model(model_path)
        
        # Get species info directly from basico, making it the source of truth.
        all_species_df = basico.get_species()
        self.species_list = sorted(list(all_species_df.index.tolist()))
        self.species_map = {name: idx for idx, name in enumerate(self.species_list)}
        
        # COPASI integration tuning
        copasi_config = config["copasi_config"]
        self.copasi_dense_num_steps: int = copasi_config["dense_num_steps"]
        self.copasi_max_internal_steps: int = copasi_config["max_internal_steps"]
        self.copasi_abs_tol: float = copasi_config["abs_tol"]
        self.copasi_rel_tol: float = copasi_config["rel_tol"]

    def evaluate(self, proposal: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Takes proposed experimental conditions and returns the "real"
        experimental data by simulating them with the ground truth k_vector.

        Args:
            proposal: A dictionary of initial concentrations, e.g.,
                      {'S': 1.0, 'cat': 0.05, 'P': 0.0}.

        Returns:
            A dictionary containing the time-course data of the experiment,
            or None if the simulation fails.
        """
        y0 = proposal.copy()
        
        # Automatically add inhibitor if the model requires it and it's not in the proposal
        if "inhibitor" in self.species_map and "inhibitor" not in y0:
            y0["inhibitor"] = self.s0_ref * self.inhibitor_conc_factor
        
        self._apply_k_vector(self.true_k_vector)

        s0_for_normalization = y0["S"]

        run_data = self._generate_and_process_run(y0)
        
        if run_data is not None:
            run_data["initial_conditions"] = y0
        
        return run_data

    def _generate_and_process_run(self, y0: dict) -> Optional[dict]:
        """
        Executes one sparse run and processes the results.
        """
        dense_solution = self._simulate_dense(y0, self.max_t)
        if dense_solution is None:
            return None

        # Determine effective t_max for sampling (from basico_kinetic_data_generator)
        final_p_val = dense_solution.y[self.species_map["P"], -1]
        if final_p_val < 1e-12:
            effective_t_max = float(self.max_t)
        else:
            p_trace = dense_solution.y[self.species_map["P"]]
            above_idx = np.where(p_trace >= final_p_val * 0.95)[0]
            effective_t_max = (
                dense_solution.t[above_idx[0]] if len(above_idx) > 0 else float(self.max_t)
            )

        # Generate sparse time points for the run
        if self.num_points > 1 and effective_t_max > 0:
            t_start_log = effective_t_max * 1e-6
            if t_start_log == 0.0:
                t_start_log = np.finfo(float).eps
            
            sparse_times = np.logspace(np.log10(t_start_log), np.log10(effective_t_max), self.num_points)
        else:
            sparse_times = np.array([0.0])

        sparse_y = np.vstack(
            [
                np.interp(sparse_times, dense_solution.t, dense_solution.y[row,:])
                for row in range(dense_solution.y.shape[0])
            ]
        )

        # Apply noise to un-normalized data
        if self.noise_level and self.noise_level > 0:
            noise = np.random.normal(0.0, self.noise_level, sparse_y.shape)
            sparse_y = sparse_y + noise
        
        s_idx = self.species_map["S"]
        p_idx = self.species_map["P"]

        return {
            "time_data": sparse_times.tolist(),
            "substrate_data": sparse_y[s_idx].tolist(),
            "product_data": sparse_y[p_idx].tolist(),
            "initial_concentration_of_catalyst": y0["cat"],
        }

    # --- Simulation helpers copied from GroundTruthGenerator ---

    def _apply_initial_conditions(self, y0: Dict[str, float]) -> None:
        for species_name in self.species_list:
            conc = y0.get(species_name, 0.0)
            basico.set_species(species_name, initial_concentration=float(conc))

    def _apply_k_vector(self, k_vector: np.ndarray) -> None:
        for value, param_key in zip(k_vector, self._reaction_parameter_keys):
            basico.set_value(param_key, float(value))

    def _simulate_dense(self, y0: Dict[str, float], max_t: int) -> Optional[_TimecourseSolutionAdapter]:
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

    def _timeseries_to_solution(self, timeseries: Any) -> Optional[_TimecourseSolutionAdapter]:
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
        if species in columns:
            return species
        bracketed_species = f"[{species}]"
        if bracketed_species in columns:
            return bracketed_species
        raise KeyError(f"Species {species} not present in COPASI time-course columns: {list(columns)}")