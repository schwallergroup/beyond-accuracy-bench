import json
from typing import List, Dict, Any, Tuple, Optional

from bench.oracle.kinetic_oracle import KineticOracle
from bench.fitters.cmaes_fitter import run_cmaes_for_multiple_models

class ExperimentEnvironment:
    """
    Encapsulates the logic for running one step of the experiment,
    including oracle evaluation, model fitting, and message formatting.
    """
    def __init__(self, oracle: KineticOracle, model_names: List[str], parallel_fitting: bool = True):
        self.oracle = oracle
        self.model_names = model_names
        self.parallel_fitting = parallel_fitting
        self.full_data_sample_for_cmaes: Dict[str, Any] = {}
        self.previous_fitting_results: Dict[str, Any] = {}
        self.last_fitting_results: Optional[List[Tuple[str, Dict[str, Any]]]] = None
        self.excluded_runs: List[str] = []

    def _run_oracle_evaluation(self, step: int, proposal_dict: Dict[str, float]):
        """
        (Private) Runs only the oracle evaluation part of an experiment step.
        """
        run_name = f"run_{step}"
        current_data_for_run = self.oracle.evaluate(proposal_dict)
        if "initial_conditions" not in current_data_for_run:
            current_data_for_run["initial_conditions"] = proposal_dict
        self.full_data_sample_for_cmaes[run_name] = current_data_for_run

    def _run_fitting(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        (Private) Runs the model fitting on the cumulative data, excluding specified runs.
        """
        data_sample_for_fitting = {
            run_name: data
            for run_name, data in self.full_data_sample_for_cmaes.items()
            if run_name not in self.excluded_runs
        }

        if not data_sample_for_fitting:
            return [(model_name, {"error": "No valid data available for fitting."}) for model_name in self.model_names]

        fitting_results = run_cmaes_for_multiple_models(
            model_names=self.model_names,
            data_sample=data_sample_for_fitting,
            previous_results=self.previous_fitting_results,
            parallel=self.parallel_fitting,
        )
        # Update the previous results cache for the next step
        for model_name, result in fitting_results:
            if isinstance(result, dict) and 'error' not in result:
                self.previous_fitting_results[model_name] = result

        self.last_fitting_results = fitting_results
        return fitting_results

    def get_representative_fit_snapshot(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Returns the latest completed fitting results in the compact format used by
        adaptive condition scorers. This snapshot only contains data available
        before the next proposal step.
        """
        if self.last_fitting_results is None:
            return None

        representative_fits: Dict[str, Dict[str, Any]] = {}
        for model_name, result in self.last_fitting_results:
            if not isinstance(result, dict) or "error" in result:
                representative_fits[model_name] = {
                    "error": result.get("error") if isinstance(result, dict) else "Unknown fitting error"
                }
                continue

            representative_fits[model_name] = {
                "neg_log_likelihood": result.get("neg_log_likelihood"),
                "param_names": result.get("param_names", []),
                "best_params": result.get("best_params", []),
                "best_k_values": result.get("best_k_values", []),
            }

        return representative_fits

    def _format_fitting_results_for_llm(self, fitting_results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """
        Formats the CMA-ES fitting results into a human-readable string for the LLM.
        """
        formatted_str = "```json\n{\n"
        results_dict = {model_name: res for model_name, res in fitting_results}
        
        sorted_items = sorted(
            results_dict.items(),
            key=lambda item: item[1].get('neg_log_likelihood', float('inf')) if isinstance(item[1], dict) else float('inf')
        )
        
        for model_name, result_dict in sorted_items:
            formatted_str += f'  "{model_name}": '
            if isinstance(result_dict, dict) and 'error' in result_dict:
                formatted_str += f'"Error: {result_dict["error"]}",\n'
            elif isinstance(result_dict, dict):
                formatted_str += f'{result_dict.get("neg_log_likelihood", "N/A")},\n'
            else:
                formatted_str += f'{result_dict},\n'

        formatted_str = formatted_str.rstrip(',\n') + "\n}\n```"
        return formatted_str

    def _get_nll_message(self, fitting_results: List[Tuple[str, Dict]]) -> str:
        return f"""I am providing you with the fitting results from all experiments conducted so far. The results are given as negative log-likelihood (NLL) values for each of the {len(fitting_results)} candidate models. A lower NLL value indicates a better fit to the data.

{self._format_fitting_results_for_llm(fitting_results)}"""

    def _get_raw_data_message(self, full_data_sample: Dict) -> str:
        truncated_data_sample = {}
        for run_name, run_data in full_data_sample.items():
            truncated_run = {
                "initial_conditions": run_data["initial_conditions"],
                "time_data": [round(t, 2) for t in run_data["time_data"][:15]],
                "product_data": [round(p, 4) for p in run_data["product_data"][:15]],
                "substrate_data": [round(s, 4) for s in run_data["substrate_data"][:15]],
            }
            if len(run_data["time_data"]) > 15:
                truncated_run["time_data"].append("...")
                truncated_run["product_data"].append("...")
                truncated_run["substrate_data"].append("...")
            truncated_data_sample[run_name] = truncated_run
            
        return f"""I am providing you with the raw time-course data from all experiments conducted so far. The data is in a JSON format, where each key (e.g., "run_1") corresponds to a single experimental run.
```json
{json.dumps(truncated_data_sample, indent=2)}
```"""

    def get_user_message(self, step: int, proposal_dict: Dict, feedback_mode: str) -> Tuple[str, List[Tuple[str, Dict]]]:
        """
        Runs a step, formats the message for the agent based on the feedback mode,
        and handles fitting failures by excluding problematic runs in future steps.
        """
        # 1. Run oracle evaluation to get new experimental data
        run_name = f"run_{step}"
        self._run_oracle_evaluation(step, proposal_dict)

        base_message = f"""We have conducted the experiment you proposed with initial conditions: {proposal_dict}.
The kinetic data has been generated."""
        
        # 2. Run fitting. At this point, the current run's data is included.
        fitting_results = self._run_fitting()

        # 3. Check if fitting failed for all models
        all_failed = all(
            res is None or (isinstance(res, dict) and 'error' in res)
            for _, res in fitting_results
        )

        # 4. If all failed, update state for the *next* run.
        special_message = ""
        if all_failed and run_name not in self.excluded_runs:
            self.excluded_runs.append(run_name)
            # Create a special message only for 'nll' and 'both' modes
            if feedback_mode in ['nll', 'both']:
                special_message = (
                    f"\n\nIMPORTANT: The fitting process failed for all candidate mechanisms with the data from the last experiment ({run_name}). "
                    "This suggests the experimental conditions might be problematic for model fitting. "
                    f"This experimental run ({run_name}) will be excluded from all future fitting steps. "
                    "Please propose a new experiment under different conditions."
                )

        # 5. Assemble the final user message based on feedback mode
        user_message = base_message + special_message

        if feedback_mode == 'nll':
            nll_message = self._get_nll_message(fitting_results)
            user_message += f"\n{nll_message}"
        elif feedback_mode == 'raw_data':
            raw_data_message = self._get_raw_data_message(self.full_data_sample_for_cmaes)
            user_message += f"\n{raw_data_message}"
        elif feedback_mode == 'both':
            nll_message = self._get_nll_message(fitting_results)
            raw_data_message = self._get_raw_data_message(self.full_data_sample_for_cmaes)
            user_message += f"\n\n{nll_message}\n\n{raw_data_message}"
        else:
            raise ValueError(f"Unknown feedback mode: {feedback_mode}")

        return user_message, fitting_results
