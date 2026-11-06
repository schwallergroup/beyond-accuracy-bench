import os
import sys
import numpy as np
import cma
from numba import jit
import inspect
import multiprocessing

from .model_definition import MODEL_INFO
from ..config import PRIOR_CONFIG, CMA_OPTIONS, CMA_OPTIONS_AFTER_FIRST_RUN

# Ensure the project root is in the path to allow imports from src
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from .bayesian_fitter import LogProb


def fit_with_cmaes(
    base_param_names: list[str],
    data_sample: dict,
    ode_func,
    species_order: list[str],
    initial_guess: np.ndarray = None,
) -> dict:
    """
    Fits kinetic data to a model using the CMA-ES algorithm.
    """
    # Set up objective function, parameters, and bounds for log-likelihood
    fixed_sigma = None
    
    try:
        log_prob = LogProb(
            multi_run_data=data_sample, 
            param_names=base_param_names + ["log_sigma"], 
            ode_func=ode_func,
            species_order=species_order,
            prior_config=PRIOR_CONFIG, 
            fixed_sigma=fixed_sigma,
        )
    except Exception as e:
        return {"error": f"Failed to initialize LogProb: {e}"}

    param_names = log_prob.param_names
    
    def objective(theta: np.ndarray) -> float:
        val = log_prob(theta)
        return -val if np.isfinite(val) else np.inf

    bounds = []
    for name in param_names:
        if 'log_(' in name and '.k' in name:
            bounds.append([log_prob.log_k_min, log_prob.log_k_max])
        elif 'log_sigma' in name:
            bounds.append([log_prob.log_sigma_min, log_prob.log_sigma_max])
    cma_bounds = list(zip(*bounds))
    result_key = 'neg_log_likelihood'
    
    model_context = {
        "param_names": log_prob.param_names,
        "n_params": log_prob.n_params,
    }

    # Select CMA options based on initial_guess
    if initial_guess is None:
        selected_cma_options = CMA_OPTIONS.copy()
    else:
        selected_cma_options = CMA_OPTIONS_AFTER_FIRST_RUN.copy()
    
    selected_cma_options['bounds'] = cma_bounds

    # If this is the first run, create an initial guess from the center of the bounds.
    if initial_guess is None:
        lows = np.array(cma_bounds[0])
        highs = np.array(cma_bounds[1])
        initial_guess = (lows + highs) / 2.0

    try:
        # sigma0 is now included in the selected_cma_options directly.
        # We need to extract it to pass as a separate argument to CMAEvolutionStrategy
        # and remove it from inopts if it's there.
        sigma0 = selected_cma_options.pop('sigma0', 1.0) # Default to 1.0 if not specified in config

        es = cma.CMAEvolutionStrategy(initial_guess, sigma0, inopts=selected_cma_options)
        es.optimize(objective)

        # Process and return results
        best_params = es.result.xbest
        
        results = {
            'best_params': best_params.tolist(),
            'param_names': param_names,
            result_key: es.result.fbest,
            'termination': es.result.stop,
            'es_result': es.result,
            'model_context': model_context
        }
        k_indices = [i for i, name in enumerate(param_names) if 'log_(' in name and '.k' in name]
        results['best_k_values'] = (10**np.array(best_params)[k_indices]).tolist()

    except Exception as e:
        results = {"error": f"CMA-ES optimization failed: {e}"}

    return results


def _prepare_model_for_fitting(model_name):
    """
    Helper function to load a model's hardcoded info 
    from model_definition.py.
    """
    try:
        if model_name not in MODEL_INFO:
            return {"error": f"Model {model_name} not found in MODEL_INFO"}

        model_info = MODEL_INFO[model_name]
        
        base_param_names = [f"log_({name})" for name in model_info["param_names"]]
        
        return {
            "base_param_names": base_param_names,
            "ode_func": model_info["ode_func"],
            "species_order": model_info["species_order"],
            "error": None
        }
    except Exception as e:
        import traceback
        tb_message = traceback.format_exc()
        return {"error": f"Failed to prepare model {model_name} for fitting: {e}\n{tb_message}"}


def run_cmaes_for_multiple_models(
    model_names: list[str],
    data_sample: dict,
    previous_results: dict = None,
    parallel: bool = True,
) -> list[tuple[str, dict]]:
    """
    Runs CMA-ES fitting for a specified list of kinetic models.
    Can run in parallel or sequentially.
    """
    if not parallel:
        return run_cmaes_for_multiple_models_sequential(
            model_names=model_names,
            data_sample=data_sample,
            previous_results=previous_results,
        )

    if previous_results is None:
        previous_results = {}
        
    # Prepare all model-specific objects in the main process
    model_preparations = {name: _prepare_model_for_fitting(name) for name in model_names}
    
    pool_args = []
    successful_models = []
    for name in model_names:
        prep = model_preparations[name]
        if prep["error"] is None:
            initial_guess = None
            if name in previous_results and 'best_params' in previous_results[name]:
                initial_guess = np.array(previous_results[name]['best_params'])
            
            pool_args.append((
                prep["base_param_names"],
                data_sample,
                prep["ode_func"],
                prep["species_order"],
                initial_guess,
            ))
            successful_models.append(name)

    if not successful_models:
         return list(model_preparations.items())

    n_cores = min(os.cpu_count() or 1, len(successful_models))

    # Run fitting in parallel
    with multiprocessing.get_context("spawn").Pool(processes=n_cores) as pool:
        async_results = pool.starmap_async(fit_with_cmaes, pool_args)
        try:
            result_dicts = async_results.get(timeout=1800)  # 30 min timeout
        except multiprocessing.TimeoutError:
            print(f"CMA-ES analysis for the sample timed out after 30 minutes.")
            for name in successful_models:
                model_preparations[name] = {"error": "Timeout"}
            return list(model_preparations.items())
        except Exception as e:
            print(f"An error occurred during parallel processing: {e}")
            for name in successful_models:
                model_preparations[name] = {"error": str(e)}
            return list(model_preparations.items())

    # Combine and process results
    final_results = model_preparations
    for i, name in enumerate(successful_models):
        result_dict = result_dicts[i]
        if isinstance(result_dict, dict) and 'error' in result_dict:
            print(f"--- CMA-ES Error for model {name} ---")
            print(result_dict['error'])
            print("------------------------------------")
        else:
            print(f"Finished CMA-ES for {name}. Stopped because: {result_dict.get('termination', 'N/A')}")
        final_results[name] = result_dict
        
    return list(final_results.items())


def run_cmaes_for_multiple_models_sequential(
    model_names: list[str],
    data_sample: dict,
    previous_results: dict = None,
) -> list[tuple[str, dict]]:
    """
    Runs CMA-ES fitting sequentially for a specified list of kinetic models.
    """
    if previous_results is None:
        previous_results = {}

    print(f"--- Preparing to analyze {len(model_names)} models sequentially ---")
    results = []
    for model_name in model_names:
        try:
            # Prepare model info inside the loop for sequential execution
            prep = _prepare_model_for_fitting(model_name)
            if prep["error"]:
                results.append((model_name, {"error": prep["error"]}))
                continue

            initial_guess = None
            if model_name in previous_results and 'best_params' in previous_results[model_name]:
                initial_guess = np.array(previous_results[model_name]['best_params'])

            result_dict = fit_with_cmaes(
                base_param_names=prep["base_param_names"],
                data_sample=data_sample,
                ode_func=prep["ode_func"],
                species_order=prep["species_order"],
                initial_guess=initial_guess,
            )
            print(f"Finished CMA-ES for {model_name}. Stopped because: {result_dict.get('termination', 'N/A')}")
            results.append((model_name, result_dict))

        except Exception as e:
            import traceback
            error_message = f"[{model_name}] Error: An error occurred during sequential execution: {e}"
            tb_message = traceback.format_exc()
            print(error_message)
            print(tb_message)
            results.append((model_name, {'error': error_message + "\n" + tb_message}))

    print(f"--- CMA-ES analysis complete for {len(results)} models ---")
    return results
