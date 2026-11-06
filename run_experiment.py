import multiprocessing
import argparse
import contextlib
import copy

try:
    import weave
    _WEAVE_AVAILABLE = True
except ImportError:
    weave = None  # type: ignore[assignment]
    _WEAVE_AVAILABLE = False
from bench.proposers.base import Proposer
from bench.proposers.random import RandomProposer
from bench.proposers.agent import AgentProposer
from bench.proposers.bayes import BayesProposer
from bench.proposers.maximin_lhs import MaximinLHSProposer
from bench.proposers.distance import DistanceProposer, MapEIGProposer
from bench.proposers.simple_heuristic import SimpleHeuristicProposer
from bench.oracle.kinetic_oracle import KineticOracle # Use actual oracle
from bench.fitters.cmaes_fitter import run_cmaes_for_multiple_models # Import CMA-ES fitter
from bench.config import (
    KINETIC_ORACLE_DATA_CONFIG, GROUND_TRUTH_GENERATOR_CONFIG,
    PRIOR_CONFIG, CMA_OPTIONS, CMA_OPTIONS_AFTER_FIRST_RUN, RANDOM_PROPOSER_CONFIG,
    AGENT_PROPOSER_CONFIG, LLM_CONFIGS
)
from bench.evaluation import ExperimentLogger, Evaluator
from bench.fitters.model_definition import MODEL_INFO
from bench.environment import ExperimentEnvironment
import sys
import os
import json
from typing import List, Dict, Any, Tuple, Optional # More specific types
from datetime import datetime
import time # Added for sleep
import traceback # Added for detailed error logging
import glob
import fnmatch
import re
from collections import deque


import pandas as pd
from scipy.integrate import solve_ivp
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# Assume model names are M1-M20 as per prompts.py
ALL_MODEL_NAMES = [f"M{i+1}" for i in range(20)]
GROUND_TRUTH_FILENAME_RE = re.compile(r"^M\d+_ground_truth_\d+\.json$")
def _shutdown_weave_safely() -> None:
    """
    Best-effort teardown for Weave client state in worker processes.
    Some Weave versions expose ``finish`` while others do not.
    """
    if not _WEAVE_AVAILABLE:
        return
    finish_fn = getattr(weave, "finish", None)
    if callable(finish_fn):
        try:
            finish_fn()
        except Exception as exc:
            print(f"Warning: weave.finish() failed during worker shutdown: {exc}")


def _ground_truth_filename_to_id(gt_filename: str) -> str:
    return gt_filename.replace('_ground_truth', '').replace('.json', '')


def _build_baseline_assignment_map(
    ground_truth_files: List[str],
    seed: int,
) -> Dict[str, int]:
    """
    Assigns a unique run-level assignment id to each selected sample.

    The mapping is created by shuffling the selected file list once using a
    global seed, then assigning ids 0..N-1 in that shuffled order. This avoids
    dependence on mechanism-ordered filenames while keeping assignments
    independent of ground-truth identity content.
    """
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(len(ground_truth_files))
    assignment_map: Dict[str, int] = {}
    for assignment_id, file_index in enumerate(permuted_indices.tolist()):
        assignment_map[ground_truth_files[file_index]] = assignment_id
    return assignment_map


def _run_sample_job(
    sample_job_args: Tuple[Any, ...],
    agent_semaphore: Optional[multiprocessing.synchronize.Semaphore],
) -> None:
    process_ground_truth_file(
        sample_job_args[0],
        sample_job_args[1],
        sample_job_args[2],
        sample_job_args[3],
        sample_job_args[4],
        sample_job_args[5],
        sample_job_args[6],
        agent_semaphore,
        sample_job_args[7],
        sample_job_args[8],
        sample_job_args[9],
        sample_job_args[10],
        sample_job_args[11],
        sample_job_args[12],
    )


def _run_samples_in_parallel_with_timeouts(
    sample_job_args_list: List[Tuple[Any, ...]],
    max_parallel_jobs: int,
    timeout_minutes: float,
    agent_semaphore: Optional[multiprocessing.synchronize.Semaphore],
    mp_ctx,
) -> None:
    """
    Runs sample jobs with forked worker processes and per-sample timeouts.
    A timed-out sample is terminated without cancelling the remaining queue.
    """
    pending_jobs = deque(sample_job_args_list)
    active_jobs: List[Dict[str, Any]] = []
    timeout_seconds = timeout_minutes * 60.0

    while pending_jobs or active_jobs:
        while pending_jobs and len(active_jobs) < max_parallel_jobs:
            sample_job_args = pending_jobs.popleft()
            sample_id = _ground_truth_filename_to_id(sample_job_args[0])
            worker = mp_ctx.Process(
                target=_run_sample_job,
                args=(sample_job_args, agent_semaphore),
                name=f"sample-worker-{sample_id}",
            )
            worker.start()
            active_jobs.append({
                "process": worker,
                "sample_id": sample_id,
                "started_at": time.monotonic(),
            })

        remaining_active_jobs: List[Dict[str, Any]] = []
        progress_made = False

        for job in active_jobs:
            worker = job["process"]
            sample_id = job["sample_id"]
            elapsed_seconds = time.monotonic() - job["started_at"]

            if worker.is_alive() and elapsed_seconds > timeout_seconds:
                print(
                    f"Sample {sample_id} timed out after {timeout_minutes} minutes. "
                    "Terminating worker."
                )
                worker.terminate()
                worker.join(timeout=10)
                if worker.is_alive() and hasattr(worker, "kill"):
                    print(f"Sample {sample_id} did not exit after terminate(); killing worker.")
                    worker.kill()
                    worker.join(timeout=10)
                progress_made = True
                continue

            if not worker.is_alive():
                worker.join()
                if worker.exitcode == 0:
                    print(f"Sample {sample_id} worker exited normally.")
                else:
                    print(f"Sample {sample_id} worker exited with code {worker.exitcode}.")
                progress_made = True
                continue

            remaining_active_jobs.append(job)

        active_jobs = remaining_active_jobs

        if (pending_jobs or active_jobs) and not progress_made:
            time.sleep(1)

def get_sanitized_llm_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a sanitized, JSON-serializable version of the LLM configuration.
    It converts class objects to their string representation and removes API keys.
    """
    if model_name not in LLM_CONFIGS:
        return None
    
    config = copy.deepcopy(LLM_CONFIGS[model_name])
    
    if "class" in config and hasattr(config["class"], "__name__"):
        config["class"] = f"{config['class'].__module__}.{config['class'].__name__}"
        
    if "params" in config and isinstance(config["params"], dict):
        config["params"] = {
            k: v for k, v in config["params"].items() 
            if "api_key" not in k.lower()
        }
        
    return config

def run_experiment_loop_cumulative(
    proposer: Proposer, 
    oracle: KineticOracle, 
    num_steps: int, 
    model_names: List[str],
    logger: ExperimentLogger,
    proposer_name: str, # Added for Weave metadata
    ground_truth_name: str, # Added for Weave metadata
    parallel: bool = True,
    agent_semaphore: Optional[multiprocessing.synchronize.Semaphore] = None,
    feedback_mode: str = 'nll',
) -> List[Dict[str, Any]]:
    """
    Runs the main experimental loop with a given proposer, oracle, and CMA-ES fitting.
    """
    history: List[Dict[str, str]] = [] # History is now a list of messages
    
    environment = ExperimentEnvironment(oracle, model_names, parallel_fitting=parallel)

    print(f"\n--- Starting experiment with {type(proposer).__name__} ---")

    for i in range(num_steps):
        step = i + 1
        print(f"\n--- Step {step}/{num_steps} ---")
        pre_step_cumulative_data = copy.deepcopy(environment.full_data_sample_for_cmaes)
        previous_representative_fits = environment.get_representative_fit_snapshot()

        # 1. Have the Proposer propose the next experiment
        if isinstance(proposer, AgentProposer) and agent_semaphore is not None:
            with agent_semaphore:
                proposer_output = proposer.propose(
                    history,
                    proposer_name=proposer_name,
                    ground_truth_name=ground_truth_name,
                    step=step,
                    num_steps=num_steps,
                    feedback_mode=feedback_mode,
                    cumulative_data=pre_step_cumulative_data,
                    previous_representative_fits=previous_representative_fits,
                )
        else:
            proposer_output = proposer.propose(
                history,
                proposer_name=proposer_name,
                ground_truth_name=ground_truth_name,
                step=step,
                num_steps=num_steps,
                feedback_mode=feedback_mode,
                cumulative_data=pre_step_cumulative_data,
                previous_representative_fits=previous_representative_fits,
            )

        raw_ai_response = proposer_output["raw_response"]
        history.append({"role": "assistant", "content": raw_ai_response})

        # 2. Check if a proposal was returned. If not, end the experiment.
        if "proposal" not in proposer_output or proposer_output["proposal"] is None:
            print(f"Agent did not provide a proposal at step {step}. Ending experiment.")
            break

        proposal_dict = proposer_output["proposal"]

        # 3. Run environment step to get data and fitting results
        user_message_content, fitting_results = environment.get_user_message(step, proposal_dict, feedback_mode)
        
        # 4. Log results
        candidate_mechanisms = proposer_output.get("candidate_mechanisms")
        logger.log_step(
            step=step,
            proposal=proposal_dict,
            fitting_results=fitting_results,
            candidate_mechanisms=candidate_mechanisms,
            pre_step_cumulative_data=pre_step_cumulative_data,
            cumulative_data=copy.deepcopy(environment.full_data_sample_for_cmaes),
            representative_fits=proposer_output.get("representative_fits"),
            condition_score=proposer_output.get("condition_score"),
        )

        # 5. Format and append the user message to history
        history.append({"role": "user", "content": user_message_content})

    print(f"\n--- Experiment finished with {type(proposer).__name__} ---")
    
    return history
    
    
def run_and_evaluate(
    proposer: Proposer, 
    ground_truth_model_name: str, 
    ground_truth_id: str, 
    ground_truth_filepath: str, 
    num_steps: int, 
    tag: str, 
    oracle_data_config: Dict[str, Any], # Added to receive the potentially overridden config
    proposer_model: Optional[str] = None, 
    parallel: bool = True, 
    agent_semaphore: Optional[multiprocessing.synchronize.Semaphore] = None, 
    feedback_mode: str = 'nll',
) -> Tuple[float, Optional[str]]:
    """
    Sets up the oracle and logger, runs the experiment loop, and evaluates the results.
    """
    print(f"\n--- Running evaluation for Ground Truth: {os.path.basename(ground_truth_filepath)} ---")

    oracle = KineticOracle(ground_truth_path=ground_truth_filepath, config=oracle_data_config)

    proposer_name = type(proposer).__name__
    run_folder_name = f"{proposer_name}_{proposer_model}" if proposer_name == "AgentProposer" and proposer_model else proposer_name
    output_dir = os.path.join("results", tag, ground_truth_id, run_folder_name)
    logger = ExperimentLogger(output_dir=output_dir)

    run_experiment_loop_cumulative(
        proposer=proposer, oracle=oracle, num_steps=num_steps, model_names=ALL_MODEL_NAMES,
        logger=logger, proposer_name=proposer_name, ground_truth_name=ground_truth_id,
        parallel=parallel, agent_semaphore=agent_semaphore, feedback_mode=feedback_mode,
    )

    evaluator = Evaluator(log_dir=output_dir, ground_truth_model_name=ground_truth_model_name)
    accuracy = evaluator.calculate_accuracy()
    best_models_per_step = evaluator.get_best_model_per_step()
    final_best_model = best_models_per_step.get(num_steps)

    print("\n--- Evaluation Summary ---")
    print(f"Proposer: {proposer_name}")
    if proposer_model: print(f"Proposer Model: {proposer_model}")
    print(f"Ground Truth File: {os.path.basename(ground_truth_filepath)}")
    print(f"Total Steps: {num_steps}")
    print(f"  - Overall Accuracy: {accuracy:.2f}%")
    print(f"  - Final Step Best Model: {final_best_model}")
    
    for step, model in best_models_per_step.items():
        is_correct = "✓" if model == ground_truth_model_name else "✗"
        print(f"  - Step {step}: Best model was {model} ({is_correct})")
    
    print("-" * 26)
    return accuracy, final_best_model


def process_ground_truth_file(
    gt_filename: str,
    gt_dir: str,
    num_experiment_steps: int,
    batch_tag: str,
    proposer_type: str,
    proposer_model: str,
    parallel_cmaes: bool,
    agent_semaphore: Optional[multiprocessing.synchronize.Semaphore],
    feedback_mode: str,
    agent_prompt_variant: str,
    oracle_data_config: Dict[str, Any],
    max_restart_attempts: int = 0,
    assignment_id: Optional[int] = None,
    use_weave: bool = True,
):
    """
    Worker function to process a single ground truth file, with restart logic.
    """
    gt_filepath = os.path.join(gt_dir, gt_filename)
    ground_truth_model_name = gt_filename.split('_')[0]
    ground_truth_id = gt_filename.replace('_ground_truth', '').replace('.json', '')

    # Determine proposer_class_name for directory naming consistency
    proposer_class_name = ""
    if proposer_type == 'agent':
        proposer_class_name = "AgentProposer"
    elif proposer_type == 'random':
        proposer_class_name = "RandomProposer"
    elif proposer_type == 'bayes':
        proposer_class_name = "BayesProposer"
    elif proposer_type == 'maximin-lhs':
        proposer_class_name = "MaximinLHSProposer"
    elif proposer_type == 'distance':
        proposer_class_name = "DistanceProposer"
    elif proposer_type == 'map-eig':
        proposer_class_name = "MapEIGProposer"
    elif proposer_type == 'simple-heuristic':
        proposer_class_name = "SimpleHeuristicProposer"
    else:
        print(f"Error: Unknown proposer type: '{proposer_type}'. Aborting sample {ground_truth_id}.")
        return # Abort if type is invalid early

    run_folder_name = f"{proposer_class_name}_{proposer_model}" if proposer_type == "agent" and proposer_model else proposer_class_name
    output_dir = os.path.join("results", batch_tag, ground_truth_id, run_folder_name)
    os.makedirs(output_dir, exist_ok=True) # Ensure base output dir exists

    restart_count = 0
    sample_completed = False

    while restart_count <= max_restart_attempts: # Loop for initial run + max_restart_attempts
        print(f"\n--- Processing Sample: {ground_truth_id} (Restart Attempt: {restart_count}/{max_restart_attempts}) ---")

        # --- Re-generate proposer (to get a fresh LLM object) ---
        proposer = None # Initialize to avoid UnboundLocalError
        if proposer_type == 'agent':
            print(f"Initializing fresh AgentProposer (model: {proposer_model})...")
            if restart_count > 0: # Delay before the second and subsequent runs
                time.sleep(5)
            proposer = AgentProposer(
                model_name=proposer_model,
                prompt_variant=agent_prompt_variant,
            )
        elif proposer_type == 'random':
            print("Initializing fresh RandomProposer...")
            if restart_count > 0: # Delay before the second and subsequent runs
                time.sleep(1) # Shorter delay for RandomProposer
            proposer = RandomProposer(assignment_id=assignment_id or 0)
        elif proposer_type == 'bayes':
            print("Initializing fresh BayesProposer...")
            if restart_count > 0:
                time.sleep(1)
            proposer = BayesProposer()
        elif proposer_type == 'maximin-lhs':
            print("Initializing fresh MaximinLHSProposer...")
            if restart_count > 0:
                time.sleep(1)
            proposer = MaximinLHSProposer(assignment_id=assignment_id or 0)
        elif proposer_type == 'distance':
            print("Initializing fresh DistanceProposer...")
            if restart_count > 0:
                time.sleep(1)
            proposer = DistanceProposer(model_names=ALL_MODEL_NAMES, parallel_fitting=parallel_cmaes)
        elif proposer_type == 'map-eig':
            print("Initializing fresh MapEIGProposer...")
            if restart_count > 0:
                time.sleep(1)
            proposer = MapEIGProposer(model_names=ALL_MODEL_NAMES, parallel_fitting=parallel_cmaes)
        elif proposer_type == 'simple-heuristic':
            print("Initializing fresh SimpleHeuristicProposer...")
            if restart_count > 0:
                time.sleep(1)
            proposer = SimpleHeuristicProposer()
        else:
            print(f"Error: Unknown proposer type: '{proposer_type}'. Aborting sample {ground_truth_id}.")
            break # Exit loop for invalid type
        
        try:
            if use_weave and _WEAVE_AVAILABLE:
                weave.init(project_name="KineticMechanismBench")
                weave_ctx = weave.attributes({
                    "batch_tag": batch_tag,
                    "ground_truth_id": ground_truth_id,
                    "feedback_mode": feedback_mode,
                    "agent_prompt_variant": agent_prompt_variant if proposer_type == 'agent' else None,
                    "proposer": proposer_type,
                    "proposer_model": proposer_model if proposer_type == 'agent' else None,
                    "restart_attempt": restart_count,
                    "baseline_assignment_id": assignment_id,
                })
            else:
                weave_ctx = contextlib.nullcontext()
            with weave_ctx:
                run_and_evaluate(
                    proposer=proposer,
                    ground_truth_model_name=ground_truth_model_name,
                    ground_truth_id=ground_truth_id,
                    ground_truth_filepath=gt_filepath,
                    num_steps=num_experiment_steps,
                    tag=batch_tag,
                    proposer_model=proposer_model if proposer_type == 'agent' else None,
                    parallel=parallel_cmaes,
                    agent_semaphore=agent_semaphore,
                    feedback_mode=feedback_mode,
                    oracle_data_config=oracle_data_config
                )

            # --- 実行後の完了チェック ---
            # output_dir 内の step_xxx_fitting_results.json ファイルの数を数える
            completed_step_files = glob.glob(os.path.join(output_dir, "step_*_fitting_results.json"))
            completed_steps_count = len(completed_step_files)

            if completed_steps_count >= num_experiment_steps:
                print(f"Sample {ground_truth_id} successfully completed all {num_experiment_steps} steps.")
                sample_completed = True
                break # 完了したのでループを抜ける
            else:
                print(f"Sample {ground_truth_id} completed only {completed_steps_count}/{num_experiment_steps} steps.")
                restart_count += 1
                if restart_count <= max_restart_attempts:
                    print(f"Attempting restart for sample {ground_truth_id} (Attempt {restart_count}/{max_restart_attempts})...")
                else:
                    print(f"Max restart attempts ({max_restart_attempts}) reached for sample {ground_truth_id}. Marking as failed.")
        except Exception as e:
            print(f"An error occurred during run for sample {ground_truth_id} (Attempt {restart_count}): {e}")
            traceback.print_exc() # デバッグ用にスタックトレースも出力

            restart_count += 1
            if restart_count <= max_restart_attempts:
                print(f"Attempting restart for sample {ground_truth_id} (Attempt {restart_count}/{max_restart_attempts})...")
            else:
                print(f"Max restart attempts ({max_restart_attempts}) reached for sample {ground_truth_id} due to error. Marking as failed.")
        finally:
            _shutdown_weave_safely()
        
    if not sample_completed:
        print(f"Sample {ground_truth_id} ultimately failed to complete after {max_restart_attempts} restarts.")


def main():
    """
    Main function to find all ground truths and run experiments for each of them.
    """
    parser = argparse.ArgumentParser(description="Run kinetic mechanism bench experiments.")
    parser.add_argument("--mode", type=str, default="mechanism-parallel", choices=["mechanism-parallel", "sample-parallel"], help="Set the parallel execution mode.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs for processing ground truth files. Only used with '--mode sample-parallel'.")
    parser.add_argument("--steps", type=int, required=True, help="Number of experimental steps.")
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Directory containing ground truth JSON files.",
    )
    parser.add_argument(
        "--sample-parallel-timeout-minutes",
        type=float,
        default=150.0,
        help=(
            "Per-sample timeout in minutes for sample-parallel mode. "
            "Timed-out samples are terminated and remaining samples continue."
        ),
    )
    parser.add_argument("--agent-concurrency", type=int, default=5, help="Maximum number of concurrent Agent executions.")
    parser.add_argument("--feedback-mode", type=str, default="nll", choices=["nll", "raw_data", "both"], help="Set the feedback mode for the agent.")
    parser.add_argument(
        "--agent-prompt-variant",
        type=str,
        default="standard",
        choices=["standard", "mechanism_hidden", "step_instruction"],
        help="Select the system prompt variant for AgentProposer. The non-standard variants require --feedback-mode nll.",
    )
    parser.add_argument(
        "--proposer",
        type=str,
        default="agent",
        choices=["agent", "random", "bayes", "maximin-lhs", "distance", "map-eig", "simple-heuristic"],
        help="The type of proposer to use. Default: agent"
    )
    parser.add_argument(
        "--proposer-model",
        type=str,
        default=AGENT_PROPOSER_CONFIG.get("model_name"),
        help=f"The name of the LLM to use for the AgentProposer. Must be a key in LLM_CONFIGS. Default: {AGENT_PROPOSER_CONFIG.get('model_name')}"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Override the noise level for the kinetic oracle data generation."
    )
    parser.add_argument(
        "--ground-truth-pattern",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional glob pattern(s) to select a subset of ground truth samples. "
            "Patterns are matched against both filenames like "
            "'M1_ground_truth_5.json' and sample ids like 'M1_5'. "
            "Example: --ground-truth-pattern 'M1_*' 'M7_2'"
        ),
    )
    parser.add_argument(
        "--max-sample-restarts",
        type=int,
        default=0,
        help="Maximum number of times to restart a sample's processing if not all steps are completed."
    )
    parser.add_argument(
        "--no-weave",
        action="store_true",
        default=False,
        help="Disable Weave logging entirely.",
    )
    args = parser.parse_args()

    if args.agent_prompt_variant in {"mechanism_hidden", "step_instruction"}:
        if args.proposer != "agent":
            raise ValueError(f"--agent-prompt-variant {args.agent_prompt_variant} is only supported with --proposer agent.")
        if args.feedback_mode != "nll":
            raise ValueError(
                f"--agent-prompt-variant {args.agent_prompt_variant} is currently only supported with --feedback-mode nll."
            )

    if args.mode == "mechanism-parallel":
        parallel_cmaes, run_samples_in_parallel = True, False
        if args.jobs > 1: print("Warning: --jobs > 1 is ignored with '--mode mechanism-parallel'.")
    elif args.mode == "sample-parallel":
        parallel_cmaes, run_samples_in_parallel = False, args.jobs > 1
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    gt_dir = args.gt_dir
    if not os.path.exists(gt_dir) or not os.listdir(gt_dir):
        print(f"Ground truth directory '{gt_dir}' is empty or does not exist. Please run `python generate_ground_truths.py` first.")
        return

    ground_truth_files = sorted(
        f for f in os.listdir(gt_dir)
        if GROUND_TRUTH_FILENAME_RE.fullmatch(f)
    )

    if args.ground_truth_pattern:
        selected_patterns = args.ground_truth_pattern

        def matches_ground_truth_pattern(filename: str) -> bool:
            sample_id = filename.replace('_ground_truth', '').replace('.json', '')
            return any(
                fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(sample_id, pattern)
                for pattern in selected_patterns
            )

        ground_truth_files = [
            filename for filename in ground_truth_files
            if matches_ground_truth_pattern(filename)
        ]
        print(
            f"Filtered ground truth files with patterns {selected_patterns}: "
            f"{len(ground_truth_files)} matched."
        )

    if not ground_truth_files:
        print("No ground truth files matched the requested selection.")
        return

    print(f"Found {len(ground_truth_files)} ground truth files to process.")

    batch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting experiment batch with tag: {batch_tag}")

    snapshot_dir = os.path.join("results", batch_tag)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Create a mutable copy of KINETIC_ORACLE_DATA_CONFIG for potential overrides
    oracle_data_config = copy.deepcopy(KINETIC_ORACLE_DATA_CONFIG)
    if args.noise_level is not None:
        print(f"Overriding KINETIC_ORACLE_DATA_CONFIG['noise_level'] to {args.noise_level}")
        oracle_data_config['noise_level'] = args.noise_level

    # Consolidate all configurations into a single file for the batch
    run_config = {
        "BASE_CONFIGS": {
            "PRIOR_CONFIG": PRIOR_CONFIG,
            "CMA_OPTIONS": CMA_OPTIONS,
            "CMA_OPTIONS_AFTER_FIRST_RUN": CMA_OPTIONS_AFTER_FIRST_RUN,
            "RANDOM_PROPOSER_CONFIG": RANDOM_PROPOSER_CONFIG,
            "KINETIC_ORACLE_DATA_CONFIG": oracle_data_config, # Use the potentially overridden config
            "GROUND_TRUTH_GENERATOR_CONFIG": GROUND_TRUTH_GENERATOR_CONFIG,
        },
        "RUN_SETTINGS": {
            "run_args": vars(args),
            "proposer_model_config": get_sanitized_llm_config(args.proposer_model) if args.proposer == 'agent' else None
        }
    }

    with open(os.path.join(snapshot_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved consolidated run configuration to {snapshot_dir}")

    baseline_assignment_map: Dict[str, int] = {}
    if args.proposer in {"random", "maximin-lhs"}:
        baseline_assignment_map = _build_baseline_assignment_map(
            ground_truth_files=ground_truth_files,
            seed=int(RANDOM_PROPOSER_CONFIG["seed"]),
        )
        assignments_payload = {
            _ground_truth_filename_to_id(filename): assignment_id
            for filename, assignment_id in baseline_assignment_map.items()
        }
        with open(os.path.join(snapshot_dir, "baseline_assignments.json"), "w") as f:
            json.dump(assignments_payload, f, indent=2, sort_keys=True)
        print(f"Saved baseline assignment ids to {os.path.join(snapshot_dir, 'baseline_assignments.json')}")
    
    use_weave = _WEAVE_AVAILABLE and not args.no_weave
    if not _WEAVE_AVAILABLE:
        print("Weave is not installed. Weave logging disabled.")
    elif args.no_weave:
        print("Weave logging disabled via --no-weave.")

    sample_job_args_list = [
        (
            gt,
            gt_dir,
            args.steps,
            batch_tag,
            args.proposer,
            args.proposer_model,
            parallel_cmaes,
            args.feedback_mode,
            args.agent_prompt_variant,
            oracle_data_config,
            args.max_sample_restarts,
            baseline_assignment_map.get(gt),
            use_weave,
        )
        for gt in ground_truth_files
    ]
    
    if run_samples_in_parallel:
        mp_ctx = multiprocessing.get_context("fork")
        agent_semaphore = mp_ctx.BoundedSemaphore(args.agent_concurrency)
        print(
            f"Running in SAMPLE-PARALLEL mode with {args.jobs} jobs "
            f"and {args.sample_parallel_timeout_minutes} minute per-sample timeout."
        )
        _run_samples_in_parallel_with_timeouts(
            sample_job_args_list=sample_job_args_list,
            max_parallel_jobs=args.jobs,
            timeout_minutes=args.sample_parallel_timeout_minutes,
            agent_semaphore=agent_semaphore,
            mp_ctx=mp_ctx,
        )
    else:
        print("Running in SEQUENTIAL or MECHANISM-PARALLEL mode.")
        agent_semaphore = multiprocessing.BoundedSemaphore(args.agent_concurrency)
        for sample_job_args in sample_job_args_list:
            _run_sample_job(sample_job_args, agent_semaphore)


if __name__ == "__main__":
    main()
