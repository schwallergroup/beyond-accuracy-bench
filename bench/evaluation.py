"""
This module provides classes for logging experimental results and evaluating them.
"""
import os
import json
from typing import List, Dict, Any, Tuple, Optional
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn.preprocessing import StandardScaler # Added import
from sklearn.decomposition import PCA # Added import

from .config import PRIOR_CONFIG, RANDOM_PROPOSER_CONFIG


class ExperimentLogger:
    """
    Logs proposals and fitting results for each step of an experiment to disk.
    """
    def __init__(self, output_dir: str):
        """
        Initializes the logger and creates the output directory.

        Args:
            output_dir: The directory where log files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Logging experiment results to: {self.output_dir}")

    def log_step(
        self,
        step: int,
        proposal: Dict[str, Any],
        fitting_results: List[Tuple[str, Dict[str, Any]]],
        candidate_mechanisms: Optional[List[str]] = None,
        pre_step_cumulative_data: Optional[Dict[str, Any]] = None,
        cumulative_data: Optional[Dict[str, Any]] = None,
        representative_fits: Optional[Dict[str, Any]] = None,
        condition_score: Optional[Dict[str, Any]] = None,
    ):
        """
        Saves the proposal, candidate mechanisms, and NLL results to JSON files.
        """
        print(f"--- [DEBUG] Inside ExperimentLogger.log_step for step: {step} in {self.output_dir} ---")
        # Save the proposal and candidate mechanisms
        proposal_log_data = {
            "proposal": proposal,
            "candidate_mechanisms": candidate_mechanisms
        }
        proposal_path = os.path.join(self.output_dir, f"step_{step:03d}_proposal.json")
        with open(proposal_path, 'w') as f:
            json.dump(proposal_log_data, f, indent=4)

        # Create a simplified dictionary with only the negative log-likelihood
        nll_results = {}
        for model_name, res_dict in fitting_results:
            if 'error' in res_dict:
                nll_results[model_name] = None  # Use None to indicate an error during fitting
            else:
                nll_results[model_name] = res_dict.get('neg_log_likelihood')

        results_path = os.path.join(self.output_dir, f"step_{step:03d}_fitting_results.json")
        with open(results_path, 'w') as f:
            json.dump(nll_results, f, indent=4)

        if pre_step_cumulative_data is not None:
            pre_data_snapshot_path = os.path.join(self.output_dir, f"step_{step:03d}_pre_data_snapshot.json")
            with open(pre_data_snapshot_path, 'w') as f:
                json.dump(pre_step_cumulative_data, f, indent=4)

        if cumulative_data is not None:
            data_snapshot_path = os.path.join(self.output_dir, f"step_{step:03d}_data_snapshot.json")
            with open(data_snapshot_path, 'w') as f:
                json.dump(cumulative_data, f, indent=4)

        if representative_fits is not None:
            fit_snapshot_path = os.path.join(self.output_dir, f"step_{step:03d}_representative_fits.json")
            with open(fit_snapshot_path, 'w') as f:
                json.dump(representative_fits, f, indent=4)

        if condition_score is not None:
            score_snapshot_path = os.path.join(self.output_dir, f"step_{step:03d}_condition_score.json")
            with open(score_snapshot_path, 'w') as f:
                json.dump(condition_score, f, indent=4)

    def log_final_answer(self, step: int, answer_output: Dict[str, Any]):
        """Saves the final answer from the agent to a dedicated file."""
        print(f"--- [DEBUG] Inside ExperimentLogger.log_final_answer for step: {step} in {self.output_dir} ---")
        log_data = {"step": step, **answer_output}
        answer_path = os.path.join(self.output_dir, "final_answer.json")
        with open(answer_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"Final answer logged to: {answer_path}")


class Evaluator:
    """
    Evaluates the results of an experiment based on logged data.
    """
    def __init__(self, log_dir: str, ground_truth_model_name: str):
        """
        Initializes the evaluator.

        Args:
            log_dir: The directory containing the logged experiment files.
            ground_truth_model_name: The name of the correct model for this experiment.
        """
        self.log_dir = log_dir
        # The name can be 'M1_0', so we extract the base model name 'M1'
        self.ground_truth_model_name = ground_truth_model_name.split('_')[0]
        self.results_per_step = self._load_results()

    def _load_results(self) -> Dict[int, Dict[str, Any]]:
        """
        Loads all fitting results from the log directory.
        """
        results = {}
        if not os.path.exists(self.log_dir):
            print(f"Warning: Log directory not found: {self.log_dir}")
            return results
            
        for filename in sorted(os.listdir(self.log_dir)):
            if filename.startswith('step_') and filename.endswith('_fitting_results.json'):
                try:
                    step = int(filename.split('_')[1])
                    filepath = os.path.join(self.log_dir, filename)
                    with open(filepath, 'r') as f:
                        results[step] = json.load(f)
                except (ValueError, IndexError):
                    continue # Ignore malformed filenames
        return results

    def get_best_model_per_step(self) -> Dict[int, str]:
        """
        Determines the best model (lowest neg_log_likelihood) for each step.
        """
        best_models = {}
        for step, results in self.results_per_step.items():
            # Filter out models where fitting failed (NLL is None)
            valid_results = {model: nll for model, nll in results.items() if nll is not None and nll != 'inf'}
            
            if not valid_results:
                best_models[step] = None
                continue
            
            # Find the model with the minimum NLL
            best_model = min(valid_results, key=valid_results.get)
            best_models[step] = best_model
            
        return best_models

    def calculate_accuracy(self) -> float:
        """
        Calculates the final accuracy: the percentage of steps where the best-fit
        model matched the ground truth model.
        """
        best_models_per_step = self.get_best_model_per_step()
        if not best_models_per_step:
            return 0.0
        
        correct_predictions = 0
        for step, best_model in best_models_per_step.items():
            if best_model == self.ground_truth_model_name:
                correct_predictions += 1
        
        accuracy = (correct_predictions / len(best_models_per_step)) * 100
        return accuracy

class MultiExperimentEvaluator:
    """
    Evaluates and compares results from multiple experiments under a single tag.
    """
    def __init__(self, results_root: str = "results"):
        """
        Initializes the multi-experiment evaluator.
        Args:
            results_root: The root directory where all experiment results are stored.
        """
        self.results_root = results_root
        self.model_names = sorted([f"M{i+1}" for i in range(20)], key=lambda x: int(x[1:])) # Consistent model name list for ordering

    def calculate_difficulty_scores(self, tag: str) -> pd.DataFrame:
        """
        Analyzes mechanism difficulty from evaluation results for a given tag.
        This logic is migrated from the original `analyze_difficulty.py` script.
        """
        results_dir = os.path.join(self.results_root, tag)
        instance_dirs = glob.glob(os.path.join(results_dir, "M*_*"))
        analysis_data = []

        for instance_dir in sorted(instance_dirs):
            instance_name = os.path.basename(instance_dir)
            gt_model_base = instance_name.split('_')[0]
            
            # Assuming difficulty is evaluated with RandomProposer
            fitting_files = glob.glob(os.path.join(instance_dir, "RandomProposer", "step_*_fitting_results.json"))
            if not fitting_files:
                continue
                
            success_count = 0
            total_trials = 0
            
            for fpath in fitting_files:
                try:
                    with open(fpath, 'r') as f:
                        results = json.load(f)
                    
                    if not results:
                        continue
                    
                    total_trials += 1
                    valid_results = {k: v for k, v in results.items() if v is not None and isinstance(v, (int, float))}
                    
                    if not valid_results or gt_model_base not in valid_results:
                        continue

                    best_model = min(valid_results, key=valid_results.get)
                    
                    if best_model == gt_model_base:
                        success_count += 1
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not read or parse {fpath}: {e}")
            
            if total_trials > 0:
                success_rate = success_count / total_trials
                analysis_data.append({
                    "name": instance_name,
                    "success_rate": success_rate,
                    "total_trials": total_trials,
                    "success_count": success_count,
                })
        
        return pd.DataFrame(analysis_data)

    def _gather_all_step_data_from_nll(self, tag: str) -> pd.DataFrame:
        """
        Gathers comprehensive data (accuracy, rank, NLLs, entropy) for EVERY 
        step of every experiment based on NLL results and returns it as a 
        single detailed DataFrame. This is the core data gathering method.
        """
        loaded_results = self._load_results_by_tag(tag, full_results=True)
        if not loaded_results:
            return pd.DataFrame()

        data_points = []
        for proposer_name, ground_truth_results in loaded_results.items():
            for ground_truth, steps in ground_truth_results.items():
                if not steps:
                    continue
                
                ground_truth_base = ground_truth.split('_')[0]
                
                for step, results in steps.items():
                    valid_results = {model: nll for model, nll in results.items() if nll is not None and nll != 'inf'}
                    
                    # Pre-initialize metrics with default failure values
                    is_correct, rank, entropy, best_nll, gt_nll, nll_diff = 0, len(self.model_names) + 1, np.nan, np.nan, np.nan, np.nan

                    if valid_results:
                        # Common calculations
                        sorted_models = sorted(valid_results.items(), key=lambda item: item[1])
                        best_model, best_nll = sorted_models[0]
                        gt_nll = valid_results.get(ground_truth_base, np.nan)

                        # Accuracy (Top-1)
                        is_correct = 1 if best_model == ground_truth_base else 0
                        
                        # Rank
                        if ground_truth_base in valid_results:
                            sorted_nlls = sorted(list(set(valid_results.values())))
                            rank = sorted_nlls.index(gt_nll) + 1

                        # Entropy
                        if len(valid_results) >= 2:
                            nlls = np.array(list(valid_results.values()))
                            log_likelihoods = -nlls
                            log_probs = log_likelihoods - logsumexp(log_likelihoods)
                            probs = np.exp(log_probs)
                            entropy = -np.sum(p * np.log(p) for p in probs if p > 0)
                        else:
                            entropy = 0.0
                        
                        # NLL diff
                        if not np.isnan(gt_nll):
                            nll_diff = gt_nll - best_nll

                    data_points.append({
                        'tag': tag,
                        'proposer': proposer_name,
                        'ground_truth': ground_truth,
                        'step': step,
                        'accuracy': is_correct,
                        'rank': rank,
                        'entropy': entropy,
                        'best_nll': best_nll,
                        'gt_nll': gt_nll,
                        'nll_diff': nll_diff
                    })

        return pd.DataFrame(data_points)

    def _gather_all_step_data_from_nll(self, tag: str) -> pd.DataFrame:
        """
        Gathers comprehensive data (accuracy, rank, NLLs, entropy) for EVERY 
        step of every experiment based on NLL results and returns it as a 
        single detailed DataFrame. This is the core data gathering method.
        """
        loaded_results = self._load_results_by_tag(tag, full_results=True)
        if not loaded_results:
            return pd.DataFrame()

        data_points = []
        for proposer_name, ground_truth_results in loaded_results.items():
            for ground_truth, steps in ground_truth_results.items():
                if not steps:
                    continue
                
                ground_truth_base = ground_truth.split('_')[0]
                
                for step, results in steps.items():
                    valid_results = {model: nll for model, nll in results.items() if nll is not None and nll != 'inf'}
                    
                    # Pre-initialize metrics with default failure values
                    best_model_name, is_correct, rank, entropy, best_nll, gt_nll, nll_diff = None, 0, len(self.model_names) + 1, np.nan, np.nan, np.nan, np.nan

                    if valid_results:
                        # Common calculations
                        sorted_models = sorted(valid_results.items(), key=lambda item: item[1])
                        best_model_name, best_nll = sorted_models[0]
                        gt_nll = valid_results.get(ground_truth_base, np.nan)

                        # Accuracy (Top-1)
                        is_correct = 1 if best_model_name == ground_truth_base else 0
                        
                        # Rank
                        if ground_truth_base in valid_results:
                            sorted_nlls = sorted(list(set(valid_results.values())))
                            rank = sorted_nlls.index(gt_nll) + 1

                        # Entropy
                        if len(valid_results) >= 2:
                            nlls = np.array(list(valid_results.values()))
                            log_likelihoods = -nlls
                            log_probs = log_likelihoods - logsumexp(log_likelihoods)
                            probs = np.exp(log_probs)
                            entropy = -np.sum(p * np.log(p) for p in probs if p > 0)
                        else:
                            entropy = 0.0
                        
                        # NLL diff
                        if not np.isnan(gt_nll):
                            nll_diff = gt_nll - best_nll

                    data_points.append({
                        'tag': tag,
                        'proposer': proposer_name,
                        'ground_truth': ground_truth,
                        'step': step,
                        'best_model_name': best_model_name,
                        'accuracy': is_correct,
                        'rank': rank,
                        'entropy': entropy,
                        'best_nll': best_nll,
                        'gt_nll': gt_nll,
                        'nll_diff': nll_diff
                    })

        return pd.DataFrame(data_points)

    def _fill_all_null_step_from_previous(
        self,
        df: pd.DataFrame,
        target_step: int = 5,
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Fill an all-null target step from the same run's immediately previous
        step if it has at least one valid NLL.
        """
        if df.empty or "best_nll" not in df.columns:
            return df, 0, 0

        filled_df = df.copy()
        fallback_columns = [
            "best_model_name",
            "accuracy",
            "rank",
            "entropy",
            "best_nll",
            "gt_nll",
            "nll_diff",
        ]
        fallback_columns = [col for col in fallback_columns if col in filled_df.columns]

        filled_count = 0
        unfilled_count = 0
        group_columns = ["proposer", "ground_truth"]

        if "nll_source_step" not in filled_df.columns:
            filled_df["nll_source_step"] = filled_df["step"]
        filled_df["used_previous_nll"] = False

        for _, run_df in filled_df.groupby(group_columns, sort=False):
            target_rows = run_df[
                (run_df["step"] == target_step) & (run_df["best_nll"].isna())
            ]
            for target_idx, target_row in target_rows.iterrows():
                previous_valid = run_df[
                    (run_df["step"] == target_row["step"] - 1) & (run_df["best_nll"].notna())
                ]

                if previous_valid.empty:
                    unfilled_count += 1
                    continue

                source_row = previous_valid.iloc[0]
                for col in fallback_columns:
                    filled_df.at[target_idx, col] = source_row[col]
                filled_df.at[target_idx, "nll_source_step"] = source_row["step"]
                filled_df.at[target_idx, "used_previous_nll"] = True
                filled_count += 1

        return filled_df, filled_count, unfilled_count

    def get_final_step_metrics(self, tag: str) -> pd.DataFrame:
        """
        Calculates aggregated metrics (Accuracy, NLL, Entropy) for a given tag,
        focusing only on experiments that completed the full number of iterations
        and had valid fitting results at the final step. If step 5 has all-null
        NLLs, step 4 is used as a final-summary fallback when available.
        """
        all_data = self._gather_all_step_data_from_nll(tag)
        if all_data.empty:
            return pd.DataFrame()

        # 1. Determine the benchmark's intended final step for each proposer
        # This is the max step reached across all runs for a proposer.
        all_data['benchmark_final_step'] = all_data.groupby('proposer')['step'].transform('max')
        
        # 2. Find the actual max step reached for each individual run
        all_data['run_max_step'] = all_data.groupby(['proposer', 'ground_truth'])['step'].transform('max')

        # 3. Fill all-null step 5 entries from step 4 for final summary metrics.
        all_data, _, _ = self._fill_all_null_step_from_previous(all_data, target_step=5)

        # 4. Filter to get data only from runs that completed all benchmark steps
        completed_runs = all_data[all_data['run_max_step'] == all_data['benchmark_final_step']]

        # 5. From those completed runs, select only the rows corresponding to the final step
        final_step_data = completed_runs[completed_runs['step'] == completed_runs['benchmark_final_step']]

        # 6. Filter out samples where neither the final step nor its fallback had valid NLLs.
        # The 'best_nll' column is NaN when no valid NLLs were found.
        final_step_data = final_step_data[final_step_data['best_nll'].notna()]

        # Clean up temporary columns before returning
        final_step_data = final_step_data.drop(columns=['benchmark_final_step', 'run_max_step'])
        
        return final_step_data.reset_index(drop=True)

    def get_fit_failure_reasons(self, tag: str) -> Dict[str, Any]:
        """
        Analyzes why samples failed to fit at the final step for each proposer.
        """
        search_path = os.path.join(self.results_root, tag, "*", "*")
        experiment_dirs = glob.glob(search_path)
        if not experiment_dirs:
            print(f"Warning: No experiment directories found for tag '{tag}' to analyze failures.")
            return {}

        per_proposer = defaultdict(list)
        for exp_dir in experiment_dirs:
            parts = exp_dir.split(os.sep)
            if len(parts) < 2:
                continue
            proposer_name = parts[-1]
            ground_truth_model = parts[-2]
            per_proposer[proposer_name].append((ground_truth_model, exp_dir))

        summary = {}
        for proposer_name, samples in per_proposer.items():
            final_step = 0
            for _, exp_dir in samples:
                step_files = glob.glob(os.path.join(exp_dir, "step_*_fitting_results.json"))
                for f_path in step_files:
                    try:
                        step = int(os.path.basename(f_path).split('_')[1])
                        final_step = max(final_step, step)
                    except (ValueError, IndexError):
                        continue

            failures = {
                "missing_results_files": 0,
                "step_filename_parse_error": 0,
                "short_run": 0,
                "final_step_parse_error": 0,
                "all_null_final_step": 0,
            }
            failure_samples = defaultdict(list)
            total_samples = len(samples)

            for ground_truth_model, exp_dir in samples:
                step_files = glob.glob(os.path.join(exp_dir, "step_*_fitting_results.json"))
                if not step_files:
                    failures["missing_results_files"] += 1
                    failure_samples["missing_results_files"].append(ground_truth_model)
                    continue
                step_files_by_step = {}
                for f_path in step_files:
                    try:
                        step = int(os.path.basename(f_path).split('_')[1])
                        step_files_by_step[step] = f_path
                    except (ValueError, IndexError):
                        continue
                if not step_files_by_step:
                    failures["step_filename_parse_error"] += 1
                    failure_samples["step_filename_parse_error"].append(ground_truth_model)
                    continue

                sample_max_step = max(step_files_by_step.keys())
                if final_step and sample_max_step < final_step:
                    failures["short_run"] += 1
                    failure_samples["short_run"].append(ground_truth_model)

                final_step_file = step_files_by_step.get(sample_max_step)
                if not final_step_file:
                    failures["final_step_parse_error"] += 1
                    failure_samples["final_step_parse_error"].append(ground_truth_model)
                    continue

                try:
                    with open(final_step_file, 'r') as f:
                        final_step_results = json.load(f)
                except (OSError, json.JSONDecodeError):
                    failures["final_step_parse_error"] += 1
                    failure_samples["final_step_parse_error"].append(ground_truth_model)
                    continue

                valid_results = {model: nll for model, nll in final_step_results.items() if nll is not None and nll != 'inf'}
                if not valid_results:
                    failures["all_null_final_step"] += 1
                    failure_samples["all_null_final_step"].append(ground_truth_model)

            summary[proposer_name] = {
                "final_step": final_step,
                "total_samples": total_samples,
                "failures": failures,
                "failure_samples": dict(failure_samples),
            }

        return summary

    def _load_results_by_tag(self, tag: str, full_results: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Loads all experiment results for a given tag, structured by proposer and ground truth.

        Args:
            tag: The experiment tag to load results for.
            full_results: If True, returns full NLL results per step. 
                          Otherwise, returns only the best model name per step.

        Returns:
            A nested dictionary.
            If full_results is False: {proposer: {ground_truth: {step: best_model_name}}}
            If full_results is True:  {proposer: {ground_truth: {step: {model: nll}}}}
        """
        results = defaultdict(dict)
        search_path = os.path.join(self.results_root, tag, "*", "*")
        
        print(f"Searching for experiment results in: {search_path}")
        experiment_dirs = glob.glob(search_path)
        
        if not experiment_dirs:
            print(f"Warning: No experiment directories found for tag '{tag}'.")
            return {}

        for exp_dir in experiment_dirs:
            try:
                # Path is expected to be .../results/<tag>/<ground_truth>/<proposer>
                parts = exp_dir.split(os.sep)
                proposer_name = parts[-1]
                ground_truth_model = parts[-2]
                
                evaluator = Evaluator(log_dir=exp_dir, ground_truth_model_name=ground_truth_model)
                
                if full_results:
                    if evaluator.results_per_step:
                        results[proposer_name][ground_truth_model] = evaluator.results_per_step
                else:
                    best_models_per_step = evaluator.get_best_model_per_step()
                    if best_models_per_step:
                        results[proposer_name][ground_truth_model] = best_models_per_step

            except IndexError:
                print(f"Warning: Could not parse directory path: {exp_dir}")
                continue
        
        print(f"Loaded results for {len(results)} proposers.")
        return dict(results)

    def _load_proposals_by_tag(self, tag: str) -> Dict[str, Dict[str, Any]]:
        """
        Loads all proposal data for a given tag, structured by proposer and ground truth.
        Specifically targets the 'candidate_mechanisms' if available.
        """
        proposals = defaultdict(dict)
        search_path = os.path.join(self.results_root, tag, "*", "*")
        
        print(f"Searching for proposal files in: {search_path}")
        experiment_dirs = glob.glob(search_path)

        if not experiment_dirs:
            print(f"Warning: No experiment directories found for tag '{tag}' to load proposals.")
            return {}

        for exp_dir in experiment_dirs:
            try:
                parts = exp_dir.split(os.sep)
                proposer_name = parts[-1]
                ground_truth_model = parts[-2]
                
                proposals[proposer_name][ground_truth_model] = {}
                
                proposal_files = sorted(glob.glob(os.path.join(exp_dir, "step_*_proposal.json")))
                for f_path in proposal_files:
                    step = int(os.path.basename(f_path).split('_')[1])
                    with open(f_path, 'r') as f:
                        data = json.load(f)
                    
                    # Store the whole proposal data for the step
                    proposals[proposer_name][ground_truth_model][step] = data

            except (IndexError, ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Could not parse proposals in directory {exp_dir}: {e}")
                continue
        
        print(f"Loaded proposals for {len(proposals)} proposers.")
        return dict(proposals)

    def generate_confusion_matrices_from_nll(self, tag: str):
        """
        Generates and saves a confusion matrix for each proposer based on the best model
        from the final negative log-likelihood (NLL).
        """
        final_step_data = self.get_final_step_metrics(tag)
        if final_step_data.empty:
            print(f"No final step results to generate confusion matrix from NLL.")
            return

        output_dir = os.path.join(self.results_root, tag)

        for proposer_name, group in final_step_data.groupby('proposer'):
            # Extract ground truth base model from the full ground truth name (e.g., M1_0 -> M1)
            y_true = group['ground_truth'].apply(lambda x: x.split('_')[0])
            
            # The best model is the one with the lowest NLL. We need to find its name.
            # The new _gather_all_step_data doesn't explicitly return the best model name,
            # but we can deduce it from accuracy and ground truth. If accuracy is 1, pred is true.
            # This is getting complicated. Let's add best_model_name to the gatherer.
            
            # --- Let's pause and rethink. The old code worked.
            loaded_results = self._load_results_by_tag(tag)
            if not loaded_results: return
            proposer_results = loaded_results.get(proposer_name)
            if not proposer_results: continue

            y_true_cm = []
            y_pred_cm = []

            for ground_truth, steps in proposer_results.items():
                if not steps: continue
                final_step = max(steps.keys())
                final_prediction = steps.get(final_step)

                if final_prediction:
                    y_true_cm.append(ground_truth.split('_')[0])
                    y_pred_cm.append(final_prediction)
            
            if not y_true_cm:
                print(f"No results to generate confusion matrix from NLL for {proposer_name}.")
                continue

            # Generate confusion matrix
            cm = confusion_matrix(y_true_cm, y_pred_cm, labels=self.model_names)
            cm_df = pd.DataFrame(cm, index=self.model_names, columns=self.model_names)

            # Plotting
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix from NLL for {proposer_name} (Tag: {tag})')
            plt.ylabel('Actual (Ground Truth)')
            plt.xlabel('Predicted')
            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(output_dir, f'confusion_matrix_from_nll_{proposer_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved confusion matrix from NLL for {proposer_name} to {save_path}")

    def generate_confusion_matrices_from_final_answer(self, tag: str):
        """
        Generates and saves a confusion matrix for each proposer based on 'final_answer.json'.
        """
        search_path = os.path.join(self.results_root, tag, "*", "*", "final_answer.json")
        final_answer_files = glob.glob(search_path)

        if not final_answer_files:
            print("No 'final_answer.json' files found to generate confusion matrix from final answer.")
            return

        # Group files by proposer
        files_by_proposer = defaultdict(list)
        for f in final_answer_files:
            # .../results/<tag>/<ground_truth>/<proposer>/final_answer.json
            parts = f.split(os.sep)
            proposer_name = parts[-2]
            files_by_proposer[proposer_name].append(f)

        output_dir = os.path.join(self.results_root, tag)

        for proposer_name, files in files_by_proposer.items():
            y_true = []
            y_pred = []
            
            for f_path in files:
                try:
                    parts = f_path.split(os.sep)
                    ground_truth_full = parts[-3]
                    ground_truth_base = ground_truth_full.split('_')[0]

                    with open(f_path, 'r') as f:
                        data = json.load(f)
                    
                    final_prediction = data.get('final_answer')

                    if final_prediction:
                        y_true.append(ground_truth_base)
                        y_pred.append(final_prediction)

                except (IndexError, json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not process file {f_path}: {e}")
                    continue

            if not y_true:
                print(f"No results to generate confusion matrix from final answer for {proposer_name}.")
                continue

            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=self.model_names)
            cm_df = pd.DataFrame(cm, index=self.model_names, columns=self.model_names)

            # Plotting
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix from Final Answer for {proposer_name} (Tag: {tag})')
            plt.ylabel('Actual (Ground Truth)')
            plt.xlabel('Predicted')
            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(output_dir, f'confusion_matrix_from_final_answer_{proposer_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved confusion matrix from final answer for {proposer_name} to {save_path}")

    def plot_accuracy_over_iterations(self, tag: str, top_n: int = 1):
        """
        Generates and saves a plot of top-N accuracy vs. iteration for all proposers,
        comparing results from NLL and candidate mechanisms if available.
        """
        # 1. Get accuracy from NLL
        df_nll = self._gather_all_step_data_from_nll(tag)
        if not df_nll.empty:
            df_nll, filled_steps, unfilled_steps = self._fill_all_null_step_from_previous(df_nll, target_step=5)
            df_nll['is_correct'] = (df_nll['rank'] <= top_n).astype(int)
            df_nll['source'] = 'NLL'
            
            total_steps = len(df_nll)
            correct_steps = df_nll['is_correct'].sum()
            all_null_steps = total_steps - df_nll['best_nll'].notna().sum()
            
            print("\n--- NLL Accuracy Calculation Summary ---")
            print(f"{filled_steps} all-null step-5 entries were filled from step 4 for accuracy.")
            print(f"{all_null_steps} steps with all null NLLs remained and were treated as incorrect (accuracy=0).")
            if unfilled_steps:
                print(f"{unfilled_steps} all-null step-5 entries could not be filled because step 4 had no valid NLL.")
            print(f"Total correct Top-{top_n} steps (accuracy=1) from NLL: {correct_steps} out of {total_steps} total steps considered.")
        
        # 2. Get accuracy from candidate_mechanisms
        df_cand, no_cand_steps, correct_cand_steps = self._gather_accuracy_data_from_candidates(tag, top_n)
        if not df_cand.empty:
            df_cand['source'] = 'Candidates'
            print("\n--- Candidate Accuracy Calculation Summary ---")
            print(f"{no_cand_steps} steps without candidate mechanisms were found and treated as incorrect (accuracy=0).")
            print(f"Total correct Top-{top_n} steps (accuracy=1) from candidates: {correct_cand_steps} out of {len(df_cand)} total steps considered.")

        # 3. Combine and plot
        if df_nll.empty and df_cand.empty:
            print("No data available to plot accuracy.")
            return

        combined_df = pd.concat([df_nll, df_cand], ignore_index=True)
        
        output_dir = os.path.join(self.results_root, tag)
        plt.figure(figsize=(12, 8))
        
        # Use style to differentiate between NLL and Candidates
        sns.lineplot(data=combined_df, x='step', y='is_correct', hue='proposer', style='source', marker='o', errorbar='ci')

        if top_n == 1:
            plot_title = f'Accuracy Over Iterations (Tag: {tag})'
            plot_ylabel = 'Accuracy'
            save_filename = 'accuracy_over_iterations.png'
        else:
            plot_title = f'Top-{top_n} Accuracy Over Iterations (Tag: {tag})'
            plot_ylabel = f'Top-{top_n} Accuracy'
            save_filename = f'top_{top_n}_accuracy_over_iterations.png'

        plt.title(plot_title)
        plt.xlabel('Iteration Step')
        plt.ylabel(plot_ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1.05)
        # Ensure x-ticks are integers if steps are available
        if not combined_df.empty and pd.api.types.is_numeric_dtype(combined_df['step']):
             plt.xticks(sorted(combined_df['step'].dropna().unique()))

        plt.legend(title='Proposer / Source')
        plt.tight_layout()

        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved combined {plot_ylabel.lower()} plot to {save_path}")

    def plot_rank_over_iterations(self, tag: str):
        """
        Generates and saves a plot of the ground truth model's rank vs. iteration,
        comparing NLL-based rank and candidate-based rank.
        """
        # 1. Get rank from NLL
        df_nll = self._gather_all_step_data_from_nll(tag)
        all_null_steps = 0
        if not df_nll.empty:
            df_nll = df_nll[df_nll['best_nll'].notna()] # Exclude null steps from rank plot
            df_nll['source'] = 'NLL'
            all_null_steps = len(self._gather_all_step_data_from_nll(tag)) - len(df_nll)
            print("\n--- NLL Rank Calculation Summary ---")
            print(f"{all_null_steps} steps with all null NLLs were found and excluded from the rank calculation.")

        # 2. Get rank from candidate_mechanisms
        df_cand, no_cand_steps = self._gather_rank_data_from_candidates(tag)
        if not df_cand.empty:
            df_cand['source'] = 'Candidates'
            print("\n--- Candidate Rank Calculation Summary ---")
            print(f"{no_cand_steps} steps without candidate mechanisms were excluded from the rank calculation.")

        # 3. Combine and plot
        if df_nll.empty and df_cand.empty:
            print("No data available to plot rank.")
            return

        combined_df = pd.concat([df_nll, df_cand], ignore_index=True)

        output_dir = os.path.join(self.results_root, tag)
        plt.figure(figsize=(12, 8))
        ax = sns.lineplot(data=combined_df, x='step', y='rank', hue='proposer', style='source', marker='o', errorbar='ci')

        plt.title(f'Ground Truth Rank Over Iterations (Tag: {tag})')
        plt.xlabel('Iteration Step')
        plt.ylabel('Rank of Ground Truth Model')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        if not combined_df.empty and pd.api.types.is_numeric_dtype(combined_df['step']):
            plt.xticks(sorted(combined_df['step'].dropna().unique()))
        plt.legend(title='Proposer / Source')
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'rank_over_iterations.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved combined rank plot to {save_path}")

    def _gather_rank_data_from_candidates(self, tag: str) -> Tuple[pd.DataFrame, int]:
        """Gathers rank data from 'candidate_mechanisms'."""
        loaded_proposals = self._load_proposals_by_tag(tag)
        if not loaded_proposals:
            return pd.DataFrame(), 0

        plot_data = []
        steps_without_candidates = 0

        for proposer_name, ground_truth_results in loaded_proposals.items():
            for ground_truth, steps in ground_truth_results.items():
                ground_truth_base = ground_truth.split('_')[0]
                for step, proposal_data in steps.items():
                    candidates = proposal_data.get('candidate_mechanisms')
                    
                    rank = len(self.model_names) + 1 # Default rank if not found
                    if isinstance(candidates, list) and candidates:
                        # Robustly extract model names
                        candidate_models = []
                        if all(isinstance(c, dict) for c in candidates):
                            candidate_models = [c.get('mechanism') for c in candidates]
                        elif all(isinstance(c, str) for c in candidates):
                            candidate_models = candidates

                        # Filter out any None or empty string values
                        candidate_models = [m for m in candidate_models if m]

                        if candidate_models: # Check if we have any valid models after parsing
                            if ground_truth_base in candidate_models:
                                rank = candidate_models.index(ground_truth_base) + 1
                        else: # The format was unexpected or the list was empty after filtering
                            steps_without_candidates += 1
                    else: # `candidates` was not a non-empty list
                        steps_without_candidates += 1
                    
                    plot_data.append({
                        'proposer': proposer_name,
                        'ground_truth': ground_truth,
                        'step': step,
                        'rank': rank,
                        'tag': tag
                    })
        
        return pd.DataFrame(plot_data), steps_without_candidates

    def _gather_accuracy_data_from_candidates(self, tag: str, top_n: int) -> Tuple[pd.DataFrame, int, int]:
        """Gathers accuracy data from 'candidate_mechanisms'."""
        df_cand_rank, steps_without_candidates = self._gather_rank_data_from_candidates(tag)
        if df_cand_rank.empty:
            return pd.DataFrame(), 0, 0
        
        df_cand_rank['is_correct'] = (df_cand_rank['rank'] <= top_n).astype(int)
        correct_steps = df_cand_rank['is_correct'].sum()

        return df_cand_rank, steps_without_candidates, correct_steps

    def plot_rank_histogram(self, tag: str):
        """
        Generates and saves a histogram of the ground truth model's rank in the final iteration.
        Compares ranks derived from NLL and from candidate mechanisms.
        """
        final_step_data = self.get_final_step_metrics(tag)
        if final_step_data.empty:
            print("No NLL data available to plot rank histogram.")
            return

        final_step_data.rename(columns={'rank': 'nll_rank'}, inplace=True)
        
        # Get candidate ranks
        cand_ranks_all_steps, _ = self._gather_rank_data_from_candidates(tag)
        if cand_ranks_all_steps.empty:
            final_cand_ranks = pd.DataFrame(columns=['proposer', 'ground_truth', 'cand_rank'])
        else:
            final_cand_idx = cand_ranks_all_steps.groupby(['proposer', 'ground_truth'])['step'].idxmax()
            final_cand_ranks = cand_ranks_all_steps.loc[final_cand_idx].reset_index()
            final_cand_ranks.rename(columns={'rank': 'cand_rank'}, inplace=True)

        # Merge NLL and Candidate ranks
        merged_df = pd.merge(
            final_step_data[['proposer', 'ground_truth', 'nll_rank']],
            final_cand_ranks[['proposer', 'ground_truth', 'cand_rank']],
            on=['proposer', 'ground_truth'],
            how='outer'
        )

        plot_df = merged_df.melt(
            id_vars=['proposer', 'ground_truth'],
            value_vars=['nll_rank', 'cand_rank'],
            var_name='source',
            value_name='rank'
        )
        plot_df['source'] = plot_df['source'].replace({'nll_rank': 'NLL', 'cand_rank': 'Candidates'})

        if plot_df.empty:
            print("No data available to plot rank histogram.")
            return
            
        output_dir = os.path.join(self.results_root, tag)

        for proposer_name, proposer_df in plot_df.groupby('proposer'):
            plt.figure(figsize=(12, 8))
            
            ax = sns.histplot(data=proposer_df, x='rank', hue='source', multiple='layer', discrete=True, alpha=0.6)
            
            plt.title(f'Histogram of Final Ground Truth Rank for {proposer_name} (Tag: {tag})')
            plt.xlabel('Rank of Ground Truth Model (Final Iteration)')
            plt.ylabel('Count of Experiments')
            plt.grid(True, linestyle='--', alpha=0.6)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # Set explicit ticks to ensure all ranks up to 21 are potentially visible
            plt.xticks(range(1, len(self.model_names) + 2)) 

            plt.legend(title='Source')
            plt.tight_layout()

            save_path = os.path.join(output_dir, f'rank_histogram_{proposer_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved rank histogram for {proposer_name} to {save_path}")

    def plot_nll_difference_over_iterations(self, tag: str):
        """
        Generates a plot of the NLL difference between the ground truth model
        and the best model over iterations.
        """
        df = self._gather_all_step_data_from_nll(tag)
        all_null_steps = 0
        if not df.empty:
             all_null_steps = df['best_nll'].isna().sum()
             df.dropna(subset=['nll_diff'], inplace=True)
        
        if df.empty:
            print("No data available to plot NLL difference.")
            return

        print("\n--- NLL Difference Calculation Summary ---")
        print(f"{all_null_steps} steps with all null NLLs were found and excluded from the calculation.")

        output_dir = os.path.join(self.results_root, tag)
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='step', y='nll_diff', hue='proposer', marker='o', errorbar='ci')

        plt.title(f'NLL Difference (True - Best) Over Iterations (Tag: {tag})')
        plt.xlabel('Iteration Step')
        plt.ylabel('NLL(True) - NLL(Best)')
        plt.grid(True, linestyle='--', alpha=0.6)
        if not df.empty and pd.api.types.is_numeric_dtype(df['step']):
            plt.xticks(sorted(df['step'].dropna().unique()))
        plt.legend(title='Proposer')
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'nll_difference_over_iterations.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved NLL difference plot to {save_path}")

    def plot_nll_range_over_iterations(self, tag: str):
        """
        Generates a plot of the NLL difference between the worst and best model
        over iterations.
        """
        loaded_results = self._load_results_by_tag(tag, full_results=True)
        if not loaded_results:
            print("No results found to plot NLL range.")
            return

        output_dir = os.path.join(self.results_root, tag)
        plot_data = []

        for proposer_name, ground_truth_results in loaded_results.items():
            for ground_truth, steps in ground_truth_results.items():
                for step, results in steps.items():
                    valid_results = {model: nll for model, nll in results.items() if nll is not None and nll != 'inf'}
                    
                    if len(valid_results) < 2:
                        nll_range = 0.0
                    else:
                        best_nll = min(valid_results.values())
                        worst_nll = max(valid_results.values())
                        nll_range = worst_nll - best_nll
                    
                    plot_data.append({
                        'proposer': proposer_name,
                        'step': step,
                        'nll_range': nll_range
                    })

        if not plot_data:
            print("No data available to plot NLL range.")
            return

        df = pd.DataFrame(plot_data)

        # Plotting
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='step', y='nll_range', hue='proposer', marker='o', errorbar='ci')

        plt.title(f'NLL Range (Worst - Best) Over Iterations (Tag: {tag})')
        plt.xlabel('Iteration Step')
        plt.ylabel('NLL(Worst) - NLL(Best)')
        plt.grid(True, linestyle='--', alpha=0.6)
        if not df.empty and pd.api.types.is_numeric_dtype(df['step']):
            plt.xticks(sorted(df['step'].dropna().unique()))
        plt.legend(title='Proposer')
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'nll_range_over_iterations.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved NLL range plot to {save_path}")

    def plot_entropy_over_iterations(self, tag: str):
        """
        Generates a plot of the Shannon entropy of the model posterior
        distribution over iterations.
        """
        df = self._gather_all_step_data_from_nll(tag)
        all_null_steps = 0
        if not df.empty:
            df, filled_steps, unfilled_steps = self._fill_all_null_step_from_previous(df, target_step=5)
            all_null_steps = df['entropy'].isna().sum()
            df.dropna(subset=['entropy'], inplace=True)

        if df.empty:
            print("No data available to plot entropy.")
            return

        print("\n--- Entropy Calculation Summary ---")
        print(f"{filled_steps} all-null step-5 entries were filled from step 4 for entropy.")
        print(f"{all_null_steps} steps with all null NLLs remained and were excluded from the calculation.")
        if unfilled_steps:
            print(f"{unfilled_steps} all-null step-5 entries could not be filled because step 4 had no valid NLL.")

        output_dir = os.path.join(self.results_root, tag)
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='step', y='entropy', hue='proposer', marker='o', errorbar='ci')

        plt.title(f'Shannon Entropy of Model Posterior Over Iterations (Tag: {tag})')
        plt.xlabel('Iteration Step')
        plt.ylabel('Shannon Entropy (nats)')
        plt.grid(True, linestyle='--', alpha=0.6)
        if not df.empty and pd.api.types.is_numeric_dtype(df['step']):
            plt.xticks(sorted(df['step'].dropna().unique()))
        plt.legend(title='Proposer')
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'entropy_over_iterations.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved entropy plot to {save_path}")


    def plot_success_failure_trajectories(self, tag: str):
        """
        Generates per-proposer plots overlaying stepwise best-NLL, entropy, and
        ground-truth rank trajectories, split by whether the run ended with the
        correct final-step prediction.
        """
        all_steps_df = self._gather_all_step_data_from_nll(tag)
        final_step_df = self.get_final_step_metrics(tag)

        if all_steps_df.empty or final_step_df.empty:
            print("No data available to plot success/failure trajectories.")
            return

        outcome_df = final_step_df[["proposer", "ground_truth", "accuracy"]].copy()
        outcome_df["outcome"] = outcome_df["accuracy"].map({1: "Success", 0: "Failure"})

        merged_df = all_steps_df.merge(
            outcome_df[["proposer", "ground_truth", "outcome"]],
            on=["proposer", "ground_truth"],
            how="inner",
        )

        if merged_df.empty:
            print("No completed runs with valid final-step fits were found for success/failure trajectory plots.")
            return

        output_dir = os.path.join(self.results_root, tag)
        palette = {"Success": "#1b9e77", "Failure": "#d95f02"}

        for proposer_name, proposer_df in merged_df.groupby("proposer"):
            if proposer_df.empty:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True)

            best_nll_df = proposer_df.dropna(subset=["best_nll"]).copy()
            entropy_df = proposer_df.dropna(subset=["entropy"]).copy()
            rank_df = proposer_df.dropna(subset=["rank"]).copy()

            if best_nll_df.empty and entropy_df.empty and rank_df.empty:
                plt.close(fig)
                continue

            for outcome, color in palette.items():
                outcome_best_nll = best_nll_df[best_nll_df["outcome"] == outcome]
                outcome_entropy = entropy_df[entropy_df["outcome"] == outcome]
                outcome_rank = rank_df[rank_df["outcome"] == outcome]

                for _, sample_df in outcome_best_nll.groupby("ground_truth"):
                    axes[0].plot(
                        sample_df["step"],
                        sample_df["best_nll"],
                        color=color,
                        alpha=0.18,
                        linewidth=1.0,
                    )

                for _, sample_df in outcome_entropy.groupby("ground_truth"):
                    axes[1].plot(
                        sample_df["step"],
                        sample_df["entropy"],
                        color=color,
                        alpha=0.18,
                        linewidth=1.0,
                    )

                for _, sample_df in outcome_rank.groupby("ground_truth"):
                    axes[2].plot(
                        sample_df["step"],
                        sample_df["rank"],
                        color=color,
                        alpha=0.18,
                        linewidth=1.0,
                    )

            if not best_nll_df.empty:
                sns.lineplot(
                    data=best_nll_df,
                    x="step",
                    y="best_nll",
                    hue="outcome",
                    palette=palette,
                    estimator="mean",
                    errorbar=("ci", 95),
                    linewidth=2.5,
                    ax=axes[0],
                    legend=False,
                )

            if not entropy_df.empty:
                sns.lineplot(
                    data=entropy_df,
                    x="step",
                    y="entropy",
                    hue="outcome",
                    palette=palette,
                    estimator="mean",
                    errorbar=("ci", 95),
                    linewidth=2.5,
                    ax=axes[1],
                    legend=False,
                )

            if not rank_df.empty:
                sns.lineplot(
                    data=rank_df,
                    x="step",
                    y="rank",
                    hue="outcome",
                    palette=palette,
                    estimator="mean",
                    errorbar=("ci", 95),
                    linewidth=2.5,
                    ax=axes[2],
                    legend=True,
                )
                axes[2].invert_yaxis()
                axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))

            axes[0].set_title("Best-NLL Trajectories")
            axes[0].set_xlabel("Iteration Step")
            axes[0].set_ylabel("Minimum NLL Across Models")
            axes[0].grid(True, linestyle="--", alpha=0.6)

            axes[1].set_title("Entropy Trajectories")
            axes[1].set_xlabel("Iteration Step")
            axes[1].set_ylabel("Shannon Entropy (nats)")
            axes[1].grid(True, linestyle="--", alpha=0.6)

            axes[2].set_title("Ground-Truth Rank Trajectories")
            axes[2].set_xlabel("Iteration Step")
            axes[2].set_ylabel("Rank of Ground Truth Model")
            axes[2].grid(True, linestyle="--", alpha=0.6)

            unique_steps = sorted(proposer_df["step"].dropna().unique())
            if unique_steps:
                for ax in axes:
                    ax.set_xticks(unique_steps)

            handles, labels = axes[2].get_legend_handles_labels()
            if handles:
                axes[2].legend(handles, labels, title="Final Outcome")

            fig.suptitle(f"Success vs Failure Trajectories for {proposer_name} (Tag: {tag})")
            fig.tight_layout()

            save_path = os.path.join(output_dir, f"success_failure_trajectories_{proposer_name}.png")
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Saved success/failure trajectory plot to {save_path}")


    def plot_top_k_membership_and_cooccurrence(self, tag: str, top_k: int = 3):
        """
        Generates per-proposer visualizations of top-k model clusters split by
        final success/failure outcome: a membership heatmap and a model
        co-occurrence matrix for each outcome group.
        """
        loaded_results = self._load_results_by_tag(tag, full_results=True)
        final_step_df = self.get_final_step_metrics(tag)

        if not loaded_results or final_step_df.empty:
            print("No data available to plot top-k membership/co-occurrence.")
            return

        outcome_df = final_step_df[["proposer", "ground_truth", "accuracy"]].copy()
        outcome_df["outcome"] = outcome_df["accuracy"].map({1: "Success", 0: "Failure"})
        outcome_lookup = {
            (row["proposer"], row["ground_truth"]): row["outcome"]
            for _, row in outcome_df.iterrows()
        }

        output_dir = os.path.join(self.results_root, tag)

        for proposer_name, ground_truth_results in loaded_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            has_any_panel = False

            for col_idx, outcome in enumerate(["Success", "Failure"]):
                membership_rows = []
                membership_labels = []
                cooccurrence = pd.DataFrame(0, index=self.model_names, columns=self.model_names, dtype=int)

                proposer_results = ground_truth_results.items()
                for ground_truth, steps in proposer_results:
                    if outcome_lookup.get((proposer_name, ground_truth)) != outcome:
                        continue

                    for step in sorted(steps.keys()):
                        results = steps[step]
                        valid_results = {
                            model: nll for model, nll in results.items()
                            if isinstance(nll, (int, float)) and np.isfinite(nll)
                        }
                        if not valid_results:
                            continue

                        top_models = [
                            model for model, _ in sorted(valid_results.items(), key=lambda item: item[1])[:top_k]
                        ]
                        row = [1 if model in top_models else 0 for model in self.model_names]
                        membership_rows.append(row)
                        membership_labels.append(f"{ground_truth}:s{step}")

                        for model_a in top_models:
                            for model_b in top_models:
                                cooccurrence.loc[model_a, model_b] += 1

                membership_ax = axes[0, col_idx]
                cooccurrence_ax = axes[1, col_idx]

                if membership_rows:
                    has_any_panel = True
                    membership_df = pd.DataFrame(
                        membership_rows,
                        index=membership_labels,
                        columns=self.model_names,
                    )
                    sns.heatmap(
                        membership_df,
                        ax=membership_ax,
                        cmap=sns.color_palette(["#f7f7f7", "#2166ac"], as_cmap=True),
                        cbar=False,
                        vmin=0,
                        vmax=1,
                        linewidths=0.1,
                        linecolor="#dddddd",
                    )
                    membership_ax.set_title(f"{outcome}: Top-{top_k} Membership")
                    membership_ax.set_xlabel("Model")
                    membership_ax.set_ylabel("Sample:Step")
                else:
                    membership_ax.axis("off")
                    membership_ax.text(0.5, 0.5, f"No {outcome.lower()} runs", ha="center", va="center")

                if membership_rows:
                    sns.heatmap(
                        cooccurrence,
                        ax=cooccurrence_ax,
                        cmap="mako",
                        square=True,
                        cbar=True,
                    )
                    cooccurrence_ax.set_title(f"{outcome}: Top-{top_k} Co-occurrence")
                    cooccurrence_ax.set_xlabel("Model")
                    cooccurrence_ax.set_ylabel("Model")
                else:
                    cooccurrence_ax.axis("off")
                    cooccurrence_ax.text(0.5, 0.5, f"No {outcome.lower()} runs", ha="center", va="center")

            if not has_any_panel:
                plt.close(fig)
                continue

            fig.suptitle(f"Top-{top_k} Model Clusters for {proposer_name} (Tag: {tag})")
            fig.tight_layout()
            save_path = os.path.join(output_dir, f"top_{top_k}_membership_cooccurrence_{proposer_name}.png")
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Saved top-{top_k} membership/co-occurrence plot to {save_path}")


    def plot_top_k_cumulative_mass_success_failure(self, tag: str, max_k: int = 3):
        """
        Generates per-proposer plots of top-k cumulative posterior mass over
        iterations, split into success and failure outcomes.
        """
        loaded_results = self._load_results_by_tag(tag, full_results=True)
        final_step_df = self.get_final_step_metrics(tag)

        if not loaded_results or final_step_df.empty:
            print("No data available to plot top-k cumulative mass.")
            return

        outcome_df = final_step_df[["proposer", "ground_truth", "accuracy"]].copy()
        outcome_df["outcome"] = outcome_df["accuracy"].map({1: "Success", 0: "Failure"})
        outcome_lookup = {
            (row["proposer"], row["ground_truth"]): row["outcome"]
            for _, row in outcome_df.iterrows()
        }

        output_dir = os.path.join(self.results_root, tag)
        palette = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

        for proposer_name, ground_truth_results in loaded_results.items():
            plot_rows = []
            for ground_truth, steps in ground_truth_results.items():
                outcome = outcome_lookup.get((proposer_name, ground_truth))
                if outcome is None:
                    continue

                for step, results in steps.items():
                    valid_results = {
                        model: nll for model, nll in results.items()
                        if isinstance(nll, (int, float)) and np.isfinite(nll)
                    }
                    if not valid_results:
                        continue

                    nlls = np.array(list(valid_results.values()), dtype=float)
                    log_likelihoods = -nlls
                    log_probs = log_likelihoods - logsumexp(log_likelihoods)
                    probs = np.sort(np.exp(log_probs))[::-1]

                    cumulative_mass = np.cumsum(probs)
                    limit = min(max_k, len(cumulative_mass))
                    for k in range(1, limit + 1):
                        plot_rows.append({
                            "proposer": proposer_name,
                            "ground_truth": ground_truth,
                            "step": step,
                            "outcome": outcome,
                            "k": k,
                            "top_k_mass": float(cumulative_mass[k - 1]),
                        })

            if not plot_rows:
                continue

            df = pd.DataFrame(plot_rows)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

            for ax, outcome in zip(axes, ["Success", "Failure"]):
                outcome_df_plot = df[df["outcome"] == outcome]
                if outcome_df_plot.empty:
                    ax.axis("off")
                    ax.text(0.5, 0.5, f"No {outcome.lower()} runs", ha="center", va="center")
                    continue

                sns.lineplot(
                    data=outcome_df_plot,
                    x="step",
                    y="top_k_mass",
                    hue="k",
                    palette=palette,
                    marker="o",
                    estimator="mean",
                    errorbar=("ci", 95),
                    linewidth=2.5,
                    ax=ax,
                )
                ax.set_title(f"{outcome}")
                ax.set_xlabel("Iteration Step")
                ax.set_ylabel("Top-k Cumulative Mass")
                ax.set_ylim(0.0, 1.02)
                ax.grid(True, linestyle="--", alpha=0.6)
                unique_steps = sorted(outcome_df_plot["step"].dropna().unique())
                if unique_steps:
                    ax.set_xticks(unique_steps)
                ax.legend(title="k")

            fig.suptitle(f"Top-k Cumulative Mass for {proposer_name} (Tag: {tag})")
            fig.tight_layout()
            save_path = os.path.join(output_dir, f"top_k_cumulative_mass_{proposer_name}.png")
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Saved top-k cumulative mass plot to {save_path}")
