# This script generates ground truths for the kinetic mechanism benchmark and
# reports validation acceptance statistics.

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import sys
from statistics import mean, median
from typing import Any, Dict, List

import numpy as np

# Ensure the 'bench' package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from bench.oracle.ground_truth_generator import GroundTruthGenerator
from bench.config import GROUND_TRUTH_GENERATOR_CONFIG


def _stable_seed(model_name: str, run_index: int, base_seed: int = 42) -> int:
    model_hash = int(hashlib.md5(model_name.encode()).hexdigest(), 16) & 0xFFFFFFFF
    return (base_seed + model_hash + run_index * 9973) & 0xFFFFFFFF


def _generate_for_model(
    model_name: str,
    config: Dict[str, Any],
    num_truths: int,
) -> Dict[str, Any]:
    model_config = copy.deepcopy(config)
    model_config["num_ground_truths"] = num_truths
    np.random.seed(_stable_seed(model_name, 0))
    generator = GroundTruthGenerator(model_name=model_name, config=model_config)
    summaries: List[Dict[str, Any]] = generator.generate_and_save_ground_truths()

    attempts = [s["attempts_until_success"] for s in summaries]
    passed_fracs = [s["passed_fraction"] for s in summaries]

    return {
        "model_name": model_name,
        "num_ground_truths": num_truths,
        "summaries": summaries,
        "attempts_until_success_mean": mean(attempts) if attempts else None,
        "attempts_until_success_median": median(attempts) if attempts else None,
        "passed_fraction_mean": mean(passed_fracs) if passed_fracs else None,
        "passed_fraction_median": median(passed_fracs) if passed_fracs else None,
    }


def generate_truths() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark ground truths with validation statistics.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of model-parallel worker processes.")
    parser.add_argument("--num-ground-truths", type=int, default=10, help="Number of ground truths to generate per model.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names, e.g. M1 M4 M16.")
    args = parser.parse_args()

    print("--- Starting Ground Truth Generation ---")

    model_dir = GROUND_TRUTH_GENERATOR_CONFIG.get("model_dir", "bench/models")
    model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".cps"))
    model_names = [os.path.splitext(f)[0] for f in model_files]
    if args.models:
        requested = set(args.models)
        model_names = [m for m in model_names if m in requested]

    print(f"Found {len(model_names)} models in '{model_dir}'.")

    gen_config = copy.deepcopy(GROUND_TRUTH_GENERATOR_CONFIG)
    gen_config["num_ground_truths"] = args.num_ground_truths
    os.makedirs(gen_config["output_dir"], exist_ok=True)

    worker_args = [(model_name, gen_config, args.num_ground_truths) for model_name in model_names]

    if args.jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            results = pool.starmap(_generate_for_model, worker_args)
    else:
        results = [_generate_for_model(*worker_arg) for worker_arg in worker_args]

    summary_path = os.path.join(gen_config["output_dir"], "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Generation Summary ---")
    for result in results:
        print(
            f"{result['model_name']}: "
            f"mean attempts={result['attempts_until_success_mean']}, "
            f"median attempts={result['attempts_until_success_median']}, "
            f"mean passed fraction={result['passed_fraction_mean']}"
        )
    print(f"Saved generation summary to {summary_path}")
    print("\n--- Ground Truth Generation Complete ---")


if __name__ == "__main__":
    generate_truths()
