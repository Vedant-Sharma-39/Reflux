# scripts/generate_p1_definitive_tasks.py
# Refactored to import all experimental parameters from a single source of truth.

import argparse
import itertools
import json
import os
import re
import sys

# Add project root to path to allow importing from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

# Import the single source of truth
try:
    from config import (
        CAMPAIGN_ID,
        PARAM_GRID,
        SIM_PARAMS_STD,
        SIM_PARAMS_LONG,
        SCALING_PARAMS,
    )
except ImportError:
    print(
        "Error: Could not import configuration from src/config.py. Make sure the file exists."
    )
    sys.exit(1)


def main():
    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    resume_task_list_path = os.path.join(
        project_root, "data", f"{CAMPAIGN_ID}_task_list.txt"
    )

    parser = argparse.ArgumentParser(
        description="Generate a list of ONLY the missing tasks for the definitive sweep."
    )
    parser.add_argument("--outfile", type=str, default=resume_task_list_path)
    parser.add_argument(
        "--clean", action="store_true", help="Delete old/invalid result files."
    )
    args = parser.parse_args()

    print(f"--- Generating Tasks for Campaign: {CAMPAIGN_ID} ---")
    all_possible_tasks = []

    def generate_tasks_from_set(param_dict, set_id, sim_params):
        num_replicates = sim_params["num_replicates"]
        param_combinations = [
            dict(zip(param_dict.keys(), v))
            for v in itertools.product(*param_dict.values())
        ]
        for i, params in enumerate(param_combinations):
            for j in range(num_replicates):
                task_params = {
                    **params,
                    **{k: v for k, v in sim_params.items() if k != "num_replicates"},
                }
                task_params["task_id"] = f"{set_id}_{i:04d}_{j}"
                all_possible_tasks.append(task_params)

    # Layer 1: High k_total Runs
    generate_tasks_from_set(
        {
            "width": [128],
            "b_m": PARAM_GRID["b_m"],
            "k_total": PARAM_GRID["k_total_high"],
            "phi": PARAM_GRID["phi"],
        },
        "main_high_k",
        SIM_PARAMS_STD,
    )
    # Layer 2: Low k_total Runs
    generate_tasks_from_set(
        {
            "width": [256],
            "b_m": PARAM_GRID["b_m"],
            "k_total": PARAM_GRID["k_total_low"],
            "phi": PARAM_GRID["phi"],
        },
        "main_low_k",
        SIM_PARAMS_LONG,
    )
    # Layer 3: Finite-Size Scaling Runs
    scaling_run_params = {
        k: v for k, v in SCALING_PARAMS.items() if k != "num_replicates"
    }
    generate_tasks_from_set(scaling_run_params, "scaling", SCALING_PARAMS)

    valid_task_ids = {task["task_id"] for task in all_possible_tasks}
    print(f"Generated a universe of {len(valid_task_ids)} unique tasks.")

    completed_task_ids = set()
    os.makedirs(results_dir, exist_ok=True)
    print(f"Scanning for completed tasks in: {results_dir}")
    for filename in os.listdir(results_dir):
        if filename.startswith("result_") and filename.endswith(".json"):
            match = re.search(r"result_(.*)\.json", filename)
            if match:
                task_id = match.group(1)
                if task_id in valid_task_ids:
                    completed_task_ids.add(task_id)
                elif args.clean:
                    print(f"  - DELETING old/invalid result file: {filename}")
                    os.remove(os.path.join(results_dir, filename))
    print(f"Found {len(completed_task_ids)} valid, completed tasks.")

    missing_tasks = [
        task for task in all_possible_tasks if task["task_id"] not in completed_task_ids
    ]
    if not missing_tasks:
        print("\nCongratulations! All definitive tasks are complete.")
        open(args.outfile, "w").close()
        exit(0)

    with open(args.outfile, "w") as f:
        for task in missing_tasks:
            f.write(json.dumps(task) + "\n")
    print(f"\nGenerated a new list of {len(missing_tasks)} MISSING tasks.")
    print(f"Output file for Slurm: {args.outfile}")


if __name__ == "__main__":
    main()
