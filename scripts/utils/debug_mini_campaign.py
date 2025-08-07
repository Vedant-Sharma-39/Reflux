# FILE: scripts/utils/debug_mini_campaign.py
#
# Runs a small, local "mini-campaign" for a given experiment to test the
# full data generation pipeline. This version is corrected to be consistent
# with the main workflow scripts.

import argparse
import itertools
import json
import os
import sys
import subprocess
import shutil
from tqdm import tqdm
import numpy as np

# --- Setup Project Root Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import from the CORRECT, consistent locations ---
# CHANGED: Import directly from the single source of truth: src/config.py
from src.config import EXPERIMENTS, PARAM_GRID

# CHANGED: Import the corrected functions from the real generation script
from scripts.utils.generate_tasks import generate_task_id, resolve_parameters


# --- Define Parameters for a Fast Local Run ---
DEBUG_OVERRIDES = {
    "num_replicates": 2,  # Run a couple of replicates to test stats
    "max_steps": 50000,
    "total_run_time": 200.0,
    "warmup_time": 50.0,
    "num_samples": 10,
    "sample_interval": 5.0,
    "length": 1024,
    "width": 128,  # Override width for speed as well
}


def generate_mini_task_list(experiment_name: str, points_per_param: int) -> list:
    """
    Generates a small, representative list of tasks for a given experiment.
    """
    # CHANGED: Get config directly from the imported EXPERIMENTS dictionary
    config = EXPERIMENTS[experiment_name]
    mini_tasks = []

    print(
        f"\n[INFO] Generating a mini-grid with ~{points_per_param} point(s) per parameter..."
    )

    for set_id, sim_set in config.get("sim_sets", {}).items():
        base_params = sim_set.get("base_params", {}).copy()
        grid_param_keys = list(sim_set["grid_params"].keys())

        mini_grid_value_arrays = []
        for key in grid_param_keys:
            grid_list_name = sim_set["grid_params"][key]
            # CHANGED: Get grid values from the imported PARAM_GRID dictionary
            full_list = PARAM_GRID[grid_list_name]

            if len(full_list) <= points_per_param:
                mini_grid_value_arrays.append(full_list)
            else:
                indices = np.linspace(
                    0, len(full_list) - 1, points_per_param, dtype=int
                )
                mini_grid_value_arrays.append([full_list[i] for i in indices])

        for combo in itertools.product(*mini_grid_value_arrays):
            instance_params = dict(zip(grid_param_keys, combo))

            # Replicate the logic from the main generator script
            task_params = {
                **base_params,
                **instance_params,
                "run_mode": config["run_mode"],
                "campaign_id": config["campaign_id"] + "_debug",
            }
            # CHANGED: Removed "experiment_name" from params to match main generator

            # Apply debug overrides for a fast run
            task_params.update(DEBUG_OVERRIDES)

            # CHANGED: Call the corrected resolve_parameters function (takes 1 arg)
            resolved_params = resolve_parameters(task_params)
            resolved_params["task_id"] = generate_task_id(resolved_params)
            mini_tasks.append(resolved_params)

    return mini_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Run a local mini-campaign for debugging."
    )
    parser.add_argument(
        "experiment_name", nargs="?", default=None, choices=list(EXPERIMENTS.keys())
    )
    parser.add_argument(
        "--points",
        type=int,
        default=2,
        help="Number of points to sample per parameter grid.",
    )
    args = parser.parse_args()

    # --- Interactive Experiment Selection ---
    experiment_name = args.experiment_name
    if not experiment_name:
        print("Please choose an experiment to debug:")
        exp_list = list(EXPERIMENTS.keys())
        for i, name in enumerate(exp_list):
            print(f"  [{i+1}] {name}")
        try:
            choice = int(input("Enter number: ")) - 1
            experiment_name = exp_list[choice]
        except (ValueError, IndexError):
            sys.exit("Invalid choice. Exiting.")
        print()

    # --- Setup Debug Environment ---
    # CHANGED: Get config directly from the imported dictionary
    config = EXPERIMENTS[experiment_name]
    debug_campaign_id = f"{config['campaign_id']}_debug"

    print(f"--- Starting Mini-Campaign Debug for: {experiment_name} ---")
    print(f"--- Debug Campaign ID: {debug_campaign_id} ---")

    debug_data_dir = os.path.join(project_root, "data", debug_campaign_id)
    if os.path.exists(debug_data_dir):
        print(f"[WARN] Deleting previous debug data in {debug_data_dir}")
        shutil.rmtree(debug_data_dir)

    raw_output_dir = os.path.join(debug_data_dir, "raw")
    os.makedirs(raw_output_dir, exist_ok=True)

    # --- Generate Mini Master Task List ---
    tasks = generate_mini_task_list(experiment_name, args.points)
    master_task_file = os.path.join(
        debug_data_dir, f"{debug_campaign_id}_master_tasks.jsonl"
    )
    with open(master_task_file, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"[INFO] Generated {len(tasks)} debug tasks and created a mini master list.")

    # --- Run the Mini-Campaign ---
    worker_script = os.path.join("src", "worker.py")  # Relative path is fine with cwd

    for params in tqdm(tasks, desc="Running debug tasks"):
        params_json = json.dumps(params)

        # IMPROVEMENT: Use cwd=project_root for maximum robustness, mirroring hpc_manager.sh
        process = subprocess.run(
            [
                sys.executable,
                worker_script,
                "--params",
                params_json,
                "--output-dir",
                raw_output_dir,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=project_root,  # This is more robust than setting ENV
        )

        if process.returncode != 0:
            tqdm.write(f"\n[ERROR] Worker failed for task_id {params.get('task_id')}.")
            tqdm.write(f"   Stderr: {process.stderr.strip()}")

    print(f"\n[SUCCESS] Mini-campaign finished.")
    print(f"Raw debug data generated in: {raw_output_dir}")

    # CHANGED: Provide correct, helpful, and copy-pasteable next steps
    print("\n--- Next Steps ---")
    consolidate_script = os.path.join("scripts", "utils", "consolidate_data.py")
    print("1. To test data consolidation, run this command from the project root:")
    print(f"   python3 {consolidate_script} {debug_campaign_id}\n")
    print(
        "2. After consolidation, you can test your plotting scripts on the debug data."
    )


if __name__ == "__main__":
    main()
