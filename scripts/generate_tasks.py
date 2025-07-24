# FILE: scripts/generate_tasks.py
# A single, unified script to generate tasks for ANY experiment.
# [CORRECTED to match legacy filename format]

import argparse
import itertools
import json
import os
import re
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    from config import EXPERIMENTS
except ImportError:
    print("FATAL: Could not import EXPERIMENTS from src/config.py.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation tasks for a specified experiment."
    )
    parser.add_argument(
        "experiment_name",
        choices=EXPERIMENTS.keys(),
        help="The name of the experiment to run, as defined in src/config.py.",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete old/invalid result files."
    )
    args = parser.parse_args()

    # --- 1. Load configuration for the chosen experiment ---
    try:
        config = EXPERIMENTS[args.experiment_name]
        CAMPAIGN_ID = config["CAMPAIGN_ID"]
        RUN_MODE = config["run_mode"]
    except KeyError as e:
        print(
            f"FATAL: Missing key {e} in configuration for experiment '{args.experiment_name}'."
        )
        sys.exit(1)

    print(f"--- Generating Tasks for Campaign: {CAMPAIGN_ID} ---")

    # --- 2. Generate all possible tasks for this experiment ---
    all_possible_tasks = []
    for set_id, sim_set in config["SIM_SETS"].items():
        base_params = sim_set["base_params"].copy()
        num_replicates = base_params.pop("num_replicates", 1)

        grid_keys = list(sim_set["grid_params"].keys())
        grid_values = [config["PARAM_GRID"][v] for v in sim_set["grid_params"].values()]
        param_combinations = [
            dict(zip(grid_keys, v)) for v in itertools.product(*grid_values)
        ]

        for params in param_combinations:
            for j in range(num_replicates):
                task_params = {**base_params, **params}
                task_params["run_mode"] = RUN_MODE
                # [FIX] If a param value (like initial_patch_size) is a string,
                # evaluate it in the context of the task parameters.
                for p_key, p_val in task_params.items():
                    if isinstance(p_val, str) and p_key != "run_mode":
                        try:
                            # This allows "width" // 2 to work.
                            task_params[p_key] = eval(p_val, {}, task_params)
                        except (NameError, TypeError):
                            # Not an expression, just a string. Keep it.
                            pass

                # --- [CORRECTED] Task ID Generation Logic ---
                param_parts = []
                # Use the original grid keys for a stable and predictable order
                for k in grid_keys:
                    v = params[k]
                    key_no_underscore = k.replace("_", "")
                    if isinstance(v, float):
                        param_parts.append(f"{key_no_underscore}{v:.3f}")
                    else:
                        param_parts.append(f"{key_no_underscore}{v}")
                param_str = "_".join(param_parts)

                # Special handling for calibration to match the 'calib_' prefix
                if set_id == "main" and args.experiment_name.startswith("calibration"):
                    task_params["task_id"] = f"calib_{param_str}_rep{j:03d}"
                else:
                    task_params["task_id"] = f"{set_id}_{param_str}_rep{j:03d}"

                # Add replicate ID to the parameters themselves
                task_params["replicate_id"] = j
                all_possible_tasks.append(task_params)

    valid_task_ids = {task["task_id"] for task in all_possible_tasks}
    print(
        f"Generated a universe of {len(valid_task_ids)} unique tasks for this experiment."
    )

    # --- 3. Check for completed tasks (this part is now correct) ---
    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    task_list_path = os.path.join(project_root, "data", f"{CAMPAIGN_ID}_task_list.txt")

    completed_task_ids = set()
    os.makedirs(results_dir, exist_ok=True)
    for filename in os.listdir(results_dir):
        if filename.startswith("result_") and filename.endswith(".json"):
            # This extracts 'calib_bm0.000_rep000' from 'result_calib_bm0.000_rep000.json'
            match = re.search(r"result_(.*)\.json", filename)
            if match:
                task_id = match.group(1)
                if task_id in valid_task_ids:
                    completed_task_ids.add(task_id)
                elif args.clean:
                    print(f"  - DELETING invalid result file: {filename}")
                    os.remove(os.path.join(results_dir, filename))

    print(f"Found {len(completed_task_ids)} valid, completed tasks.")
    missing_tasks = [
        t for t in all_possible_tasks if t["task_id"] not in completed_task_ids
    ]

    if not missing_tasks:
        print(f"\nAll tasks for campaign '{CAMPAIGN_ID}' are complete.")
        # Ensure the task file is empty if all tasks are done
        open(task_list_path, "w").close()
        return

    # Sort tasks by ID to make the task list deterministic
    missing_tasks.sort(key=lambda t: t["task_id"])

    with open(task_list_path, "w") as f:
        for task in missing_tasks:
            f.write(json.dumps(task) + "\n")

    print(f"\nGenerated a new list of {len(missing_tasks)} MISSING tasks.")
    print(f"Output file for Slurm: {task_list_path}")


if __name__ == "__main__":
    main()
