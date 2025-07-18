# scripts/generate_calibration_tasks.py
# Generates the task list for the calibration campaign.

import argparse
import itertools
import json
import os
import re
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

# Import from the dedicated calibration config
try:
    from config_calibration import (
        CAMPAIGN_ID,
        SIM_PARAMS_CALIBRATION,
        PARAM_GRID_CALIBRATION,
    )
except ImportError:
    print("Error: Could not import from src/config_calibration.py.")
    sys.exit(1)


def main():
    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    task_list_path = os.path.join(project_root, "data", f"{CAMPAIGN_ID}_task_list.txt")

    parser = argparse.ArgumentParser(
        description="Generate tasks for the calibration sweep."
    )
    parser.add_argument("--outfile", type=str, default=task_list_path)
    parser.add_argument(
        "--clean", action="store_true", help="Delete old/invalid result files."
    )
    args = parser.parse_args()

    print(f"--- Generating Tasks for Calibration Campaign: {CAMPAIGN_ID} ---")

    all_possible_tasks = []
    num_replicates = SIM_PARAMS_CALIBRATION["num_replicates"]
    base_params = {
        k: v for k, v in SIM_PARAMS_CALIBRATION.items() if k != "num_replicates"
    }

    param_combinations = [
        dict(zip(PARAM_GRID_CALIBRATION.keys(), v))
        for v in itertools.product(*PARAM_GRID_CALIBRATION.values())
    ]

    for i, params in enumerate(param_combinations):
        for j in range(num_replicates):
            task_params = {**params, **base_params}
            # Create a unique, descriptive task ID
            task_params["task_id"] = f"calib_bm{params['b_m']:.3f}_rep{j:03d}"
            all_possible_tasks.append(task_params)

    valid_task_ids = {task["task_id"] for task in all_possible_tasks}
    print(f"Generated a universe of {len(valid_task_ids)} unique tasks.")

    completed_task_ids = set()
    os.makedirs(results_dir, exist_ok=True)
    for filename in os.listdir(results_dir):
        if filename.startswith("result_") and filename.endswith(".json"):
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
        print("\nAll calibration tasks are complete.")
        open(args.outfile, "w").close()
        return

    with open(args.outfile, "w") as f:
        for task in missing_tasks:
            f.write(json.dumps(task) + "\n")
    print(f"\nGenerated a new list of {len(missing_tasks)} MISSING tasks.")
    print(f"Output file for Slurm: {args.outfile}")


if __name__ == "__main__":
    main()
