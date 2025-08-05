# FILE: scripts/utils/generate_tasks.py
# [DEFINITIVE VERSION]
# This is the definitive task generator, designed to be compatible with the
# provided legacy config.py and the modern, robust launch workflow.

import argparse
import itertools
import json
import os
import sys
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    from config import EXPERIMENTS
except ImportError as e:
    print(
        f"FATAL: Could not import EXPERIMENTS from src/config.py. Error: {e}",
        file=sys.stderr,
    )
    print(
        "Please ensure src/config.py exists and has no syntax errors.", file=sys.stderr
    )
    sys.exit(1)


def generate_task_id(params: dict) -> str:
    id_defining_params = {
        k: v
        for k, v in params.items()
        if k
        not in [
            "max_steps",
            "campaign_id",
            "experiment_name",
            "task_id",
            "warmup_time",
            "num_samples",
            "sample_interval",
        ]
    }
    # Convert lists/tuples to a canonical string representation for stable hashing
    for key, value in id_defining_params.items():
        if isinstance(value, (list, tuple)):
            id_defining_params[key] = str(value)

    sorted_items = sorted(id_defining_params.items())
    param_str = "&".join([f"{k}={v}" for k, v in sorted_items])
    return str(abs(hash(param_str)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation tasks for a specified experiment."
    )
    parser.add_argument(
        "experiment_name",
        choices=EXPERIMENTS.keys(),
        help="The name of the experiment to run.",
    )
    args = parser.parse_args()

    try:
        config = EXPERIMENTS[args.experiment_name]
        campaign_id = config["CAMPAIGN_ID"]
        run_mode = config["run_mode"]
        param_grid = config["PARAM_GRID"]
    except KeyError as e:
        print(
            f"FATAL: Missing key {e} in config for '{args.experiment_name}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"--- Generating Tasks for Campaign: {campaign_id} ---")

    data_dir = os.path.join(project_root, "data", campaign_id)
    analysis_dir = os.path.join(data_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    all_possible_tasks = []
    for set_id, sim_set in config["SIM_SETS"].items():
        base_params = sim_set.get("base_params", {}).copy()
        num_replicates = base_params.pop("num_replicates", 1)
        grid_param_names = list(sim_set["grid_params"].keys())
        grid_list_names = [sim_set["grid_params"][k] for k in grid_param_names]
        grid_value_arrays = [param_grid[list_name] for list_name in grid_list_names]
        param_combinations = itertools.product(*grid_value_arrays)

        for combo in param_combinations:
            instance_params = dict(zip(grid_param_names, combo))
            for j in range(num_replicates):
                task_params = {**base_params, **instance_params}
                task_params["replicate"] = j
                task_params["run_mode"] = run_mode
                task_params["sim_set"] = set_id
                task_params["campaign_id"] = campaign_id
                task_params["experiment_name"] = args.experiment_name

                # [FIXED] Robustly resolve parameter aliases.
                # A parameter's string value can refer to another parameter's key
                # or a key in the experiment's PARAM_GRID. We loop to resolve
                # chained references (e.g., A -> B -> value) and use list() to
                # iterate over a copy for safe modification.
                for _ in range(5):  # Limit iterations to prevent infinite loops
                    resolved_something = False
                    for p_key, p_val in list(task_params.items()):
                        if not isinstance(p_val, str):
                            continue

                        # Priority 1: Resolve from another parameter in the task
                        if p_val in task_params and task_params[p_val] is not p_val:
                            task_params[p_key] = task_params[p_val]
                            resolved_something = True
                        # Priority 2: Resolve from the main PARAM_GRID
                        elif p_val in param_grid:
                            task_params[p_key] = param_grid[p_val]
                            resolved_something = True

                    if not resolved_something:
                        break  # Exit if a pass completes with no changes

                task_params["task_id"] = generate_task_id(task_params)
                all_possible_tasks.append(task_params)

    print(
        f"Generated a universe of {len(all_possible_tasks)} unique tasks for this experiment."
    )

    total_task_file = os.path.join(data_dir, f"{campaign_id}_total_tasks.txt")
    with open(total_task_file, "w") as f:
        f.write(str(len(all_possible_tasks)))

    master_summary_file = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )
    completed_task_ids = set()
    if os.path.exists(master_summary_file):
        try:
            df_completed = pd.read_csv(master_summary_file, usecols=["task_id"])
            completed_task_ids = set(df_completed["task_id"].astype(str))
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            pass

    print(
        f"Found {len(completed_task_ids)} completed tasks in the master summary file."
    )
    missing_tasks = [
        t for t in all_possible_tasks if t["task_id"] not in completed_task_ids
    ]

    resume_task_file = os.path.join(data_dir, f"{campaign_id}_resume_tasks.txt")
    if not missing_tasks:
        print(f"\nAll tasks for campaign '{campaign_id}' are complete.")
        open(resume_task_file, "w").close()
        return

    missing_tasks.sort(key=lambda t: t["task_id"])
    with open(resume_task_file, "w") as f:
        for task in missing_tasks:
            f.write(json.dumps(task) + "\n")

    print(f"\nGenerated a new list of {len(missing_tasks)} MISSING tasks.")
    print(f"Output file for Slurm: {resume_task_file}")


if __name__ == "__main__":
    main()
