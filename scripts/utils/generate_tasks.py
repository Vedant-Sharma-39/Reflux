# FILE: scripts/utils/generate_tasks.py
# [v_SIMPLE] Generates the definitive master task list for a campaign.
# This script is run ONCE per campaign. It no longer checks for completion status.

import argparse
import itertools
import json
import os
import sys
from typing import Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config_loader import get_experiment_config, EXPERIMENTS


def generate_task_id(params: dict) -> str:
    """Generates a unique, deterministic ID from a dictionary of parameters."""
    id_defining_params = {
        k: v
        for k, v in params.items()
        if k not in ["campaign_id", "experiment_name", "run_mode"]
    }
    for key, value in id_defining_params.items():
        if isinstance(value, dict):
            id_defining_params[key] = json.dumps(value, sort_keys=True)

    sorted_items = sorted(id_defining_params.items())
    param_str = "&".join([f"{k}={v}" for k, v in sorted_items])
    return str(abs(hash(param_str)))


def resolve_parameters(
    params: Dict[str, Any], grid_values: Dict[str, Any]
) -> Dict[str, Any]:
    """Robustly resolves parameter references against other params or the global grid."""
    resolved = params.copy()
    for _ in range(10):  # Safety break
        substitutions_made = False
        for key, value in list(resolved.items()):
            if isinstance(value, str):
                if value in resolved and resolved[value] is not value:
                    resolved[key] = resolved[value]
                    substitutions_made = True
                elif value in grid_values:
                    resolved[key] = grid_values[value]
                    substitutions_made = True
        if not substitutions_made:
            return resolved
    raise RuntimeError(f"Could not resolve all parameter references in: {params}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the master task list for an experiment."
    )
    parser.add_argument(
        "experiment_name",
        choices=EXPERIMENTS.keys(),
        help="The name of the experiment.",
    )
    args = parser.parse_args()

    config = get_experiment_config(args.experiment_name)
    campaign_id = config["campaign_id"]
    param_grid_values = config["PARAM_GRID_VALUES"]
    data_dir = os.path.join(project_root, "data", campaign_id)
    os.makedirs(data_dir, exist_ok=True)

    master_task_file = os.path.join(data_dir, f"{campaign_id}_master_tasks.jsonl")
    if os.path.exists(master_task_file):
        print(f"Master task file already exists: {master_task_file}")
        print("Delete it if you want to regenerate.")
        sys.exit(0)

    print(f"--- Generating Master Task List for Campaign: {campaign_id} ---")

    all_tasks = []
    for sim_set in config.get("sim_sets", []):
        base_params = sim_set.get("base_params", {}).copy()
        num_replicates = base_params.pop("num_replicates", 1)
        grid_param_keys = list(sim_set["grid_params"].keys())
        grid_value_lists = [
            param_grid_values[sim_set["grid_params"][k]] for k in grid_param_keys
        ]

        for combo in itertools.product(*grid_value_lists):
            instance_params = dict(zip(grid_param_keys, combo))
            for j in range(num_replicates):
                task_params = {
                    **base_params,
                    **instance_params,
                    "replicate": j,
                    "run_mode": config["run_mode"],
                    "campaign_id": campaign_id,
                }
                resolved_params = resolve_parameters(task_params, param_grid_values)
                resolved_params["task_id"] = generate_task_id(resolved_params)
                all_tasks.append(resolved_params)

    with open(master_task_file, "w") as f:
        for task in sorted(all_tasks, key=lambda t: t["task_id"]):
            f.write(json.dumps(task) + "\n")

    print(f"Successfully generated {len(all_tasks)} total tasks.")
    print(f"Master task list saved to: {master_task_file}")


if __name__ == "__main__":
    main()
