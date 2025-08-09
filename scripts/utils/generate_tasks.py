# FILE: scripts/utils/generate_tasks.py

import argparse
import itertools
import json
import os
import sys
from typing import Dict, Any, List
import hashlib

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import EXPERIMENTS, PARAM_GRID


def generate_task_id(params: dict) -> str:
    """Generates a unique, deterministic ID from parameters using SHA-1."""
    id_defining_params = {
        k: v for k, v in params.items() if k not in ["campaign_id", "run_mode"]
    }

    def deep_sort(obj):
        if isinstance(obj, dict):
            return sorted((k, deep_sort(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(deep_sort(x) for x in obj)
        return obj

    param_str = json.dumps(deep_sort(id_defining_params), separators=(",", ":"))
    hasher = hashlib.sha1(param_str.encode("utf-8"))
    return hasher.hexdigest()


def resolve_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    resolved = params.copy()
    for _ in range(10):
        substitutions_made = False
        for key, value in list(resolved.items()):
            if isinstance(value, str):
                if value in resolved and resolved[value] is not value:
                    resolved[key] = resolved[value]
                    substitutions_made = True
                elif value in PARAM_GRID:
                    resolved[key] = PARAM_GRID[value]
                    substitutions_made = True
        if not substitutions_made:
            return resolved
    raise RuntimeError(f"Could not resolve parameter references in: {params}")


def generate_tasks_for_experiment(experiment_name: str) -> List[Dict[str, Any]]:
    config = EXPERIMENTS[experiment_name]
    campaign_id = config["campaign_id"]
    all_tasks = []

    for set_id, sim_set in config.get("sim_sets", {}).items():
        base_params = sim_set.get("base_params", {}).copy()
        num_replicates = base_params.pop("num_replicates", 1)
        grid_param_keys = list(sim_set["grid_params"].keys())
        grid_value_lists = [
            PARAM_GRID[sim_set["grid_params"][k]] for k in grid_param_keys
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
                resolved_params = resolve_parameters(task_params)
                resolved_params["task_id"] = generate_task_id(resolved_params)
                all_tasks.append(resolved_params)
    return all_tasks


def main():
    """This script is not intended for direct execution."""
    print(
        "This script provides helper functions for manage.py and is not meant to be run directly."
    )
    print(
        "Please use 'python3 manage.py launch <experiment_name>' to generate or update task lists."
    )


if __name__ == "__main__":
    main()
