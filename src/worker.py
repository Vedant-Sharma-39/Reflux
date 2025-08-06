# FILE: src/worker.py
# [v_SIMPLE] This worker executes a single simulation task and writes its
# output directly to a file in the specified output directory.

import argparse
import json
import sys
import os
import traceback
import pandas as pd
from typing import Dict, Any

# --- Robust Path Setup ---
# Use an environment variable set by the runner script for robustness.
project_root = os.getenv("PROJECT_ROOT")
if not project_root:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Use absolute imports ---
from src.core.model import GillespieSimulation
from src.core.metrics import (
    MetricsManager,
    SectorWidthTracker,
    InterfaceRoughnessTracker,
    SteadyStatePropertiesTracker,
    FrontDynamicsTracker,
    TimeSeriesTracker,
)
# Note: config_loader is not strictly needed here anymore if env_map is resolved
# during task generation, but we keep it for now for compatibility.
from src.config_loader import EXPERIMENTS


RUN_MODE_CONFIG = {
    "calibration": {"tracker_class": SectorWidthTracker},
    "diffusion": {"tracker_class": InterfaceRoughnessTracker},
    "phase_diagram": {
        "tracker_class": SteadyStatePropertiesTracker,
        "tracker_params": ["warmup_time", "num_samples", "sample_interval"],
    },
    "relaxation": {
        "tracker_class": TimeSeriesTracker,
        "tracker_params": {"log_interval": "sample_interval"},
    },
    "bet_hedging": {
        "tracker_class": FrontDynamicsTracker,
        "tracker_params": ["log_q_interval"],
    },
}


def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single simulation and returns the results dictionary."""
    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown or unsupported run_mode: '{run_mode}'")

    # The parameter resolution should ideally happen in generate_tasks.py.
    # This block is for backwards compatibility or debugging.
    if "environment_map" in params and isinstance(params.get("environment_map"), str):
        map_name = params["environment_map"]
        if "experiment_name" in params:
            # This is a bit of a hack; better to resolve before the worker
            exp_name = params["experiment_name"]
            # Need to get the full config to access the param grid
            from src.config_loader import get_experiment_config
            full_exp_config = get_experiment_config(exp_name)
            params["environment_map"] = full_exp_config["PARAM_GRID_VALUES"].get(map_name)

    manager = MetricsManager()
    config = RUN_MODE_CONFIG[run_mode]
    tracker_class = config["tracker_class"]

    # --- START: REPLACE THIS BLOCK ---
    tracker_kwargs = {}
    tracker_param_config = config.get("tracker_params", {})

    if isinstance(tracker_param_config, list):

        for key in tracker_param_config:
            if key in params:
                tracker_kwargs[key] = params[key]
    elif isinstance(tracker_param_config, dict):

        for tracker_arg, sim_param_key in tracker_param_config.items():
            if sim_param_key in params:
                tracker_kwargs[tracker_arg] = params[sim_param_key]

    manager.add_tracker(tracker_class, **tracker_kwargs)

    sim = GillespieSimulation(metrics_manager=manager, **params)
    max_steps = params.get("max_steps", 30_000_000)
    max_time = params.get("total_run_time", float("inf"))

    while sim.time < max_time and sim.step_count < max_steps:
        active, boundary_hit = sim.step()
        manager.after_step_hook()
        if not active or boundary_hit or manager.is_done():
            break

    results = manager.finalize()

    # Combine original params with the results for a complete record.
    final_output = {**params, **results}

    # Post-processing for bet-hedging can stay the same
    if run_mode == "bet_hedging" and "front_dynamics" in final_output:
        df_history = pd.DataFrame(final_output.pop("front_dynamics"))
        cycle_len_q = params.get("patch_width", 0) * len(params.get("environment_map", []))
        warmup_q = params.get("warmup_cycles_for_stats", 4) * cycle_len_q if cycle_len_q > 0 else 0.0
        stats_df = df_history[df_history["mean_front_q"] > warmup_q]
        final_output.update({
            "avg_front_speed": stats_df["front_speed"].mean() if not stats_df.empty else 0.0,
            "var_front_speed": stats_df["front_speed"].var(ddof=1) if len(stats_df) > 1 else 0.0,
            "avg_rho_M": stats_df["mutant_fraction"].mean() if not stats_df.empty else 0.0,
        })

    return final_output


def main():
    parser = argparse.ArgumentParser(
        description="Run a single simulation and save output to a file."
    )
    parser.add_argument(
        "--params", required=True, help="JSON string of simulation parameters."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save the output JSON file."
    )
    args = parser.parse_args()

    params = {}
    output_path = None
    try:
        params = json.loads(args.params)
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task parameters must include a 'task_id'.")

        # Define the output path based on the task_id
        output_path = os.path.join(args.output_dir, f"{task_id}.json")
        
        # --- Idempotency Check ---
        # If the final output file already exists, do nothing.
        if os.path.exists(output_path):
            print(f"Output for task {task_id} already exists. Skipping.")
            sys.exit(0)

        result_data = run_simulation(params)
        
        # Ensure the output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Write the final, single JSON output
        with open(output_path, "w") as f:
            json.dump(result_data, f, allow_nan=True, indent=2)

        print(f"Successfully completed task {task_id}.")

    except Exception:
        # If something goes wrong, write an error file instead.
        task_id = params.get("task_id", "unknown_task")
        error_path = os.path.join(args.output_dir, f"{task_id}.error")
        error_output = {
            "task_id": task_id,
            "error": traceback.format_exc(),
            "params": params,
        }
        with open(error_path, "w") as f:
            json.dump(error_output, f, indent=2)
        
        # Also print to stderr for immediate visibility in logs
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
