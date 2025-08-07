# FILE: src/worker.py
# This worker executes a single simulation task defined by a self-contained JSON object
# and writes its output directly to a file in the specified output directory.

import argparse
import json
import sys
import os
import traceback
import pandas as pd
from typing import Dict, Any

# --- Robust Path Setup ---
project_root = os.getenv("PROJECT_ROOT")
if not project_root:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.model import GillespieSimulation
from src.core.metrics import (
    MetricsManager,
    SectorWidthTracker,
    InterfaceRoughnessTracker,
    SteadyStatePropertiesTracker,
    TimeSeriesTracker,
    FrontDynamicsTracker,
)

RUN_MODE_CONFIG = {
    "calibration": {"tracker_class": SectorWidthTracker, "tracker_params": {}},
    "diffusion": {"tracker_class": InterfaceRoughnessTracker, "tracker_params": {}},
    "phase_diagram": {
        "tracker_class": SteadyStatePropertiesTracker,
        "tracker_params": {
            "warmup_time": "warmup_time",
            "num_samples": "num_samples",
            "sample_interval": "sample_interval",
        },
    },
    "bet_hedging": {
        "tracker_class": FrontDynamicsTracker,
        "tracker_params": {"log_q_interval": "log_q_interval"},
    },
    "relaxation": {
        "tracker_class": TimeSeriesTracker,
        "tracker_params": {"log_interval": "sample_interval"},
    },
}


def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single simulation and returns the results dictionary."""
    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown or unsupported run_mode: '{run_mode}'")

    manager = MetricsManager()
    config = RUN_MODE_CONFIG[run_mode]
    tracker_class = config["tracker_class"]

    tracker_kwargs = {
        tracker_arg: params[sim_param_key]
        for tracker_arg, sim_param_key in config["tracker_params"].items()
        if sim_param_key in params
    }
    manager.add_tracker(tracker_class, **tracker_kwargs)

    METADATA_KEYS = [
        "task_id",
        "campaign_id",
        "run_mode",
        "num_replicates",
        "replicate",
    ]
    sim_params = {
        key: value for key, value in params.items() if key not in METADATA_KEYS
    }
    sim = GillespieSimulation(metrics_manager=manager, **sim_params)

    max_steps = params.get("max_steps", 30_000_000)
    max_time = params.get("total_run_time", float("inf"))

    # --- [IMPROVEMENT] Add explicit termination reason tracking ---
    termination_reason = "unknown"

    while sim.time < max_time and sim.step_count < max_steps:
        active, boundary_hit = sim.step()
        manager.after_step_hook()
        if not active:
            termination_reason = "no_active_sites"
            break
        if boundary_hit:
            termination_reason = "boundary_hit"
            break
        if manager.is_done():
            termination_reason = "tracker_is_done"
            break

    # After the loop, check if it timed out
    if termination_reason == "unknown":
        if sim.step_count >= max_steps:
            termination_reason = "max_steps_reached"
        elif sim.time >= max_time:
            termination_reason = "max_time_reached"

    results = manager.finalize()
    final_output = {**params, **results}

    # Add the new termination info to the output file for easier debugging
    final_output["termination_reason"] = termination_reason
    final_output["final_sim_time"] = sim.time
    final_output["final_sim_steps"] = sim.step_count
    # --- [END IMPROVEMENT] ---

    if run_mode == "bet_hedging" and "front_dynamics" in final_output:
        # ... (rest of the function is unchanged)
        df_history = pd.DataFrame(final_output.pop("front_dynamics"))
        if not df_history.empty:
            env_map = params.get("environment_map", {})
            patch_width = params.get("patch_width", 0)
            cycle_len_q = (
                patch_width * len(env_map) if patch_width > 0 and env_map else 0.0
            )
            warmup_q = params.get("warmup_cycles_for_stats", 4) * cycle_len_q
            stats_df = df_history[df_history["mean_front_q"] > warmup_q]
            final_output.update(
                {
                    "avg_front_speed": (
                        stats_df["front_speed"].mean() if not stats_df.empty else 0.0
                    ),
                    "var_front_speed": (
                        stats_df["front_speed"].var(ddof=1)
                        if len(stats_df) > 1
                        else 0.0
                    ),
                    "avg_rho_M": (
                        stats_df["mutant_fraction"].mean()
                        if not stats_df.empty
                        else 0.0
                    ),
                }
            )

    return final_output


def main():
    parser = argparse.ArgumentParser(description="Run a single simulation task.")
    parser.add_argument(
        "--params", required=True, help="JSON string of simulation parameters."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save the output file."
    )
    args = parser.parse_args()

    params = {}
    try:
        params = json.loads(args.params)
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task parameters must include a 'task_id'.")

        final_output_path = os.path.join(args.output_dir, f"{task_id}.json")
        final_error_path = os.path.join(args.output_dir, f"{task_id}.error")
        tmp_output_path = os.path.join(args.output_dir, f".tmp_{task_id}_{os.getpid()}")

        if os.path.exists(final_output_path) or os.path.exists(final_error_path):
            # Suppress output for clean logs, but exit successfully.
            sys.exit(0)

        os.makedirs(args.output_dir, exist_ok=True)
        result_data = run_simulation(params)

        with open(tmp_output_path, "w") as f:
            json.dump(result_data, f, allow_nan=True, separators=(",", ":"))

        os.rename(tmp_output_path, final_output_path)

        print(f"Successfully completed task {task_id}.")

    except Exception:
        task_id = params.get("task_id", "unknown_task")
        final_error_path = os.path.join(args.output_dir, f"{task_id}.error")
        tmp_error_path = os.path.join(
            args.output_dir, f".tmp_err_{task_id}_{os.getpid()}"
        )

        error_output = {
            "task_id": task_id,
            "error": traceback.format_exc(),
            "params": params,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        with open(tmp_error_path, "w") as f:
            json.dump(error_output, f, indent=2)
        os.rename(tmp_error_path, final_error_path)

        # Print the error traceback to stderr so it's captured in the Slurm log
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
