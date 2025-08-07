# FILE: src/worker.py
# This worker executes a single simulation task defined by a self-contained JSON object
# and writes its output directly to a file in the specified output directory.
# [v2 - Added 'bet_hedging_converged' run mode for rigorous fitness calculation]

import argparse
import json
import sys
import os
import traceback
import pandas as pd
import numpy as np
from collections import deque
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
    "diffusion": {
        "tracker_class": InterfaceRoughnessTracker,
        "tracker_params": {"capture_interval": 0.5},
    },
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
    "bet_hedging_converged": {
        "tracker_class": None,  # This mode has a custom loop
        "tracker_params": {},
    },
}


def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single simulation and returns the results dictionary."""
    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown or unsupported run_mode: '{run_mode}'")

    # --- Custom loop for rigorous bet-hedging fitness calculation ---
    if run_mode == "bet_hedging_converged":
        sim = GillespieSimulation(**params)

        env_map = params.get("environment_map", {})
        patch_width = params.get("patch_width", 0)
        cycle_q = patch_width * len(env_map) if patch_width > 0 and env_map else 0.0
        if cycle_q <= 0:
            raise ValueError(
                "bet_hedging_converged requires patch_width and environment_map."
            )

        max_cycles = params.get("max_cycles", 50)
        conv_window = params.get("convergence_window_cycles", 5)
        conv_threshold = params.get("convergence_threshold", 0.01)

        cycle_speeds = deque(maxlen=conv_window)
        termination_reason = "max_cycles_reached"
        last_cycle_time = 0.0
        cycles_completed = 0

        for cycle in range(1, max_cycles + 1):
            target_q = cycle * cycle_q

            while sim.mean_front_position < target_q:
                active, boundary_hit = sim.step()
                if not active or boundary_hit:
                    termination_reason = "stalled_or_boundary_hit"
                    break

            if termination_reason != "max_cycles_reached":
                break

            time_for_cycle = sim.time - last_cycle_time
            speed_this_cycle = cycle_q / time_for_cycle if time_for_cycle > 1e-9 else 0
            cycle_speeds.append(speed_this_cycle)
            last_cycle_time = sim.time
            cycles_completed = cycle

            if len(cycle_speeds) == conv_window:
                mean_speed = np.mean(cycle_speeds)
                if mean_speed > 1e-9:
                    std_dev = np.std(cycle_speeds, ddof=1)
                    if (std_dev / mean_speed) < conv_threshold:
                        termination_reason = "converged"
                        break

        final_fitness = np.mean(cycle_speeds) if cycle_speeds else 0.0
        final_variance = np.var(cycle_speeds, ddof=1) if len(cycle_speeds) > 1 else 0.0

        results = {
            "avg_front_speed": final_fitness,
            "var_front_speed": final_variance,
            "avg_rho_M": sim.mutant_fraction,
            "num_cycles_completed": cycles_completed,
        }
        final_output = {**params, **results}
        final_output["termination_reason"] = termination_reason
        final_output["final_sim_time"] = sim.time
        final_output["final_sim_steps"] = sim.step_count
        return final_output

    # --- Standard logic for all other run modes ---
    manager = MetricsManager()
    config = RUN_MODE_CONFIG[run_mode]
    tracker_class = config["tracker_class"]
    if tracker_class is None:
        raise ValueError(
            f"Run mode '{run_mode}' is not fully implemented in the standard loop."
        )

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
        key: value
        for key, value in params.items()
        if key not in METADATA_KEYS and key not in tracker_kwargs
    }
    sim = GillespieSimulation(metrics_manager=manager, **sim_params)

    max_steps = params.get("max_steps", 30_000_000)
    max_time = params.get("total_run_time", float("inf"))
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

    if termination_reason == "unknown":
        if sim.step_count >= max_steps:
            termination_reason = "max_steps_reached"
        elif sim.time >= max_time:
            termination_reason = "max_time_reached"

    results = manager.finalize()
    final_output = {**params, **results}
    final_output["termination_reason"] = termination_reason
    final_output["final_sim_time"] = sim.time
    final_output["final_sim_steps"] = sim.step_count

    if run_mode == "bet_hedging" and "front_dynamics" in final_output:
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

        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
