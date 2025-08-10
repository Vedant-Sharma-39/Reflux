# FILE: src/worker.py (Corrected avg_rho_M Calculation)

import argparse
import gzip
import json
import os
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

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
    RecoveryDynamicsTracker,
    HomogeneousDynamicsTracker,
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
        "tracker_class": None,
        "tracker_params": {},
    },
    "recovery_dynamics": {
        "tracker_class": RecoveryDynamicsTracker,
        "tracker_params": {
            "timeseries_interval": "timeseries_interval",
            "warmup_time_ss": "warmup_time_ss",
            "num_samples_ss": "num_samples_ss",
            "sample_interval_ss": "sample_interval_ss",
        },
    },
    "homogeneous_dynamics": {
        "tracker_class": HomogeneousDynamicsTracker,
        "tracker_params": {
            "warmup_time": "warmup_time",
            "num_samples": "num_samples",
            "sample_interval": "sample_interval",
        },
    },
    "visualization": {
        "tracker_class": None,  # This mode has a custom loop
        "tracker_params": {},
    },
}


def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    run_mode = params.get("run_mode")
    if "env_definition" in params and isinstance(params["env_definition"], str):
        params["env_definition"] = json.loads(params["env_definition"])

    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown run_mode: '{run_mode}'")

    if run_mode == "visualization":
        # Pass necessary parameters for the model to handle plotting
        params["output_dir_viz"] = os.path.join(
            project_root, "data", params["campaign_id"], "images"
        )
        sim = GillespieSimulation(**params)
        max_cycles = params.get("max_cycles", 15)
        # Calculate total distance to simulate based on cycles
        cycle_q = sim.cycle_q_viz if hasattr(sim, "cycle_q_viz") else 0
        target_q = max_cycles * cycle_q if cycle_q > 0 else sim.length - 2

        # Run the simulation for the specified number of cycles
        while sim.mean_front_position < target_q:
            active, boundary_hit = sim.step()
            if not active or boundary_hit:
                break

        # The result is the images on disk, so we just return a success message
        return {"status": "completed", "task_id": params.get("task_id")}

    if run_mode == "bet_hedging_converged":
        sim = GillespieSimulation(**params)
        env_map = params.get("environment_map", {})
        patch_width = params.get("patch_width", 0)
        cycle_q = 0.0
        if patch_width > 0 and env_map:
            cycle_q = patch_width * len(env_map)
        elif "env_definition" in params:  # Handle new env def
            env_def = params["env_definition"]
            if isinstance(env_def, str):
                env_def = json.loads(env_def)
            cycle_q = (
                sum(p["width"] for p in env_def.get("patches", []))
                if not env_def.get("scrambled")
                else env_def.get("cycle_length", 0)
            )
        if cycle_q <= 0:
            raise ValueError("bet_hedging_converged requires a defined cycle length.")

        max_cycles = params.get("max_cycles", 50)
        conv_window = params.get("convergence_window_cycles", 5)
        conv_threshold = params.get("convergence_threshold", 0.01)

        # --- FIX: Create deques for both speed and mutant fraction ---
        cycle_speeds = deque(maxlen=conv_window)
        rho_m_samples = deque(maxlen=conv_window)
        # --- END FIX ---

        termination_reason = "max_cycles_reached"
        last_cycle_time = 0.0
        cycles_completed = 0
        for cycle in range(1, max_cycles + 1):
            target_q = sim.mean_front_position + cycle_q
            start_time_cycle = sim.time
            while sim.mean_front_position < target_q:
                active, boundary_hit = sim.step()
                if not active or boundary_hit:
                    termination_reason = "stalled_or_boundary_hit"
                    break
            if termination_reason != "max_cycles_reached":
                break

            time_for_cycle = sim.time - start_time_cycle
            speed_this_cycle = cycle_q / time_for_cycle if time_for_cycle > 1e-9 else 0

            # --- FIX: Sample both metrics at the end of each cycle ---
            cycle_speeds.append(speed_this_cycle)
            rho_m_samples.append(sim.mutant_fraction)
            # --- END FIX ---

            cycles_completed = cycle
            if len(cycle_speeds) == conv_window:
                mean_speed = np.mean(cycle_speeds)
                if (
                    mean_speed > 1e-9
                    and (np.std(cycle_speeds, ddof=1) / mean_speed) < conv_threshold
                ):
                    termination_reason = "converged"
                    break

        # --- FIX: Calculate results from the sampled deques ---
        results = {
            "avg_front_speed": np.mean(cycle_speeds) if cycle_speeds else 0.0,
            "var_front_speed": (
                np.var(cycle_speeds, ddof=1) if len(cycle_speeds) > 1 else 0.0
            ),
            "avg_rho_M": np.mean(rho_m_samples) if rho_m_samples else 0.0,
            "num_cycles_completed": cycles_completed,
        }
        # --- END FIX ---

        final_output = {**params, **results}
        final_output["termination_reason"] = termination_reason
        return final_output

    manager = MetricsManager()
    config = RUN_MODE_CONFIG[run_mode]
    tracker_class = config["tracker_class"]
    tracker_kwargs = {
        k: params[v] for k, v in config["tracker_params"].items() if v in params
    }
    manager.add_tracker(tracker_class, **tracker_kwargs)
    sim_params = {
        k: v
        for k, v in params.items()
        if k
        not in ["task_id", "campaign_id", "run_mode", "num_replicates", "replicate"]
        and k not in tracker_kwargs
    }
    sim = GillespieSimulation(metrics_manager=manager, **sim_params)
    max_steps, max_time = params.get("max_steps", 30_000_000), params.get(
        "total_run_time", float("inf")
    )
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
        termination_reason = (
            "max_steps_reached" if sim.step_count >= max_steps else "max_time_reached"
        )
    results = manager.finalize()
    final_output = {**params, **results}
    final_output["termination_reason"] = termination_reason
    return final_output


def main():
    parser = argparse.ArgumentParser(description="Run a single simulation task.")
    parser.add_argument(
        "--params", required=True, help="JSON string of simulation parameters."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save bulky output files (like timeseries).",
    )
    args = parser.parse_args()

    params = {}
    try:
        params = json.loads(args.params)
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task parameters must include a 'task_id'.")

        result_data = run_simulation(params)

        if params.get("run_mode") == "relaxation" and "timeseries" in result_data:
            timeseries_data = result_data.pop("timeseries")
            ts_dir = Path(args.output_dir).parent / "timeseries_raw"
            ts_dir.mkdir(exist_ok=True)
            ts_path = ts_dir / f"ts_{task_id}.json.gz"
            with gzip.open(ts_path, "wt", encoding="utf-8") as f_gz:
                json.dump(timeseries_data, f_gz)

        print(json.dumps(result_data, allow_nan=True, separators=(",", ":")))

    except Exception:
        task_id = params.get("task_id", "unknown_task")
        error_output = {
            "task_id": task_id,
            "error": traceback.format_exc(),
            "params": params,
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
