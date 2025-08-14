# FILE: src/worker.py (Final, Simplified Version)

import argparse
import gzip
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

# --- Robust Path Setup ---
# Ensures `from src...` imports work when this script is run as a module.
project_root = os.getenv("PROJECT_ROOT")
if not project_root:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.model import GillespieSimulation
from src.core.metrics import (
    MetricsManager,
    BoundaryDynamicsTracker,
    InterfaceRoughnessTracker,
    SteadyStatePropertiesTracker,
    FrontConvergenceTracker,
    CyclicTimeSeriesTracker,
    RelaxationConvergenceTracker,
)

# This dictionary maps the `run_mode` string from config.py to the
# appropriate MetricTracker class and the parameter names it expects.
RUN_MODE_CONFIG = {
    "calibration": {"tracker_class": BoundaryDynamicsTracker, "tracker_params": {}},
    "diffusion": {
        "tracker_class": InterfaceRoughnessTracker,
        "tracker_params": {"capture_interval": "capture_interval"},
    },
    "phase_diagram": {
        "tracker_class": SteadyStatePropertiesTracker,
        "tracker_params": {
            "warmup_time": "warmup_time",
            "num_samples": "num_samples",
            "sample_interval": "sample_interval",
        },
    },
    "bet_hedging_converged": {
        "tracker_class": FrontConvergenceTracker,
        "tracker_params": {
            "max_duration": "max_cycles",
            "duration_unit": "'cycles'",
            "convergence_window": "convergence_window_cycles",
            "convergence_threshold": "convergence_threshold",
        },
    },
    "homogeneous_converged": {
        "tracker_class": FrontConvergenceTracker,
        "tracker_params": {
            "max_duration": "max_run_time",
            "duration_unit": "'time'",
            "convergence_window": "convergence_window",
            "convergence_threshold": "convergence_threshold",
            "convergence_check_interval": "convergence_check_interval",
        },
    },
    "cyclic_timeseries": {
        "tracker_class": CyclicTimeSeriesTracker,
        "tracker_params": {
            "warmup_cycles": "warmup_cycles",
            "measure_cycles": "measure_cycles",
            "sample_interval": "sample_interval",
        },
    },
    "relaxation_converged": {
        "tracker_class": RelaxationConvergenceTracker,
        "tracker_params": {
            "sample_interval": "sample_interval",
            "convergence_window": "convergence_window",
            "convergence_threshold": "convergence_threshold",
        },
    },
}

# List of keys for data that is too large to store in the main summary CSV.
BULKY_DATA_KEYS = [
    "timeseries",
    "trajectory",
    "roughness_sq_trajectory",
    "front_dynamics",
]


def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """A single, unified function to run any simulation task."""
    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown run_mode: '{run_mode}'")

    # The worker trusts that the `params` dictionary is already fully resolved.

    manager = MetricsManager(params)
    config = RUN_MODE_CONFIG[run_mode]
    manager.add_tracker(config["tracker_class"], config["tracker_params"])

    sim = GillespieSimulation(**params)
    manager.register_simulation(sim)

    max_steps = params.get("max_steps", 50_000_000)
    max_time = params.get("max_run_time", float("inf"))

    # Generic, simple simulation loop. All complex logic is handled by the tracker.
    while sim.time < max_time and sim.step_count < max_steps:
        active, boundary_hit = sim.step()
        manager.after_step_hook()
        if not active or boundary_hit or manager.is_done():
            break

    results = manager.finalize()

    # Use a default termination reason if the tracker doesn't provide one.
    if "termination_reason" not in results:
        reason = (
            "max_steps_reached" if sim.step_count >= max_steps else "max_time_reached"
        )
        if not active:
            reason = "no_active_sites"
        if boundary_hit:
            reason = "boundary_hit"
        results["termination_reason"] = reason

    # Return a merged dictionary of original params and final results.
    # Note: we return the original params dict to ensure task_id consistency.
    return {**params, **results}


def main():
    parser = argparse.ArgumentParser(description="Run a single simulation task.")
    parser.add_argument(
        "--params", required=True, help="JSON string of simulation parameters."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save raw chunk files."
    )
    args = parser.parse_args()

    params = {}
    try:
        # The params from the JSONL task file are loaded and used directly.
        # No normalization or resolution is needed here anymore.
        params = json.loads(args.params)
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task parameters must include a 'task_id'.")

        result_data = run_simulation(params)

        # Separate bulky data (e.g., timeseries) for separate storage.
        for key in BULKY_DATA_KEYS:
            if key in result_data:
                bulky_data = result_data.pop(key)
                subdir_name = "timeseries" if "timeseries" in key else "trajectories"
                data_dir = Path(args.output_dir).parent / subdir_name
                data_dir.mkdir(exist_ok=True)
                prefix = "ts" if "timeseries" in key else "traj"
                out_path = data_dir / f"{prefix}_{task_id}.json.gz"
                with gzip.open(out_path, "wt", encoding="utf-8") as f_gz:
                    json.dump(bulky_data, f_gz)

        # Print the lightweight summary data to stdout as a single JSON line.
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
