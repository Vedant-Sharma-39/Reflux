# FILE: src/worker.py (Corrected for JSON Serialization)

import argparse
import gzip
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

# --- Add numpy import for the fix ---
import numpy as np

# --- Robust Path Setup ---
project_root = os.getenv("PROJECT_ROOT")
if not project_root:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.model import GillespieSimulation
from src.core.metrics import (
    MetricsManager,
    MetricTracker,
    BoundaryDynamicsTracker,
    InterfaceRoughnessTracker,
    SteadyStatePropertiesTracker,
    FrontConvergenceTracker,
    CyclicTimeSeriesTracker,
    RelaxationConvergenceTracker,
    FixationTimeTracker,
)

# ... (RUN_MODE_CONFIG and BULKY_DATA_KEYS are unchanged) ...
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
    "visualization": {"tracker_class": MetricTracker, "tracker_params": {}},
    "fixation_analysis": {
        "tracker_class": FixationTimeTracker,
        "tracker_params": {},  # No special params needed
    },
}
BULKY_DATA_KEYS = [
    "timeseries",
    "trajectory",
    "roughness_sq_trajectory",
    "front_dynamics",
]


# --- NEW HELPER CLASS FOR THE FIX ---
class NumpyEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles NumPy data types.
    This prevents `TypeError: Object of type int64 is not JSON serializable`.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# --- END OF NEW HELPER CLASS ---


def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    # (This function is unchanged)
    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown run_mode: '{run_mode}'")
    manager = MetricsManager(params)
    config = RUN_MODE_CONFIG[run_mode]
    manager.add_tracker(config["tracker_class"], config["tracker_params"])
    sim = GillespieSimulation(**params)
    manager.register_simulation(sim)
    max_steps = params.get("max_steps", 50_000_000)
    max_time = params.get("max_run_time", float("inf"))
    while sim.time < max_time and sim.step_count < max_steps:
        active, boundary_hit = sim.step()
        manager.after_step_hook()
        if not active or boundary_hit or manager.is_done():
            break
    results = manager.finalize()
    if "termination_reason" not in results:
        reason = (
            "max_steps_reached" if sim.step_count >= max_steps else "max_time_reached"
        )
        if not active:
            reason = "no_active_sites"
        if boundary_hit:
            reason = "boundary_hit"
        results["termination_reason"] = reason
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
        params = json.loads(args.params)
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("Task parameters must include a 'task_id'.")

        result_data = run_simulation(params)

        for key in BULKY_DATA_KEYS:
            if key in result_data:
                bulky_data = result_data.pop(key)
                subdir_name = "timeseries" if "timeseries" in key else "trajectories"
                data_dir = Path(args.output_dir).parent / subdir_name
                data_dir.mkdir(exist_ok=True)
                prefix = "ts" if "timeseries" in key else "traj"
                out_path = data_dir / f"{prefix}_{task_id}.json.gz"
                with gzip.open(out_path, "wt", encoding="utf-8") as f_gz:
                    # --- FIX IS ALSO APPLIED HERE FOR CONSISTENCY ---
                    json.dump(bulky_data, f_gz, cls=NumpyEncoder)

        # --- THIS IS THE CORRECTED LINE ---
        # We now pass our custom encoder class to json.dumps().
        print(
            json.dumps(
                result_data, allow_nan=True, separators=(",", ":"), cls=NumpyEncoder
            )
        )
        # --- END OF CORRECTION ---

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
