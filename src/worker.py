# FILE: src/worker.py (Corrected - Standalone Aif Model)

import argparse
import gzip
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any
import numpy as np

# --- Robust Path Setup ---
project_root = os.getenv("PROJECT_ROOT")
if not project_root:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CORRECTED IMPORTS ---
# All necessary classes are imported. GillespieRadialSimulation is removed.
from src.core.model import GillespieSimulation
from src.core.model_transient import GillespieTransientStateSimulation
from src.core.metrics_aif import AifMetricsManager, AifSectorTrajectoryTracker

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
    "aif_width_analysis": {
        "tracker_class": AifMetricsManager,
        "tracker_params": {"max_steps": "max_steps"},
    },
    
    "aif_sector_trajectory": {
        "tracker_class": AifSectorTrajectoryTracker,
        "tracker_params": {},
    },
    
    "visualization": {"tracker_class": MetricTracker, "tracker_params": {}},
    "fixation_analysis": {"tracker_class": FixationTimeTracker, "tracker_params": {}},
}
BULKY_DATA_KEYS = [
    "timeseries",
    "trajectory",
    "roughness_sq_trajectory",
    "front_dynamics",
    "sector_trajectory",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    # --- CORRECTED Dynamic Class Selection Logic ---
    SimClass = None
    if "b_sus" in params:
        # The presence of a 'b_sus' parameter is the unique trigger for the Aif model.
        SimClass = GillespieAifReplication
    elif params.get("switching_lag_duration", 0.0) > 0.0:
        # The transient state (microlag) model.
        SimClass = GillespieTransientStateSimulation
    else:
        # The default linear front model.
        SimClass = GillespieSimulation
    # --- END OF CORRECTION ---

    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown run_mode: '{run_mode}'")

    manager = MetricsManager(params)
    config = RUN_MODE_CONFIG[run_mode]
    manager.add_tracker(config["tracker_class"], config["tracker_params"])

    sim = SimClass(**params)

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
                
                if "timeseries" in key:
                    subdir_name = "timeseries_data"
                elif "trajectory" in key:
                    subdir_name = "trajectories"
                elif key == "final_population":
                    subdir_name = "populations"
                else:
                    subdir_name = "trajectories"
                
                data_dir = Path(args.output_dir).parent / subdir_name
                data_dir.mkdir(exist_ok=True)
                prefix = "ts" if "timeseries" in key else "traj"
                out_path = data_dir / f"{prefix}_{task_id}.json.gz"
                with gzip.open(out_path, "wt", encoding="utf-8") as f_gz:
                    json.dump(bulky_data, f_gz, cls=NumpyEncoder)

        print(
            json.dumps(
                result_data, allow_nan=True, separators=(",", ":"), cls=NumpyEncoder
            )
        )

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
