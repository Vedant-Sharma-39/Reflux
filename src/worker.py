# FILE: src/worker.py (Definitive, Cleaned for "Fossil Record" Workflow)

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
project_root = os.getenv("PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Model and Tracker Imports ---
from src.core.model import GillespieSimulation
from src.core.model_transient import GillespieTransientStateSimulation
from src.core.model_aif import AifModelSimulation
from src.core.metrics import (
    MetricsManager, MetricTracker, BoundaryDynamicsTracker, InterfaceRoughnessTracker,
    SteadyStatePropertiesTracker, FrontConvergenceTracker, FixationTimeTracker,
    HomogeneousDynamicsTracker, InvasionOutcomeTracker
)
# The ONLY AIF tracker needed for this workflow is imported:
from src.core.metrics_aif import AifMetricsManager

# --- Configuration for Run Modes ---
# This dictionary maps the 'run_mode' from config.py to the correct MetricTracker.
RUN_MODE_CONFIG = {
    # --- Standard Trackers for Linear Front Models ---
    "calibration": {"tracker_class": BoundaryDynamicsTracker},
    "diffusion": {"tracker_class": InterfaceRoughnessTracker},
    "phase_diagram": {"tracker_class": SteadyStatePropertiesTracker},
    "bet_hedging_converged": {"tracker_class": FrontConvergenceTracker},
    "fixation_analysis": {"tracker_class": FixationTimeTracker},
    "HomogeneousDynamicsTracker": {"tracker_class": HomogeneousDynamicsTracker},
    "invasion_outcome": {"tracker_class": InvasionOutcomeTracker},
    
    # --- The Tracker for AIF "Fossil Record" Analysis ---
    "aif_width_analysis": {"tracker_class": AifMetricsManager},

    # --- Generic Tracker for Visualization Runs ---
    "visualization": {"tracker_class": MetricTracker},
}

# --- Keys for data to be saved in separate files ---
BULKY_DATA_KEYS = [
    "final_population",  # <-- The essential key for this workflow
    # Other keys for other experiments can be listed here
    "timeseries",
    "trajectory",
    "sector_trajectory",
]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    # Select simulation engine based on parameters
    SimClass = AifModelSimulation if "b_sus" in params else GillespieSimulation

    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown or misconfigured run_mode: '{run_mode}'")

    manager = MetricsManager(params)
    config = RUN_MODE_CONFIG[run_mode]
    manager.add_tracker(config["tracker_class"], config.get("tracker_params", {}))

    sim = SimClass(**params)
    manager.register_simulation(sim)

    max_steps = params.get("max_steps", 50_000_000)
    while sim.step_count < max_steps:
        active, boundary_hit = sim.step()
        manager.after_step_hook()
        if not active or manager.is_done() or boundary_hit:
            break
            
    results = manager.finalize()
    if "termination_reason" not in results:
        reason = "max_steps_reached"
        if not active: reason = "no_active_sites"
        if boundary_hit: reason = "boundary_hit"
        results["termination_reason"] = reason
        
    return {**params, **results}

def main():
    parser = argparse.ArgumentParser(description="Run a single simulation task.")
    parser.add_argument("--params", required=True, help="JSON string of parameters.")
    parser.add_argument("--output-dir", required=True, help="Directory for raw data.")
    args = parser.parse_args()

    params = {}
    try:
        params = json.loads(args.params)
        task_id = params.get("task_id")
        result_data = run_simulation(params)

        for key in BULKY_DATA_KEYS:
            if key in result_data:
                bulky_data = result_data.pop(key)
                
                if key == "final_population":
                    subdir_name, prefix = "populations", "pop"
                elif "trajectory" in key:
                    subdir_name, prefix = "trajectories", "traj"
                elif "timeseries" in key:
                    subdir_name, prefix = "timeseries", "ts"
                else:
                    subdir_name, prefix = "other_data", "dat"
                
                data_dir = Path(args.output_dir).parent / subdir_name
                data_dir.mkdir(exist_ok=True)
                out_path = data_dir / f"{prefix}_{task_id}.json.gz"
                
                with gzip.open(out_path, "wt", encoding="utf-8") as f_gz:
                    json.dump(bulky_data, f_gz, cls=NumpyEncoder)

        print(json.dumps(result_data, allow_nan=True, separators=(",", ":"), cls=NumpyEncoder))
    except Exception:
        task_id = params.get("task_id", "unknown_task")
        error_output = {"task_id": task_id, "error": traceback.format_exc(), "params": params}
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()