# FILE: src/worker.py
# [DEFINITIVE VERSION]
# A single, unified worker that uses a configuration-driven approach to select
# metric trackers and run the appropriate simulation loop.

import argparse
import json
import sys
import os
import traceback
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
core_path = os.path.join(project_root, "src/core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from model import GillespieSimulation
from metrics import (
    MetricsManager,
    SectorWidthTracker,
    InterfaceRoughnessTracker,
    SteadyStatePropertiesTracker,
    FrontDynamicsTracker,
    TimeSeriesTracker,
)

RUN_MODE_CONFIG = {
    "calibration": {"tracker_class": SectorWidthTracker, "tracker_params": {}},
    "diffusion": {"tracker_class": InterfaceRoughnessTracker, "tracker_params": {}},
    "structure_analysis": {
        "tracker_class": SteadyStatePropertiesTracker,
        "tracker_params": {
            "warmup_time": "warmup_time",
            "num_samples": "num_samples",
            "sample_interval": "sample_interval",
        },
    },
    "steady_state": {
        "tracker_class": SteadyStatePropertiesTracker,
        "tracker_params": {
            "warmup_time": "warmup_time",
            "num_samples": "num_samples",
            "sample_interval": "sample_interval",
        },
    },
    "correlation_analysis": {
        "tracker_class": SteadyStatePropertiesTracker,
        "tracker_params": {
            "warmup_time": "warmup_time",
            "num_samples": "num_samples",
            "sample_interval": "sample_interval",
        },
    },
    "timeseries_from_pure_state": {
        "tracker_class": TimeSeriesTracker,
        "tracker_params": {"log_interval": "log_interval"},
    },
    "perturbation": {
        "tracker_class": TimeSeriesTracker,
        "tracker_params": {
            "log_interval": "total_run_time",
            "sample_interval": "sample_interval",
        },
    },
    "spatial_fluctuation_analysis": {
        "tracker_class": FrontDynamicsTracker,
        "tracker_params": {"log_q_interval": "log_q_interval"},
    },
}


def run_simulation(params: dict):
    run_mode = params.get("run_mode")
    if run_mode not in RUN_MODE_CONFIG:
        raise ValueError(f"Unknown run_mode: {run_mode}")
    manager = MetricsManager()
    config = RUN_MODE_CONFIG[run_mode]
    tracker_class = config["tracker_class"]
    tracker_kwargs = {
        key: params[param_name]
        for key, param_name in config.get("tracker_params", {}).items()
    }
    manager.add_tracker(tracker_class, **tracker_kwargs)
    sim = GillespieSimulation(metrics_manager=manager, **params)
    max_steps = params.get("max_steps", 30_000_000)
    max_time = params.get("max_sim_time", params.get("total_run_time", 20000.0))

    for _ in range(max_steps):
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break
        if sim.time >= max_time:
            break
        if hasattr(manager, "is_done") and manager.is_done():
            break

    results = manager.finalize()
    if run_mode == "spatial_fluctuation_analysis" and "front_dynamics" in results:
        df_history = pd.DataFrame(results.pop("front_dynamics"))
        cycle_len_q = sum(w for _, w in sim.patch_sequence) if sim.patch_sequence else 0
        warmup_q = params.get("warmup_cycles_for_stats", 4) * cycle_len_q
        stats_df = df_history[df_history["mean_front_q"] > warmup_q]
        results.update(
            {
                "avg_front_speed": (
                    stats_df["front_speed"].mean() if not stats_df.empty else 0.0
                ),
                "var_front_speed": (
                    stats_df["front_speed"].var(ddof=1) if len(stats_df) > 1 else 0.0
                ),
                "avg_rho_M": (
                    stats_df["mutant_fraction"].mean() if not stats_df.empty else 0.0
                ),
            }
        )
        results["timeseries"] = df_history.to_dict("records")
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified simulation worker.")
    parser.add_argument(
        "--params", required=True, help="JSON string of simulation parameters."
    )
    args = parser.parse_args()
    DELIMITER = "---WORKER_PAYLOAD_SEPARATOR---"
    params = {}
    try:
        params = json.loads(args.params)
        result_data = run_simulation(params)
        summary_output = {**params, **result_data}
        timeseries_output = {"task_id": params["task_id"], "timeseries": []}
        if "timeseries" in summary_output:
            timeseries_output["timeseries"] = summary_output.pop("timeseries")
        print(json.dumps(summary_output))
        print(DELIMITER)
        print(json.dumps(timeseries_output))
    except Exception as e:
        error_output = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "params": args.params,
        }
        print(json.dumps(error_output), file=sys.stderr)
        task_id = params.get("task_id", "unknown_task")
        print(json.dumps({"params": params, "task_id": task_id, "error": "failed"}))
        print(DELIMITER)
        print(json.dumps({"task_id": task_id, "timeseries": []}))


if __name__ == "__main__":
    main()
