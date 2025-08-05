# FILE: src/worker.py
# A unified worker script that is called by Slurm to run a single simulation.
# It dispatches to the correct simulation and metric tracking logic based on
# the 'run_mode' specified in the input parameters.

import argparse
import json
import sys
import os
import traceback
from collections import deque
import numpy as np
import pandas as pd

# This ensures the script can find other modules in the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.linear_model import GillespieSimulation
from src.fluctuating_model import FluctuatingGillespieSimulation
from src.metrics import (
    MetricsManager,
    SectorWidthTracker,
    InterfaceRoughnessTracker,
    FrontPropertiesTracker,
    FrontDynamicsTracker,
    TimeSeriesTracker,
)
from src.config import EXPERIMENTS


# --- ### DEFINITIVE VERSION WITH BACKWARD COMPATIBILITY ### ---
def run_spatial_fluctuation_analysis(params):
    """
    Runs analysis for fluctuating environments. It is backward compatible and can
    handle both old configs with `patch_width` and new configs with
    `environment_patch_sequence`.
    """
    sim_params = params.copy()
    env_map_name = sim_params.pop("environment_map")
    # Use campaign_id from params, fallback to a default if not present
    campaign_id = sim_params.pop(
        "campaign_id", next(iter(EXPERIMENTS.values()))["CAMPAIGN_ID"]
    )
    actual_env_map = (
        EXPERIMENTS.get(campaign_id, {}).get("PARAM_GRID", {}).get(env_map_name)
    )
    if actual_env_map is None:
        # Fallback for older configs where campaign_id might not match experiment name
        for exp in EXPERIMENTS.values():
            if env_map_name in exp.get("PARAM_GRID", {}):
                actual_env_map = exp["PARAM_GRID"][env_map_name]
                break
    if actual_env_map is None:
        raise ValueError(
            f"Could not find environment map named '{env_map_name}' in any experiment config."
        )

    # --- ### BACKWARD COMPATIBILITY SHIM ### ---
    if "environment_patch_sequence" in sim_params:
        patch_sequence_name = sim_params.pop("environment_patch_sequence")
        actual_patch_sequence = EXPERIMENTS[campaign_id]["PARAM_GRID"][
            patch_sequence_name
        ]
        sim_params["environment_patch_sequence"] = actual_patch_sequence
        patch0_width = actual_patch_sequence[0][1]
        patch1_width = actual_patch_sequence[1][1]
    elif "patch_width" in sim_params:
        patch_width = sim_params.pop("patch_width")
        actual_patch_sequence = [(0, patch_width), (1, patch_width)]
        sim_params["environment_patch_sequence"] = actual_patch_sequence
        patch0_width = patch_width
        patch1_width = patch_width
    else:
        raise ValueError(
            "Missing 'patch_width' or 'environment_patch_sequence' in simulation parameters."
        )
    # --- ### END SHIM ### ---

    constructor_args = FluctuatingGillespieSimulation.__init__.__code__.co_varnames
    sim_constructor_params = {
        k: v for k, v in sim_params.items() if k in constructor_args
    }

    manager = MetricsManager()
    sim_constructor_params["metrics_manager"] = manager
    sim = FluctuatingGillespieSimulation(
        environment_map=actual_env_map, **sim_constructor_params
    )

    tracker = FrontDynamicsTracker(sim, log_q_interval=params["log_q_interval"])
    manager.add_tracker(tracker)
    manager.register_simulation(sim)
    manager.initialize_all()

    cycle_length_q = sum(width for _, width in actual_patch_sequence)
    next_cycle_boundary_q = cycle_length_q
    convergence_window = params.get("convergence_window", 10)
    cycle_boundary_points = deque([(0.0, 0.0)], maxlen=convergence_window + 1)
    min_cycles_for_check = params.get("convergence_min_cycles", 5)

    while sim.mean_front_position < params["length"]:
        did_step, boundary_hit = sim.step()
        if not did_step or boundary_hit:
            break
        if cycle_length_q > 0 and sim.mean_front_position >= next_cycle_boundary_q:
            cycle_boundary_points.append((sim.time, sim.mean_front_position))
            next_cycle_boundary_q += cycle_length_q
            if len(cycle_boundary_points) > min_cycles_for_check:
                times = np.array([p[0] for p in cycle_boundary_points])
                positions = np.array([p[1] for p in cycle_boundary_points])
                delta_t = np.diff(times)
                delta_q = np.diff(positions)
                cycle_speeds = np.divide(
                    delta_q, delta_t, out=np.zeros_like(delta_q), where=delta_t != 0
                )
                if len(cycle_speeds) >= convergence_window:
                    speed_mean = np.mean(cycle_speeds)
                    if (
                        speed_mean > 1e-9
                        and (np.std(cycle_speeds, ddof=1) / speed_mean)
                        < params["convergence_threshold"]
                    ):
                        break

    df_history = tracker.get_dataframe()
    warmup_cycles = params.get("warmup_cycles_for_stats", 4)
    warmup_q = warmup_cycles * cycle_length_q
    stats_df = df_history[df_history["mean_front_q"] > warmup_q]

    task_id = params.get("task_id", "unknown_task")
    timeseries_dir = os.path.join(project_root, "data", campaign_id, "timeseries")
    os.makedirs(timeseries_dir, exist_ok=True)
    ts_path = os.path.join(timeseries_dir, f"ts_{task_id}.json.gz")
    if not df_history.empty:
        df_history.to_json(ts_path, orient="records", compression="gzip")

    results = {
        "avg_front_speed": (
            stats_df["front_speed"].mean() if not stats_df.empty else 0.0
        ),
        "var_front_speed": (
            stats_df["front_speed"].var(ddof=1) if len(stats_df) > 1 else 0.0
        ),
        "avg_rho_M": (
            stats_df["mutant_fraction"].mean() if not stats_df.empty else 0.0
        ),
        "var_rho_M": (
            stats_df["mutant_fraction"].var(ddof=1) if len(stats_df) > 1 else 0.0
        ),
        "final_q": sim.mean_front_position,
        "patch0_width": patch0_width,
        "patch1_width": patch1_width,
    }
    return results


def run_timeseries_from_pure_mutant_sim(params):
    manager = MetricsManager()
    # The config sets initial_mutant_patch_size = width
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
        initial_condition_type="patch",
        initial_mutant_patch_size=params["width"],
        metrics_manager=manager,
    )

    tracker = TimeSeriesTracker(sim, log_interval=params["log_interval"])
    manager.add_tracker(tracker)

    max_time = params.get("max_sim_time", 5000.0)
    while sim.time < max_time:
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break

    return {"timeseries": tracker.get_timeseries()}


def run_transient_survival_analysis(params):
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
        initial_condition_type="patch",
        initial_mutant_patch_size=0,
    )
    survived = False
    in_hostile_zone = False
    hostile_patch_start_q = params["safe_patch_width"]
    while sim.mean_front_position < params["length"]:
        if not in_hostile_zone and sim.mean_front_position >= hostile_patch_start_q:
            in_hostile_zone = True
            sim.b_wt = 0.0
            sim._update_rates()
        did_step, boundary_hit = sim.step()
        if boundary_hit:
            survived = True
            break
        if not did_step:
            survived = False
            break
    return {"survived": 1 if survived else 0}


def run_calibration_sim(params):
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params.get("k_total", 0.0),
        phi=params.get("phi", 0.0),
        initial_condition_type="patch",
        initial_mutant_patch_size=params["initial_mutant_patch_size"],
    )
    tracker = SectorWidthTracker(sim)
    max_steps = params.get("max_steps", 1_000_000)
    for _ in range(max_steps):
        did_step, boundary_hit = sim.step()
        tracker.after_step_hook()
        if not did_step or boundary_hit:
            break
    return {"trajectory": tracker.get_trajectory()}


def run_timeseries_from_pure_state_sim(params):
    """
    Runs a simulation from a pure initial state to measure relaxation dynamics.
    """
    manager = MetricsManager()
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
        initial_condition_type="patch",
        initial_mutant_patch_size=params["initial_mutant_patch_size"],
        metrics_manager=manager,
    )

    tracker = TimeSeriesTracker(sim, log_interval=params["log_interval"])
    manager.add_tracker(tracker)
    manager.register_simulation(sim)

    max_time = params.get("max_sim_time", 5000.0)
    while sim.time < max_time:
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break
        # Hook is called inside sim.step()

    return {"timeseries": tracker.get_timeseries()}


def run_diffusion_sim(params):
    sim = GillespieSimulation(
        width=params["width"],
        length=params.get("length", 4096),
        b_m=params["b_m"],
        k_total=params.get("k_total", 0.0),
        phi=params.get("phi", 0.0),
        initial_condition_type=params.get("initial_condition_type", "mixed"),
        initial_mutant_patch_size=params.get("initial_mutant_patch_size", 0),
    )
    tracker = InterfaceRoughnessTracker(sim)
    max_steps = params.get("max_steps", 10_000_000)
    for _ in range(max_steps):
        did_step, boundary_hit = sim.step()
        tracker.after_step_hook()
        if not did_step or boundary_hit:
            break
    return {"roughness_trajectory": tracker.get_roughness_trajectory()}


def run_structure_analysis_sim(params):
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
        initial_condition_type=params.get("initial_condition_type", "mixed"),
    )
    manager = MetricsManager()
    tracker = FrontPropertiesTracker(
        sim,
        warmup_time=params["warmup_time"],
        num_samples=params["num_samples"],
        sample_interval=params["sample_interval"],
    )
    manager.add_tracker(tracker)
    manager.register_simulation(sim)
    manager.initialize_all()
    while sim.time < (
        params["warmup_time"] + params["num_samples"] * params["sample_interval"] + 100
    ):
        did_step, boundary_hit = sim.step()
        manager.after_step_hook()
        if not did_step or boundary_hit:
            break
    return tracker.get_results()


def run_perturbation_sim(params):
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
        initial_condition_type=params.get("initial_condition_type", "mixed"),
    )
    manager = MetricsManager()
    tracker = TimeSeriesTracker(sim, sample_interval=params["sample_interval"])
    manager.add_tracker(tracker)
    manager.register_simulation(sim)

    pulse_state = "pre_pulse"
    pulse_end_time = params["pulse_start_time"] + params["pulse_duration"]

    while sim.time < params["total_run_time"]:
        if pulse_state == "pre_pulse" and sim.time >= params["pulse_start_time"]:
            sim.set_switching_rate(params["k_total_pulse"], sim.phi_base)
            pulse_state = "in_pulse"
        elif pulse_state == "in_pulse" and sim.time >= pulse_end_time:
            sim.set_switching_rate(sim.k_total_base, sim.phi_base)
            pulse_state = "post_pulse"

        did_step, boundary_hit = sim.step()
        manager.after_step_hook()
        if not did_step or boundary_hit:
            break
    df = tracker.get_dataframe()
    return {"timeseries": df.to_dict(orient="records")}


RUN_MODES = {
    "calibration": run_calibration_sim,
    "diffusion": run_diffusion_sim,
    "structure_analysis": run_structure_analysis_sim,
    "perturbation": run_perturbation_sim,
    "spatial_fluctuation_analysis": run_spatial_fluctuation_analysis,
    "transient_survival_analysis": run_transient_survival_analysis,
    "timeseries_from_pure_state": run_timeseries_from_pure_state_sim,  # New entry
    "correlation_analysis": run_structure_analysis_sim,
    "steady_state": run_structure_analysis_sim,
}


def main():
    parser = argparse.ArgumentParser(description="Unified Reflux simulation worker.")
    parser.add_argument(
        "--params", required=True, help="JSON string of simulation parameters."
    )
    args = parser.parse_args()
    try:
        params = json.loads(args.params)
        run_mode = params.get("run_mode")
        if not run_mode or run_mode not in RUN_MODES:
            raise ValueError(f"Invalid or missing 'run_mode': {run_mode}")
        result_data = RUN_MODES[run_mode](params)
        final_output = {**params, **result_data}
    except Exception as e:
        final_output = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "params": args.params,
        }
    print(json.dumps(final_output))


if __name__ == "__main__":
    main()
