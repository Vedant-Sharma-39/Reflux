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
)
from src.config import EXPERIMENTS


# --- [DEFINITIVELY CORRECTED] Worker function with CYCLE-AWARE logic ---
def run_spatial_fluctuation_analysis(params):
    """
    Runs analysis for fluctuating environments with dynamic stopping criteria
    based on the convergence of the running average speed.
    """
    sim_params = params.copy()

    env_map_name = sim_params.pop("environment_map")
    sim_params.pop("campaign_id", None)

    exp_name = params.get("campaign_id", "spatial_bet_hedging_v1")
    actual_env_map = EXPERIMENTS[exp_name]["PARAM_GRID"][env_map_name]

    constructor_args = FluctuatingGillespieSimulation.__init__.__code__.co_varnames
    sim_constructor_params = {
        k: v for k, v in sim_params.items() if k in constructor_args
    }

    # --- [THE FIX] ---
    # 1. Create the MetricsManager.
    manager = MetricsManager()
    # 2. Add the manager to the parameters that will be passed to the constructor.
    sim_constructor_params["metrics_manager"] = manager

    # 3. Create the simulation. It will now be aware of the manager.
    sim = FluctuatingGillespieSimulation(
        environment_map=actual_env_map, **sim_constructor_params
    )

    # 4. Create and register the tracker with the manager.
    tracker = FrontDynamicsTracker(sim, log_q_interval=params["log_q_interval"])
    manager.add_tracker(tracker)
    manager.initialize_all()

    running_avg_history = deque(maxlen=params["convergence_window"])

    num_patches = len(actual_env_map)
    cycle_length = params["patch_width"] * num_patches
    min_q_for_convergence_check = params["convergence_min_cycles"] * cycle_length

    while sim.mean_front_position < params["length"]:
        did_step, boundary_hit = sim.step()
        if not did_step or boundary_hit:
            break

        if sim.mean_front_position > min_q_for_convergence_check:
            df = tracker.get_dataframe()
            if len(df) > params["convergence_window"]:
                current_running_avg = df["front_speed"].expanding().mean().iloc[-1]
                running_avg_history.append(current_running_avg)
                if len(running_avg_history) == params["convergence_window"]:
                    history_mean = np.mean(running_avg_history)
                    if history_mean > 1e-9:
                        coeff_of_variation = (
                            np.std(running_avg_history, ddof=1) / history_mean
                        )
                        if coeff_of_variation < params["convergence_threshold"]:
                            break

    df_history = tracker.get_dataframe()
    stats_df = df_history[df_history["mean_front_q"] > params["warmup_q_for_stats"]]

    results = {
        "avg_front_speed": (
            stats_df["front_speed"].mean() if not stats_df.empty else 0.0
        ),
        "var_front_speed": (
            stats_df["front_speed"].var(ddof=1) if len(stats_df) > 1 else 0.0
        ),
        "avg_rho_M": stats_df["mutant_fraction"].mean() if not stats_df.empty else 0.0,
        "var_rho_M": (
            stats_df["mutant_fraction"].var(ddof=1) if len(stats_df) > 1 else 0.0
        ),
        "final_q": sim.mean_front_position,
    }
    return results


# ... [Other worker functions remain unchanged] ...
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
    tracker = FrontDynamicsTracker(sim, log_q_interval=params["sample_interval"])
    pulse_on = False
    while sim.time < params["total_run_time"]:
        if not pulse_on and sim.time >= params["pulse_start_time"]:
            sim.set_switching_rate(params["k_total_pulse"])
            pulse_on = True
        if pulse_on and sim.time >= (
            params["pulse_start_time"] + params["pulse_duration"]
        ):
            sim.set_switching_rate(params["k_total"])
            pulse_on = False
        did_step, boundary_hit = sim.step()
        tracker.after_step_hook()
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
