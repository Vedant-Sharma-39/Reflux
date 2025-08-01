# FILE: src/worker.py
# [DEFINITIVE, CONSOLIDATED VERSION]
# This is the complete, unified worker script. It handles all legacy, spatial,
# and perturbation run modes, using the correct, finalized trackers and simulation models.

import argparse
import json
import traceback
import numpy as np
from collections import deque

# --- Main simulation models ---
from linear_model import GillespieSimulation
from fluctuating_model import FluctuatingGillespieSimulation

# --- All necessary trackers from the final metrics.py ---
from metrics import (
    MetricsManager,
    SteadyStateTracker,
    SectorWidthTracker,
    InterfaceRoughnessTracker,
    CorrelationAndStructureTracker,
    FrontPropertiesTracker,
    FrontDynamicsTracker,
)

# ==============================================================================
# RUN FUNCTIONS FOR EACH EXPERIMENT TYPE
# ==============================================================================


def run_steady_state_sim(params):
    """For legacy 'steady_state' experiments."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        metrics_manager=manager,
        **{
            k: v
            for k, v in params.items()
            if k in GillespieSimulation.__init__.__code__.co_varnames
        },
    )
    tracker = SteadyStateTracker(sim, params["warmup_time"], params["sample_interval"])
    manager.add_tracker(tracker)
    total_run_time = params.get(
        "total_run_time", params["warmup_time"] + 200 * params["sample_interval"]
    )
    while sim.time < total_run_time:
        if not sim.step()[0]:
            break
    return {"avg_rho_M": tracker.get_steady_state_mutant_fraction()}


def run_calibration_sim(params):
    """For 'calibration' experiments measuring sector boundary drift."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        metrics_manager=manager,
        **{
            k: v
            for k, v in params.items()
            if k in GillespieSimulation.__init__.__code__.co_varnames
        },
    )
    tracker = SectorWidthTracker(sim)
    manager.add_tracker(tracker)
    step_count = 0
    max_steps = params.get("max_steps", 5_000_000)
    while len(sim.m_front_cells) > 0 and step_count < max_steps:
        if not sim.step()[0]:
            break
        step_count += 1
    return {"trajectory": tracker.get_trajectory()}


def run_diffusion_sim(params):
    """For 'diffusion' experiments measuring front roughness (KPZ-like)."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        metrics_manager=manager,
        **{
            k: v
            for k, v in params.items()
            if k in GillespieSimulation.__init__.__code__.co_varnames
        },
    )
    tracker = InterfaceRoughnessTracker(sim)
    manager.add_tracker(tracker)
    step_count = 0
    max_steps = params.get("max_steps", 10_000_000)
    while step_count < max_steps:
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break
        step_count += 1
    return {"roughness_trajectory": tracker.get_roughness_trajectory()}


def run_structure_analysis_sim(params):
    """For modern 'structure_analysis' experiments using the lightweight tracker."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        metrics_manager=manager,
        **{
            k: v
            for k, v in params.items()
            if k in GillespieSimulation.__init__.__code__.co_varnames
        },
    )
    tracker = FrontPropertiesTracker(
        sim, params["warmup_time"], params["num_samples"], params["sample_interval"]
    )
    manager.add_tracker(tracker)
    manager.initialize_all()
    total_run_time = (
        params["warmup_time"]
        + (params["num_samples"] * params["sample_interval"])
        + 10.0
    )
    while sim.time < total_run_time:
        if not sim.step()[0]:
            break
    manager.finalize_all()
    return tracker.get_results()


def run_perturbation_sim(params):
    """
    Runs a simulation with a temporary "pulse" of a high switching rate.
    This function correctly uses the standard GillespieSimulation model.
    """
    manager = MetricsManager()
    sim = GillespieSimulation(
        metrics_manager=manager,
        **{
            k: v
            for k, v in params.items()
            if k in GillespieSimulation.__init__.__code__.co_varnames
        },
    )
    tracker = FrontDynamicsTracker(sim, log_interval=params["sample_interval"])
    manager.add_tracker(tracker)
    manager.initialize_all()

    k_total_base = params["k_total"]
    k_total_pulse = params.get("k_total_pulse", k_total_base)
    pulse_start_time = params.get("pulse_start_time", float("inf"))
    pulse_duration = params.get("pulse_duration", 0)
    pulse_end_time = pulse_start_time + pulse_duration
    total_run_time = params.get("total_run_time", 4000.0)
    pulse_is_active = False

    while sim.time < total_run_time:
        if not pulse_is_active and sim.time >= pulse_start_time:
            sim.set_switching_rate(k_total_pulse)
            pulse_is_active = True
        elif pulse_is_active and sim.time >= pulse_end_time:
            sim.set_switching_rate(k_total_base)
            pulse_is_active = False

        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break

    manager.finalize_all()
    return {"timeseries": tracker.get_dataframe().to_dict("records")}


def run_spatial_fluctuation_sim_converged(params, return_full_timeseries=False):
    manager = MetricsManager()
    from config import EXPERIMENTS

    exp_name = next(
        (
            name
            for name, conf in EXPERIMENTS.items()
            if conf["CAMPAIGN_ID"] == params.get("campaign_id")
        ),
        None,
    )
    if not exp_name:
        raise ValueError("Could not find experiment config for campaign")
    env_map_name = params["environment_map"]
    actual_env_map = EXPERIMENTS[exp_name]["PARAM_GRID"][env_map_name]

    # This is the key fix for the TypeError
    unpacked_params = {k: v for k, v in params.items() if k != "environment_map"}

    sim = FluctuatingGillespieSimulation(
        metrics_manager=manager,
        environment_map=actual_env_map,
        **{
            k: v
            for k, v in unpacked_params.items()
            if k in FluctuatingGillespieSimulation.__init__.__code__.co_varnames
        },
    )
    tracker = FrontDynamicsTracker(sim, log_q_interval=params["log_q_interval"])
    manager.add_tracker(tracker)
    manager.initialize_all()

    # Dynamic stopping logic... (omitted for brevity, but it's the full loop)
    min_q_for_check = params.get("convergence_min_q", 200.0)
    check_window_size = params.get("convergence_window", 50)
    convergence_threshold = params.get("convergence_threshold", 0.01)
    running_avg_history = deque(maxlen=check_window_size)
    while sim.mean_front_position < params["length"]:
        did_step, boundary_hit = sim.step()
        if not did_step or boundary_hit:
            break
        if sim.mean_front_position > min_q_for_check:
            df = tracker.get_dataframe()
            if len(df) > check_window_size:
                current_running_avg = df["front_speed"].expanding().mean().iloc[-1]
                running_avg_history.append(current_running_avg)
                if len(running_avg_history) == check_window_size:
                    history_mean = np.mean(running_avg_history)
                    if (
                        history_mean > 1e-6
                        and (np.std(running_avg_history, ddof=1) / history_mean)
                        < convergence_threshold
                    ):
                        break

    final_df = tracker.get_dataframe()
    warmup_q = params.get("warmup_q_for_stats", 100.0)
    stats_df = final_df[final_df["mean_front_q"] > warmup_q]
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
    if return_full_timeseries:
        results["timeseries_df"] = final_df
    return results


# ==============================================================================
# MAIN WORKER DISPATCHER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Definitive simulation worker.")
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="JSON string of simulation parameters.",
    )
    args = parser.parse_args()
    params = json.loads(args.params)
    result = params.copy()
    run_mode = params.get("run_mode")

    run_functions = {
        "steady_state": run_steady_state_sim,
        "calibration": run_calibration_sim,
        "diffusion": run_diffusion_sim,
        "structure_analysis": run_structure_analysis_sim,
        "perturbation": run_perturbation_sim,
        "spatial_fluctuation_analysis": run_spatial_fluctuation_sim_converged,
    }

    try:
        if run_mode in run_functions:
            sim_results = run_functions[run_mode](params)
            result.update(sim_results)
        else:
            raise ValueError(f"Unsupported run_mode: '{run_mode}'")
    except Exception:
        result["error"] = traceback.format_exc()

    print(json.dumps(result))


if __name__ == "__main__":
    main()
