# FILE: src/worker.py
# [DEFINITIVELY CORRECTED & COMPLETE v4]
# This version fixes a critical bug where the MetricsManager was not being
# passed to the GillespieSimulation, preventing tracker hooks from being called.
# All run functions are now corrected to properly link the simulation and manager.

import argparse
import json
import traceback
import numpy as np
from linear_model import GillespieSimulation
from metrics import (
    MetricsManager,
    SectorWidthTracker,
    InterfaceRoughnessTracker,
    FrontPropertiesTracker,
    CorrelationAndStructureTracker,
    SteadyStateTracker,
)


# ==============================================================================
# CORRECTED RUN FUNCTIONS
# ==============================================================================


def run_steady_state_sim(params):
    """[CORRECTED] For legacy 'steady_state' experiments (e.g., p1_definitive_v2)."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        width=params["width"],
        length=params.get("length", 50000),
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
        initial_condition_type=params.get("initial_condition_type", "mixed"),
        metrics_manager=manager,
    )
    tracker = SteadyStateTracker(sim, params["warmup_time"], params["sample_interval"])
    manager.add_tracker(tracker)
    manager.initialize_all()

    total_run_time = params.get(
        "total_run_time", params["warmup_time"] + 200 * params["sample_interval"]
    )
    while sim.time < total_run_time:
        active, _ = sim.step()
        if not active:
            break

    manager.finalize_all()
    avg_rho_M = tracker.get_steady_state_mutant_fraction()
    return {"avg_rho_M": avg_rho_M if not np.isnan(avg_rho_M) else -1.0}


def run_correlation_analysis_sim(params):
    """[CORRECTED] For legacy 'correlation_analysis' experiments using the heavy g(r) tracker."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        width=params["width"],
        length=params.get("length", 50000),
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params.get("phi", 0.0),
        initial_condition_type=params.get("initial_condition_type", "mixed"),
        metrics_manager=manager,
    )
    corr_tracker = CorrelationAndStructureTracker(
        sim, params["warmup_time"], params["num_samples"], params["sample_interval"]
    )
    rho_tracker = SteadyStateTracker(
        sim, params["warmup_time"], params["sample_interval"]
    )
    manager.add_tracker(corr_tracker)
    manager.add_tracker(rho_tracker)
    manager.initialize_all()

    total_run_time = (
        params["warmup_time"]
        + (params["num_samples"] * params["sample_interval"])
        + 10.0
    )
    while sim.time < total_run_time:
        active, _ = sim.step()
        if not active:
            break

    manager.finalize_all()
    results = corr_tracker.get_results()
    avg_rho_M = rho_tracker.get_steady_state_mutant_fraction()
    results["avg_rho_M"] = avg_rho_M if not np.isnan(avg_rho_M) else -1.0
    return results


def run_calibration_sim(params):
    """[CORRECTED] For 'calibration' experiments measuring sector boundary drift."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params.get("k_total", 0.0),
        phi=params.get("phi", 0.0),
        initial_condition_type=params.get("initial_condition_type", "patch"),
        initial_mutant_patch_size=params.get("initial_mutant_patch_size", 0),
        metrics_manager=manager,
    )
    tracker = SectorWidthTracker(sim)
    manager.add_tracker(tracker)

    max_steps = params.get("max_steps", 5_000_000)
    step_count = 0
    while len(sim.m_front_cells) > 0 and step_count < max_steps:
        active, _ = sim.step()
        if not active:
            break
        step_count += 1
    return {"trajectory": tracker.get_trajectory()}


def run_diffusion_sim(params):
    """[CORRECTED] For 'diffusion' experiments measuring front roughness (KPZ-like)."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params.get("k_total", 0.0),
        phi=params.get("phi", 0.0),
        initial_condition_type=params.get("initial_condition_type", "mixed"),
        initial_mutant_patch_size=params.get("initial_mutant_patch_size", 0),
        metrics_manager=manager,
    )
    tracker = InterfaceRoughnessTracker(sim)
    manager.add_tracker(tracker)

    max_steps = params.get("max_steps", 10_000_000)
    step_count = 0
    while step_count < max_steps:
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break
        step_count += 1
    return {"roughness_trajectory": tracker.get_roughness_trajectory()}


def run_structure_analysis_sim(params):
    """[CORRECTED] For modern 'structure_analysis' experiments using the lightweight tracker."""
    manager = MetricsManager()
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
        initial_condition_type=params.get("initial_condition_type", "mixed"),
        initial_mutant_patch_size=params.get("initial_mutant_patch_size", 0),
        metrics_manager=manager,
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
        active, _ = sim.step()
        if not active:
            break

    manager.finalize_all()
    return tracker.get_results()


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
    try:
        if run_mode == "steady_state":
            sim_results = run_steady_state_sim(params)
        elif run_mode == "calibration":
            sim_results = run_calibration_sim(params)
        elif run_mode == "diffusion":
            sim_results = run_diffusion_sim(params)
        elif run_mode == "correlation_analysis":
            sim_results = run_correlation_analysis_sim(params)
        elif run_mode == "structure_analysis":
            sim_results = run_structure_analysis_sim(params)
        else:
            raise ValueError(f"Unsupported run_mode: '{run_mode}'")
        result.update(sim_results)
    except Exception:
        result["error"] = traceback.format_exc()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
