# FILE: src/worker.py
#

import argparse
import json
import traceback
import numpy as np
from linear_model import GillespieSimulation
from metrics import (
    MetricsManager,
    SteadyStateTracker,
    SectorWidthTracker,
    SpatialCorrelationTracker,
    InterfaceRoughnessTracker,
)


def run_steady_state_sim(params):
    """Corresponds to the "p1_definitive_v2" experiment."""
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params["phi"],
    )
    manager = MetricsManager()
    manager.register_simulation(sim)
    tracker = SteadyStateTracker(
        sim,
        warmup_time=params["warmup_time"],
        sample_interval=params["sample_interval"],
    )
    manager.add_tracker(tracker)
    total_run_time = params.get(
        "total_run_time", params["warmup_time"] + 200 * params["sample_interval"]
    )
    while sim.time < total_run_time:
        active, _ = sim.step()
        if not active:
            break
        manager.after_step()
    avg_rho_M = tracker.get_steady_state_mutant_fraction()
    return {"avg_rho_M": avg_rho_M if not np.isnan(avg_rho_M) else -1.0}


def run_calibration_sim(params):
    """Corresponds to the "calibration_v4" experiment."""
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=0.0,
        phi=0.0,
        initial_mutant_patch_size=params["initial_patch_size"],
    )
    manager = MetricsManager()
    manager.register_simulation(sim)
    tracker = SectorWidthTracker(sim, capture_interval=1.0)
    manager.add_tracker(tracker)
    max_steps = params.get("max_steps", 3_000_000)
    step_count = 0
    while len(sim.m_front_cells) > 0 and step_count < max_steps:
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break
        step_count += 1
        manager.after_step()
    return {"trajectory": tracker.get_trajectory(), "final_q": sim.mean_front_position}


def run_correlation_sim(params):
    """Corresponds to the "spatial_structure_v1" experiment."""
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=params["k_total"],
        phi=params.get("phi", 0.0),
    )
    manager = MetricsManager()
    manager.register_simulation(sim)
    corr_tracker = SpatialCorrelationTracker(
        sim,
        warmup_time=params["warmup_time"],
        num_samples=params["num_samples"],
        sample_interval=params["sample_interval"],
    )
    manager.add_tracker(corr_tracker)

    # Also track rho_M for potential future use, since it's cheap
    rho_tracker = SteadyStateTracker(
        sim,
        warmup_time=params["warmup_time"],
        sample_interval=params["sample_interval"],
    )
    manager.add_tracker(rho_tracker)

    total_run_time = (
        params["warmup_time"]
        + (params["num_samples"] * params["sample_interval"])
        + 1.0
    )
    while sim.time < total_run_time:
        active, _ = sim.step()
        if not active:
            break
        manager.after_step()
    correlation_function = corr_tracker.get_correlation_function()
    avg_rho_M = rho_tracker.get_steady_state_mutant_fraction()
    return {
        "g_r": correlation_function,
        "avg_rho_M": avg_rho_M if not np.isnan(avg_rho_M) else -1.0,
    }


def run_diffusion_sim(params):
    """Runs a sim from a flat front to measure interface roughness."""
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=1.0,
        k_total=0.0,
        phi=0.0,
        initial_mutant_patch_size=0,
    )
    manager = MetricsManager()
    manager.register_simulation(sim)
    tracker = InterfaceRoughnessTracker(sim, capture_interval=0.5)
    manager.add_tracker(tracker)
    manager.initialize_all()

    max_steps = params.get("max_steps", 5_000_000)
    step_count = 0
    while step_count < max_steps:
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break
        step_count += 1
        manager.after_step()

    return {"roughness_trajectory": tracker.get_roughness_trajectory()}


def main():
    parser = argparse.ArgumentParser(description="Unified simulation worker.")
    parser.add_argument(
        "--params", type=str, required=True, help="JSON string of parameters."
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
        elif run_mode == "correlation_analysis":
            sim_results = run_correlation_sim(params)
        elif run_mode == "diffusion":
            sim_results = run_diffusion_sim(params)
        else:
            raise ValueError(f"Unknown run_mode: '{run_mode}'")
        result.update(sim_results)
    except Exception:
        result["error"] = traceback.format_exc()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
