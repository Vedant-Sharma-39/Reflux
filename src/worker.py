# FILE: src/worker.py
# A single, unified worker that dispatches to the correct simulation
# logic based on the 'run_mode' parameter.

import argparse
import json
import traceback
import numpy as np
from linear_model import GillespieSimulation
from metrics import MetricsManager, SteadyStateTracker, SectorWidthTracker


def run_steady_state_sim(params):
    """Runs a sim for a fixed time and measures the steady-state average."""
    # This worker uses the steady-state tracker.
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
    manager.initialize_all()

    while sim.time < params["total_run_time"]:
        active, _ = sim.step()
        if not active:
            break  # Stop if simulation stalls
        manager.after_step()

    avg_rho_M = tracker.get_steady_state_mutant_fraction()
    return {"avg_rho_M": avg_rho_M if not np.isnan(avg_rho_M) else -1.0}


def run_calibration_sim(params):
    """Runs a sim until extinction and tracks the sector width trajectory."""
    # This worker uses the sector width tracker and overrides k_total/phi.
    sim = GillespieSimulation(
        width=params["width"],
        length=params["length"],
        b_m=params["b_m"],
        k_total=0.0,
        phi=0.0,  # Calibration specific overrides
        initial_mutant_patch_size=params["initial_patch_size"],
    )
    manager = MetricsManager()
    manager.register_simulation(sim)
    tracker = SectorWidthTracker(sim, capture_interval=1.0)
    manager.add_tracker(tracker)
    manager.initialize_all()

    max_steps = params.get("max_steps", 3_000_000)
    step_count = 0
    while len(sim.m_front_cells) > 0 and step_count < max_steps:
        active, boundary_hit = sim.step()
        if not active or boundary_hit:
            break
        step_count += 1
        manager.after_step()

    return {"trajectory": tracker.get_trajectory(), "final_q": sim.mean_front_position}


# --- Main Dispatcher ---
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
        else:
            raise ValueError(f"Unknown run_mode: '{run_mode}'")

        result.update(sim_results)

    except Exception:
        result["error"] = traceback.format_exc()

    print(json.dumps(result))


if __name__ == "__main__":
    main()
