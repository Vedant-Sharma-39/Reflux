# src/worker.py
# Corrected standalone worker script.

import argparse
import json
import numpy as np
import traceback


def run_simulation_from_args(params, is_debug=False):
    # --- Always import necessary modules ---
    from linear_model import GillespieSimulation
    from metrics import MetricsManager, SteadyStateTracker

    result = params.copy()

    try:
        # --- Always instantiate the simulation and trackers ---
        sim = GillespieSimulation(
            width=params["width"],
            length=params["length"],
            b_m=params["b_m"],
            k_total=params["k_total"],
            phi=params["phi"],
        )
        manager = MetricsManager()
        manager.register_simulation(sim)
        ss_tracker = SteadyStateTracker(
            sim,
            warmup_time=params["warmup_time"],
            sample_interval=params["sample_interval"],
        )
        manager.add_tracker(ss_tracker)
        manager.initialize_all()

        # --- Debug logging is optional ---
        if is_debug:
            print("\n--- Starting DEBUG Log for Parameters ---")
            print(params)
            print("\nTime\t\tWT_Front\tM_Front\t\tTotal_Rate")
            print("-" * 50)
            next_log_time = 0.0

        # --- Main simulation loop ---
        while sim.time < params["total_run_time"]:
            if is_debug and sim.time >= next_log_time:
                wt_c = len(sim.wt_front_cells)
                m_c = len(sim.m_front_cells)
                print(f"{sim.time:.2f}\t\t{wt_c}\t\t{m_c}\t\t{sim.total_rate:.2f}")
                next_log_time += 20.0

            did_step, _ = sim.step()
            
            if manager:
                manager.after_step()

            if not did_step:
                # Stall condition, break the loop
                break

        # --- Always get the result after the loop ---
        avg_rho_M = ss_tracker.get_steady_state_mutant_fraction()
        result["avg_rho_M"] = avg_rho_M if not np.isnan(avg_rho_M) else -1.0

    except Exception as e:
        result["avg_rho_M"] = -2.0
        result["error"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a single Gillespie simulation.")
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="A JSON string of simulation parameters.",
    )
    parser.add_argument(
        "--debug-log", action="store_true", help="Enable detailed time-lapse logging."
    )
    args = parser.parse_args()

    params = json.loads(args.params)
    result = run_simulation_from_args(params, is_debug=args.debug_log)

    # Only print JSON if not in debug mode
    if not args.debug_log:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
