# src/phase_worker.py
# [OPTIMIZED] A dedicated worker for the phase transition experiment that measures
# survival probability. It now stops as soon as the WT population goes extinct.

import argparse
import json
import traceback


def run_phase_simulation(params):
    """Runs a single simulation and returns 1 if WT survived, 0 otherwise."""
    from linear_model import GillespieSimulation
    from metrics import MetricsManager, SurvivalTracker  # Use the new tracker

    result = params.copy()
    try:
        manager = MetricsManager()
        sim = GillespieSimulation(
            width=params["width"],
            length=params["length"],
            b_m=params["b_m"],
            k_total=params["k_total"],
            phi=params["phi"],
            initial_mutant_patch_size=0,  # Start with an all-WT front
            metrics_manager=manager,
        )

        tracker = SurvivalTracker(sim)
        manager.add_tracker(tracker)
        manager.initialize_all()

        # --- [OPTIMIZED] Main Simulation Loop ---
        # The loop now stops as soon as the wild-type front disappears,
        # which is the absorbing event we are studying.
        while len(sim.wt_front_cells) > 0 and sim.time < params["total_run_time"]:
            active, boundary_hit = sim.step()
            # If population dies or hits wall, this secondary check handles it.
            if not active or boundary_hit:
                break

        # Finalize is called after the loop stops for any reason.
        # The SurvivalTracker will correctly report the final state of wt_front_cells.
        manager.finalize_all()
        result["wt_survived"] = tracker.wt_survived

    except Exception:
        result["wt_survived"] = -1  # Error code
        result["error"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a single phase transition survival simulation."
    )
    parser.add_argument("--params", type=str, required=True)
    args = parser.parse_args()
    params = json.loads(args.params)
    result = run_phase_simulation(params)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
