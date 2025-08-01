# FILE: scripts/viz/run_end_to_end_test.py
# [DEFINITIVE, VISUAL VERSION]
# This final test script now includes a TQDM progress bar to visualize
# the progress of the initial simulation run.

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from tqdm import tqdm

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "scripts"))

from config import EXPERIMENTS
from fluctuating_model import FluctuatingGillespieSimulation
from metrics import MetricsManager, FrontDynamicsTracker
from analyze_spatial_strategies import plot_landscapes, plot_example_timeseries

# ==============================================================================
# 1. DEFINE THE SINGLE TEST SCENARIO
# ==============================================================================
TEST_PARAMS = {
    "b_m": 0.85,
    "phi": -0.9,
    "k_total": 0.1,
    "patch_width": 120,
    "width": 256,
    "length": 1536,
    "initial_condition_type": "mixed",
    "environment_map": "env_bet_hedging",
    "campaign_id": "spatial_bet_hedging_v1",
    "log_q_interval": 2.0,
    "convergence_min_q": 300.0,
    "convergence_window": 50,
    "convergence_threshold": 0.01,
    "warmup_q_for_stats": 150.0,
}

FIGURES_DIR = os.path.join(project_root, "figures", "final_end_to_end_test")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================================
# 2. MAIN TEST EXECUTION
# ==============================================================================
def main():
    print("--- Running Final End-to-End Workflow Test ---")

    # --- STAGE 1: Simulate using the 'smart' logic WITH a progress bar ---
    print("\n[1/3] Running a single smart simulation...")

    # --- [THE VISUALIZATION FIX] ---
    # We replicate the worker's logic here to wrap it in a progress bar.

    # Setup simulation objects
    exp_name = "spatial_bet_hedging_v1"
    env_map_name = TEST_PARAMS["environment_map"]
    actual_env_map = EXPERIMENTS[exp_name]["PARAM_GRID"][env_map_name]

    sim_params = TEST_PARAMS.copy()
    sim_params["environment_map"] = actual_env_map
    unpacked_sim_params = {
        k: v for k, v in sim_params.items() if k != "environment_map"
    }

    sim = FluctuatingGillespieSimulation(
        environment_map=actual_env_map,
        **{
            k: v
            for k, v in unpacked_sim_params.items()
            if k in FluctuatingGillespieSimulation.__init__.__code__.co_varnames
        },
    )
    manager = MetricsManager()
    tracker = FrontDynamicsTracker(sim, log_q_interval=sim_params["log_q_interval"])
    manager.add_tracker(tracker)
    manager.initialize_all()

    # Dynamic stopping loop wrapped in TQDM
    running_avg_history = deque(maxlen=sim_params["convergence_window"])
    with tqdm(
        total=sim_params["length"], desc="    Sim Progress (q)", unit=" pos"
    ) as pbar:
        last_q = 0
        while sim.mean_front_position < sim_params["length"]:
            did_step, boundary_hit = sim.step()
            if not did_step or boundary_hit:
                break

            # Update progress bar
            current_q = sim.mean_front_position
            pbar.update(current_q - last_q)
            last_q = current_q

            # Check for convergence
            if current_q > sim_params["convergence_min_q"]:
                df = tracker.get_dataframe()
                if len(df) > sim_params["convergence_window"]:
                    current_running_avg = df["front_speed"].expanding().mean().iloc[-1]
                    running_avg_history.append(current_running_avg)
                    if len(running_avg_history) == sim_params["convergence_window"]:
                        history_mean = np.mean(running_avg_history)
                        if (
                            history_mean > 1e-6
                            and (np.std(running_avg_history, ddof=1) / history_mean)
                            < sim_params["convergence_threshold"]
                        ):
                            pbar.set_description("    Converged!")
                            break

    # Calculate final results from the tracker data
    df_history = tracker.get_dataframe()
    warmup_q = sim_params.get("warmup_q_for_stats", 100.0)
    stats_df = df_history[df_history["mean_front_q"] > warmup_q]
    summary_results = {
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
    print("    Simulation finished. Final summary statistics:")
    print(f"    {summary_results}")

    # --- The rest of the script is unchanged ---
    print("\n[2/3] Generating convergence diagnostic plot...")
    if not df_history.empty:
        df_history["speed_running_avg"] = df_history["front_speed"].expanding().mean()
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(
            df_history["mean_front_q"],
            df_history["speed_running_avg"],
            "r-",
            lw=2.5,
            label="Running Average Speed",
        )
        ax.axvline(
            summary_results["final_q"],
            color="k",
            ls="--",
            lw=2,
            label=f"Simulation Stopped at q={summary_results['final_q']:.1f}",
        )
        ax.axvline(
            sim_params["convergence_min_q"],
            color="gray",
            ls=":",
            label="Convergence Check Start",
        )
        ax.set_title("Diagnostic: Convergence of Running Average Speed")
        ax.set_xlabel("Mean Front Position (q)")
        ax.set_ylabel("Speed")
        ax.legend()
        plt.savefig(
            os.path.join(FIGURES_DIR, "final_test_convergence_plot.png"), dpi=120
        )
        plt.close()
        print("    Convergence plot saved.")

    print("\n[3/3] Testing final analysis and plotting functions...")
    mock_df_avg = pd.DataFrame([{**TEST_PARAMS, **summary_results}])
    plot_landscapes(
        mock_df_avg,
        "avg_front_speed",
        "Mock Fitness Landscape",
        os.path.join(FIGURES_DIR, "final_test_mock_heatmap.png"),
    )
    print("    Mock heatmap saved successfully.")

    timeseries_dir = os.path.join(FIGURES_DIR, "final_test_timeseries")
    os.makedirs(timeseries_dir, exist_ok=True)
    plot_example_timeseries(sim_params, timeseries_dir)
    print("    Final timeseries plot saved successfully.")

    print("\n--- End-to-End Test Complete ---")


if __name__ == "__main__":
    main()
