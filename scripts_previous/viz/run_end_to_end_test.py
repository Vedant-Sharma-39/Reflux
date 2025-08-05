# FILE: scripts/viz/run_end_to_end_test.py
# [DEFINITIVE, VISUAL VERSION]
# This script now visually tests the "smart", CYCLE-AWARE convergence logic.

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import deque

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
exp_config = EXPERIMENTS["spatial_bet_hedging_v1"]
TEST_PARAMS = exp_config["SIM_SETS"]["main_scan"]["base_params"].copy()
TEST_PARAMS.update(
    {
        "b_m": 0.85,
        "phi": -0.9,
        "k_total": 0.1,
        # --- [NEW] Test the most challenging case: large patch width ---
        "patch_width": 120,
    }
)


FIGURES_DIR = os.path.join(project_root, "figures", "final_end_to_end_test")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================================
# 2. MAIN TEST EXECUTION
# ==============================================================================
def main():
    print("--- Running Final End-to-End Workflow Test (Cycle-Aware) ---")
    print("\n[1/3] Running a single smart simulation...")

    env_map_name = TEST_PARAMS["environment_map"]
    actual_env_map = exp_config["PARAM_GRID"][env_map_name]

    sim_params = TEST_PARAMS.copy()
    sim_params["environment_map"] = actual_env_map

    constructor_args = FluctuatingGillespieSimulation.__init__.__code__.co_varnames
    final_sim_params = {k: v for k, v in sim_params.items() if k in constructor_args}

    manager = MetricsManager()
    final_sim_params["metrics_manager"] = manager

    sim = FluctuatingGillespieSimulation(**final_sim_params)

    tracker = FrontDynamicsTracker(sim, log_q_interval=sim_params["log_q_interval"])
    manager.add_tracker(tracker)
    manager.initialize_all()

    # --- Replicate the new worker logic for visualization ---
    running_avg_history = deque(maxlen=sim_params["convergence_window"])
    num_patches = len(actual_env_map)
    cycle_length = sim_params["patch_width"] * num_patches
    min_q_for_check = sim_params["convergence_min_cycles"] * cycle_length

    with tqdm(
        total=sim_params["length"], desc="    Sim Progress (q)", unit=" pos"
    ) as pbar:
        last_q = 0
        while sim.mean_front_position < sim_params["length"]:
            did_step, boundary_hit = sim.step()
            if not did_step or boundary_hit:
                break

            current_q = sim.mean_front_position
            pbar.update(current_q - last_q)
            last_q = current_q

            if current_q > min_q_for_check:
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

    df_history = tracker.get_dataframe()
    if df_history.empty:
        sys.exit("FATAL: No tracker data generated.")

    stats_df = df_history[df_history["mean_front_q"] > sim_params["warmup_q_for_stats"]]
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

    # --- STAGE 2: Generate convergence diagnostic plot ---
    print("\n[2/3] Generating convergence diagnostic plot...")
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
    # --- [NEW] Visualize the cycle-aware check start ---
    ax.axvline(
        min_q_for_check,
        color="gray",
        ls=":",
        lw=3,
        label=f"Convergence Check Start (after {sim_params['convergence_min_cycles']} cycles)",
    )
    ax.set_title("Diagnostic: Convergence of Running Average Speed (Cycle-Aware)")
    ax.set_xlabel("Mean Front Position (q)")
    ax.set_ylabel("Speed")
    ax.legend()
    plt.savefig(
        os.path.join(FIGURES_DIR, "final_test_cycle_aware_convergence.png"), dpi=120
    )
    plt.close()
    print("    Convergence plot saved.")

    # --- STAGE 3: Test final analysis and plotting functions ---
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
