# FILE: scripts/viz/visualize_bet_hedging_dynamics.py
# A dedicated script to visually inspect the spatial dynamics of different
# bet-hedging strategies in a fluctuating environment.

import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EXPERIMENTS
from fluctuating_model import FluctuatingGillespieSimulation, Wildtype, Mutant
from hex_utils import HexPlotter

# ==============================================================================
# 1. DEFINE THE KEY STRATEGIES TO VISUALIZE
# ==============================================================================
# We choose a fixed environment and genetic background, and vary the strategy.
BASE_PARAMS = {
    "b_m": 0.85,
    "patch_width": 60,
}

STRATEGIES_TO_VISUALIZE = [
    {
        "name": "No_Switching",
        "k_total": 0.0,
        "phi": 0.0,
        "description": "Baseline: Segregated domains, no adaptation.",
    },
    {
        "name": "Slow_Unbiased_Switching",
        "k_total": 0.05,
        "phi": 0.0,
        "description": "Slow mixing, domains should slowly erode.",
    },
    {
        "name": "Fast_Polluting_Switching",
        "k_total": 1.0,
        "phi": -0.9,
        "description": "Optimal-like strategy: fast switching biased towards the specialist (mutant).",
    },
    {
        "name": "Fast_Purging_Switching",
        "k_total": 1.0,
        "phi": 0.9,
        "description": "Fast switching biased towards the generalist (wild-type).",
    },
]

# ==============================================================================
# 2. SIMULATION & VISUALIZATION CONFIGURATION
# ==============================================================================
SIM_CONFIG = {
    "width": 128,
    "length": 512,  # Short length for quick visualization
    "initial_condition_type": "mixed",
    "max_sim_q": 400.0,  # Run until the front reaches this position
    "snapshot_interval_q": 80.0,  # Take snapshot every 80 units of front position
}

BASE_FIGURES_DIR = os.path.join(project_root, "figures", "bet_hedging_dynamics_viz")
os.makedirs(BASE_FIGURES_DIR, exist_ok=True)

COLOR_MAP = {Wildtype: "#0d3b66", Mutant: "#FFC300"}  # High contrast colors
LABELS = {Wildtype: "Generalist (WT)", Mutant: "Specialist (M)"}


# ==============================================================================
# 3. MAIN VISUALIZATION SCRIPT
# ==============================================================================
def main():
    print("--- Starting Bet-Hedging Dynamics Visualization ---")

    # Load the environment map once
    env_map_name = "env_bet_hedging"
    actual_env_map = EXPERIMENTS["spatial_bet_hedging_v1"]["PARAM_GRID"][env_map_name]

    for strategy in STRATEGIES_TO_VISUALIZE:
        print(f"\n--- Running Visualization for: {strategy['name']} ---")
        print(f"    k_total = {strategy['k_total']:.2f}, phi = {strategy['phi']:.2f}")
        print(f"    Hypothesis: {strategy['description']}")

        # --- Setup simulation and plotter for this run ---
        sim_params = {
            "width": SIM_CONFIG["width"],
            "length": SIM_CONFIG["length"],
            "initial_condition_type": SIM_CONFIG["initial_condition_type"],
            "environment_map": actual_env_map,
            **BASE_PARAMS,
            **{k: v for k, v in strategy.items() if k not in ["name", "description"]},
        }

        sim = FluctuatingGillespieSimulation(**sim_params)
        plotter = HexPlotter(hex_size=2.0, labels=LABELS, colormap=COLOR_MAP)

        # --- Create a dedicated directory for this run's snapshots ---
        run_dir = os.path.join(BASE_FIGURES_DIR, strategy["name"])
        os.makedirs(run_dir, exist_ok=True)
        print(f"    Saving snapshots to: {run_dir}")

        # --- Run simulation and take snapshots based on front position ---
        next_snapshot_q = 0.0
        frame_count = 0
        with tqdm(
            total=SIM_CONFIG["max_sim_q"], desc=f"  {strategy['name']:<25}", unit="q"
        ) as pbar:
            while sim.mean_front_position < SIM_CONFIG["max_sim_q"]:
                if sim.mean_front_position >= next_snapshot_q:
                    title = (
                        f"Strategy: {strategy['name']}\n"
                        f"$k_{{total}}$ = {strategy['k_total']:.2f}, $\\phi$ = {strategy['phi']:.2f}\n"
                        f"Front Position q = {sim.mean_front_position:.1f}"
                    )

                    plotter.plot_population(
                        sim.population,
                        title=title,
                        wt_front=sim.wt_front_cells,
                        m_front=sim.m_front_cells,
                    )

                    snapshot_filename = os.path.join(
                        run_dir, f"snapshot_{frame_count:03d}.png"
                    )
                    plotter.save_figure(snapshot_filename, dpi=150)
                    frame_count += 1
                    next_snapshot_q += SIM_CONFIG["snapshot_interval_q"]

                # Run one step and update progress bar
                last_q = sim.mean_front_position
                did_step, boundary_hit = sim.step()
                pbar.update(sim.mean_front_position - last_q)

                if not did_step or boundary_hit:
                    print(
                        f"      Simulation stalled or hit boundary at q = {sim.mean_front_position:.2f}"
                    )
                    pbar.update(SIM_CONFIG["max_sim_q"])  # Complete the bar
                    break

        # Save final state
        title = (
            f"FINAL STATE - Strategy: {strategy['name']}\n"
            f"$k_{{total}}$ = {strategy['k_total']:.2f}, $\\phi$ = {strategy['phi']:.2f}\n"
            f"Front Position q = {sim.mean_front_position:.1f}"
        )
        plotter.plot_population(
            sim.population,
            title=title,
            wt_front=sim.wt_front_cells,
            m_front=sim.m_front_cells,
        )
        snapshot_filename = os.path.join(
            run_dir, f"snapshot_{frame_count:03d}_final.png"
        )
        plotter.save_figure(snapshot_filename, dpi=150)

        plt.close(plotter.fig)  # Free up memory before next run

    print("\n--- Visualization complete. ---")


if __name__ == "__main__":
    main()
