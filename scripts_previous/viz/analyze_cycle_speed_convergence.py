# FILE: scripts/viz/analyze_cycle_speed_convergence.py
# A new diagnostic script to test the convergence of the average front speed
# by measuring it over full environmental cycles.

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EXPERIMENTS
from fluctuating_model import FluctuatingGillespieSimulation, Wildtype, Mutant
from hex_utils import HexPlotter, Hex

# ... [SIM_PARAMS and other constants remain the same] ...
SIM_PARAMS = {
    "name": "Fast_Polluting_Strategy",
    "width": 128,
    "length": 4096,
    "b_m": 0.85,
    "k_total": 1.0,
    "phi": -0.9,
    "patch_width": 60,
    "initial_condition_type": "mixed",
}
MAX_SIM_Q = 3000.0
FIGURES_DIR = os.path.join(project_root, "figures", "convergence_analysis")
os.makedirs(FIGURES_DIR, exist_ok=True)
COLOR_MAP = {Wildtype: "#0d3b66", Mutant: "#FFC300"}
LABELS = {Wildtype: "Generalist (WT)", Mutant: "Specialist (M)"}
ENV_COLOR_MAP = {Wildtype: "#a9d6e5", Mutant: "#feeaa7"}


# --- [CORRECTED] Background Drawing Function ---
def draw_environment_background(
    ax, hex_plotter, patch_width, num_patches, max_q, env_map
):
    """Draws colored vertical spans to represent the environmental patches."""
    for i in range(int(np.ceil(max_q / patch_width))):
        patch_idx = i % num_patches
        patch_info = env_map.get(patch_idx, {})

        favored_type = Wildtype
        if patch_info.get("b_m", 0) > patch_info.get("b_wt", 1.0):
            favored_type = Mutant

        color = ENV_COLOR_MAP.get(favored_type, "lightgrey")

        # Convert the patch boundaries (in q-coordinates) to the plotter's x-coordinates
        x_start, _ = hex_plotter.hex_to_cartesian(
            Hex(i * patch_width, 0, -i * patch_width)
        )
        x_end, _ = hex_plotter.hex_to_cartesian(
            Hex((i + 1) * patch_width, 0, -(i + 1) * patch_width)
        )

        # Add a small buffer to the span to align with hexagon centers
        ax.axvspan(
            x_start - hex_plotter.size * 0.75,
            x_end - hex_plotter.size * 0.75,
            color=color,
            alpha=0.3,
            zorder=-10,
        )


def main():
    print(f"--- Running Cycle Speed Convergence Analysis for: {SIM_PARAMS['name']} ---")

    env_map_name = "env_bet_hedging"
    actual_env_map = EXPERIMENTS["spatial_bet_hedging_v1"]["PARAM_GRID"][env_map_name]
    num_patches = len(actual_env_map)
    cycle_length = SIM_PARAMS["patch_width"] * num_patches

    constructor_args = FluctuatingGillespieSimulation.__init__.__code__.co_varnames
    sim_constructor_params = {
        k: v for k, v in SIM_PARAMS.items() if k in constructor_args
    }

    manager = MetricsManager()
    sim_constructor_params["metrics_manager"] = manager
    sim = FluctuatingGillespieSimulation(
        environment_map=actual_env_map, **sim_constructor_params
    )

    cycle_data = [{"time": 0.0, "q": 0.0}]
    next_cycle_boundary_q = cycle_length

    with tqdm(total=MAX_SIM_Q, desc="Simulating Cycles", unit="q") as pbar:
        while sim.mean_front_position < MAX_SIM_Q:
            last_q = sim.mean_front_position
            did_step, boundary_hit = sim.step()
            if not did_step or boundary_hit:
                pbar.set_description("Stalled/Finished")
                break
            pbar.update(sim.mean_front_position - last_q)
            if sim.mean_front_position >= next_cycle_boundary_q:
                cycle_data.append({"time": sim.time, "q": sim.mean_front_position})
                next_cycle_boundary_q += cycle_length

    print("\n--- Processing cycle data ---")
    df = pd.DataFrame(cycle_data)
    df["cycle_number"] = np.arange(len(df))
    df["delta_q"] = df["q"].diff()
    df["delta_t"] = df["time"].diff()
    df["cycle_speed"] = df["delta_q"] / df["delta_t"]
    df["running_avg_speed"] = df["cycle_speed"].expanding().mean()

    print("--- Generating final plot ---")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(20, 16),
        gridspec_kw={"height_ratios": [1.2, 1]},
        constrained_layout=True,
    )
    fig.suptitle(f"Speed Convergence Analysis: {SIM_PARAMS['name']}", fontsize=22)

    plotter = HexPlotter(hex_size=1.5, labels=LABELS, colormap=COLOR_MAP, ax=ax1)

    draw_environment_background(
        ax1, plotter, SIM_PARAMS["patch_width"], num_patches, sim.length, actual_env_map
    )

    ax1.set_title(f"Final State at q = {sim.mean_front_position:.1f}")
    plotter.plot_population(
        sim.population, wt_front=sim.wt_front_cells, m_front=sim.m_front_cells
    )

    ax2.plot(
        df["cycle_number"],
        df["cycle_speed"],
        "o-",
        c="darkcyan",
        alpha=0.6,
        label="Speed per Cycle",
    )
    ax2.plot(
        df["cycle_number"],
        df["running_avg_speed"],
        "o-",
        c="purple",
        lw=2.5,
        label="Running Average Speed",
    )

    if len(df) > 5:
        converged_speed = df["cycle_speed"].iloc[-5:].mean()
        ax2.axhline(
            converged_speed,
            color="red",
            ls="--",
            label=f"Final Converged Speed ≈ {converged_speed:.3f}",
        )

    ax2.set_title("Convergence of Average Speed per Environmental Cycle")
    ax2.set_xlabel("Cycle Number")
    ax2.set_ylabel("Average Front Speed (in cycle)")
    ax2.set_xlim(left=0.5)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, which="both", ls="--")
    ax2.legend(fontsize=14)

    filename = os.path.join(
        FIGURES_DIR, f"convergence_debug_with_running_avg_{SIM_PARAMS['name']}.png"
    )
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"\nAnalysis complete. Plot saved to: {filename}")
    print(df[["cycle_number", "cycle_speed", "running_avg_speed"]].dropna().round(4))


if __name__ == "__main__":
    main()
