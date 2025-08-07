# FILE: scripts/utils/debug_front_speed.py
# A dedicated script to visually and quantitatively test the front speed
# of a single simulation run.

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.model import GillespieSimulation, Wildtype, Mutant
from src.core.hex_utils import HexPlotter

# ==============================================================================
# 1. DEFINE PARAMETERS FOR THE TEST RUN
# ==============================================================================
TEST_PARAMS = {
    "width": 128,
    "length": 1024,
    "initial_condition_type": "mixed",
    "environment_map": {0: {"b_wt": 1.0}, 1: {"b_wt": 0.0, "b_m": 1.0}},
    "b_m": 0.95,
    "phi": 0,
    "k_total": 0.02,
    "patch_width": 30,
}

DEBUG_CONFIG = {
    "max_steps": 100000,
    "log_interval_steps": 500,
    "snapshot_interval_steps": 25000,
}

# --- Output Directory ---
OUTPUT_DIR = os.path.join(project_root, "debug_images", "front_speed_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 2. MAIN VERIFICATION SCRIPT
# ==============================================================================
def main():
    print("--- Running Front Speed Visualization and Debug Test ---")
    print(f"Parameters: {TEST_PARAMS}")
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # Use a professional plot style
    plt.style.use('seaborn-v0_8-paper')

    sim = GillespieSimulation(**TEST_PARAMS)
    plotter = HexPlotter(
        hex_size=2.0,
        labels={Wildtype: "WT", Mutant: "M"},
        colormap={Wildtype: "#0d3b66", Mutant: "#FFC300"},
    )

    history = []

    for step in tqdm(range(DEBUG_CONFIG["max_steps"]), desc="Simulating"):
        if step % DEBUG_CONFIG["log_interval_steps"] == 0:
            history.append(
                {"step": step, "time": sim.time, "q_position": sim.mean_front_position}
            )

        if step > 0 and step % DEBUG_CONFIG["snapshot_interval_steps"] == 0:
            title = f"Step: {step}, Time: {sim.time:.2f}, Q: {sim.mean_front_position:.2f}"
            # Pass the front cells to the plotter for highlighting
            plotter.plot_population(
                sim.population,
                title=title,
                wt_front=sim.wt_front_cells.keys(),
                m_front=sim.m_front_cells.keys(),
            )
            snapshot_path = os.path.join(OUTPUT_DIR, f"snapshot_step_{step}.png")
            plotter.save_figure(snapshot_path, dpi=300)

        active, _ = sim.step()
        if not active:
            print("\nSimulation stalled.")
            break

    if not history:
        print("No data was recorded. Exiting.")
        return

    df = pd.DataFrame(history)
    fit_df = df[df["step"] > DEBUG_CONFIG["max_steps"] * 0.1]
    if len(fit_df) < 2:
        print("Not enough data to perform a fit.")
        return

    slope, intercept, r_value, _, _ = linregress(fit_df["time"], fit_df["q_position"])
    measured_speed = slope

    print("\n--- Verification Results ---")
    print(f"Calculated Average Front Speed: {measured_speed:.4f} (sites/time)")
    print(f"Linear Fit R-squared: {r_value**2:.4f}")

    # Create the final, publication-quality speed plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot raw data with transparency
    ax.plot(
        df["time"],
        df["q_position"],
        "o",
        markersize=4,
        alpha=0.4,
        label="Mean Front Position (Data)",
    )

    # Plot the best-fit line
    fit_time = fit_df["time"]
    fit_line = intercept + slope * fit_time
    ax.plot(
        fit_time,
        fit_line,
        "-",
        color="#D62728", # A strong red color
        lw=2.5,
        label=f"Linear Fit (Speed = {measured_speed:.4f})",
    )

    ax.set_title("Front Propagation Over Time", fontsize=16, weight='bold')
    ax.set_xlabel("Simulation Time", fontsize=14)
    ax.set_ylabel("Mean Front Position (q-coordinate)", fontsize=14)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "final_speed_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nFinal speed plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()