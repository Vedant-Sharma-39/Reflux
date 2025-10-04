# FILE: scripts/debug_aif_streaks.py (CORRECTED with Origin-Based Filtering)

import sys
from pathlib import Path
import json
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Simulation and Analysis Functions ---
from src.core.model_aif import AifModelSimulation
from scripts.analyze_aif_streaks import find_streaks_via_bfs
from src.utils.analysis_helpers import load_population_data, measure_width_radial_binning

def main():
    """
    Orchestrates a single simulation-to-analysis pipeline for debugging.
    FILTERS STREAKS based on ORIGIN and a minimum SIZE threshold for cohesion.
    """
    print("--- Standalone AIF Streak Debugging & Visualization Script ---")

    # Part 1: Run Simulation (Unchanged)
    # ... (simulation params and run logic is exactly the same) ...
    print("\n[1/3] Configuring and running a short simulation...")
# FILE: scripts/debug_aif_streaks.py

# ... inside the main() function ...
    
    params = {
        "campaign_id": "debug_aif_streaks",
        "simulation_class": "AifModelSimulation",
        
        # Use the new correlated initial condition
        "initial_condition_type": "aif_droplet_grf",
        
        # Control the size of the initial clusters
        "correlation_length": 1,
        
        "initial_droplet_radius": 150,
        "initial_resistant_fraction": 0.1, # Preserves the 90/10 ratio
        "max_steps": 250_000,
        "b_sus": 1.0, 
        "b_res": 0.95, 
        "b_comp": 1.0, 
        "k_res_comp": 0.0,
        
        # The radial_bias_strength parameter is now completely removed.
    }

    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_pop_file = output_dir / "aif_streaks_for_debug.json.gz"
    sim = AifModelSimulation(**params)
    with tqdm(total=params["max_steps"], desc="Simulating") as pbar:
        while sim.step_count < params["max_steps"]:
            active, _ = sim.step()
            pbar.update(1)
            if not active: break
    final_pop_data = [{"q": h.q, "r": h.r, "type": t} for h, t in sim.population.items()]
    with gzip.open(temp_pop_file, "wt", encoding="utf-8") as f: json.dump(final_pop_data, f)
    print(f"Final population data saved to: {temp_pop_file}")

    # Part 2: Analyze with Origin and Size Filtering
    print("\n[2/3] Analyzing the final population to identify and measure streaks...")

    df_pop = load_population_data(temp_pop_file)
    df_resistant = df_pop[df_pop['type'].isin({2, 3})].copy()
    df_susceptible = df_pop[~df_pop['type'].isin({2, 3})]
    
    df_streaks_all = find_streaks_via_bfs(df_resistant)

    # Filter by ORIGIN
    print("Filtering for streaks originating at the initial front...")
    initial_radius = params['initial_droplet_radius']
    INITIAL_FRONT_SHELL_WIDTH = 0.10
    radius_upper_bound = initial_radius
    radius_lower_bound = initial_radius * (1 - INITIAL_FRONT_SHELL_WIDTH)
    origin_streak_ids = []
    for streak_id, group in df_streaks_all.groupby('streak_id'):
        if ((group['radius'] >= radius_lower_bound) & (group['radius'] <= radius_upper_bound)).any():
            origin_streak_ids.append(streak_id)
    df_streaks_origin = df_streaks_all[df_streaks_all['streak_id'].isin(origin_streak_ids)].copy()
    
    # NEW AND CRITICAL: Filter by Minimum Size
    MIN_STREAK_SIZE_THRESHOLD = 20
    valid_streak_ids = []
    for streak_id, group in df_streaks_origin.groupby('streak_id'):
        if len(group) >= MIN_STREAK_SIZE_THRESHOLD:
            valid_streak_ids.append(streak_id)
    df_streaks_filtered = df_streaks_origin[df_streaks_origin['streak_id'].isin(valid_streak_ids)].copy()
    print(f"Found {len(valid_streak_ids)} significant streaks with >{MIN_STREAK_SIZE_THRESHOLD} cells.")

    # Part 3: Analyze each VALID, COHESIVE streak
    all_streak_trajectories = []
    for streak_id in valid_streak_ids:
        df_single_streak = df_streaks_filtered[df_streaks_filtered['streak_id'] == streak_id]
        df_trajectory = measure_width_radial_binning(df_single_streak.copy())
        if not df_trajectory.empty:
            df_trajectory['streak_id'] = streak_id
            df_trajectory['arc_length'] = df_trajectory['mean_radius'] * df_trajectory['width_rad']
            all_streak_trajectories.append(df_trajectory)

    if not all_streak_trajectories:
        sys.exit("Analysis failed: No significant streaks were found after filtering.")
    df_final = pd.concat(all_streak_trajectories)

    # Part 4: Visualization
    print("\n[3/3] Generating final analysis plot...")
    # ... (Plotting code is identical) ...
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.suptitle(f"Debug Analysis of Significant Clonal Streaks (Filtered)", fontsize=24, y=0.98)
    ax1.set_title("Final Streak Identification (Large, Cohesive Streaks Only)", fontsize=18)
    ax1.scatter(df_susceptible['x'], df_susceptible['y'], c='lightgrey', s=1, alpha=0.3)
    sns.scatterplot(
        data=df_streaks_filtered, x='x', y='y', hue='streak_id', palette='tab20',
        s=5, legend=None, ax=ax1
    )
    ax1.set_xlabel("X Coordinate"); ax1.set_ylabel("Y Coordinate")
    ax1.set_aspect('equal', 'box')
    ax2.set_title("Streak Arc Length vs. Radius", fontsize=18)
    sns.lineplot(
        data=df_final, x='mean_radius', y='arc_length', hue='streak_id',
        palette='tab20', legend='full', ax=ax2, lw=2.5
    )
    ax2.set_xlabel("Radius from Colony Center"); ax2.set_ylabel("Measured Arc Length (Linear Units)")
    ax2.grid(True, which='both', linestyle=':')
    ax2.set_ylim(bottom=0)
    output_path = output_dir / "debug_streak_analysis_result_filtered.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Standalone analysis complete! Figure saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()