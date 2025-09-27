# FILE: scripts/paper_figures/fig_aif_equilibrium_analysis.py (NEW FILE)

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing
import os
from scipy.stats import linregress

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data
from scripts.paper_figures.fig_aif_spatial_trajectories import process_simulation_output

def calculate_trajectory_slope(df_group: pd.DataFrame) -> pd.Series:
    """
    For a single simulation's trajectory, calculate the slope of
    arc length vs. radius. This quantifies whether the sector is
    growing, shrinking, or stable.
    """
    # Use only the later part of the trajectory for a stable slope
    start_radius = df_group['mean_radius'].quantile(0.25)
    stable_trajectory = df_group[df_group['mean_radius'] > start_radius]
    
    if len(stable_trajectory) < 5:
        return pd.Series({"growth_rate": np.nan})

    # Perform linear regression
    slope, _, _, _, _ = linregress(stable_trajectory['mean_radius'], stable_trajectory['arc_length'])
    return pd.Series({"growth_rate": slope})

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    df_summary = load_aggregated_data(campaign_id, str(PROJECT_ROOT))
    if df_summary.empty: sys.exit(f"Data for '{campaign_id}' is empty.")

    pop_dir = PROJECT_ROOT / "data" / campaign_id / "populations"
    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    if not pop_dir.exists(): sys.exit(f"Population directory not found: {pop_dir}")

    tasks = [{"pop_file_path": pop_dir / f"pop_{r['task_id']}.json.gz", "b_res": r["b_res"], 
              "initial_width": r["sector_width_initial"], "replicate": r["replicate"]} for _, r in df_summary.iterrows()]

    num_workers = max(1, os.cpu_count() - 1)
    print(f"Starting parallel analysis of {len(tasks)} population files...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_simulation_output, tasks), total=len(tasks)))
    
    all_measurements = [df for df in results if df is not None and not df.empty]
    if not all_measurements: sys.exit("No valid measurements extracted.")
    df_full = pd.concat(all_measurements)

    # --- New Analysis: Calculate Growth Rate for Each Trajectory ---
    print("Calculating growth rate (slope) for each trajectory...")
    # Add a unique ID for each individual simulation run
    df_full['run_id'] = df_full['b_res'].astype(str) + '_' + df_full['initial_width'].astype(str) + '_' + df_full['replicate'].astype(str)
    
    growth_rates = df_full.groupby('run_id').apply(calculate_trajectory_slope).reset_index()
    
    # Merge the growth rates back with the parameters
    df_params = df_full[['run_id', 'b_res', 'initial_width']].drop_duplicates()
    df_analysis = pd.merge(growth_rates, df_params, on='run_id')

    # --- Visualization of the Key Finding ---
    print("Generating summary plot of sector growth rate...")
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 9))

    sns.lineplot(
        data=df_analysis,
        x='b_res',
        y='growth_rate',
        hue='initial_width',
        palette='viridis',
        marker='o',
        ms=10,
        lw=3,
        ci=95,
        ax=ax
    )

    # Highlight the equilibrium line where growth rate is zero
    ax.axhline(0, color='r', ls='--', lw=2.5, label='Stable Arc Length (Growth Rate = 0)')
    
    # Find and highlight the critical b_res value
    # We can do this by finding where the average trend crosses zero
    # For simplicity, we'll visually inspect, but a numerical fit could find the exact point.
    
    ax.set(
        xlabel="Resistant Fitness, $b_{res}$",
        ylabel="Sector Growth Rate (Slope of Arc Length vs. Radius)",
        title="Finding the Critical Fitness for Stable Sector Size"
    )
    ax.legend(title='Initial Sector Width (cells)')
    ax.grid(True, which='both', ls=':')

    # Annotate the different regimes
    ax.text(0.8, ax.get_ylim()[1]*0.8, 'Collapse Regime\n(Selection > Expansion)', ha='center', fontsize=14, color='navy')
    ax.text(0.98, ax.get_ylim()[1]*0.8, 'Growth Regime\n(Expansion > Selection)', ha='center', fontsize=14, color='darkgreen')

    output_path = figure_dir / "fig_aif_equilibrium_point_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Equilibrium analysis plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()