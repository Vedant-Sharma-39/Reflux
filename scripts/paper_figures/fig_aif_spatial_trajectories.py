# FILE: scripts/paper_figures/fig_aif_spatial_trajectories.py (NEW FILE)

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing
import os

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data
from scripts.analyze_aif_sectors import (
    load_population_data, identify_sectors_by_angle,
    refine_sectors_by_neighborhood, measure_width_radial_binning,
)

def process_simulation_output(task_info: dict) -> pd.DataFrame | None:
    """
    WORKER FUNCTION: Analyzes a single simulation's final population file
    and returns the full width-vs-radius trajectory.
    """
    pop_file_path = task_info["pop_file_path"]
    if not pop_file_path.exists(): return None
    
    # Run the full spatial analysis pipeline
    df_pop = load_population_data(pop_file_path)
    df_resistant = df_pop[df_pop['type'].isin({2, 3})].copy()
    
    # Handle extinctions gracefully
    if df_resistant.empty:
        # Return a single point at (0,0) to represent extinction
        res_df = pd.DataFrame([{"mean_radius": 0, "arc_length": 0, "width_rad": 0}])
        res_df["b_res"] = task_info["b_res"]
        res_df["initial_width"] = task_info["initial_width"]
        res_df["replicate"] = task_info["replicate"]
        return res_df

    df_clustered = identify_sectors_by_angle(df_resistant)
    df_refined = refine_sectors_by_neighborhood(df_clustered)
    df_width = measure_width_radial_binning(df_refined)

    if df_width.empty:
        res_df = pd.DataFrame([{"mean_radius": 0, "arc_length": 0, "width_rad": 0}])
    else:
        df_width['arc_length'] = df_width['mean_radius'] * df_width['width_rad']
        res_df = df_width

    res_df["b_res"] = task_info["b_res"]
    res_df["initial_width"] = task_info["initial_width"]
    res_df["replicate"] = task_info["replicate"]
    return res_df

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    df_summary = load_aggregated_data(campaign_id, str(PROJECT_ROOT))
    if df_summary.empty: sys.exit(f"Data for '{campaign_id}' is empty.")

    pop_dir = PROJECT_ROOT / "data" / campaign_id / "populations"
    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    if not pop_dir.exists(): sys.exit(f"Population directory not found: {pop_dir}")

    # Create the list of tasks for the worker pool
    tasks = []
    for _, row in df_summary.iterrows():
        tasks.append({
            "pop_file_path": pop_dir / f"pop_{row['task_id']}.json.gz",
            "b_res": row["b_res"],
            "initial_width": row["sector_width_initial"],
            "replicate": row["replicate"],
        })

    num_workers = max(1, os.cpu_count() - 1)
    print(f"Starting parallel analysis of {len(tasks)} population files...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_simulation_output, tasks), total=len(tasks)))
    
    all_measurements = [df for df in results if df is not None and not df.empty]
    if not all_measurements: sys.exit("No valid measurements extracted.")
    df_full = pd.concat(all_measurements)

    # --- Comprehensive Trajectory Visualization ---
    print("Generating detailed spatial trajectory plot...")
    sns.set_theme(style="whitegrid", context="talk")

    # Facet by BOTH initial width (rows) and fitness cost (columns)
    g = sns.FacetGrid(
        df_full,
        row="initial_width",
        col="b_res",
        height=5,
        aspect=1.2,
        sharex=True,
        sharey=True, # Share axes for direct comparison
        margin_titles=True
    )
    
    # Map the plotting functions to the grid
    # 1. Plot each replicate's trajectory as a faint line
    g.map_dataframe(
        sns.lineplot,
        x="mean_radius",
        y="arc_length",
        units="replicate",
        estimator=None,
        color="gray",
        alpha=0.2,
        zorder=1
    )
    
    # 2. Overlay a robust, smoothed average line
    g.map_dataframe(
        sns.regplot,
        x="mean_radius",
        y="arc_length",
        scatter=False,
        lowess=True,
        line_kws={'color': 'teal', 'lw': 3.5, 'zorder': 2}
    )

    g.set_axis_labels("Radius from Colony Center", "Sector Arc Length (Linear Units)")
    
    # Set titles for rows and columns
    g.set_titles(col_template="$b_{{res}}$ = {col_name:.4f}", row_template="Initial Width = {row_name}")
    
    g.fig.suptitle("Sector Spatial Trajectories vs. Fitness and Initial Size", y=1.03, fontsize=24)
    g.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle

    output_path = figure_dir / "fig_aif_spatial_trajectories_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Detailed trajectory plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()