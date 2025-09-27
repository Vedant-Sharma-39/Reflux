# FILE: scripts/paper_figures/fig_aif_fitted_trajectories.py (NEW FILE)

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
# We can reuse the same Stage 1 worker function as it correctly processes the raw data
from scripts.paper_figures.fig_aif_growth_rate_analysis import process_simulation_output

def calculate_linear_fit(df_group: pd.DataFrame, fit_start_radius: float = 100.0) -> pd.Series:
    """
    Takes a group of trajectories for one parameter set, calculates the mean
    trajectory, and returns the SLOPE and INTERCEPT of a linear fit to its late stage.
    """
    max_radius = df_group['mean_radius'].max()
    if pd.isna(max_radius) or max_radius < fit_start_radius:
        return pd.Series({"slope": np.nan, "intercept": np.nan})

    bins = np.arange(0, max_radius + 5, 5)
    df_group['radius_bin'] = pd.cut(df_group['mean_radius'], bins)
    
    mean_trajectory = df_group.groupby('radius_bin', observed=True)['arc_length'].mean().reset_index()
    mean_trajectory['mean_radius'] = mean_trajectory['radius_bin'].apply(lambda x: x.mid)
    mean_trajectory['mean_radius'] = mean_trajectory['mean_radius'].astype(float)

    stable_mean_trajectory = mean_trajectory[mean_trajectory['mean_radius'] > fit_start_radius].dropna()

    if len(stable_mean_trajectory) < 4:
        return pd.Series({"slope": np.nan, "intercept": np.nan})
    else:
        # linregress returns slope, intercept, r_value, p_value, std_err
        slope, intercept, _, _, _ = linregress(stable_mean_trajectory['mean_radius'], stable_mean_trajectory['arc_length'])
        return pd.Series({"slope": slope, "intercept": intercept})

def plot_fit_line(data, fit_start_radius=100.0, **kwargs):
    """A custom plotting function for FacetGrid to draw the calculated linear fit."""
    ax = plt.gca()
    # Get the single slope and intercept value for this facet's data
    slope = data['slope'].iloc[0]
    intercept = data['intercept'].iloc[0]

    if pd.isna(slope) or pd.isna(intercept):
        return

    # Create the x-points for the line, from the fit start to the max radius
    x_max = data['mean_radius'].max()
    x_fit = np.array([fit_start_radius, x_max])
    
    # Calculate the corresponding y-points using the fit
    y_fit = intercept + slope * x_fit
    
    # Plot the line
    ax.plot(x_fit, y_fit, **kwargs)

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
    
    # --- Stage 1: Process raw data (same as before) ---
    print(f"--- Stage 1: Processing {len(tasks)} population files ---")
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_simulation_output, tasks), total=len(tasks), desc="Processing Files"))
    df_full = pd.concat([df for df in results if df is not None and not df.empty])

    # --- Stage 2: Calculate linear fit for each parameter set ---
    print(f"\n--- Stage 2: Calculating linear fits for each parameter group ---")
    df_fits = df_full.groupby(['b_res', 'initial_width']).apply(calculate_linear_fit).reset_index()

    # Merge the fit parameters (slope, intercept) back into the main DataFrame
    df_plot_data = pd.merge(df_full, df_fits, on=['b_res', 'initial_width'])

    # --- Stage 3: Visualization ---
    print("\n--- Stage 3: Generating final plot with fitted lines ---")
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.FacetGrid(
        df_plot_data, row="initial_width", col="b_res",
        height=5, aspect=1.2, sharey=True, margin_titles=True
    )
    
    # 1. Map the raw trajectories as faint gray lines
    g.map_dataframe(
        sns.lineplot, x="mean_radius", y="arc_length", units="replicate",
        estimator=None, color="gray", alpha=0.2, zorder=1
    )
    
    # 2. Map our custom function to draw the red best-fit line
    g.map_dataframe(
        plot_fit_line, color='crimson', lw=3.5, zorder=2, label="Late-Stage Linear Fit"
    )

    g.set_axis_labels("Radius from Colony Center", "Sector Arc Length (Linear Units)")
    g.set_titles(col_template="$b_{{res}}$ = {col_name:.4f}", row_template="Initial Width = {row_name}")
    g.fig.suptitle("Sector Spatial Trajectories with Late-Stage Linear Fit", y=1.03, fontsize=24)
    g.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = figure_dir / "fig_aif_fitted_trajectories.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Fitted trajectory plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()