# FILE: scripts/paper_figures/fig_aif_distribution_analysis.py

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
CELL_SIZE_UM = 5.0 # For secondary axis calibration

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    processed_data_path = analysis_dir / "processed_spatial_trajectories.csv.gz"

    if not processed_data_path.exists():
        sys.exit(f"Processed data file not found. Run 'python scripts/utils/process_aif_trajectories.py' first.")
    
    print(f"Loading pre-processed data from: {processed_data_path.name}")
    df_full = pd.read_csv(processed_data_path)

    # --- Select a representative condition to plot ---
    # We choose a mid-range initial width and a fitness cost where sectors persist
    # but are under selection, analogous to the paper's data.
    b_res_target = 0.9700
    width_target = 40
    df_subset = df_full[(np.isclose(df_full['b_res'], b_res_target)) & (df_full['initial_width'] == width_target)].copy()

    if df_subset.empty:
        sys.exit(f"No data found for the target condition: b_res={b_res_target}, initial_width={width_target}")

    print(f"\nAnalyzing condition: b_res={b_res_target}, initial_width={width_target}")
    
    # --- Bin data by radius to calculate statistics ---
    max_radius = df_subset['mean_radius'].max()
    bins = np.arange(0, max_radius + 20, 20) # Use 20-unit radius bins
    df_subset['radius_bin'] = pd.cut(df_subset['mean_radius'], bins=bins)

    df_stats = df_subset.groupby('radius_bin', observed=True)['arc_length'].agg(
        ['mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    df_stats.columns = ['radius_bin', 'mean', 'median', 'q25', 'q75']
    df_stats['radius_mid'] = df_stats['radius_bin'].apply(lambda b: b.mid)

    # --- Generate Plot ---
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Median and IQR
    ax.plot(df_stats['radius_mid'], df_stats['median'], 'o-', color='crimson', lw=3, label='Median')
    ax.fill_between(df_stats['radius_mid'], df_stats['q25'], df_stats['q75'], color='crimson', alpha=0.2, label='Interquartile Range')

    # Plot Mean
    ax.plot(df_stats['radius_mid'], df_stats['mean'], 'o--', color='darkblue', lw=2.5, label='Mean')
    
    ax.set_title(f'Sector Width vs. Radius (Initial Width = {width_target}, $b_{{res}}$ = {b_res_target:.4f})')
    ax.set_xlabel('Radius, r [simulation units]')
    ax.set_ylabel('Width, w [cells]')
    ax.legend()
    ax.grid(True, which='both', linestyle=':')
    ax.set_ylim(bottom=0)

    # Add secondary axis in microns
    secax_x = ax.secondary_xaxis('top', functions=(lambda r: r * CELL_SIZE_UM, lambda r: r / CELL_SIZE_UM))
    secax_x.set_xlabel('Radius, r [µm]')
    secax_y = ax.secondary_yaxis('right', functions=(lambda w: w * CELL_SIZE_UM, lambda w: w / CELL_SIZE_UM))
    secax_y.set_ylabel('Width, w [µm]')

    figure_dir = PROJECT_ROOT / "figures"; figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / "fig_aif_width_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Figure 2c analog saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()