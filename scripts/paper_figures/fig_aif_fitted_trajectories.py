# FILE: scripts/paper_figures/fig_a_if_fitted_trajectories.py (Corrected and Optimized)

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

def calculate_true_mean_trajectory(df_group: pd.DataFrame) -> pd.DataFrame:
    """Calculates the true mean trajectory, accounting for survivorship bias by padding extinct trajectories with zeros."""
    if df_group.empty: return pd.DataFrame()
    max_radius_global = df_group['mean_radius'].max()
    if pd.isna(max_radius_global) or max_radius_global < 1: return pd.DataFrame()
    analysis_max_radius = max_radius_global * 0.90
    common_radius_grid = np.arange(0, analysis_max_radius, 2.0)
    interpolated_trajectories = []
    for replicate_id, replicate_df in df_group.groupby('replicate'):
        if replicate_df.empty: continue
        replicate_df = replicate_df.sort_values('mean_radius')
        x_original, y_original = replicate_df['mean_radius'].values, replicate_df['arc_length'].values
        y_interpolated = np.interp(common_radius_grid, x_original, y_original, right=0)
        max_radius_replicate = x_original.max()
        y_interpolated[common_radius_grid > max_radius_replicate] = 0.0
        interpolated_trajectories.append(y_interpolated)
    if not interpolated_trajectories: return pd.DataFrame()
    all_trajectories_array = np.vstack(interpolated_trajectories)
    true_mean_arc_length = all_trajectories_array.mean(axis=0)
    df_true_mean = pd.DataFrame({'mean_radius': common_radius_grid, 'mean_arc_length': true_mean_arc_length})
    return df_true_mean

def calculate_conditional_mean_trajectory(df_group: pd.DataFrame) -> pd.DataFrame:
    """Calculates the mean trajectory of SURVIVORS ONLY at each radius."""
    if df_group.empty: return pd.DataFrame()
    
    # --- THIS IS THE FIX for SettingWithCopyWarning ---
    # Explicitly work on a copy to avoid modifying the original data slice.
    df = df_group.copy()
    # --- END OF FIX ---
    
    max_radius = df['mean_radius'].max()
    analysis_max_radius = max_radius * 0.9
    bins = np.arange(0, analysis_max_radius, 2.0)
    
    df['radius_bin'] = pd.cut(df['mean_radius'], bins=bins, right=False)
    
    binned_stats = df.groupby('radius_bin', observed=True).agg(
        conditional_mean_arc_length=('arc_length', 'mean')
    ).reset_index()
    
    binned_stats['mean_radius'] = binned_stats['radius_bin'].apply(lambda b: b.mid)
    return binned_stats[['mean_radius', 'conditional_mean_arc_length']]

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    processed_data_path = analysis_dir / "processed_spatial_trajectories.csv.gz"
    if not processed_data_path.exists():
        sys.exit(f"Processed data file not found. Run 'python scripts/utils/process_aif_trajectories.py aggregate' first.")
    print(f"Loading pre-processed data from: {processed_data_path.name}")
    df_full = pd.read_csv(processed_data_path)

    # --- REFACTORED: Remove the inefficient pre-calculation loops ---
    # print(f"\n--- Calculating (1) True Mean and (2) Conditional Mean trajectories ---")
    # df_true_means = df_full.groupby(['b_res', 'initial_width']).apply(calculate_true_mean_trajectory).reset_index()
    # df_conditional_means = df_full.groupby(['b_res', 'initial_width']).apply(calculate_conditional_mean_trajectory).reset_index()

    print("\n--- Generating individual diagnostic plots in a single pass ---")
    output_dir = PROJECT_ROOT / "figures" / "aif_fitted_trajectories_individual"
    output_dir.mkdir(exist_ok=True)
    print(f"Plots will be saved in: {output_dir}")

    sns.set_theme(style="whitegrid", context="talk")
    
    # --- REFACTORED: Use a single loop for efficiency ---
    grouped = df_full.groupby(['b_res', 'initial_width'])
    for (b_res_val, width_val), group_df in tqdm(grouped, total=len(grouped)):
        
        # --- All calculations are now done inside this single loop ---
        mean_df_subset = calculate_true_mean_trajectory(group_df)
        cond_mean_df_subset = calculate_conditional_mean_trajectory(group_df)

        # Start plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        local_max_radius = group_df['mean_radius'].max()
        analysis_max_radius = local_max_radius * 0.90
        plot_group_df = group_df[group_df['mean_radius'] <= analysis_max_radius]
        
        sns.lineplot(data=plot_group_df, x="mean_radius", y="arc_length", units="replicate", 
                     estimator=None, color="gray", alpha=0.15, zorder=1, ax=ax)
        
        if not mean_df_subset.empty:
            sns.lineplot(data=mean_df_subset, x="mean_radius", y="mean_arc_length", 
                         color='crimson', lw=3.5, zorder=3, ax=ax, label='True Mean (incl. extinctions)')

        if not cond_mean_df_subset.empty:
            sns.lineplot(data=cond_mean_df_subset, x="mean_radius", y="conditional_mean_arc_length",
                         color='royalblue', lw=3.5, zorder=2, ax=ax, label='Conditional Mean (survivors only)')

        local_max_arc = group_df['arc_length'][group_df['mean_radius'] <= analysis_max_radius].max()
        ax.set_xlim(0, analysis_max_radius * 1.05 if analysis_max_radius > 0 else 10)
        ax.set_ylim(0, local_max_arc * 1.05 if local_max_arc > 0 else 10)
        
        ax.set_title(f"Initial Width = {width_val},  $b_{{res}}$ = {b_res_val:.4f}")
        ax.set_xlabel("Radius from Colony Center")
        ax.set_ylabel("Sector Arc Length (Linear Units)")
        ax.grid(True, which='both', linestyle=':')
        ax.legend()

        filename = f"width_{width_val}_bres_{b_res_val:.4f}.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        
        plt.close(fig)

    print(f"\n✅ Successfully generated {len(grouped)} individual plot files.")

if __name__ == "__main__":
    main()