# FILE: scripts/paper_figures/fig_aif_distribution_analysis.py (Corrected with Flipped Axes Layout)

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.gridspec as gridspec

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS

def get_binned_stats(df_group: pd.DataFrame) -> pd.DataFrame:
    if df_group.empty: return pd.DataFrame()
    max_radius_global = df_group['mean_radius'].max()
    if pd.isna(max_radius_global) or max_radius_global < 1: return pd.DataFrame()
    analysis_max_radius = max_radius_global * 0.90
    common_radius_grid = np.arange(0, analysis_max_radius, 25.0)
    all_replicates_padded = []
    for _, replicate_df in df_group.groupby('replicate'):
        if replicate_df.empty: continue
        replicate_df = replicate_df.sort_values('mean_radius')
        x_original, y_original = replicate_df['mean_radius'].values, replicate_df['arc_length'].values
        y_interpolated = np.interp(common_radius_grid, x_original, y_original, right=0)
        max_radius_replicate = x_original.max()
        y_interpolated[common_radius_grid > max_radius_replicate] = 0.0
        all_replicates_padded.append(y_interpolated)
    if not all_replicates_padded: return pd.DataFrame()
    df_padded = pd.DataFrame(np.vstack(all_replicates_padded).T, index=common_radius_grid)
    stats_df = pd.DataFrame({
        'mean_radius': common_radius_grid,
        'mean': df_padded.mean(axis=1),
        'median': df_padded.median(axis=1),
        'q25': df_padded.quantile(0.25, axis=1),
        'q75': df_padded.quantile(0.75, axis=1),
    })
    n_boot = 100
    boot_means = np.array([df_padded.sample(frac=1, replace=True, axis=1).mean(axis=1) for _ in range(n_boot)])
    stats_df['mean_ci_lower'] = np.percentile(boot_means, 2.5, axis=0)
    stats_df['mean_ci_upper'] = np.percentile(boot_means, 97.5, axis=0)
    return stats_df

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    processed_data_path = analysis_dir / "processed_spatial_trajectories.csv.gz"
    if not processed_data_path.exists():
        sys.exit(f"Processed data file not found. Run 'python scripts/utils/process_aif_trajectories.py aggregate' first.")
    print(f"Loading pre-processed data from: {processed_data_path.name}")
    df_full = pd.read_csv(processed_data_path)

    b_res_target, width_target = 0.9500, 40
    df_subset = df_full[(np.isclose(df_full['b_res'], b_res_target)) & (df_full['initial_width'] == width_target)].copy()
    if df_subset.empty: sys.exit(f"No data found for b_res={b_res_target} and width={width_target}.")

    print("\n--- Calculating robust binned statistics (Mean, Median, Quantiles) ---")
    stats_df = get_binned_stats(df_subset)

    print("\n--- Generating distribution analysis plot with flipped axes ---")
    sns.set_theme(style="ticks", context="talk")
    
    # --- NEW LAYOUT: Use GridSpec for a top and bottom panel ---
    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1) # Share the X-axis (arc length)
    # --- END NEW LAYOUT ---

    # === Panel A: Mean vs Median with Flipped Axes ===
    ax1.plot(stats_df['mean'], stats_df['mean_radius'], 'o-', color='royalblue', label='Mean')
    ax1.fill_betweenx(stats_df['mean_radius'], stats_df['mean_ci_lower'], stats_df['mean_ci_upper'], color='royalblue', alpha=0.3)
    ax1.plot(stats_df['median'], stats_df['mean_radius'], 's-', color='crimson', label='Median')
    ax1.fill_betweenx(stats_df['mean_radius'], stats_df['q25'], stats_df['q75'], color='crimson', alpha=0.3)
    
    ax1.set_ylabel("Radius from Colony Center, r")
    ax1.tick_params(axis='x', labelbottom=False) # Hide x-axis labels on the top plot
    ax1.legend()
    ax1.grid(True, linestyle=':')
    ax1.set_xlim(left=0)
    fig.suptitle(f"Initial Width = {width_target}, $b_{{res}}$ = {b_res_target:.4f}", fontsize=18)

    # === Panel B: Ridgeline Plot (Violin Plot) Aligned Below ===
    bins = np.linspace(0, stats_df['mean_radius'].max(), 10)
    df_subset['radius_bin_joy'] = pd.cut(df_subset['mean_radius'], bins=bins)
    df_subset['radius_bin_str'] = df_subset['radius_bin_joy'].apply(lambda x: f"{x.left:.0f}-{x.right:.0f}" if pd.notna(x) else "").astype(str)
    
    df_joyplot = df_subset.dropna(subset=['radius_bin_joy'])[['arc_length', 'radius_bin_str']]
    joyplot_order = sorted(df_joyplot['radius_bin_str'].unique(), key=lambda s: float(s.split('-')[0]) if '-' in s else -1)
    
    # Plotting vertically, with radius bins on the y-axis
    sns.violinplot(data=df_joyplot, y='radius_bin_str', x='arc_length', ax=ax2,
                   density_norm='width', inner=None, orient='h', hue='radius_bin_str',
                   palette='viridis_r', cut=0, order=joyplot_order, legend=False)

    ax2.set_xlabel("Sector Arc Length, w")
    ax2.set_ylabel("") # Remove redundant y-axis title
    ax2.grid(True, linestyle=':')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    figure_dir = PROJECT_ROOT / "figures"; figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / "fig_aif_distribution_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Distribution analysis plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()