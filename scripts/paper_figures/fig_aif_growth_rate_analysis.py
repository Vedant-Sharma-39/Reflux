# FILE: scripts/paper_figures/fig_aif_growth_rate_analysis.py (Fast, No PyArrow)

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import linregress

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS

def calculate_late_stage_slope(df_group: pd.DataFrame, fit_start_radius: float = 100.0) -> pd.Series:
    max_radius = df_group['mean_radius'].max()
    if pd.isna(max_radius) or max_radius < fit_start_radius:
        return pd.Series({"growth_rate": np.nan})
    bins = np.arange(0, max_radius + 5, 5)
    df_group['radius_bin'] = pd.cut(df_group['mean_radius'], bins)
    mean_trajectory = df_group.groupby('radius_bin', observed=True)['arc_length'].mean().reset_index()
    mean_trajectory['mean_radius'] = mean_trajectory['radius_bin'].apply(lambda x: x.mid)
    mean_trajectory['mean_radius'] = mean_trajectory['mean_radius'].astype(float)
    stable_mean_trajectory = mean_trajectory[mean_trajectory['mean_radius'] > fit_start_radius].dropna()
    if len(stable_mean_trajectory) < 4:
        return pd.Series({"growth_rate": np.nan})
    else:
        slope = linregress(stable_mean_trajectory['mean_radius'], stable_mean_trajectory['arc_length']).slope
        return pd.Series({"growth_rate": slope})

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    
    # --- LOAD FROM GZIPPED CSV ---
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    processed_data_path = analysis_dir / "processed_spatial_trajectories.csv.gz"
    if not processed_data_path.exists():
        sys.exit(f"Processed data file not found. Run 'scripts/utils/process_aif_trajectories.py' first.")
    print(f"Loading pre-processed data from: {processed_data_path.name}")
    df_full = pd.read_csv(processed_data_path)

    print("Fitting mean trajectories to calculate growth rates...")
    df_analysis = df_full.groupby(['b_res', 'initial_width']).apply(calculate_late_stage_slope).reset_index()

    # --- VISUALIZATION ---
    print("Generating definitive growth rate plot...")
    sns.set_theme(style="whitegrid", context="talk", rc={"grid.linestyle": ":"})
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.lineplot(data=df_analysis, x='b_res', y='growth_rate', hue='initial_width', palette="viridis", marker='o', ms=12, lw=3.5, ax=ax)
    ax.axhline(0, color='crimson', ls='--', lw=2.5, label='Stable Arc Length (Growth Rate = 0)')
    ax.set(xlabel="Resistant Fitness, $b_{res}$", ylabel="Growth Rate of Mean Trajectory (Slope for r > 100)", title="Finding the Critical Fitness for Stable Sector Size")
    ax.legend(title='Initial Sector Width (cells)', fontsize=14)
    ax.grid(True, which='both')

    figure_dir = PROJECT_ROOT / "figures"
    output_path = figure_dir / "fig_aif_growth_rate_analysis_final.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Definitive growth rate analysis plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()