# FILE: scripts/paper_figures/fig_aif_kaplan_meier_survival.py

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from lifelines import KaplanMeierFitter

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    processed_data_path = analysis_dir / "processed_spatial_trajectories.csv.gz"
    if not processed_data_path.exists():
        sys.exit(f"Processed data file not found. Run 'scripts/utils/process_aif_trajectories.py' first.")
    
    print(f"Loading pre-processed data from: {processed_data_path.name}")
    df_full = pd.read_csv(processed_data_path)

    print("Preparing data for Kaplan-Meier survival analysis...")
    # 1. For each sector, find its maximum radius (its 'duration')
    df_max_radius = df_full.groupby(['b_res', 'initial_width', 'replicate'])['mean_radius'].max().reset_index(name='duration')

    # 2. Determine if the event (extinction) was observed or if the data was 'censored'
    # A sector is 'censored' if it survived to the end of the simulation.
    max_possible_radius = df_max_radius['duration'].max()
    censoring_radius = max_possible_radius * 0.95 # A robust threshold
    df_max_radius['event_observed'] = (df_max_radius['duration'] < censoring_radius).astype(int)

    # Get initial radius from config to set plot limits
    initial_radius = EXPERIMENTS["aif_definitive_spatial_scan"]["sim_sets"]["main"]["base_params"]["initial_droplet_radius"]

    print("Generating Kaplan-Meier survival plot...")
    sns.set_theme(style="whitegrid", context="talk")

    # --- Create a Faceted Plot ---
    # One subplot for each initial_width, with b_res as colored lines
    initial_widths = sorted(df_max_radius['initial_width'].unique())
    b_res_levels = sorted(df_max_radius['b_res'].unique())
    
    palette = sns.color_palette('viridis_r', n_colors=len(b_res_levels))
    
    fig, axes = plt.subplots(1, len(initial_widths), figsize=(7 * len(initial_widths), 7), sharey=True)
    if len(initial_widths) == 1: axes = [axes] # Handle case of single subplot

    for i, width in enumerate(tqdm(initial_widths, desc="Plotting facets")):
        ax = axes[i]
        ax.set_title(f"Initial Width = {width} cells", fontsize=18)

        for b_res in b_res_levels:
            # Filter data for the specific condition
            subset = df_max_radius[(df_max_radius['initial_width'] == width) & (df_max_radius['b_res'] == b_res)]
            if subset.empty: continue

            T = subset['duration']       # The lifetime of each sector
            E = subset['event_observed'] # 1 if it died, 0 if it was censored
            
            kmf = KaplanMeierFitter()
            kmf.fit(T, event_observed=E, label=f'{b_res:.4f}')
            kmf.plot_survival_function(ax=ax, ci_show=True, color=palette[b_res_levels.index(b_res)], lw=3)
        
        ax.set_xlabel("Colony Radius")
        ax.grid(True, which='both', linestyle=':')
        ax.set_xlim(left=initial_radius, right=max_possible_radius + 10)
        ax.set_yscale('log')

    axes[0].set_ylabel("Sector Survival Probability")
    axes[0].set_ylim(1e-3, 1.1) # Set y-limits to match paper
    
    # Adjust legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title='$b_{res}$', bbox_to_anchor=(1.01, 0.85), loc='upper left')
    for ax in axes: ax.get_legend().remove()

    fig.suptitle("Kaplan-Meier Survival Analysis of Sector Collapse (No Rescue)", y=1.03, fontsize=24)
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    figure_dir = PROJECT_ROOT / "figures"
    output_path = figure_dir / "fig_aif_kaplan_meier_survival.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Figure 2f analog saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()