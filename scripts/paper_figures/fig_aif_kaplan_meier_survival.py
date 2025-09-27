# FILE: scripts/paper_figures/fig_aif_kaplan_meier_survival.py (Definitive, Corrected API Version)

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
from src.io.data_loader import load_aggregated_data

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    processed_data_path = analysis_dir / "processed_spatial_trajectories.csv.gz"
    if not processed_data_path.exists():
        sys.exit(f"Processed data file not found. Run 'scripts/utils/process_aif_trajectories.py' first.")
    print(f"Loading pre-processed data from: {processed_data_path.name}")
    df_full = pd.read_csv(processed_data_path)

    print("Preparing data for Kaplan-Meier analysis...")
    df_max_radius = df_full.groupby(['b_res', 'initial_width', 'replicate'])['mean_radius'].max().reset_index(name='duration')
    max_possible_radius = df_max_radius['duration'].max()
    censoring_radius = max_possible_radius * 0.95
    df_max_radius['event_observed'] = (df_max_radius['duration'] < censoring_radius).astype(int)
    initial_radius = EXPERIMENTS["aif_definitive_spatial_scan"]["sim_sets"]["main"]["base_params"]["initial_droplet_radius"]

    print("Generating Kaplan-Meier survival plot...")
    sns.set_theme(style="whitegrid", context="talk")

    initial_widths = sorted(df_max_radius['initial_width'].unique())
    b_res_levels = sorted(df_max_radius['b_res'].unique())
    palette = sns.color_palette('viridis_r', n_colors=len(b_res_levels))
    color_map = dict(zip(b_res_levels, palette))

    fig, axes = plt.subplots(1, len(initial_widths), figsize=(7 * len(initial_widths), 7), sharey=True)
    if len(initial_widths) == 1: axes = [axes]

    for i, width in enumerate(initial_widths):
        ax = axes[i]
        ax.set_title(f"Initial Width = {width} cells", fontsize=18)

        for b_res in b_res_levels:
            subset = df_max_radius[(df_max_radius['initial_width'] == width) & (df_max_radius['b_res'] == b_res)]
            if subset.empty: continue

            T = subset['duration']
            E = subset['event_observed']
            
            kmf = KaplanMeierFitter()
            kmf.fit(T, event_observed=E, label=f'{b_res:.4f}')
            
            # --- THIS IS THE CORRECTED METHOD ---
            # Instead of a 'timeline' kwarg, we plot the survival function and its
            # confidence interval manually. This gives us full control and avoids the API error.
            
            # Get the survival probability and confidence intervals from the fitter
            survival_df = kmf.survival_function_
            ci_df = kmf.confidence_interval_
            
            # Plot the main survival curve
            ax.plot(survival_df.index, survival_df[kmf.label], color=color_map[b_res], lw=3)
            
            # Plot the confidence interval as a shaded region
            ax.fill_between(
                ci_df.index,
                ci_df[f'{kmf.label}_lower_0.95'],
                ci_df[f'{kmf.label}_upper_0.95'],
                color=color_map[b_res],
                alpha=0.2
            )
        
        ax.set_xlabel("Colony Radius")
        ax.grid(True, which='both', linestyle=':')
        ax.set_xlim(left=initial_radius, right=max_possible_radius + 10)

    axes[0].set_ylabel("Sector Survival Probability")
    axes[0].set_ylim(-0.05, 1.05)
    
    fig.suptitle("Kaplan-Meier Survival Analysis of Sector Collapse", y=1.03, fontsize=24)

    # Re-create the legend manually for a clean look
    legend_elements = [plt.Line2D([0], [0], color=color_map[br], lw=3, label=f'{br:.4f}') for br in b_res_levels]
    fig.legend(handles=legend_elements[::-1], title='$b_{res}$', bbox_to_anchor=(1.01, 0.85), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    figure_dir = PROJECT_ROOT / "figures"
    output_path = figure_dir / "fig_aif_kaplan_meier_survival_final.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Final, artifact-free Kaplan-Meier plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()