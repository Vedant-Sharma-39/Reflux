# FILE: scripts/calibration/calibrate_with_paper_data.py
#
# A dedicated script to perform a proportional calibration of the Reflux-AIF model
# against the original experimental data from Aif et al. (2022).

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from lifelines import KaplanMeierFitter

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Configuration ---
# Path to the paper's original data file
PAPER_DATA_H5_PATH = PROJECT_ROOT / "data" / "experimental_data.h5"

# Path to your simulation's pre-processed trajectory data
REFLUX_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "aif_calibration_droplet_v1" / "analysis" / "processed_spatial_trajectories.csv.gz"

# The equilibrium width from the paper's text/model (in micrometers)
W_EQ_PAPER = 50.0

def load_paper_experimental_data(h5_path: Path) -> pd.DataFrame:
    """Loads the 'no rescue' survival data from the paper's HDF5 file."""
    print(f"Loading paper's experimental data from: {h5_path.name}")
    data = {}
    # These keys correspond to the (CHX=50, BED=0, treatment=10) "no rescue" case
    group_path = "50/0/10"
    with h5py.File(h5_path, 'r') as f:
        data['radius_um'] = f[f"{group_path}/radii"][:]
        numbers = f[f"{group_path}/numbers"][:]
        # Summing across types (uncomp, comp, mixed) to get total clones
        total_clones = numbers[:, 0] + numbers[:, 1] + numbers[:, 2]
    
    # Calculate survival probability, handling potential division by zero
    n0 = total_clones[0] if total_clones.size > 0 else 1
    data['survival_prob'] = total_clones / n0
    return pd.DataFrame(data)

def calculate_reflux_w_eq(reflux_data_path: Path) -> float:
    """Calculates the characteristic equilibrium width from the Reflux simulation data."""
    print(f"Calculating w_eq from Reflux data: {reflux_data_path.name}")
    df = pd.read_csv(reflux_data_path)
    
    # Use the late-stage part of the expansion for a stable measurement
    max_radius = df['mean_radius'].max()
    equilibrium_start_radius = max_radius * 0.5
    df_late_stage = df[df['mean_radius'] > equilibrium_start_radius]

    # The median is robust to outliers
    w_eq_sim = df_late_stage['arc_length'].median()
    return w_eq_sim

def calculate_reflux_survival_curve(reflux_data_path: Path) -> pd.DataFrame:
    """Calculates the survival curve from Reflux data using Kaplan-Meier."""
    print(f"Calculating survival curve from Reflux data...")
    df = pd.read_csv(reflux_data_path)

    # Each sector is a unique trajectory
    df['sector_unique_id'] = df['replicate'].astype(str) + '_' + df['sector_id'].astype(str)
    
    # For each sector, find its maximum radius (duration)
    df_max_radius = df.groupby('sector_unique_id')['mean_radius'].max().reset_index(name='duration')
    
    # Determine if the event (extinction) was observed or if the data was censored
    # An event is observed if the sector disappeared before the end of the simulation.
    # We define "the end" as 95% of the maximum radius reached by ANY sector.
    max_possible_radius = df['mean_radius'].max()
    censoring_radius = max_possible_radius * 0.95
    df_max_radius['event_observed'] = (df_max_radius['duration'] < censoring_radius).astype(int)

    # Fit the Kaplan-Meier model
    kmf = KaplanMeierFitter()
    kmf.fit(df_max_radius['duration'], event_observed=df_max_radius['event_observed'])
    
    survival_df = kmf.survival_function_.reset_index()
    survival_df.columns = ['radius_sim', 'survival_prob']
    return survival_df

def main():
    """Main calibration and plotting pipeline."""
    if not PAPER_DATA_H5_PATH.exists():
        sys.exit(f"Error: Paper's data file not found at {PAPER_DATA_H5_PATH}")
    if not REFLUX_PROCESSED_DATA_PATH.exists():
        sys.exit(f"Error: Reflux pre-processed data not found at {REFLUX_PROCESSED_DATA_PATH}. Run the pre-processing script first.")

    # --- Stage 1: Load and calculate all necessary data ---
    df_paper = load_paper_experimental_data(PAPER_DATA_H5_PATH)
    w_eq_sim = calculate_reflux_w_eq(REFLUX_PROCESSED_DATA_PATH)
    df_reflux_survival = calculate_reflux_survival_curve(REFLUX_PROCESSED_DATA_PATH)

    print("\n--- Calibration Values ---")
    print(f"  Paper Equilibrium Width (w_eq_paper): {W_EQ_PAPER:.2f} µm")
    print(f"  Reflux Equilibrium Width (w_eq_sim):   {w_eq_sim:.2f} [simulation units]")
    print("--------------------------\n")

    # --- Stage 2: Proportional Scaling (Dimensionless Radius) ---
    df_paper['dimensionless_radius'] = df_paper['radius_um'] / W_EQ_PAPER
    df_reflux_survival['dimensionless_radius'] = df_reflux_survival['radius_sim'] / w_eq_sim

    # --- Stage 3: Plotting for Direct Comparison ---
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot Paper's Experimental Data
    ax.plot(
        df_paper['dimensionless_radius'],
        df_paper['survival_prob'],
        marker='o', ms=8, linestyle='none', color='black',
        label=f"Aif et al. (2022) Data (w_eq = {W_EQ_PAPER:.1f} µm)"
    )

    # Plot Your Reflux Simulation Data
    ax.plot(
        df_reflux_survival['dimensionless_radius'],
        df_reflux_survival['survival_prob'],
        linestyle='-', lw=3.5, color='crimson',
        label=f"Reflux-AIF Model (w_eq = {w_eq_sim:.2f} cells)"
    )

    ax.set_yscale('log')
    ax.set_xlabel('Dimensionless Radius,  $r / w_{eq}$')
    ax.set_ylabel('Sector Survival Probability, $P_{surv}$')
    ax.set_title('Proportional Calibration of Inflation-Selection Balance')
    ax.legend(fontsize=14)
    ax.grid(True, which="both", linestyle=":")
    ax.set_ylim(bottom=1e-3, top=1.2) # Match paper's y-axis limits

    output_dir = PROJECT_ROOT / "figures" / "calibration"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "proportional_calibration_survival.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✅ Calibration plot saved successfully to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()