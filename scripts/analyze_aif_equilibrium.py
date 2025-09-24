# FILE: scripts/analyze_aif_equilibrium.py
#
# Analyzes the output of the 'aif_equilibrium_finding' experiment.
# It plots sector width trajectories to empirically find the equilibrium
# width for different fitness costs (b_res).

import sys
import json
import gzip
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Add project root to Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS


def load_trajectory_data(traj_file_path: Path) -> pd.DataFrame:
    """Loads a single gzipped trajectory data file."""
    if not traj_file_path.exists():
        return pd.DataFrame()
    with gzip.open(traj_file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def main():
    campaign_id = EXPERIMENTS["aif_equilibrium_finding"]["campaign_id"]
    data_dir = PROJECT_ROOT / "data" / campaign_id
    traj_dir = data_dir / "trajectories"
    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)

    if not traj_dir.exists():
        sys.exit(
            f"Trajectory data directory not found: {traj_dir}\n"
            f"Please run and consolidate the '{campaign_id}' campaign first."
        )

    df_summary = pd.read_csv(
        data_dir / "analysis" / f"{campaign_id}_summary_aggregated.csv"
    )

    all_trajectories = []
    for _, row in tqdm(
        df_summary.iterrows(), total=len(df_summary), desc="Loading trajectories"
    ):
        task_id = row["task_id"]
        traj_file = traj_dir / f"traj_{task_id}.json.gz"
        df_traj = load_trajectory_data(traj_file)
        if not df_traj.empty:
            # Add parameters from the summary file to each trajectory
            df_traj["b_res"] = row["b_res"]
            df_traj["initial_width"] = row["sector_width_initial"]
            df_traj["replicate"] = row["replicate"]
            all_trajectories.append(df_traj)

    if not all_trajectories:
        sys.exit("No valid trajectory data could be loaded.")

    df_full = pd.concat(all_trajectories)

    # Convert radians to degrees for easier interpretation
    df_full["sector_width_deg"] = np.rad2deg(df_full["sector_width_rad"])

    # --- Visualization ---
    print("Generating FacetGrid plot...")
    sns.set_theme(style="whitegrid", context="talk")

    # Use FacetGrid to create a column for each b_res value
    g = sns.FacetGrid(
        df_full,
        col="b_res",
        hue="initial_width",
        col_wrap=3,
        height=6,
        aspect=1.2,
        palette="viridis",
    )

    # Map the lineplot to the grid
    g.map_dataframe(sns.lineplot, x="colony_radius", y="sector_width_deg", alpha=0.7)

    # Add titles and labels
    g.set_axis_labels("Colony Radius", "Sector Width (Degrees)")
    g.set_titles("b_res = {col_name:.3f}")
    g.add_legend(title="Initial Width")
    g.fig.suptitle("Sector Width Dynamics vs. Fitness Cost", y=1.02, fontsize=24)

    # Save the figure
    output_path = figure_dir / "aif_equilibrium_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nAnalysis figure saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
