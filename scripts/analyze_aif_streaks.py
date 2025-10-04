# FILE: scripts/analyze_aif_streaks.py (CORRECTED with Origin-Based Filtering)

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import argparse

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.analysis_helpers import load_population_data, measure_width_radial_binning
from src.core.hex_utils import Hex

# --- Core Streak Identification Logic (Unchanged) ---
def find_streaks_via_bfs(df_resistant: pd.DataFrame) -> pd.DataFrame:
    """Identifies spatially connected sectors (streaks) in a population of cells."""
    # ... (This function is correct and remains unchanged)
    if df_resistant.empty:
        return df_resistant
    print("Building spatial neighborhood graph...")
    hex_to_idx = {Hex(q, r, -q - r): i for i, (q, r) in enumerate(zip(df_resistant['q'], df_resistant['r']))}
    adj = {i: [] for i in range(len(df_resistant))}
    for h, idx in hex_to_idx.items():
        for neighbor_hex in h.neighbors():
            if neighbor_hex in hex_to_idx:
                neighbor_idx = hex_to_idx[neighbor_hex]
                adj[idx].append(neighbor_idx)
    print(f"Identifying streaks using BFS...")
    streak_ids = -np.ones(len(df_resistant), dtype=int)
    current_streak_id = 0
    for i in range(len(df_resistant)):
        if streak_ids[i] == -1:
            q = deque([i])
            streak_ids[i] = current_streak_id
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if streak_ids[v] == -1:
                        streak_ids[v] = current_streak_id
                        q.append(v)
            current_streak_id += 1
    df_resistant['streak_id'] = streak_ids
    print(f"Found {current_streak_id} potential streaks.")
    return df_resistant

# --- Main Analysis and Visualization Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="Analyze clonal streaks from a single AIF simulation run.")
    parser.add_argument("pop_file", type=Path, help="Path to the gzipped population file (e.g., data/.../populations/pop_...json.gz)")
    parser.add_argument("--initial-radius", type=int, required=True, help="The initial droplet radius used in the simulation (e.g., 60).")
    args = parser.parse_args()

    if not args.pop_file.exists():
        sys.exit(f"Error: Population file not found at {args.pop_file}")

    # 1. Load Data
    df_pop = load_population_data(args.pop_file)
    df_resistant = df_pop[df_pop['type'].isin({2, 3})].copy()
    df_susceptible = df_pop[~df_pop['type'].isin({2, 3})]

    # 2. Identify ALL potential streaks
    df_streaks_all = find_streaks_via_bfs(df_resistant)

    # 3. Filter by ORIGIN (Unchanged)
    print("Filtering for streaks originating at the initial front...")
    INITIAL_FRONT_SHELL_WIDTH = 0.10
    radius_upper_bound = args.initial_radius
    radius_lower_bound = args.initial_radius * (1 - INITIAL_FRONT_SHELL_WIDTH)
    origin_streak_ids = []
    for streak_id, group in df_streaks_all.groupby('streak_id'):
        if ((group['radius'] >= radius_lower_bound) & (group['radius'] <= radius_upper_bound)).any():
            origin_streak_ids.append(streak_id)
    df_streaks_origin = df_streaks_all[df_streaks_all['streak_id'].isin(origin_streak_ids)].copy()
    print(f"Found {len(origin_streak_ids)} streaks originating from the initial front.")

    # --- NEW AND CRITICAL: Filter by Minimum Size ---
    # This removes the fragmented "archipelagos" that break the analysis.
    # A successful, cohesive streak will have many hundreds or thousands of cells.
    MIN_STREAK_SIZE_THRESHOLD = 200 # Increase this if noise persists

    valid_streak_ids = []
    for streak_id, group in df_streaks_origin.groupby('streak_id'):
        if len(group) >= MIN_STREAK_SIZE_THRESHOLD:
            valid_streak_ids.append(streak_id)
            
    df_streaks_filtered = df_streaks_origin[df_streaks_origin['streak_id'].isin(valid_streak_ids)].copy()
    print(f"Found {len(valid_streak_ids)} significant streaks with >{MIN_STREAK_SIZE_THRESHOLD} cells.")
    # --- END OF NEW FILTERING LOGIC ---

    # 4. Analyze each VALID, COHESIVE streak
    all_streak_trajectories = []
    print("Measuring width vs. radius for each significant streak...")
    for streak_id in valid_streak_ids:
        df_single_streak = df_streaks_filtered[df_streaks_filtered['streak_id'] == streak_id]
        df_trajectory = measure_width_radial_binning(df_single_streak.copy())
        if not df_trajectory.empty:
            df_trajectory['streak_id'] = streak_id
            df_trajectory['arc_length'] = df_trajectory['mean_radius'] * df_trajectory['width_rad']
            all_streak_trajectories.append(df_trajectory)

    if not all_streak_trajectories:
        sys.exit("Analysis failed: No significant streaks were found after filtering.")
        
    df_final = pd.concat(all_streak_trajectories)

    # 5. Visualization
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.suptitle(f"Analysis of Significant Clonal Streaks for {args.pop_file.stem}", fontsize=24, y=0.98)

    ax1.set_title("Final Streak Identification (Large, Cohesive Streaks Only)", fontsize=18)
    ax1.scatter(df_susceptible['x'], df_susceptible['y'], c='lightgrey', s=1, alpha=0.3)
    sns.scatterplot(
        data=df_streaks_filtered, x='x', y='y', hue='streak_id', palette='tab20',
        s=5, legend=None, ax=ax1
    )
    ax1.set_xlabel("X Coordinate"); ax1.set_ylabel("Y Coordinate")
    ax1.set_aspect('equal', 'box')

    ax2.set_title("Streak Arc Length vs. Radius", fontsize=18)
    sns.lineplot(
        data=df_final, x='mean_radius', y='arc_length', hue='streak_id',
        palette='tab20', legend='full', ax=ax2, lw=2.5 # Turn legend on to see IDs
    )
    ax2.set_xlabel("Radius from Colony Center"); ax2.set_ylabel("Measured Arc Length (Linear Units)")
    ax2.grid(True, which='both', linestyle=':')
    ax2.set_ylim(bottom=0)

    output_dir = PROJECT_ROOT / "figures" / "streak_analysis"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{args.pop_file.stem}_analysis_filtered.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis figure saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()