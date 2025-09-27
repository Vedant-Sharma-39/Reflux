# FILE: scripts/analyze_aif_sectors.py (Complete, Final Version)
#
# An exploratory tool to test different algorithms for identifying and measuring
# sectors in the AIF model from a single saved population file.
#
# Key Features:
# 1. Angular Clustering + k-Nearest Neighbors Refinement for robust sector identification.
# 2. LOWESS (Locally Weighted Scatterplot Smoothing) for superior trendline generation.
# 3. An analysis cutoff to prevent edge artifacts in width measurement.
# 4. Verbosity flags in analysis functions to allow for silent, reusable operation.

import sys
import json
import gzip
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
from scipy import stats
import statsmodels.api as sm

# --- Add project root to Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Configuration & Constants ---
DATA_FILE_PATH = PROJECT_ROOT / "figures" / "debug_runs" / "aif_multisector_final_pop.json.gz"
RESISTANT_TYPES = {2, 3}  # Type 2 (Resistant) and 3 (Compensated)
ANGULAR_GAP_THRESHOLD = 1.0
LOWESS_FRAC = 0.15
# Cutoff to prevent edge artifacts: we only analyze data up to this
# percentage of a sector's maximum measured radius.
ANALYSIS_RANGE_PERCENTILE = 0.95


# --- Helper Functions ---
def _axial_to_cartesian(q: int, r: int, size: float = 1.0) -> tuple[float, float]:
    """Converts axial hex coordinates to cartesian for plotting."""
    x = size * (3.0 / 2.0 * q)
    y = size * (np.sqrt(3) / 2.0 * q + np.sqrt(3) * r)
    return x, y

def load_population_data(file_path: Path) -> pd.DataFrame:
    """Loads and preprocesses the saved population data from a gzipped JSON file."""
    if not file_path.exists():
        sys.exit(f"Data file not found: {file_path}\n"
                 "Please run 'scripts/debug_aif_sector_analysis.py' first to generate it.")
    
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        pop_data = json.load(f)
    
    df = pd.DataFrame(pop_data)
    
    # Pre-calculate cartesian and polar coordinates for all cells
    cartesian_coords = np.array([_axial_to_cartesian(q, r) for q, r in zip(df['q'], df['r'])])
    df['x'] = cartesian_coords[:, 0]
    df['y'] = cartesian_coords[:, 1]
    df['radius'] = np.linalg.norm(cartesian_coords, axis=1)
    df['angle'] = np.arctan2(df['y'], df['x'])
    
    return df

def refine_sectors_by_neighborhood(df: pd.DataFrame, k: int = 7, iterations: int = 2, verbose: bool = True) -> pd.DataFrame:
    """Corrects sector assignments based on a majority vote of spatial neighbors."""
    if verbose:
        print(f"Refining sector labels using {k}-nearest neighbors ({iterations} iterations)...")
    if df.empty or 'sector_id' not in df.columns: return df

    kdtree = KDTree(df[['x', 'y']].to_numpy())
    
    for i in range(iterations):
        _, indices = kdtree.query(df[['x', 'y']], k=k)
        all_sector_ids = df['sector_id'].to_numpy()
        neighbor_ids = all_sector_ids[indices]
        modes, _ = stats.mode(neighbor_ids, axis=1, keepdims=True)
        df['sector_id'] = modes.flatten()
        if verbose:
            print(f"  Refinement pass {i+1} complete.")
        
    return df

# --- Main Clustering and Measurement Algorithms ---
def identify_sectors_by_angle(df_resistant: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Pass 1: Identifies sectors by finding large gaps in the angular distribution."""
    if verbose:
        print("Identifying sectors using Angular Clustering...")
    if df_resistant.empty:
        df_resistant['sector_id'] = -1
        return df_resistant

    df_sorted = df_resistant.sort_values('angle').copy()
    angles = df_sorted['angle'].to_numpy()
    gaps = np.diff(angles)
    wrap_around_gap = (angles[0] + 2 * np.pi) - angles[-1]
    all_gaps = np.append(gaps, wrap_around_gap)

    is_break_point = all_gaps > ANGULAR_GAP_THRESHOLD
    sector_labels = np.cumsum(is_break_point)
    # Handle wrap-around case where the first and last point are in the same sector
    if sector_labels.size > 0 and sector_labels[-1] > 0 and is_break_point[-1]:
         sector_labels = (sector_labels - sector_labels[-1]) % (sector_labels[-1])

    df_sorted['sector_id'] = sector_labels
    num_sectors = len(df_sorted['sector_id'].unique())
    if verbose:
        print(f"Found {num_sectors} distinct sectors initially.")
    return df_sorted

def measure_width_radial_binning(sector_df: pd.DataFrame) -> pd.DataFrame:
    """Measures sector width using robust radial binning, avoiding edge artifacts."""
    if sector_df.empty or len(sector_df) < 10: return pd.DataFrame()

    # Define a cutoff radius to avoid artifacts at the sparse outer edge.
    max_sector_radius = sector_df['radius'].max()
    analysis_cutoff_radius = max_sector_radius * ANALYSIS_RANGE_PERCENTILE

    bins = np.arange(0, max_sector_radius + 2.0, 2.0)
    sector_df['radius_bin'] = pd.cut(sector_df['radius'], bins, right=False)
    
    results = []
    # Use observed=True to adopt new, efficient Pandas behavior and silence warnings.
    for bin_interval, group in sector_df.groupby("radius_bin", observed=True):
        # Apply the cutoff: skip bins that are too close to the sector's edge.
        if bin_interval.mid > analysis_cutoff_radius:
            continue
            
        if len(group) < 5: continue
        angles = group['angle'].to_numpy()
        p_lower, p_upper = np.percentile(angles, [5, 95])
        span = (p_upper - p_lower + 2 * np.pi) % (2 * np.pi)
        results.append({'mean_radius': bin_interval.mid, 'width_rad': span})
        
    df_results = pd.DataFrame(results)
    df_results['method'] = 'Radial Binning'
    return df_results
    
def main():
    """Main analysis and visualization pipeline for a single simulation file."""
    # 1. Load and preprocess data
    df_pop = load_population_data(DATA_FILE_PATH)
    df_resistant = df_pop[df_pop['type'].isin(RESISTANT_TYPES)].copy()
    
    # 2. Run the analysis pipeline
    df_angular_clustered = identify_sectors_by_angle(df_resistant, verbose=True)
    df_refined_clustered = refine_sectors_by_neighborhood(df_angular_clustered, verbose=True)
    
    # 3. Measure width for each identified sector
    all_analysis_results = []
    unique_sectors = sorted(df_refined_clustered['sector_id'].unique())
    for sector_id in unique_sectors:
        df_sector = df_refined_clustered[df_refined_clustered['sector_id'] == sector_id]
        results_binning = measure_width_radial_binning(df_sector.copy())
        if not results_binning.empty:
            results_binning['sector_id'] = sector_id
            all_analysis_results.append(results_binning)

    if not all_analysis_results:
        sys.exit("Analysis failed: No sectors with sufficient data found after filtering.")
        
    df_final = pd.concat(all_analysis_results).sort_values(by=['sector_id', 'mean_radius']).dropna()

    # 4. Smooth the data for visualization
    print(f"Applying LOWESS smoothing with frac={LOWESS_FRAC}...")
    smoothed_dfs = []
    for sector_id, group in df_final.groupby('sector_id'):
        if len(group) < 3: continue
        smoothed = sm.nonparametric.lowess(endog=group['width_rad'], exog=group['mean_radius'], frac=LOWESS_FRAC)
        df_smooth = pd.DataFrame(smoothed, columns=['mean_radius', 'smoothed_width'])
        df_smooth['sector_id'] = sector_id
        smoothed_dfs.append(df_smooth)
    
    if not smoothed_dfs:
        sys.exit("Smoothing failed, not enough data points remain after filtering.")
    df_smoothed_final = pd.concat(smoothed_dfs)

    # --- 5. Visualization ---
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.suptitle("Sector Analysis using Angular Clustering + Neighborhood Refinement", fontsize=24, y=0.98)

    # Panel A: Diagnostic plot to verify clustering
    ax1.set_title("Final Sector Identification", fontsize=18)
    df_susceptible = df_pop[~df_pop['type'].isin(RESISTANT_TYPES)]
    ax1.scatter(df_susceptible['x'], df_susceptible['y'], c='lightgrey', s=1, alpha=0.3)
    sns.scatterplot(
        data=df_refined_clustered, x='x', y='y', hue='sector_id', palette='viridis',
        s=5, legend='full', ax=ax1
    )
    ax1.set_xlabel("X Coordinate"); ax1.set_ylabel("Y Coordinate")
    ax1.set_aspect('equal', 'box'); ax1.legend(title="Refined Sector ID")

    # Panel B: Width vs. Radius analysis plot
    ax2.set_title("Sector Width vs. Radius (LOWESS Smoothed)", fontsize=18)
    sns.scatterplot(
        data=df_final, x='mean_radius', y='width_rad', hue='sector_id', palette='viridis',
        alpha=0.4, s=50, legend=False, ax=ax2
    )
    sns.lineplot(
        data=df_smoothed_final, x='mean_radius', y='smoothed_width', hue='sector_id',
        palette='viridis', lw=3.5, legend='full', ax=ax2
    )
    ax2.set_xlabel("Radius from Colony Center"); ax2.set_ylabel("Measured Sector Width (Radians)")
    ax2.grid(True, which='both', linestyle=':'); 
    ax2.legend(title="Sector ID")
    ax2.set_ylim(bottom=0)

    # --- 6. Save Final Figure ---
    output_path = PROJECT_ROOT / "figures" / "debug_runs" / "aif_sector_analysis_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analysis figure saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()