# FILE: src/utils/analysis_helpers.py (NEW FILE)

import sys
import json
import gzip
from pathlib import Path
import pandas as pd
import numpy as np

# --- Helper Functions ---
def _axial_to_cartesian(q: int, r: int, size: float = 1.0) -> tuple[float, float]:
    """Converts axial hex coordinates to cartesian for plotting."""
    x = size * (3.0 / 2.0 * q)
    y = size * (np.sqrt(3) / 2.0 * q + np.sqrt(3) * r)
    return x, y

def load_population_data(file_path: Path) -> pd.DataFrame:
    """Loads and preprocesses the saved population data from a gzipped JSON file."""
    if not file_path.exists():
        # A more informative error message
        sys.exit(f"Data file not found: {file_path}\n"
                 "Please ensure the path is correct and the simulation has generated the file.")

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

def measure_width_radial_binning(sector_df: pd.DataFrame, analysis_range_percentile=0.95) -> pd.DataFrame:
    """
    Measures sector width using robust radial binning.
    CORRECTED to handle angle wrap-around by finding the largest gap between cells.
    """
    if sector_df.empty or len(sector_df) < 10: return pd.DataFrame()

    max_sector_radius = sector_df['radius'].max()
    analysis_cutoff_radius = max_sector_radius * analysis_range_percentile

    bins = np.arange(0, max_sector_radius + 5.0, 5.0) # Slightly larger bins can be more stable
    sector_df['radius_bin'] = pd.cut(sector_df['radius'], bins, right=False)

    results = []
    for bin_interval, group in sector_df.groupby("radius_bin", observed=True):
        if bin_interval.mid > analysis_cutoff_radius:
            continue

        if len(group) < 5: continue
        
        # --- START OF THE CRITICAL FIX ---
        
        # Get angles and sort them from -pi to +pi
        angles = np.sort(group['angle'].to_numpy())
        
        # Calculate the gaps between consecutive sorted angles
        gaps = np.diff(angles)
        
        # Calculate the wrap-around gap between the last and first angle
        wrap_around_gap = (angles[0] + 2 * np.pi) - angles[-1]
        
        # The true angular width is 2*pi minus the largest gap found
        max_gap = np.max(np.append(gaps, wrap_around_gap))
        span = 2 * np.pi - max_gap
        
        # --- END OF THE CRITICAL FIX ---

        results.append({'mean_radius': bin_interval.mid, 'width_rad': span})

    df_results = pd.DataFrame(results)
    df_results['method'] = 'Radial Binning (Corrected)'
    return df_results
