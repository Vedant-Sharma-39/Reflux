# FILE: scripts/utils/process_aif_trajectories.py (Using Gzipped CSV)

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import multiprocessing
import os

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data
from scripts.analyze_aif_sectors import (
    load_population_data, identify_sectors_by_angle,
    refine_sectors_by_neighborhood, measure_width_radial_binning,
)

def process_simulation_output(task_info: dict) -> pd.DataFrame | None:
    """WORKER FUNCTION: Analyzes a single population file and returns its spatial trajectory."""
    pop_file_path = task_info["pop_file_path"]
    if not pop_file_path.exists(): return None
    
    df_pop = load_population_data(pop_file_path)
    df_resistant = df_pop[df_pop['type'].isin({2, 3})].copy()
    if df_resistant.empty:
        df = pd.DataFrame([{"mean_radius": 0, "arc_length": 0, "width_rad": 0}])
    else:
        df_clustered = identify_sectors_by_angle(df_resistant, verbose=False)
        df_refined = refine_sectors_by_neighborhood(df_clustered, verbose=False)
        df = measure_width_radial_binning(df_refined)
        if df.empty:
             df = pd.DataFrame([{"mean_radius": 0, "arc_length": 0, "width_rad": 0}])
        else:
            df['arc_length'] = df['mean_radius'] * df['width_rad']
    
    df["b_res"] = task_info["b_res"]
    df["initial_width"] = task_info["initial_width"]
    df["replicate"] = task_info["replicate"]
    return df

def main():
    campaign_id = EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"]
    print(f"--- Pre-processing AIF Trajectories for campaign: {campaign_id} ---")
    
    df_summary = load_aggregated_data(campaign_id, str(PROJECT_ROOT))
    if df_summary.empty: sys.exit(f"Data for '{campaign_id}' is empty.")

    data_dir = PROJECT_ROOT / "data" / campaign_id
    pop_dir = data_dir / "populations"
    analysis_dir = data_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    if not pop_dir.exists(): sys.exit(f"Population directory not found: {pop_dir}")

    tasks = [{"pop_file_path": pop_dir / f"pop_{r['task_id']}.json.gz", "b_res": r["b_res"], 
              "initial_width": r["sector_width_initial"], "replicate": r["replicate"]} for _, r in df_summary.iterrows()]

    num_workers = max(1, os.cpu_count() - 1)
    print(f"Processing {len(tasks)} population files using {num_workers} workers...")
    
    all_measurements = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        pbar = tqdm(pool.imap_unordered(process_simulation_output, tasks), total=len(tasks), desc="Processing Files")
        for result_df in pbar:
            if result_df is not None:
                all_measurements.append(result_df)
    
    if not all_measurements: sys.exit("No valid measurements extracted.")
    df_processed = pd.concat(all_measurements)

    # --- SAVE TO GZIPPED CSV (NO PYARROW NEEDED) ---
    output_path = analysis_dir / "processed_spatial_trajectories.csv.gz"
    df_processed.to_csv(output_path, index=False, compression='gzip')
    
    print(f"\n✅ Successfully processed {len(df_summary)} simulations.")
    print(f"   Intermediate data saved to: {output_path}")

if __name__ == "__main__":
    main()