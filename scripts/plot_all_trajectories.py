#!/usr/bin/env python3
# FILE: scripts/analyze_circular_drift_diffusion.py
#
# Analyzes trajectory files from a circular colony simulation to estimate
# the drift (selection) and diffusion (genetic drift) parameters based on
# the theory from Hallatschek & Nelson (2010).

import sys
import json
import gzip
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_ID = "aif_online_scan_v1"
RESISTANT, COMPENSATED = 2, 3
MUTANT_TYPES = [RESISTANT, COMPENSATED]

# We need the parameter map to group results by condition
# (this assumes a summary file exists from the previous, more complex script)
# If not, we can adapt the script to parse parameters from filenames if needed.
SUMMARY_CSV_PATH = PROJECT_ROOT / "data" / CAMPAIGN_ID / "analysis" / f"{CAMPAIGN_ID}_summary_aggregated.csv"

# --- Main Analysis Logic ---

def process_file_for_analysis(file_path: Path) -> pd.DataFrame:
    """Loads a single trajectory file and prepares it for drift-diffusion analysis."""
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        rows = data if isinstance(data, list) else data.get("sector_trajectory", [])
        if not rows: return pd.DataFrame()

        df = pd.DataFrame(rows)

        # 1. Identify the founding mutant lineage
        df_mutants = df[df['type'].isin(MUTANT_TYPES)].copy()
        if df_mutants.empty: return pd.DataFrame()

        min_radius = df_mutants['radius'].min()
        founding_root_id = df_mutants[df_mutants['radius'] == min_radius]['root_sid'].iloc[0]

        lineage_df = df_mutants[df_mutants['root_sid'] == founding_root_id].copy()
        
        # Ensure data is sorted by radius to calculate differences correctly
        lineage_df = lineage_df.sort_values(by='radius').reset_index(drop=True)
        
        # Skip trajectories that are too short to analyze
        if len(lineage_df) < 5: return pd.DataFrame()

        # 2. Convert width to angle (Φ = X / r)
        # Avoid division by zero, although radius should always be > 0
        lineage_df['angle'] = lineage_df['width_cells'] / lineage_df['radius'].replace(0, 1e-6)
        
        # 3. Calculate the inverse radius and its step-by-step changes
        lineage_df['inv_radius'] = 1 / lineage_df['radius']

        # 4. Calculate step-by-step changes (displacements)
        lineage_df['d_angle'] = lineage_df['angle'].diff()
        lineage_df['d_inv_radius'] = lineage_df['inv_radius'].diff().abs() # Change in 1/r is negative as r grows

        # Return only the valid displacement data
        return lineage_df[['d_angle', 'd_inv_radius']].dropna()

    except Exception:
        return pd.DataFrame() # Return empty on any error

def main():
    print(f"--- Drift & Diffusion Analysis for Campaign: {CAMPAIGN_ID} ---")

    # 1. Load parameter map to know which file belongs to which condition
    if not SUMMARY_CSV_PATH.exists():
        sys.exit(f"[ERROR] Summary file not found: {SUMMARY_CSV_PATH}")
    
    param_map = pd.read_csv(SUMMARY_CSV_PATH, usecols=["task_id", "b_res", "sector_width_initial"])
    param_map = param_map.rename(columns={"task_id": "run_id"}).drop_duplicates("run_id")

    traj_dir = PROJECT_ROOT / "data" / CAMPAIGN_ID / "trajectories"

    # 2. Process all files and collect displacement data
    all_displacements = []
    
    pbar = tqdm(param_map.itertuples(), total=len(param_map), desc="Processing files")
    for row in pbar:
        file_path = traj_dir / f"traj_{row.run_id}.json.gz"
        if not file_path.exists(): continue
        
        displacements_df = process_file_for_analysis(file_path)
        
        if not displacements_df.empty:
            displacements_df['b_res'] = row.b_res
            displacements_df['initial_width'] = row.sector_width_initial
            all_displacements.append(displacements_df)
    
    if not all_displacements:
        sys.exit("No valid trajectory data found to analyze.")

    df_all = pd.concat(all_displacements, ignore_index=True)

    # 3. Analyze results for each experimental condition
    print("\n--- Results ---")
    grouped = df_all.groupby(['b_res', 'initial_width'])

    for (b_res, w0), group in grouped:
        print(f"\nCondition: b_res = {b_res:.4f}, initial_width = {w0}")

        # --- Calculate Drift ---
        # Drift is the mean angular change per unit change in inverse radius
        # A positive drift means the angle tends to increase (beneficial)
        # We divide by the mean change in 1/r to get a rate
        mean_d_angle = group['d_angle'].mean()
        mean_d_inv_r = group['d_inv_radius'].mean()
        drift_rate = mean_d_angle / mean_d_inv_r if mean_d_inv_r != 0 else 0
        
        print(f"  Drift (Selection):")
        print(f"    - Mean angular change per step ⟨ΔΦ⟩: {mean_d_angle:.6f} radians")
        print(f"    - Normalized Drift Rate: {drift_rate:.4f}")
        if drift_rate > 0.1: print("    - Interpretation: Likely beneficial mutation (b_res > 1.0)")
        elif drift_rate < -0.1: print("    - Interpretation: Likely deleterious mutation (b_res < 1.0)")
        else: print("    - Interpretation: Likely neutral (b_res ≈ 1.0)")

        # --- Calculate Diffusion ---
        # From Hallatschek Eq(6): ⟨(ΔΦ)²⟩ = 4 * D_x * Δ(1/r)
        # So, D_x = ⟨(ΔΦ)²⟩ / (4 * Δ(1/r))
        # This is the slope of Mean Squared Displacement vs. 4*Δ(1/r)
        
        # We calculate it from the mean of all squared displacements
        mean_sq_d_angle = (group['d_angle'] ** 2).mean()
        
        # Calculate the diffusion constant D_x
        # We assume the drift is small, so ⟨(ΔΦ)²⟩ ≈ Var(ΔΦ) + ⟨ΔΦ⟩²
        diffusion_constant_Dx = mean_sq_d_angle / (4 * mean_d_inv_r) if mean_d_inv_r != 0 else 0
        
        print(f"  Diffusion (Genetic Drift):")
        print(f"    - Mean squared angular change ⟨(ΔΦ)²⟩: {mean_sq_d_angle:.8f} radians²")
        print(f"    - Deduced Diffusion Constant D_x: {diffusion_constant_Dx:.4f} (cells)")
        print(f"    - Interpretation: Quantifies the strength of random fluctuations (genetic drift).")

if __name__ == "__main__":
    main()