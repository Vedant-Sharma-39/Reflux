# FILE: scripts/paper_figures/sup_fig_aif_growth_calibration.py

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import json
import gzip
from tqdm import tqdm
import matplotlib

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# --- Add project root to Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data

def fit_through_origin(x, y):
    """
    Performs a linear regression of the form y = m*x (intercept is zero).
    Returns the slope 'm'.
    """
    # Using the formula: m = sum(x*y) / sum(x^2)
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y) / np.dot(x, x)

def main():
    campaign_id = EXPERIMENTS["aif_growth_timeseries"]["campaign_id"]
    df_summary = load_aggregated_data(campaign_id, str(PROJECT_ROOT))
    if df_summary.empty: sys.exit(f"Data for '{campaign_id}' is empty.")

    ts_dir = PROJECT_ROOT / "data" / campaign_id / "timeseries"
    figure_dir = PROJECT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)

    # --- Step 1 & 2: Load data and calculate individual growth rates ---
    # (This part remains the same as the previous correct version)
    all_ts_data = []
    for _, row in tqdm(df_summary.iterrows(), desc="Loading timeseries"):
        ts_file_path = ts_dir / f"ts_{row['task_id']}.json.gz"
        if not ts_file_path.exists(): continue
        with gzip.open(ts_file_path, "rt") as f: ts_list = json.load(f)
        df_ts = pd.DataFrame(ts_list)
        df_ts['b'] = row['b_sus']
        df_ts['replicate'] = row['replicate']
        all_ts_data.append(df_ts)
        
    df_full = pd.concat(all_ts_data)
    
    growth_rate_results = []
    for params, group_df in tqdm(df_full.groupby(['b', 'replicate']), desc="Calculating slopes"):
        b_val, _ = params
        if len(group_df) < 5: continue
        fit = linregress(group_df['time'], group_df['radius'])
        growth_rate_results.append({"b": b_val, "growth_rate": fit.slope})

    df_rates = pd.DataFrame(growth_rate_results)
    df_plot_data = df_rates.groupby('b').agg(
        mean_growth_rate=('growth_rate', 'mean'),
        std_growth_rate=('growth_rate', 'std')
    ).reset_index()

    # --- Step 3: Generate the polished 2-panel figure ---
    print("Generating polished 2-panel calibration figure...")
    sns.set_theme(style="ticks", context="paper")
    
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300, constrained_layout=True)
    fig.suptitle("Supplementary Figure: Calibration of Cellular Birth Rate Parameter", fontsize=12, weight='bold')

    # Panel A (unchanged)
    sns.lineplot(data=df_full, x="time", y="radius", hue="b", palette="coolwarm", ci='sd', lw=2, ax=axA)
    axA.set_title("(A) Colony Growth Dynamics", fontsize=10)
    axA.set_xlabel("Simulation Time", fontsize=10)
    axA.set_ylabel("Median Colony Radius", fontsize=10)
    legA = axA.legend(title="Birth Rate (b)", fontsize=8)
    plt.setp(legA.get_title(), fontsize=8)
    
    # --- Panel B with Physically-Constrained Fit ---
    axB.errorbar(
        df_plot_data["b"], df_plot_data["mean_growth_rate"],
        yerr=df_plot_data["std_growth_rate"], fmt='o',
        markersize=6, capsize=4, color="#003f5c", label="Mean Growth Rate"
    )
    
    # --- THIS IS THE FIX ---
    # Perform the linear fit forced through the origin
    slope_forced = fit_through_origin(df_plot_data["b"], df_plot_data["mean_growth_rate"])
    
    x_fit = np.array([0, df_plot_data["b"].max()])
    y_fit = slope_forced * x_fit
    axB.plot(x_fit, y_fit, color="#d35400", ls='--', lw=2,
             label=f"Fit (Growth Rate = {slope_forced:.3f} * b)")
    
    axB.set_title("(B) Quantified Growth Rate", fontsize=10)
    axB.set_xlabel("Cellular Birth Rate Parameter (b)", fontsize=10)
    axB.set_ylabel("Colony Radial Growth Rate", fontsize=10)
    axB.legend(fontsize=8)
    axB.set_xlim(left=0) # Start x-axis at 0
    axB.set_ylim(bottom=0) # Start y-axis at 0
    
    for ax in [axA, axB]:
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, which='both', linestyle=':', alpha=0.7)
    
    sns.despine(fig)
    
    # --- Print the new, single-parameter calibration result ---
    print("\n--- RIGOROUS CALIBRATION RESULTS ---")
    print(f"Proportionality Constant (Slope): {slope_forced:.4f} (Radius units / Time units) per b")
    # ---

    output_path = figure_dir / "sup_fig_aif_growth_calibration.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\n✅ Physically-constrained calibration figure saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()