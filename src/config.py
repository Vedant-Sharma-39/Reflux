# src/config.py
# SINGLE SOURCE OF TRUTH for the Phase 1 Definitive Campaign, Version 2.
# This version adds data density in critical regions identified from the v1 analysis.

import numpy as np

# --- 1. Campaign Identifier ---
CAMPAIGN_ID = "p1_definitive_v2"

# --- 2. Core Simulation Parameters (The Full Sweep) ---
PARAM_GRID = {
    # ADDED a new b_m value to better resolve fitness-dependent trends.
    "b_m": [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95],
    # REFINED k_total ranges for higher density in the crossover region.
    "k_total_low": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],  # Increased density
    "k_total_mid": [0.15, 0.2, 0.3, 0.5, 0.75],  # New mid-range
    "k_total_high": [
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        7.0,
        10.0,
        20.0,
        50.0,
        100.0,
    ],  # Increased density
    # Kept phi the same, it was sufficient.
    "phi": np.linspace(-1.0, 1.0, 21).tolist(),
}
# Combine all k_total ranges for analysis convenience
PARAM_GRID["k_total_all"] = sorted(
    list(
        set(
            PARAM_GRID["k_total_low"]
            + PARAM_GRID["k_total_mid"]
            + PARAM_GRID["k_total_high"]
        )
    )
)

# --- 3. HPC & Simulation Settings ---
SIM_PARAMS_STD = {
    "length": 50000,
    "total_run_time": 1000.0,
    "warmup_time": 500.0,
    "sample_interval": 10.0,
    "num_replicates": 8,
}
SIM_PARAMS_LONG = {
    "length": 50000,
    "total_run_time": 4000.0,
    "warmup_time": 2000.0,
    "sample_interval": 20.0,
    "num_replicates": 24,  # INCREASED replicates for low-k
}
SCALING_PARAMS = {  # No changes needed
    "width": [32, 64, 128, 256],
    "b_m": [0.8],
    "k_total": [1.0, 20.0],
    "phi": [-0.5],
    "num_replicates": 16,
}

# --- 4. Analysis & Plotting Parameters (More Principled Selection) ---
ANALYSIS_PARAMS = {
    # Figure 3: Now a 3-panel plot for a clearer trend
    "slice_plot_b_m": [0.65, 0.8, 0.95],
    # Figure 5: A more systematic selection of bias levels
    "fitness_cost_plot_f_M": [0.5, 0.75, 0.95],  # Neutral, Strong, Very Strong Bias
    # Figure 7: Still a good choice
    "crossover_fit_f_M": 0.75,
}
