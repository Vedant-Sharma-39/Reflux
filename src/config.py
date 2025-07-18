# FILE: src/config.py
# ==============================================================================
# SINGLE SOURCE OF TRUTH for ALL simulation campaigns.
#
# To define a new experiment, add a new entry to the main EXPERIMENTS
# dictionary. Each experiment needs:
#
# - CAMPAIGN_ID: A unique string for naming directories and files.
# - run_mode: A keyword ('steady_state', 'calibration') that tells the
#             unified worker (src/worker.py) which simulation logic to run.
# - HPC_PARAMS: A dictionary of resources needed for each Slurm job.
# - PARAM_GRID: The universe of all parameter values to be swept.
# - SIM_SETS: Defines the specific parameter combinations for this experiment.
# ==============================================================================

import numpy as np

EXPERIMENTS = {
    # --- Experiment 1: The main phase diagram analysis ---
    "p1_definitive_v2": {
        "CAMPAIGN_ID": "p1_definitive_v2",
        "run_mode": "steady_state",
        "HPC_PARAMS": {
            "time": "0-08:00:00",
            "mem": "4G",
            "sims_per_task": 100,  # Good for many short/medium-length sims
        },
        "PARAM_GRID": {
            "b_m": [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95],
            "k_total_low": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
            "k_total_mid": [0.15, 0.2, 0.3, 0.5, 0.75],
            "k_total_high": [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0, 100.0],
            "phi": np.linspace(-1.0, 1.0, 21).tolist(),
            "width_scaling": [32, 64, 128, 256],
        },
        "SIM_SETS": {
            "main_high_k": {
                "base_params": {
                    "width": 128,
                    "length": 50000,
                    "num_replicates": 8,
                    "total_run_time": 1000.0,
                    "warmup_time": 500.0,
                    "sample_interval": 10.0,
                },
                "grid_params": {"k_total": "k_total_high", "b_m": "b_m", "phi": "phi"},
            },
            "main_low_k": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "num_replicates": 24,
                    "total_run_time": 4000.0,
                    "warmup_time": 2000.0,
                    "sample_interval": 20.0,
                },
                "grid_params": {"k_total": "k_total_low", "b_m": "b_m", "phi": "phi"},
            },
            "main_mid_k": {
                "base_params": {
                    "width": 128,
                    "length": 50000,
                    "num_replicates": 12,
                    "total_run_time": 2000.0,
                    "warmup_time": 1000.0,
                    "sample_interval": 15.0,
                },
                "grid_params": {"k_total": "k_total_mid", "b_m": "b_m", "phi": "phi"},
            },
            "scaling": {
                "base_params": {
                    "length": 50000,
                    "num_replicates": 16,
                    "total_run_time": 1000.0,
                    "warmup_time": 500.0,
                    "sample_interval": 10.0,
                    "b_m": 0.8,
                    "k_total": 1.0,
                    "phi": -0.5,
                },
                "grid_params": {"width": "width_scaling"},
            },
        },
        # Parameters used by analysis scripts, not the launcher
        "ANALYSIS_PARAMS": {
            "slice_plot_b_m": [0.65, 0.8, 0.95],
            "fitness_cost_plot_f_M": [0.5, 0.75, 0.95],
            "crossover_fit_f_M": 0.75,
        },
    },
    # --- Experiment 2: Calibration of drift velocity ---
    "calibration_v4": {
        "CAMPAIGN_ID": "calibration_v4_deleterious_focus",
        "run_mode": "calibration",
        "HPC_PARAMS": {
            "time": "0-12:00:00",  # Calibration runs can be long and have high variance
            "mem": "2G",  # But they are memory-light
            "sims_per_task": 50,  # Smaller chunk size is good for high-variance run times
        },
        "PARAM_GRID": {
            # Densely sample the deleterious range (b_m < 1)
            "b_m": np.unique(
                np.concatenate(
                    [
                        np.linspace(0, 0.8, 100),
                        np.linspace(0.80, 0.98, 20),
                        np.linspace(0.985, 0.995, 5),
                        np.array([1.0]),  # Include the neutral case
                    ]
                )
            ).tolist()
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 512,
                    "initial_patch_size": 80,
                    "max_steps": 3_000_000,
                    "num_replicates": 500,
                },
                "grid_params": {"b_m": "b_m"},
            }
        },
    },
    # --- Add your next great experiment here! ---
    # "my_new_experiment": {
    #     "CAMPAIGN_ID": "a_unique_name_for_folders",
    #     "run_mode": "some_new_mode", # Requires adding logic to src/worker.py
    #     "HPC_PARAMS": { ... },
    #     "PARAM_GRID": { ... },
    #     "SIM_SETS": { ... },
    # },
}
