# FILE: src/config.py
# ==============================================================================
# SINGLE SOURCE OF TRUTH for ALL simulation campaigns.
# ==============================================================================

import numpy as np

EXPERIMENTS = {
    # --- Experiment 1: The main phase diagram analysis for mean mutant fraction ---
    "p1_definitive_v2": {
        "CAMPAIGN_ID": "p1_definitive_v2",
        "run_mode": "steady_state",
        "HPC_PARAMS": {
            "time": "0-08:00:00",
            "mem": "4G",
            "sims_per_task": 100,
        },
        "PARAM_GRID": {
            "b_m": [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95],
            "k_total_low": np.logspace(-2, -1, 6).tolist(),
            "k_total_mid": np.logspace(-0.82, 0, 5).tolist(),
            "k_total_high": np.logspace(0.18, 2, 10).tolist(),
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
            "time": "0-12:00:00",
            "mem": "2G",
            "sims_per_task": 50,
        },
        "PARAM_GRID": {
            "b_m": np.unique(
                np.concatenate(
                    [
                        np.linspace(0, 0.8, 100),
                        np.linspace(0.80, 0.98, 20),
                        np.linspace(0.985, 0.995, 5),
                        np.array([1.0]),
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
    # --- Experiment 3: Definitive analysis of spatial structure and criticality ---
    "spatial_structure_v1": {
        "CAMPAIGN_ID": "spatial_structure_v1_fm75",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {
            "time": "0-12:00:00",
            "mem": "4G",
            "sims_per_task": 10,
        },
        "PARAM_GRID": {
            "b_m": [0.5, 0.8, 0.95],
            "k_total": np.logspace(-2, 2, 20).tolist(),
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "phi": -0.5,
                    "num_replicates": 24,
                    "warmup_time": 2000.0,
                    "num_samples": 100,
                    "sample_interval": 10.0,
                },
                "grid_params": {"b_m": "b_m", "k_total": "k_total"},
            }
        },
    },
    # --- Experiment 4: Refined KPZ Diffusion/Roughening Analysis ---
    "diffusion_v2_refined": {
        "CAMPAIGN_ID": "diffusion_v2_refined_neutral",
        "run_mode": "diffusion",
        "HPC_PARAMS": {
            "time": "1-00:00:00",  # Request 1 full day for long runs
            "mem": "4G",
            "sims_per_task": 25,
        },
        "PARAM_GRID": {
            "width": [32, 64, 128, 256, 512],
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "length": 4096,
                    "b_m": 1.0,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "initial_mutant_patch_size": 0,
                    "max_steps": 25_000_000,
                    "num_replicates": 400,
                },
                "grid_params": {"width": "width"},
            }
        },
    },
}
