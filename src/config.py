# FILE: src/config.py (Updated with the Optimized Figure 5 Campaign)

import numpy as np

# ==============================================================================
# 1. FINAL PARAMETER GRIDS
# (This section is unchanged and remains correct)
# ==============================================================================
PARAM_GRID = {
    "bm_final_wide": np.round(np.linspace(0.1, 1.0, 10), 2).tolist(),
    "bm_final_narrow": np.round(np.linspace(0.8, 1.0, 5), 2).tolist(),
    "phi_final_full": np.round(np.linspace(-1.0, 1.0, 9), 2).tolist(),
    "phi_final_asymmetric": np.round(np.linspace(-1.0, 0.5, 7), 2).tolist(),
    "k_total_final_log": np.round(np.logspace(-2, 2, 20), 5).tolist(),
    "k_total_final_extended": np.round(np.logspace(-2.5, 1.5, 13), 4).tolist(),
    "k_zero": [0],
    "width_scan": [64, 128, 256, 512],
    "patch_width_scan": [30, 60, 120],
    "ic_control_scan": [0],
    "env_bet_hedging": {"0": {"b_wt": 1.0}, "1": {"b_wt": 0.0, "b_m": 1.0}},
    "env_asymmetric_cycle_refined": [
        {
            "name": "30_90",
            "patches": [
                {"id": 0, "width": 30, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 90, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
        {
            "name": "60_60",
            "patches": [
                {"id": 0, "width": 60, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 60, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
        {
            "name": "90_30",
            "patches": [
                {"id": 0, "width": 90, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 30, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
        {
            "name": "scrambled_60_60",
            "scrambled": True,
            "cycle_length": 120,
            "avg_patch_width": 60,
            "patches": [
                {"id": 0, "proportion": 0.5, "params": {"b_wt": 1.0}},
                {"id": 1, "proportion": 0.5, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
    ],
    "bm_visualization": [0.8, 0.95],
    "k_total_visualization": [0.02, 0.5],
    "phi_visualization": [0.0],
    "patch_width_visualization": [60],
}


# ==============================================================================
# 2. FINAL EXPERIMENT DEFINITIONS
# ==============================================================================
EXPERIMENTS = {
    # --- Figure 1-3 Campaigns Remain Unchanged ---
    "boundary_analysis": {
        "campaign_id": "fig1_boundary_analysis",
        "run_mode": "calibration",
        "hpc_params": {"time": "06:00:00", "mem": "1G", "sims_per_task": 100},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 512,
                    "length": 1024,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 500,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 128,
                    "max_steps": 2000000,
                },
                "grid_params": {"b_m": "bm_final_wide"},
            }
        },
    },
    "kpz_scaling": {
        "campaign_id": "fig1_kpz_scaling",
        "run_mode": "diffusion",
        "hpc_params": {"time": "05:00:00", "mem": "2G", "sims_per_task": 50},
        "sim_sets": {
            "main": {
                "base_params": {
                    "length": 4096,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 250,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 0,
                    "max_steps": 5000000,
                },
                "grid_params": {"b_m": "bm_final_wide", "width": "width_scan"},
            }
        },
    },
    "phase_diagram": {
        "campaign_id": "fig2_phase_diagram",
        "run_mode": "phase_diagram",
        "hpc_params": {"time": "05:30:00", "mem": "2G", "sims_per_task": 150},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 4096,
                    "num_replicates": 50,
                    "warmup_time": 500.0,
                    "num_samples": 200,
                    "sample_interval": 10.0,
                    "initial_condition_type": "mixed",
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                },
            }
        },
    },
    "bet_hedging_final": {
        "campaign_id": "fig3_bet_hedging_final",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "11:00:00", "mem": "2G", "sims_per_task": 100},
        "sim_sets": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "environment_map": "env_bet_hedging",
                    "num_replicates": 32,
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                    "patch_width": "patch_width_scan",
                },
            },
            "controls": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "initial_condition_type": "patch",
                    "environment_map": "env_bet_hedging",
                    "num_replicates": 32,
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "patch_width": "patch_width_scan",
                    "initial_mutant_patch_size": "ic_control_scan",
                },
            },
        },
    },
    # --- REMOVED old/redundant relaxation/recovery experiments ---
    # --- NEW, OPTIMIZED CAMPAIGN FOR FIGURE 5 ---
    "timescale_analysis": {
        "campaign_id": "fig5_timescales",
        "run_mode": "timescale_dynamics",  # The dedicated, efficient run mode
        "hpc_params": {"time": "08:00:00", "mem": "2G", "sims_per_task": 40},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "num_replicates": 50,
                    # --- TARGETED PARAMETERS ---
                    "b_m": 0.5,  # Corresponds to s = -0.50 (Medium Selection)
                    "phi": 0.0,  # Corresponds to Unbiased Switching
                    # --- Relaxation Part ---
                    "relaxation_max_time": 2000.0,
                    # --- Tracking Part ---
                    "tracking_patch_width": 60,
                    "environment_map": "env_bet_hedging",
                    "warmup_cycles": 20,
                    "measure_cycles": 10,
                    # --- Shared High-Resolution Output ---
                    "timeseries_sample_interval": 5.0,
                },
                "grid_params": {
                    "k_total": "k_total_final_extended",
                },
            }
        },
    },
    # --- Other campaigns as before ---
    "asymmetric_patches": {
        "campaign_id": "fig5_asymmetric_patches",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "09:00:00", "mem": "2G", "sims_per_task": 150},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "num_replicates": 32,
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_asymmetric",
                    "k_total": "k_total_final_log",
                    "env_definition": "env_asymmetric_cycle_refined",
                },
            }
        },
    },
    "homogeneous_fitness_cost": {
        "campaign_id": "sup_homogeneous_cost",
        "run_mode": "homogeneous_converged",
        "hpc_params": {"time": "06:00:00", "mem": "2G", "sims_per_task": 50},
        "sim_sets": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "num_replicates": 40,
                    "initial_condition_type": "mixed",
                    "max_run_time": 8000.0,
                    "convergence_check_interval": 100.0,
                    "convergence_window": 5,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                },
            },
            "controls": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "num_replicates": 40,
                    "initial_condition_type": "mixed",
                    "max_run_time": 8000.0,
                    "convergence_check_interval": 100.0,
                    "convergence_window": 5,
                    "convergence_threshold": 0.01,
                    "phi": 0.0,
                },
                "grid_params": {"b_m": "bm_final_wide", "k_total": "k_zero"},
            },
        },
    },
}
