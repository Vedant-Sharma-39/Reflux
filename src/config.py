# FILE: src/config.py
# The definitive configuration for the final publication run.
# HPC parameters have been recalculated for efficiency and completeness.

import numpy as np

# ==============================================================================
# 1. FINAL PARAMETER GRIDS
# These grids provide comprehensive coverage for the final analysis.
# ==============================================================================
PARAM_GRID = {
    # --- Selection Grids (b_m) ---
    "bm_final_wide": np.round(
        np.linspace(0.1, 1.0, 10), 2
    ).tolist(),  # s from -0.9 to 0.0
    "bm_final_narrow": np.round(
        np.linspace(0.8, 1.0, 5), 2
    ).tolist(),  # s from -0.2 to 0.0
    # --- Switching Bias Grids (phi) ---
    "phi_final_full": np.round(
        np.linspace(-1.0, 1.0, 9), 2
    ).tolist(),  # Full symmetric range
    "phi_final_asymmetric": np.round(
        np.linspace(-1.0, 0.5, 7), 2
    ).tolist(),  # Focus on polluting/unbiased
    # --- Switching Rate Grids (k_total) ---
    "k_total_final_log": np.round(np.logspace(-2, 1, 10), 4).tolist(),  # 0.01 to 10
    "k_zero": [0],  # Special case for no switching
    # --- Geometric Grids ---
    "width_scan": [64, 128, 256, 512],
    "patch_width_scan": [30, 60, 120],
    "ic_control_scan": [0],  # Pure WT (generalist) control
    # --- Environment Definitions ---
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
    "bm_visualization": [0.8, 0.95],  # s = -0.2 and s = -0.05
    "k_total_visualization": [0.02, 0.5],  # A low and a high switching rate
    "phi_visualization": [0.0],  # Unbiased switching is better for demo
    "patch_width_visualization": [60],  # A representative patch width
}

# ==============================================================================
# 2. FINAL EXPERIMENT DEFINITIONS
# ==============================================================================
EXPERIMENTS = {
    # --- Figure 1 Campaigns ---
    "boundary_analysis": {
        "campaign_id": "fig1_boundary_analysis",
        "run_mode": "calibration",
        # Total simulations: 10 (b_m) * 200 (reps) = 2,000
        "hpc_params": {"time": "05:30:00", "mem": "1G", "sims_per_task": 50},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 512,
                    "length": 1024,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 200,
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
        # Total simulations: 10 (b_m) * 4 (width) * 100 (reps) = 4,000
        "hpc_params": {"time": "04:30:00", "mem": "2G", "sims_per_task": 40},
        "sim_sets": {
            "main": {
                "base_params": {
                    "length": 4096,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 100,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 0,
                    "max_steps": 5000000,
                },
                "grid_params": {"b_m": "bm_final_wide", "width": "width_scan"},
            }
        },
    },
    # --- Figure 2 Campaign ---
    "phase_diagram": {
        "campaign_id": "fig2_phase_diagram",
        "run_mode": "phase_diagram",
        # Total simulations: 10 (b_m) * 9 (phi) * 10 (k) * 50 (reps) = 45,000
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
    # --- Figure 3 Campaign ---
    "bet_hedging_final": {
        "campaign_id": "fig3_bet_hedging_final",
        "run_mode": "bet_hedging_converged",
        # Total simulations: (5*9*10*3*32) + (5*3*1*32) = 43,200 + 480 = 43,680
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
    # --- Figure 4 Campaign ---
    "relaxation_dynamics": {
        "campaign_id": "fig4_relaxation",
        "run_mode": "relaxation",
        # Total simulations: 10 (b_m) * 10 (k) * 50 (reps) = 5,000
        "hpc_params": {"time": "04:30:00", "mem": "2G", "sims_per_task": 100},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "phi": 0.0,
                    "num_replicates": 50,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": "width",
                    "total_run_time": 4000.0,
                    "sample_interval": 15.0,
                },
                "grid_params": {"b_m": "bm_final_wide", "k_total": "k_total_final_log"},
            }
        },
    },
    # --- Figure 5 Campaign ---
    "asymmetric_patches": {
        "campaign_id": "fig5_asymmetric_patches",
        "run_mode": "bet_hedging_converged",
        # Total simulations: 5 (b_m) * 7 (phi) * 10 (k) * 4 (env) * 32 (reps) = 44,800
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
    "visualization_test": {
        "campaign_id": "viz_final_paper",
        "run_mode": "visualization",
        "hpc_params": {"time": "00:30:00", "mem": "4G", "sims_per_task": 1},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 128,
                    "length": 1024,
                    "num_replicates": 1,
                    "max_snapshots": 5,
                    "snapshot_q_offset": 2.0,
                    "environment_map": "env_bet_hedging",
                    "initial_condition_type": "mixed",
                },
                "grid_params": {
                    "b_m": "bm_visualization",
                    "k_total": "k_total_visualization",
                    "phi": "phi_visualization",
                    "patch_width": "patch_width_visualization",
                },
            },
        },
    },
    "recovery_timescale": {
        "campaign_id": "fig4_recovery_timescale",
        "run_mode": "recovery_dynamics",
        "hpc_params": {"time": "05:00:00", "mem": "2G", "sims_per_task": 80},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": "width",
                    "num_replicates": 60,
                    "total_run_time": 5000.0,
                    "timeseries_interval": 10.0,
                    "warmup_time_ss": 4000.0,
                    "num_samples_ss": 100,
                    "sample_interval_ss": 10.0,
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                },
            }
        },
    },
    # --- UPDATED: HOMOGENEOUS FITNESS COST with CONVERGENCE ---
    "homogeneous_fitness_cost": {
        "campaign_id": "sup_homogeneous_cost",
        "run_mode": "homogeneous_converged",  # <-- Use the new purpose-built run mode
        "hpc_params": {"time": "06:00:00", "mem": "2G", "sims_per_task": 50},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "num_replicates": 40,
                    "initial_condition_type": "mixed",
                    # --- NEW Convergence-based parameters ---
                    "max_run_time": 8000.0,
                    "convergence_check_interval": 100.0,  # Time units
                    "convergence_window": 5,  # Number of intervals to check
                    "convergence_threshold": 0.01,  # 1% relative stdev
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_full",
                    "k_total": "k_zero",
                },
            }
        },
    },
}
