# FILE: src/config.py
# The definitive configuration file for all experiments.
# [v3 - Standardized Slurm time format to HH:MM:SS for universal compatibility]

# ==============================================================================
# 1. PARAMETER GRIDS (Reusable parameter lists)
# ==============================================================================
PARAM_GRID = {
    "bm_deleterious_wide": [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],
    "bm_deleterious_narrow": [0.75, 0.85, 0.95],
    "width_scan": [64, 128, 256, 512],
    "phi_scan": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "phi_scan_bet_hedging": [-1.0, -0.9, -0.7, -0.5, -0.2, 0.0, 0.5],
    "k_total_scan_coarse": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    "k_total_scan_fine": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    "k_total_scan_bet_hedging": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    "patch_width_scan": [30, 60, 120],
    "ic_control_scan": [0],  # Pure WT (generalist) control

    # --- Refined Parameter Grids for Asymmetric Patch Experiment ---
    "phi_scan_asymmetric": [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.2, 0.0],
    "env_asymmetric_cycle_refined": [
        {
            "name": "30_90",
            "cycle_length": 120,
            "patches": [
                {"id": 0, "width": 30, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 90, "params": {"b_wt": 0.0, "b_m": 1.0}}
            ]
        },
        {
            "name": "60_60",
            "cycle_length": 120,
            "patches": [
                {"id": 0, "width": 60, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 60, "params": {"b_wt": 0.0, "b_m": 1.0}}
            ]
        },
        {
            "name": "90_30",
            "cycle_length": 120,
            "patches": [
                {"id": 0, "width": 90, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 30, "params": {"b_wt": 0.0, "b_m": 1.0}}
            ]
        },
        {
            "name": "scrambled_60_60",
            "scrambled": True,
            "cycle_length": 120,
            "avg_patch_width": 60,
            "patches": [
                {"id": 0, "proportion": 0.5, "params": {"b_wt": 1.0}},
                {"id": 1, "proportion": 0.5, "params": {"b_wt": 0.0, "b_m": 1.0}}
            ]
        }
    ],

    "env_bet_hedging": {
        0: {"b_wt": 1.0},
        1: {"b_wt": 0.0, "b_m": 1.0},
    },
}

# ==============================================================================
# 2. EXPERIMENT DEFINITIONS
# ==============================================================================
EXPERIMENTS = {
    "boundary_analysis": {
        "campaign_id": "fig1_boundary_analysis",
        "run_mode": "calibration",
        "hpc_params": {
            "time": "02:00:00",
            "mem": "1G",
            "sims_per_task": 50,
        },  # FIX: 0-2 -> 02
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
                "grid_params": {"b_m": "bm_deleterious_wide"},
            }
        },
    },
    "kpz_scaling": {
        "campaign_id": "fig1_kpz_scaling",
        "run_mode": "diffusion",
        "hpc_params": {
            "time": "02:00:00",
            "mem": "1G",
            "sims_per_task": 40,
        },  # FIX: 0-12 -> 12
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
                "grid_params": {"b_m": "bm_deleterious_wide", "width": "width_scan"},
            }
        },
    },
    "phase_diagram": {
        "campaign_id": "fig2_phase_diagram",
        "run_mode": "phase_diagram",
        "hpc_params": {
            "time": "01:00:00",
            "mem": "1G",
            "sims_per_task": 50,
        },  # FIX: 0-04 -> 04
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
                    "b_m": "bm_deleterious_wide",
                    "phi": "phi_scan",
                    "k_total": "k_total_scan_coarse",
                },
            }
        },
    },
    "relaxation_dynamics": {
        "campaign_id": "fig4_relaxation",
        "run_mode": "relaxation",
        "hpc_params": {
            "time": "01:00:00",
            "mem": "2G",
            "sims_per_task": 40,
        },  # FIX: 0-06 -> 06
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
                "grid_params": {
                    "b_m": "bm_deleterious_wide",
                    "k_total": "k_total_scan_coarse",
                },
            }
        },
    },
    "bet_hedging_final": {
        "campaign_id": "fig3_bet_hedging_final",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {
            "time": "01:00:00",
            "mem": "2G",
            "sims_per_task": 20,
        },
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
                    "b_m": "bm_deleterious_narrow",
                    "phi": "phi_scan_bet_hedging",
                    "k_total": "k_total_scan_bet_hedging",
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
                    "b_m": "bm_deleterious_narrow",
                    "patch_width": "patch_width_scan",
                    "initial_mutant_patch_size": "ic_control_scan",
                },
            },
        },
    },
    "asymmetric_patches": {
        "campaign_id": "fig5_asymmetric_patches",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {
            "time": "02:00:00",
            "mem": "2G",
            "sims_per_task": 20,
        },
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
                    "b_m": "bm_deleterious_narrow",
                    "phi": "phi_scan_asymmetric",
                    "k_total": "k_total_scan_bet_hedging",
                    "env_definition": "env_asymmetric_cycle_refined",
                },
            }
        },
    },
}
