# FILE: src/config.py (Comprehensive Version)
# This configuration is a superset of the old and new configs. It is designed
# to be able to run all previous experiments using the new, flexible parameterization
# scheme, and maps them to the new directory/campaign structure.

import numpy as np

PARAM_GRID = {
    # --- Core Parameter Scans ---
    "bm_final_wide": np.round(np.linspace(0.1, 1.0, 10), 2).tolist(),
    "phi_final_full": np.round(np.linspace(-1.0, 1.0, 9), 2).tolist(),
    "phi_final_asymmetric": np.round(np.linspace(-1.0, 0.5, 7), 2).tolist(),
    "k_total_final_log": np.round(np.logspace(-2, 2, 20), 5).tolist(),
    "k_total_fig5_scan": np.round(np.logspace(-2.5, 1.5, 13), 4).tolist(),
    "k_zero": [0],
    "width_scan": [64, 128, 256, 512],
    # A single value for b_m=1.0, used to replicate old experiments
    "b_m_one": [1.0],
    # --- EXPANDED, FLEXIBLE ENVIRONMENT DEFINITIONS ---
    # These now cover all geometries from the old config. They intentionally omit `b_m`
    # so it can be scanned as an independent grid parameter.
    "env_definitions": {
        # --- Templates for Symmetric Environments ---
        "symmetric_strong_scan_bm_30w": {
            "name": "symmetric_strong_scan_bm_30w",
            "patches": [
                {"id": 0, "width": 30, "params": {"b_wt": 1.0, "b_m": 0.0}},
                {"id": 1, "width": 30, "params": {"b_wt": 0.0}},  # b_m injected here
            ],
        },
        "symmetric_strong_scan_bm_60w": {
            "name": "symmetric_strong_scan_bm_60w",
            "patches": [
                {"id": 0, "width": 60, "params": {"b_wt": 1.0, "b_m": 0.0}},
                {"id": 1, "width": 60, "params": {"b_wt": 0.0}},  # b_m injected here
            ],
        },
        "symmetric_strong_scan_bm_120w": {
            "name": "symmetric_strong_scan_bm_120w",
            "patches": [
                {"id": 0, "width": 120, "params": {"b_wt": 1.0, "b_m": 0.0}},
                {"id": 1, "width": 120, "params": {"b_wt": 0.0}},  # b_m injected here
            ],
        },
        # --- Templates for Asymmetric Environments ---
        "asymmetric_90_30_scan_bm": {
            "name": "asymmetric_90_30_scan_bm",
            "patches": [
                {"id": 0, "width": 90, "params": {"b_wt": 1.0, "b_m": 0.0}},
                {"id": 1, "width": 30, "params": {"b_wt": 0.0}},  # b_m injected here
            ],
        },
        "asymmetric_30_90_scan_bm": {
            "name": "asymmetric_30_90_scan_bm",
            "patches": [
                {"id": 0, "width": 30, "params": {"b_wt": 1.0, "b_m": 0.0}},
                {"id": 1, "width": 90, "params": {"b_wt": 0.0}},  # b_m injected here
            ],
        },
        # --- Template for Scrambled Environment ---
        "scrambled_60_60_scan_bm": {
            "name": "scrambled_60_60_scan_bm",
            "scrambled": True,
            "cycle_length": 120,
            "avg_patch_width": 60,
            "patches": [
                {"id": 0, "proportion": 0.5, "params": {"b_wt": 1.0, "b_m": 0.0}},
                {
                    "id": 1,
                    "proportion": 0.5,
                    "params": {"b_wt": 0.0},
                },  # b_m injected here
            ],
        },
        # --- Legacy environment map for new fig5 tracking analysis ---
        "env_bet_hedging_legacy": {"0": {"b_wt": 1.0, "b_m": 0.0}, "1": {"b_wt": 0.0}},
        # --- Unchanged debug definition ---
        "debug_viz_strong": {
            "name": "debug_viz_35_35",
            "patches": [
                {"id": 0, "width": 35, "params": {"b_wt": 1.0, "b_m": 0.0}},
                {"id": 1, "width": 35, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
    },
}

EXPERIMENTS = {
    # --- FIG 1 & 2: HOMOGENEOUS RUNS (Unchanged) ---
    "boundary_analysis": {
        "campaign_id": "fig1_boundary_analysis",
        "run_mode": "calibration",
        "hpc_params": {"time": "02:00:00"},
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
                },
                "grid_params": {"b_m": "bm_final_wide"},
            }
        },
    },
    "kpz_scaling": {
        "campaign_id": "fig1_kpz_scaling",
        "run_mode": "diffusion",
        "hpc_params": {"time": "02:00:00"},
        "sim_sets": {
            "main": {
                "base_params": {
                    "length": 4096,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 250,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 0,
                },
                "grid_params": {"b_m": "bm_final_wide", "width": "width_scan"},
            }
        },
    },
    "phase_diagram": {
        "campaign_id": "fig2_phase_diagram",
        "run_mode": "phase_diagram",
        "hpc_params": {"time": "02:00:00"},
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
    # --- FIG 3: SYMMETRIC BET HEDGING (Refactored) ---
    "bet_hedging_final": {
        "campaign_id": "fig3_bet_hedging_final",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "10:00:00", "mem": "2G", "sims_per_task": 100},
        "sim_sets": {
            "main_scan": {
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
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                    "b_m": "bm_final_wide",  # This is the primary scan
                    "env_definition": [
                        "symmetric_strong_scan_bm_60w"
                    ],  # Fixed geometry
                },
            }
        },
    },
    "bet_hedging_controls": {
        "campaign_id": "fig3_controls",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "10:00:00", "mem": "2G", "sims_per_task": 100},
        "sim_sets": {
            "legacy_geometries": {
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
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                    "b_m": "b_m_one",  # Fix b_m=1.0 to match old runs
                    "env_definition": [
                        "symmetric_strong_scan_bm_30w",
                        "symmetric_strong_scan_bm_120w",
                    ],
                },
            }
        },
    },
    # --- FIG 4: ASYMMETRIC ADAPTATION (Refactored) ---
    "asymmetric_adaptation": {
        "campaign_id": "fig4_asymmetric_adaptation",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "10:00:00", "mem": "2G", "sims_per_task": 100},
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
                    "phi": "phi_final_asymmetric",
                    "k_total": "k_total_final_log",
                    "b_m": "bm_final_wide",  # This is the primary scan
                    "env_definition": ["asymmetric_90_30_scan_bm"],  # Fixed geometry
                },
            }
        },
    },
    "asymmetric_controls": {
        "campaign_id": "fig4_relaxation",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "10:00:00", "mem": "2G", "sims_per_task": 100},
        "sim_sets": {
            "legacy_geometries": {
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
                    "phi": "phi_final_asymmetric",
                    "k_total": "k_total_final_log",
                    "b_m": "b_m_one",  # Fix b_m=1.0 to match old runs
                    "env_definition": [
                        "asymmetric_30_90_scan_bm",
                        "scrambled_60_60_scan_bm",
                    ],
                },
            }
        },
    },
    # --- FIG 5: DYNAMICS & TIMESCALES (Refactored) ---
    "relaxation_analysis": {
        "campaign_id": "fig5_relaxation",
        "run_mode": "relaxation_converged",
        "hpc_params": {"time": "02:00:00", "mem": "1G", "sims_per_task": 25},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 2048,
                    "num_replicates": 40,
                    "b_m": 0.5,
                    "phi": 0.0,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 256,
                    "sample_interval": 5.0,
                    "convergence_window": 10,
                    "convergence_threshold": 0.005,
                },
                "grid_params": {"k_total": "k_total_fig5_scan"},
            }
        },
    },
    "tracking_analysis": {
        "campaign_id": "fig5_tracking",
        "run_mode": "cyclic_timeseries",
        "hpc_params": {"time": "04:00:00", "mem": "2G", "sims_per_task": 50},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "num_replicates": 20,
                    "phi": 0.0,
                    "initial_condition_type": "mixed",
                    "environment_map": "env_bet_hedging_legacy",
                    "patch_width": 60,
                    "b_m": 0.5,
                    "warmup_cycles": 10,
                    "measure_cycles": 5,
                    "sample_interval": 5.0,
                },
                "grid_params": {"k_total": "k_total_fig5_scan"},
            }
        },
    },
    "timescales_legacy_tracking": {
        "campaign_id": "fig5_timescales",
        "run_mode": "cyclic_timeseries",
        "hpc_params": {"time": "04:00:00", "mem": "2G", "sims_per_task": 50},
        "sim_sets": {
            "legacy_run": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "num_replicates": 20,
                    "phi": 0.0,
                    "initial_condition_type": "mixed",
                    "warmup_cycles": 10,
                    "measure_cycles": 5,
                    "sample_interval": 5.0,
                },
                "grid_params": {
                    "k_total": "k_total_fig5_scan",
                    "b_m": "b_m_one",  # Fix b_m=1.0 to match old run
                    "env_definition": ["symmetric_strong_scan_bm_60w"],
                },
            }
        },
    },
    # --- DEBUGGING (Unchanged) ---
    "debug_bet_hedging": {
        "campaign_id": "debug_bet_hedging_viz",
        "run_mode": "visualization",
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 128,
                    "length": 1024,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 64,
                    "env_definition": "debug_viz_strong",
                    "k_total": 0.1,
                    "phi": 0.0,
                    "b_m": 1.0,
                },
            }
        },
    },
        "supplementary_homogeneous_cost": {
        "campaign_id": "figS1_homogeneous_cost",
        "run_mode": "homogeneous_dynamics", # A run mode similar to phase_diagram
        "hpc_params": {"time": "02:00:00"},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 4096,
                    "num_replicates": 50,
                    "phi": 0.0, # Unbiased switching
                    "warmup_time": 500.0,
                    "num_samples": 200,
                    "sample_interval": 10.0,
                    "initial_condition_type": "mixed",
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "k_total": "k_total_final_log",
                },
            }
        },
    },
}


