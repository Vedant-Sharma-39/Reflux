# FILE: src/config.py (Corrected)

import numpy as np


# =============================================================================
# === DYNAMIC ENVIRONMENT GENERATION
# =============================================================================
def _generate_gamma_environments(means, fano_factors):
    """Generates a suite of Gamma-distributed environments programmatically."""
    env_dict = {}
    # Base patches represent the two environmental states (e.g., S and G in the paper)
    # Here, WT is favored in patch 0, Mutant in patch 1.
    base_patches = [
        {"id": 0, "proportion": 0.5, "params": {"b_wt": 1.0, "b_m": 0.5}},
        {"id": 1, "proportion": 0.5, "params": {"b_wt": 0.0, "b_m": 1.0}},
    ]

    for mean in means:
        for fano in fano_factors:
            name = f"gamma_mean_{mean}_fano_{fano}"
            env_dict[name] = {
                "name": name,
                "scrambled": True,  # This flag triggers the new logic in the model
                "patch_width_distribution": "gamma",
                "mean_patch_width": mean,
                "fano_factor": fano,
                "patches": base_patches,
                # Provide an effective cycle length for compatibility with older trackers.
                "cycle_length": 16384,
            }
    return env_dict


# =============================================================================
# === PARAMETER GRIDS (Pruned to only include necessary items)
# =============================================================================
PARAM_GRID = {
    # --- Core Parameter Scans ---
    "bm_final_wide": np.round(np.linspace(0.1, 1.0, 10), 2).tolist(),
    "phi_final_full": np.round(np.linspace(-1.0, 1.0, 9), 2).tolist(),
    "k_total_final_log": np.round(np.logspace(-2, 2, 20), 5).tolist(),
    "width_scan": [64, 128, 256, 512],
    "correlation_length_scan": np.round(
        np.logspace(0, 2.5, 10), 2
    ).tolist(),  # from 1 to ~316
    # --- Focused Grids for Definitive Experiment ---
    "k_total_focused_log": np.round(np.logspace(-2, 2, 25), 5).tolist(),
    "bm_focused_scan": [1.0, 0.9, 0.5, 0.25],
    "phi_focused_scan": [0.0, 0.5, -0.5],
    "bm_definitive_scan": [1.0, 0.9, 0.7, 0.5, 0.2],
    "switching_lag_duration_scan": np.round(np.logspace(-2, 2, 10), 5).tolist(),
    "switching_lag_dense_scan": np.unique(
        np.round(
            np.concatenate(
                [
                    np.logspace(-2, -0.5, 5),  # Sparse sampling of very low lags
                    np.logspace(-0.5, 1.3, 15),  # DENSE sampling of the critical region
                    np.logspace(1.3, 2, 5),  # Sparse sampling of high lags
                ]
            ),
            5,
        )
    ).tolist(),
    # --- Environment Definitions ---
    "env_definitions": {
        # --- For Preserved `bet_hedging_final` Experiment ---
        "symmetric_refuge_30w": {
            "name": "symmetric_refuge_30w",
            "patches": [
                {"id": 0, "width": 30, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 30, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
        "symmetric_refuge_60w": {
            "name": "symmetric_refuge_60w",
            "patches": [
                {"id": 0, "width": 60, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 60, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
        "symmetric_refuge_120w": {
            "name": "symmetric_refuge_120w",
            "patches": [
                {"id": 0, "width": 120, "params": {"b_wt": 1.0}},
                {"id": 1, "width": 120, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
        # --- For New `evolutionary_phase_diagram` and `invasion_probability` Experiments ---
        **_generate_gamma_environments(
            means=[30, 60, 120, 240], fano_factors=[1, 10, 30, 60]
        ),
        # --- For Debugging ---
        "debug_viz_refuge": {
            "name": "debug_viz_refuge",
            "patches": [
                {"id": 0, "width": 35, "params": {"b_wt": 1.0, "b_m": 0.5}},
                {"id": 1, "width": 35, "params": {"b_wt": 0.0, "b_m": 1.0}},
            ],
        },
    },
}

# =============================================================================
# === EXPERIMENT DEFINITIONS
# =============================================================================
EXPERIMENTS = {
    # =========================================================================
    # === PRESERVED EXPERIMENTS (Matching Existing Data)
    # =========================================================================
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
                    "num_replicates": 5,
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
        "hpc_params": {"time": "10:00:00", "mem": "2G", "sims_per_task": 20},
        "sim_sets": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "num_replicates": 16,
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                    "b_m": "bm_final_wide",
                    "env_definition": [
                        "symmetric_refuge_30w",
                        "symmetric_refuge_60w",
                        "symmetric_refuge_120w",
                    ],
                },
            }
        },
    },
    # =========================================================================
    # === NEW EXPERIMENTS (For Future Data Generation)
    # =========================================================================
    "homogeneous_timeseries": {
        "campaign_id": "homogeneous_timeseries_analysis_v3",
        "run_mode": "relaxation_converged",
        "hpc_params": {"time": "02:00:00", "sims_per_task": 10},
        "sim_sets": {
            # --- SET 1: Tailored for SLOW switching (low k) ---
            "slow_k_set": {
                "base_params": {
                    "width": 256,
                    "length": 4096,
                    "num_replicates": 10,
                    "initial_condition_type": "patch",
                    # Long run, sample less frequently
                    "sample_interval": 10.0,
                    "convergence_window": 50,  # Needs a long window to confirm stability
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": [0.2, 0.5, 0.8],
                    "phi": [-1.0, 0.0, 0.5],
                    "k_total": [0.01, 0.1],  # Low k values
                    "initial_mutant_patch_size": [26, 128, 230],
                },
            },
            # --- SET 2: Tailored for MEDIUM switching (crossover k) ---
            "medium_k_set": {
                "base_params": {
                    "width": 256,
                    "length": 4096,
                    "num_replicates": 10,
                    "initial_condition_type": "patch",
                    # Medium run, medium sampling
                    "sample_interval": 1.0,
                    "convergence_window": 50,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": [0.2, 0.5, 0.8],
                    "phi": [-1.0, 0.0, 0.5],
                    "k_total": [1.0],  # Crossover k value
                    "initial_mutant_patch_size": [26, 128, 230],
                },
            },
            # --- SET 3: Tailored for FAST switching (high k) ---
            "fast_k_set": {
                "base_params": {
                    "width": 256,
                    "length": 4096,
                    "num_replicates": 10,
                    "initial_condition_type": "patch",
                    # Short run, sample very frequently
                    "sample_interval": 0.1,
                    "convergence_window": 50,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": [0.2, 0.5, 0.8],
                    "phi": [-1.0, 0.0, 0.5],
                    "k_total": [10.0],  # High k value
                    "initial_mutant_patch_size": [26, 128, 230],
                },
            },
        },
    },
    # --- NEW EXPERIMENT 2: Environmental Interaction for Transient Lag ---
    "lag_vs_selection_definitive": {
        "campaign_id": "fig_final_lag_vs_selection_definitive",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "3:00:00", "mem": "2G", "sims_per_task": 50},
        "sim_sets": {
            "reversible_scan": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "num_replicates": 32,
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.05,
                },
                "grid_params": {
                    "k_total": "k_total_focused_log",
                    # --- FIX 1: Use the more comprehensive b_m scan ---
                    "b_m": "bm_focused_scan",
                    "phi": "phi_focused_scan",
                    "switching_lag_duration": "switching_lag_dense_scan",
                    "env_definition": [
                        "symmetric_refuge_30w",
                        "symmetric_refuge_60w",
                        "symmetric_refuge_120w",
                    ],
                },
            },
            "irreversible_baseline": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "num_replicates": 32,
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.05,
                    "phi": -1.0,
                    "k_total": 0.0,
                    "switching_lag_duration": 0.0,
                },
                "grid_params": {
                    # --- This is already correct and matches FIX 1 ---
                    "b_m": "bm_definitive_scan",
                    "env_definition": [
                        "symmetric_refuge_30w",
                        "symmetric_refuge_60w",
                        "symmetric_refuge_120w",
                    ],
                },
            },
        },
    },
    "inherent_cost_fitness": {
        "campaign_id": "fig_inherent_cost_fitness",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "10:00:00", "mem": "2G", "sims_per_task": 20},
        "sim_sets": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "num_replicates": 16,
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.01,
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                    "env_definition": [
                        {
                            "name": "inherent_cost_30w",
                            "patches": [
                                {"id": 0, "width": 30, "params": {"b_wt": 1.0}},
                                {"id": 1, "width": 30, "params": {"b_wt": 0.0}},
                            ],
                        },
                        {
                            "name": "inherent_cost_60w",
                            "patches": [
                                {"id": 0, "width": 60, "params": {"b_wt": 1.0}},
                                {"id": 1, "width": 60, "params": {"b_wt": 0.0}},
                            ],
                        },
                        {
                            "name": "inherent_cost_120w",
                            "patches": [
                                {"id": 0, "width": 120, "params": {"b_wt": 1.0}},
                                {"id": 1, "width": 120, "params": {"b_wt": 0.0}},
                            ],
                        },
                    ],
                },
            }
        },
    },
    "evolutionary_phase_diagram": {
        "campaign_id": "evolutionary_phase_diagram",
        "run_mode": "HomogeneousDynamicsTracker",
        "hpc_params": {"time": "2:00:00", "mem": "2G", "sims_per_task": 15},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "num_replicates": 32,
                    "warmup_time": 4000.0,
                    "num_samples": 200,
                    "sample_interval": 50.0,
                },
                "grid_params": {
                    "k_total": "k_total_final_log",
                    "b_m": [1.0],
                    "phi": [0.0],
                    "env_definition": [
                        *_generate_gamma_environments(
                            means=[30, 60, 120, 240], fano_factors=[1, 10, 30, 60]
                        ).keys(),
                    ],
                },
            }
        },
    },
    "homogeneous_fitness_cost": {
        "campaign_id": "homogeneous_fitness_cost",
        "run_mode": "HomogeneousDynamicsTracker",
        "hpc_params": {"time": "04:00:00"},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 4096,
                    "num_replicates": 20,
                    "warmup_time": 500.0,
                    "num_samples": 200,
                    "sample_interval": 10.0,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 0,
                },
                "grid_params": {
                    "b_m": "bm_final_wide",
                    "phi": "phi_final_full",
                    "k_total": "k_total_final_log",
                },
            }
        },
    },
    "invasion_probability_analysis": {
        "campaign_id": "invasion_probability",
        "run_mode": "invasion_outcome",
        "hpc_params": {"time": "02:00:00", "sims_per_task": 100},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 512,
                    "num_replicates": 200,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 4,
                    "progress_threshold_frac": 0.25,
                    "progress_check_time": 20000.0,
                },
                "grid_params": {
                    "phi": [-1.0, 0.0, 0.5],
                    "k_total": [0.01, 0.1, 1.0, 10.0],
                    "env_definition": [
                        "symmetric_refuge_60w",
                        "gamma_mean_120_fano_30",
                    ],
                    "b_m": [0.5, 0.8],
                },
            }
        },
    },
    
    "aif_definitive_spatial_scan": {
        "campaign_id": "aif_definitive_spatial_scan_v2", # Versioning is good practice
        "run_mode": "aif_width_analysis",
        "hpc_params": {"time": "04:30:00", "mem": "4G", "sims_per_task": 25}, # Slightly more time
        "sim_sets": {
            "main": {
                "base_params": {
                    # --- Refined Robust Parameters ---
                    "initial_condition_type": "sector",
                    "initial_droplet_radius": 60,  # INCREASED for robustness
                    "num_sectors": 1,
                    "num_replicates": 250,
                    "max_steps": 350000,
                    # --- AIF Physics ---
                    "b_sus": 1.0, "b_comp": 1.0, "k_res_comp": 0.0,
                },
                "grid_params": {
                    # --- REFINED EXPERT b_res SCAN ---
                    # This non-linear scan focuses resolution on the critical region near 1.0
                    "b_res": np.unique(np.round(np.concatenate([
                        np.arange(0.7, 0.9, 0.1),    # Coarse scan for strong disadvantage
                        np.arange(0.92, 0.98, 0.05),  # Medium scan for weak disadvantage
                        np.arange(0.99, 1.01, 0.05),  # HIGH DENSITY scan around neutral point
                    ]), 4)).tolist(),
                    
                    # Scan initial width to make the dataset future-proof
                    "sector_width_initial": [20, 40, 60], # Adjusted to new radius
                },
            }
        },
    },

    # =========================================================================
    # === DEBUGGING
    # =========================================================================
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
                    "env_definition": "debug_viz_refuge",
                    "k_total": 0.1,
                    "phi": 0.0,
                    "b_m": 0.5,
                },
            }
        },
    },
    "debug_fragmentation_viz": {
        "campaign_id": "debug_fragmentation_viz",
        "run_mode": "visualization",
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 128,
                    "length": 1024,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "b_m": 0.80,
                    "initial_mutant_patch_size": 32,
                },
            }
        },
    },
    "deleterious_invasion_dynamics": {
        "campaign_id": "deleterious_invasion_dynamics",
        "run_mode": "fixation_analysis",
        "hpc_params": {"time": "02:30:00", "sims_per_task": 100},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 512,
                    "length": 2048,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 250,
                    "initial_condition_type": "grf_threshold",
                },
                "grid_params": {
                    "correlation_length": "correlation_length_scan",
                    "b_m": [0.7, 0.8, 0.9, 0.95],
                    "initial_mutant_patch_size": [32, 64, 128, 256],
                },
            }
        },
    },
    "debug_microlag_viz": {
        "campaign_id": "debug_microlag_viz",
        "run_mode": "visualization",  # This enables the plotter
        "sim_sets": {
            "main": {
                "base_params": {
                    # Use the same base setup as the other viz debug
                    "width": 128,
                    "length": 1024,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 64,
                    "env_definition": "debug_viz_refuge",
                    "b_m": 0.5,
                    "phi": 0.0,
                    # --- CORE PARAMETERS FOR THIS TEST ---
                    # Set a non-zero switching rate that will be "unlocked" later.
                    # A slightly higher k makes the change more dramatic and visible.
                    "k_total": 0.5,
                    # Set a lag time that is long enough to see the initial "no-switching"
                    # phase clearly, but not so long the simulation ends before it's over.
                    "switching_arrest_time": 100.0,
                },
            }
        },
    },
    "debug_transient_state_viz": {
        "campaign_id": "debug_transient_state_viz",
        "run_mode": "visualization",
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 128,
                    "length": 1024,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 64,
                    "env_definition": "debug_viz_refuge",
                    "b_m": 0.5,
                    "phi": 0.0,
                    # A high switching rate makes the transient states appear frequently.
                    "k_total": 0.1,
                    # The duration each cell will be "stuck" when it switches.
                    "switching_lag_duration": 0.5,
                },
            }
        },
    },
}
