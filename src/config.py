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
# FILE: src/config.py (in the PARAM_GRID dictionary)

    "switching_lag_focused_scan": np.unique(
        np.round(
            np.concatenate([
                # 1. Sparsely sample the negligible region to establish the baseline
                np.logspace(-2, -1, 4),  # from 0.01 to 0.1

                # 2. DENSELY sample the critical drop-off region
                np.logspace(-1, 1.5, 15), # from 0.1 to ~31.6

                # 3. Sparsely sample the prohibitive plateau region to confirm it's flat
                np.logspace(1.5, 3, 5),   # from ~31.6 to 1000
            ]),
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
                    "width": 512,
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
    "lag_vs_environment_scan": {
        "campaign_id": "lag_vs_environment_scan",
        "run_mode": "bet_hedging_converged",
        "hpc_params": {"time": "04:00:00", "mem": "2G", "sims_per_task": 50},
        "sim_sets": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 8192, # Sufficient length for convergence
                    "initial_condition_type": "mixed",
                    "num_replicates": 8, # A good number for averaging
                    "phi": 0.0, # Symmetric switching
                    
                    # Tracker parameters for robust convergence
                    "max_cycles": 50,
                    "convergence_window_cycles": 5,
                    "convergence_threshold": 0.02,
                },
                "grid_params": {
                    # --- The three parameters you want to scan ---
                    "b_m": "bm_definitive_scan",
                    "k_total": "k_total_focused_log",
                    "switching_lag_duration": "switching_lag_focused_scan",
                    
                    # --- The bet-hedging environments to test ---
                    "env_definition": [
                        "symmetric_refuge_30w",
                        "symmetric_refuge_60w",
                        "symmetric_refuge_120w",
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
                    "width": 512,
                    "length": 4096,
                    "num_replicates": 30,
                    "warmup_time": 50.0,
                    "num_samples": 300,
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
    
    # Paste these two entries into EXPERIMENTS in src/config.py

    "aif_paper_rimbands_calibrated": {
        "campaign_id": "aif_rimbands_cal_v1",
        "run_mode": "aif_width_analysis",
        "hpc_params": {"time": "04:00:00", "mem": "4G", "sims_per_task": 80},
        "sim_sets": {
            "main": {
                "base_params": {
                    # Paper-like many-clone setting (treatment at ~1.8–2.0 mm radius)
                    "initial_condition_type": "aif_front_bands",
                    "initial_droplet_radius": 365,   # ≈ 1827 μm / 5 μm per cell
                    "num_sectors": 1,                # ignored by front_bands
                    "num_bands": 16,                 # modest red fraction on rim
                    "band_width": 5,                 # ≈ 25 μm (paper’s ~20–25 μm)
                    # physics (constant selection from paper s0 = 0.013)
                    "b_sus": 1.0, "b_res": 0.987, "b_comp": 1.0, "k_res_comp": 0.0,
                    # online logging (ΔR-driven)
                    "sector_metrics_dr": 1.0,
                    "sector_metrics_interval": 0,
                    "radius_check_interval": 24,
                    # sectorization hygiene
                    "front_denoise_window": 5, "min_island_len": 3,
                    "sid_iou_thresh": 0.15, "sid_center_delta": 0.10, "death_hysteresis": 3,
                    # run long enough to add a few hundred ΔR samples
                    "max_steps": 350_000
                },
                "grid_params": {
                    # explore a small band-width spread around the paper’s initial widths
                    "band_width": [4, 5, 6],   # 20, 25, 30 μm
                    # vary number of bands to modulate initial rim fraction
                    "num_bands": [12, 16, 20],
                    # optional mild selection variation around s0=0.013
                    "b_res": [0.985, 0.987, 0.990],
                    # replicates for robust survivor stats
                    "replicate": list(range(24))
                }
            }
        }
    },

    "aif_paper_singlesector_calibrated": {
        "campaign_id": "aif_singlesector_cal_v1",
        "run_mode": "aif_width_analysis",
        "hpc_params": {"time": "04:30:00", "mem": "4G", "sims_per_task": 80},
        "sim_sets": {
            # Diagnostic single-sector sweep spanning below/above wc ≈ 56 cells (280 μm)
            "single_sector": {
                "base_params": {
                    "initial_condition_type": "sector",
                    "initial_droplet_radius": 400,   # ≈ 2.0 mm; use 600 if you want a longer rim
                    "num_sectors": 1,
                    # physics: paper constant selection s0 = 0.013
                    "b_sus": 1.0, "b_res": 0.987, "b_comp": 1.0, "k_res_comp": 0.0,
                    # online logging
                    "sector_metrics_dr": 1.0,        # use 0.75 if you want denser sampling
                    "sector_metrics_interval": 0,
                    "radius_check_interval": 24,
                    # sectorization hygiene
                    "front_denoise_window": 5, "min_island_len": 3,
                    "sid_iou_thresh": 0.15, "sid_center_delta": 0.10, "death_hysteresis": 3,
                    # runtime
                    "max_steps": 380_000
                },
                "grid_params": {
                    # widths in cells (μm = cells * 5): 20, 60, 120, 240, 360 μm
                    "sector_width_initial": [4, 12, 24, 48, 72],
                    # allow a slight sweep around s0 for sensitivity (optional)
                    "b_res": [0.985, 0.987, 0.990],
                    "replicate": list(range(24))
                }
            }
        }
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
    
"debug_speed_tracker": {
    "campaign_id": "debug_speed_tracker",
    # This run_mode ensures the correct tracker would be selected by the worker
    "run_mode": "bet_hedging_converged",
    "sim_sets": {
        "main": {
            "base_params": {
                "width": 128,
                "length": 2048, # A bit longer to allow for a few cycles
                "initial_condition_type": "patch",
                "initial_mutant_patch_size": 64,
                "env_definition": "debug_viz_refuge", # Has a cycle length of 70
                "b_m": 0.8,
                "phi": 0.0,
                "k_total": 10, # A moderate switching rate
                # A large lag that will have a very clear impact on speed
                "switching_lag_duration":10,
            },
            # These are the parameters for the tracker itself
            "tracker_params": {
                "max_cycles": 10,
                "convergence_window_cycles": 3,
                "convergence_threshold": 0.01,
            }
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
                    "k_total": 10,
                    # The duration each cell will be "stuck" when it switches.
                    "switching_lag_duration": 20000,
                },
            }
        },
    },
}
