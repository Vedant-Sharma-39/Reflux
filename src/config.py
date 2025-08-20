# FILE: src/config.py (Cleaned Version: Preserves Existing Data, Adds New Experiments)

import numpy as np


# =============================================================================
# === DYNAMIC ENVIRONMENT GENERATION
# =============================================================================
def _generate_gamma_environments(means, fano_factors):
    """Generates a suite of Gamma-distributed environments programmatically."""
    env_dict = {}
    base_patches = [
        {"id": 0, "proportion": 0.5, "params": {"b_wt": 1.0}},
        {"id": 1, "proportion": 0.5, "params": {"b_wt": 0.0, "b_m": 1.0}},
    ]

    for mean in means:
        for fano in fano_factors:
            name = f"gamma_mean_{mean}_fano_{fano}"
            env_dict[name] = {
                "name": name,
                "scrambled": True,
                "patch_width_distribution": "gamma",
                "mean_patch_width": mean,
                "fano_factor": fano,
                "patches": base_patches,
                # --- THIS IS THE FIX ---
                "cycle_length": 16384,  # Provide an effective cycle length for the tracker
                # --- END FIX ---
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
    "evolutionary_phase_diagram": {
        "campaign_id": "evolutionary_phase_diagram",
        # --- SOLUTION STEP 1: Change the run_mode ---
        # This tracker is designed for measuring steady-state speed after a warmup period,
        # which is exactly what we need for a random environment.
        "run_mode": "HomogeneousDynamicsTracker",
        "hpc_params": {"time": "12:00:00", "mem": "2G", "sims_per_task": 15},
        "sim_sets": {
            "main": {
                # --- SOLUTION STEP 2: Replace base_params with the correct ones ---
                "base_params": {
                    "width": 256,
                    "length": 16384,
                    "initial_condition_type": "mixed",
                    "num_replicates": 32,
                    # Let the front travel for 4000 time units to stabilize.
                    # This replaces the confusing 'max_cycles'.
                    "warmup_time": 4000.0,
                    # After warmup, take 200 measurements of speed.
                    "num_samples": 200,
                    # Take one measurement every 50 time units.
                    "sample_interval": 50.0,
                },
                "grid_params": {
                    "k_total": "k_total_final_log",
                    "b_m": [0.5],
                    "phi": [0.0],
                    "env_definition": [
                        # You can keep the periodic one as a control if you like.
                        # The analysis script will correctly ignore it.
                        "symmetric_refuge_60w",
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
        # --- FIX 1: Use the new, purpose-built run mode ---
        "run_mode": "invasion_outcome",
        "hpc_params": {"time": "02:00:00", "sims_per_task": 100},
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 512,  # Length can be modest, we care about width
                    "num_replicates": 200,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 4,
                    # --- FIX 2: Add tracker-specific parameters ---
                    # If patch isn't 25% of the width by t=20000, end the run.
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
        "run_mode": "visualization",  # This enables the plotter
        "sim_sets": {
            "main": {
                "base_params": {
                    "width": 128,  # Smaller width for a focused view
                    "length": 1024,
                    "k_total": 0.0,  # No switching
                    "phi": 0.0,
                    "b_m": 0.80,  # Disadvantaged case is most dramatic
                    "initial_mutant_patch_size": 32,  # 25% mutants
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
                    "num_replicates": 250,  # Increased for better stats
                    "initial_condition_type": "grf_threshold",
                },
                "grid_params": {
                    "correlation_length": "correlation_length_scan",
                    # A finer grid of deleterious fitness values
                    "b_m": [0.7, 0.8, 0.9, 0.95],
                    # A wider range of initial mutant numbers (fractions)
                    "initial_mutant_patch_size": [32, 64, 128, 256],
                },
            }
        },
    },
}
