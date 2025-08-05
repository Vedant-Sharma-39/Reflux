# FILE: src/config.py
# [v_FINAL_ROBUST - PUBLICATION TRACK]
# This version is now fully and consistently refactored. ALL numpy calls are
# pre-calculated, making the EXPERIMENTS dictionary a pure, safe data structure.

import numpy as np

# ==============================================================================
# 1. PRE-CALCULATED PARAMETER ARRAYS (FOR ALL EXPERIMENTS)
# ==============================================================================

# For boundary_dynamics_vs_selection
s_fig1a_scan = np.linspace(-0.8, 0.8, 33)
bm_fig1a_scan = s_fig1a_scan + 1.0

# For front_morphology_vs_selection
s_fig1b_scan = np.linspace(-0.8, 0.8, 17)
bm_fig1b_scan = s_fig1b_scan + 1.0

# For hierarchical_criticality_global
s_global_scan = np.linspace(-0.8, 0.4, 13)
bm_global_scan = s_global_scan + 1.0
phi_global_scan = np.linspace(-1.0, 1.0, 11)
k_global_scan = np.logspace(-2.5, 2.5, 50)

# For hierarchical_criticality_focused
s_dip_region_scan = np.linspace(-0.5, -0.1, 21)
bm_dip_region_scan = s_dip_region_scan + 1.0
k_dip_region_scan = np.logspace(np.log10(0.05), np.log10(0.8), 40)
phi_focused_scan = np.linspace(-1.0, 1.0, 11)

# For exp_relaxation_dynamics
s_deleterious_scan = np.linspace(-0.8, 0.0, 9)
bm_deleterious_scan = s_deleterious_scan + 1.0
k_relaxation_scan = np.logspace(-2, 2, 20)

# For spatial_bet_hedging_final
phi_bet_hedging_scan = np.linspace(-1.0, 1.0, 11)
k_bet_hedging_scan = np.unique(np.logspace(-3, 1.5, 40))

# For asymmetric_environment_v1
phi_asymmetric_scan = np.linspace(-1.0, 0.0, 6)
k_asymmetric_scan = np.logspace(-2.5, 1.0, 30)


# ==============================================================================
# MAIN EXPERIMENT DEFINITIONS
# ==============================================================================

EXPERIMENTS = {
    # ==============================================================================
    # 1. EXPERIMENTS FOR FIGURE 1: THE PHYSICAL CANVAS
    # ==============================================================================
    "boundary_dynamics_vs_selection": {
        "CAMPAIGN_ID": "v2_boundary_dynamics_symmetric",
        "run_mode": "calibration",
        "HPC_PARAMS": {"time": "0-12:00:00", "mem": "2G", "sims_per_task": 60},
        "PARAM_GRID": {
            "s_symmetric_scan": s_fig1a_scan.tolist(),
            "b_m_from_s_scan": bm_fig1a_scan.tolist(),
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "width": 512,
                    "length": 1024,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 400,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 128,
                    "max_steps": 5_000_000,
                },
                "grid_params": {"b_m": "b_m_from_s_scan"},
            }
        },
    },
    "front_morphology_vs_selection": {
        "CAMPAIGN_ID": "v2_front_morphology_symmetric",
        "run_mode": "diffusion",
        "HPC_PARAMS": {"time": "0-18:00:00", "mem": "4G", "sims_per_task": 40},
        "PARAM_GRID": {
            "s_symmetric_scan": s_fig1b_scan.tolist(),
            "b_m_from_s_scan": bm_fig1b_scan.tolist(),
            "width_scan": [64, 128, 256, 512, 768],
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "length": 4096,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "num_replicates": 200,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 0,
                    "max_steps": 10_000_000,
                },
                "grid_params": {"b_m": "b_m_from_s_scan", "width": "width_scan"},
            }
        },
    },
    # ==============================================================================
    # 2. EXPERIMENTS FOR FIGURE 2 & 5: PHASE DIAGRAM & CRITICALITY
    # ==============================================================================
    "hierarchical_criticality_global": {
        "CAMPAIGN_ID": "v_final_hierarchical_global",
        "run_mode": "structure_analysis",
        "HPC_PARAMS": {"time": "04:00:00", "mem": "4G", "sims_per_task": 40},
        "PARAM_GRID": {
            "s_global_scan": s_global_scan.tolist(),
            "b_m_from_s_scan": bm_global_scan.tolist(),
            "phi_full_scan": phi_global_scan.tolist(),
            "k_total_global_scan": k_global_scan.tolist(),
        },
        "SIM_SETS": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "num_replicates": 32,
                    "warmup_time": 800.0,
                    "num_samples": 300,
                    "sample_interval": 15.0,
                    "initial_condition_type": "mixed",
                },
                "grid_params": {
                    "b_m": "b_m_from_s_scan",
                    "phi": "phi_full_scan",
                    "k_total": "k_total_global_scan",
                },
            }
        },
    },
    "hierarchical_criticality_focused": {
        "CAMPAIGN_ID": "v_final_hierarchical_focused",
        "run_mode": "structure_analysis",
        "HPC_PARAMS": {"time": "0-06:00:00", "mem": "4G", "sims_per_task": 60},
        "PARAM_GRID": {
            "s_dip_region": s_dip_region_scan.tolist(),
            "b_m_dip_region": bm_dip_region_scan.tolist(),
            "k_dip_region": k_dip_region_scan.tolist(),
            "phi_full_scan": phi_focused_scan.tolist(),
        },
        "SIM_SETS": {
            "focused_scan": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "num_replicates": 64,
                    "warmup_time": 800.0,
                    "num_samples": 300,
                    "sample_interval": 15.0,
                    "initial_condition_type": "mixed",
                },
                "grid_params": {
                    "b_m": "b_m_dip_region",
                    "phi": "phi_full_scan",
                    "k_total": "k_dip_region",
                },
            }
        },
    },
    "exp_relaxation_dynamics": {
        "CAMPAIGN_ID": "v_final_relaxation_from_mutant",
        "run_mode": "timeseries_from_pure_state",
        "HPC_PARAMS": {"time": "0-08:00:00", "mem": "4G", "sims_per_task": 50},
        "PARAM_GRID": {
            "s_deleterious_scan": s_deleterious_scan.tolist(),
            "b_m_from_s_scan": bm_deleterious_scan.tolist(),
            "phi_slices": [-0.5, 0.0, 0.5],
            "k_total_scan": k_relaxation_scan.tolist(),
        },
        "SIM_SETS": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "num_replicates": 40,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": "width",
                    "max_sim_time": 5000.0,
                    "log_interval": 20.0,
                },
                "grid_params": {
                    "b_m": "b_m_from_s_scan",
                    "phi": "phi_slices",
                    "k_total": "k_total_scan",
                },
            }
        },
    },
    # ==============================================================================
    # 3. EXPERIMENTS FOR FIGURE 3 & 4: BET-HEDGING
    # ==============================================================================
    "spatial_bet_hedging_final": {
        "CAMPAIGN_ID": "v_final_spatial_bet_hedging",
        "run_mode": "spatial_fluctuation_analysis",
        "HPC_PARAMS": {"time": "1-04:00:00", "mem": "2G", "sims_per_task": 20},
        "PARAM_GRID": {
            "b_m_scan": [0.75, 0.85, 0.95],
            "phi_symmetric_scan": phi_bet_hedging_scan.tolist(),
            "k_total_focused_scan": k_bet_hedging_scan.tolist(),
            "patch_width_scan": [30, 60, 120],
            "env_bet_hedging": {0: {"b_wt": 1.0}, 1: {"b_wt": 0.0, "b_m": 1.0}},
        },
        "SIM_SETS": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "initial_condition_type": "mixed",
                    "environment_map": "env_bet_hedging",
                    "num_replicates": 32,
                    "log_q_interval": 2.0,
                    "warmup_cycles_for_stats": 4,
                },
                "grid_params": {
                    "b_m": "b_m_scan",
                    "phi": "phi_symmetric_scan",
                    "k_total": "k_total_focused_scan",
                    "patch_width": "patch_width_scan",
                },
            },
        },
    },
    "asymmetric_environment_v1": {
        "CAMPAIGN_ID": "v1_asymmetric_env_bet_hedging",
        "run_mode": "spatial_fluctuation_analysis",
        "HPC_PARAMS": {"time": "1-00:00:00", "mem": "2G", "sims_per_task": 30},
        "PARAM_GRID": {
            "b_m_scan": [0.75, 0.9],
            "phi_polluting_scan": phi_asymmetric_scan.tolist(),
            "k_total_scan": k_asymmetric_scan.tolist(),
            "env_map_bet_hedging": {0: {"b_wt": 1.0}, 1: {"b_wt": 0.0, "b_m": 1.0}},
            "env_asym_30_90": [(0, 30), (1, 90)],
            "env_asym_60_60": [(0, 60), (1, 60)],
            "env_asym_90_30": [(0, 90), (1, 30)],
            "asymmetric_sequences": [
                "env_asym_30_90",
                "env_asym_60_60",
                "env_asym_90_30",
            ],
        },
        "SIM_SETS": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 8192,
                    "initial_condition_type": "mixed",
                    "environment_map": "env_map_bet_hedging",
                    "num_replicates": 32,
                    "log_q_interval": 2.0,
                    "warmup_cycles_for_stats": 4,
                },
                "grid_params": {
                    "b_m": "b_m_scan",
                    "phi": "phi_polluting_scan",
                    "k_total": "k_total_scan",
                    "environment_patch_sequence": "asymmetric_sequences",
                },
            },
        },
    },
}
