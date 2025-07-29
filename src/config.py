# FILE: src/config.py
# [DEFINITIVE VERSION]
# This is the single source of truth for ALL simulation campaigns. It preserves
# legacy experiments for reproducibility and adds the new, targeted campaigns
# for the final publication.

import numpy as np

EXPERIMENTS = {
    # ==============================================================================
    # LEGACY CAMPAIGNS (Completed Work - Kept for Reproducibility)
    # ==============================================================================
    "p1_definitive_v2": {
        "CAMPAIGN_ID": "p1_definitive_v2",
        "run_mode": "steady_state",
        "HPC_PARAMS": {"time": "0-08:00:00", "mem": "4G", "sims_per_task": 100},
        "PARAM_GRID": {
            "b_m": [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95],
            "k_total_low": np.logspace(-2, -1, 6).tolist(),
            "k_total_mid": np.logspace(-0.82, 0, 5).tolist(),
            "k_total_high": np.logspace(0.18, 2, 10).tolist(),
            "phi": np.linspace(-1.0, 1.0, 21).tolist(),
            "width_scaling": [32, 64, 128, 256],
        },
        "SIM_SETS": {  # Note: base_params are omitted as they were not in the original file provided
            "main_high_k": {
                "grid_params": {"k_total": "k_total_high", "b_m": "b_m", "phi": "phi"}
            },
            "main_low_k": {
                "grid_params": {"k_total": "k_total_low", "b_m": "b_m", "phi": "phi"}
            },
            "main_mid_k": {
                "grid_params": {"k_total": "k_total_mid", "b_m": "b_m", "phi": "phi"}
            },
            "scaling": {"grid_params": {"width": "width_scaling"}},
        },
    },
    "calibration_v4": {
        "CAMPAIGN_ID": "calibration_v4_deleterious_focus",
        "run_mode": "calibration",
        "HPC_PARAMS": {"time": "0-12:00:00", "mem": "2G", "sims_per_task": 50},
        "PARAM_GRID": {
            "b_m": np.unique(
                np.concatenate(
                    [
                        np.linspace(0, 0.8, 100),
                        np.linspace(0.80, 0.98, 20),
                        np.linspace(0.985, 0.995, 5),
                        [1.0],
                    ]
                )
            ).tolist()
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "width": 256,
                    "length": 512,
                    "initial_mutant_patch_size": 80,
                    "max_steps": 3_000_000,
                    "num_replicates": 500,
                    "k_total": 0.0,
                    "phi": 0.0,
                    "initial_condition_type": "patch",
                },
                "grid_params": {"b_m": "b_m"},
            }
        },
    },
    "diffusion_v2_refined": {
        "CAMPAIGN_ID": "diffusion_v2_refined_neutral",
        "run_mode": "diffusion",
        "HPC_PARAMS": {"time": "1-00:00:00", "mem": "4G", "sims_per_task": 25},
        "PARAM_GRID": {"width": [32, 64, 128, 256, 512]},
        "SIM_SETS": {
            "main": {"base_params": {"b_m": 1.0, "initial_condition_type": "mixed"}}
        },  # Assuming neutral and mixed
    },
    "spatial_structure_v1": {
        "CAMPAIGN_ID": "spatial_structure_v1_fm75",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {"time": "0-12:00:00", "mem": "4G", "sims_per_task": 10},
        "PARAM_GRID": {
            "b_m": [0.5, 0.8, 0.95],
            "k_total": np.logspace(-2, 2, 20).tolist(),
        },
        "SIM_SETS": {
            "main": {"base_params": {"phi": -0.5, "initial_condition_type": "mixed"}}
        },
    },
    "criticality_mapping_v1": {
        "CAMPAIGN_ID": "criticality_v1_full_map",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {"time": "03:00:00", "mem": "4G", "sims_per_task": 60},
        "PARAM_GRID": {
            "b_m_scan": [0.5, 0.7, 0.8, 0.95],
            "phi_scan": [-1.0, -0.5, 0.0, 0.5],
            "k_total": np.logspace(-2, 2, 25).tolist(),
        },
        "SIM_SETS": {
            "scan_s": {
                "base_params": {"phi": 0.0},
                "grid_params": {"b_m": "b_m_scan", "k_total": "k_total"},
            },
            "scan_phi": {
                "base_params": {"b_m": 0.8},
                "grid_params": {"phi": "phi_scan", "k_total": "k_total"},
            },
        },
    },
    "full_phase_diagram_v1": {
        "CAMPAIGN_ID": "full_3d_scan_v1",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {"time": "03:00:00", "mem": "4G", "sims_per_task": 60},
        "PARAM_GRID": {
            "b_m_full": np.linspace(0.5, 1.0, 6).tolist(),
            "phi_full": np.linspace(-1.0, 1.0, 11).tolist(),
            "k_total_full": np.logspace(-2, 2, 25).tolist(),
        },
        "SIM_SETS": {
            "main_scan": {
                "grid_params": {
                    "b_m": "b_m_full",
                    "phi": "phi_full",
                    "k_total": "k_total_full",
                }
            }
        },
    },
    "criticality_v2": {
        "CAMPAIGN_ID": "criticality_v2_global_lores",
        "run_mode": "structure_analysis",
        "HPC_PARAMS": {"time": "0-04:00:00", "mem": "4G", "sims_per_task": 50},
        "PARAM_GRID": {
            "b_m_global_scan": [s + 1.0 for s in np.linspace(-0.8, 0.0, 17).tolist()],
            "k_total_global_scan": np.logspace(-2, 2, 25).tolist(),
        },
        "SIM_SETS": {
            "main": {
                "base_params": {"phi": 0.0, "initial_condition_type": "mixed"},
                "grid_params": {
                    "b_m": "b_m_global_scan",
                    "k_total": "k_total_global_scan",
                },
            }
        },
    },
    "criticality_v3": {
        "CAMPAIGN_ID": "criticality_v3_dip_focus",
        "run_mode": "structure_analysis",
        "HPC_PARAMS": {"time": "0-02:00:00", "mem": "2G", "sims_per_task": 100},
        "PARAM_GRID": {
            "b_m_local_scan": [s + 1.0 for s in np.linspace(-0.3, -0.1, 21).tolist()],
            "k_total_local_scan": np.logspace(
                np.log10(0.1), np.log10(0.8), 40
            ).tolist(),
        },
        "SIM_SETS": {
            "main": {
                "base_params": {"phi": 0.0, "initial_condition_type": "mixed"},
                "grid_params": {
                    "b_m": "b_m_local_scan",
                    "k_total": "k_total_local_scan",
                },
            }
        },
    },
    "criticality_irreversible_v2_global": {
        "CAMPAIGN_ID": "criticality_v2_irreversible_global",
        "run_mode": "structure_analysis",
        "HPC_PARAMS": {"time": "0-04:00:00", "mem": "4G", "sims_per_task": 50},
        "PARAM_GRID": {
            "b_m_from_s_scan": [s + 1.0 for s in np.linspace(-0.8, 0.0, 17).tolist()],
            # Very wide but sparse k_total range to find the peaks
            "k_total_global": np.logspace(-2.5, 2.5, 50).tolist(),
        },
        "SIM_SETS": {
            "main_scan": {
                "base_params": {  # Same base params as before
                    "width": 256,
                    "length": 50000,
                    "phi": -1.0,
                    "num_replicates": 32,
                    "warmup_time": 800.0,
                    "num_samples": 300,
                    "sample_interval": 15.0,
                    "initial_condition_type": "mixed",
                },
                "grid_params": {
                    "b_m": "b_m_from_s_scan",
                    "k_total": "k_total_global",
                },
            }
        },
    },
    "full_phase_diagram_v1": {
        "CAMPAIGN_ID": "full_3d_scan_v1_coarse",
        "run_mode": "structure_analysis",
        "HPC_PARAMS": {"time": "0-12:00:00", "mem": "4G", "sims_per_task": 100},
        "PARAM_GRID": {
            # Coarser scan for selection, focusing on interesting regions
            "b_m_coarse_scan": [s + 1.0 for s in [-0.8, -0.6, -0.4, -0.2, 0.0]],
            # A representative set of switching biases
            "phi_coarse_scan": [-1.0, -0.5, 0.0, 0.5, 1.0],
            # A sparse but wide k_total scan to find the peaks
            "k_total_global": np.logspace(-2.5, 2.5, 50).tolist(),
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
                    "b_m": "b_m_coarse_scan",
                    "phi": "phi_coarse_scan",
                    "k_total": "k_total_global",
                },
            }
        },
    },
    "boundary_dynamics_vs_selection": {
        "CAMPAIGN_ID": "v1_boundary_dynamics",
        "run_mode": "calibration",
        "HPC_PARAMS": {"time": "0-10:00:00", "mem": "2G", "sims_per_task": 80},
        "PARAM_GRID": {
            "b_m_scan": [s + 1.0 for s in np.linspace(-0.8, 0.0, 17).tolist()]
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "width": 512,
                    "length": 1024,
                    "k_total": 0.0,
                    "phi": 0.0,
                    # Increased replicates for a cleaner d_eff measurement
                    "num_replicates": 400,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 128,
                    "max_steps": 5_000_000,
                },
                "grid_params": {"b_m": "b_m_scan"},
            }
        },
    },
    "boundary_dynamics_vs_selection": {
        "CAMPAIGN_ID": "v1_boundary_dynamics",
        "run_mode": "calibration",
        "HPC_PARAMS": {"time": "0-10:00:00", "mem": "2G", "sims_per_task": 80},
        "PARAM_GRID": {
            "b_m_scan": [s + 1.0 for s in np.linspace(-0.8, 0.0, 17).tolist()]
        },
        "SIM_SETS": {
            "main": {
                "base_params": {
                    "width": 512,
                    "length": 1024,
                    "k_total": 0.0,
                    "phi": 0.0,
                    # Increased replicates for a cleaner d_eff measurement
                    "num_replicates": 400,
                    "initial_condition_type": "patch",
                    "initial_mutant_patch_size": 128,
                    "max_steps": 5_000_000,
                },
                "grid_params": {"b_m": "b_m_scan"},
            }
        },
    },
    "front_morphology_vs_selection": {
        "CAMPAIGN_ID": "v1_front_morphology",
        "run_mode": "diffusion",
        "HPC_PARAMS": {"time": "0-16:00:00", "mem": "4G", "sims_per_task": 50},
        "PARAM_GRID": {
            "b_m_scan": [s + 1.0 for s in np.linspace(-0.8, 0.0, 17).tolist()],
            # Increased max width for better saturation data
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
                "grid_params": {"b_m": "b_m_scan", "width": "width_scan"},
            }
        },
    },
}
