# FILE: src/config.py
# ==============================================================================
# SINGLE SOURCE OF TRUTH for ALL simulation campaigns.
# This file contains final, optimized parameters based on diagnostic runs.
# ==============================================================================

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
        "SIM_SETS": {
            "main_high_k": {"base_params": {}, "grid_params": {}},
            "main_low_k": {"base_params": {}, "grid_params": {}},
            "main_mid_k": {"base_params": {}, "grid_params": {}},
            "scaling": {"base_params": {}, "grid_params": {}},
        },
        "ANALYSIS_PARAMS": {},
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
                        np.array([1.0]),
                    ]
                )
            ).tolist()
        },
        "SIM_SETS": {"main": {"base_params": {}, "grid_params": {}}},
    },
    "diffusion_v2_refined": {
        "CAMPAIGN_ID": "diffusion_v2_refined_neutral",
        "run_mode": "diffusion",
        "HPC_PARAMS": {"time": "1-00:00:00", "mem": "4G", "sims_per_task": 25},
        "PARAM_GRID": {"width": [32, 64, 128, 256, 512]},
        "SIM_SETS": {"main": {"base_params": {}, "grid_params": {}}},
    },
    "spatial_structure_v1": {
        "CAMPAIGN_ID": "spatial_structure_v1_fm75",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {"time": "0-12:00:00", "mem": "4G", "sims_per_task": 10},
        "PARAM_GRID": {
            "b_m": [0.5, 0.8, 0.95],
            "k_total": np.logspace(-2, 2, 20).tolist(),
        },
        "SIM_SETS": {"main": {"base_params": {}, "grid_params": {}}},
    },
    # ==============================================================================
    # ACTIVE CAMPAIGNS (Optimized and Ready to Run)
    # ==============================================================================
    # --- Experiment 5: The Definitive Criticality Campaign (Targeted 2D Scans) ---
    "criticality_mapping_v1": {
        "CAMPAIGN_ID": "criticality_v1_full_map",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {
            "time": "03:00:00",  # OPTIMIZED: For larger chunk size.
            "mem": "4G",
            "sims_per_task": 60,  # OPTIMIZED: To respect cluster submission limits.
        },
        "PARAM_GRID": {
            "b_m_scan": [0.5, 0.7, 0.8, 0.95],
            "phi_scan": [-1.0, -0.5, 0.0, 0.5],
            "k_total": np.logspace(-2, 2, 25).tolist(),
        },
        "SIM_SETS": {
            # Set 1: Map k_c vs. s at zero bias
            "scan_s": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "phi": 0.0,
                    "num_replicates": 24,
                    # Parameters below are derived from diagnostic script `analyze_sampling_precision.py`
                    "warmup_time": 500.0,
                    "num_samples": 225,
                    "sample_interval": 20.0,
                },
                "grid_params": {"b_m": "b_m_scan", "k_total": "k_total"},
            },
            # Set 2: Map k_c vs. phi at fixed selection
            "scan_phi": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "b_m": 0.8,
                    "num_replicates": 24,
                    # Parameters below are derived from diagnostic script `analyze_sampling_precision.py`
                    "warmup_time": 500.0,
                    "num_samples": 225,
                    "sample_interval": 20.0,
                },
                "grid_params": {"phi": "phi_scan", "k_total": "k_total"},
            },
        },
    },
    # --- Experiment 6: Comprehensive 3D Scan of the Full Phase Diagram ---
    "full_phase_diagram_v1": {
        "CAMPAIGN_ID": "full_3d_scan_v1",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {
            "time": "03:00:00",  # OPTIMIZED: For larger chunk size.
            "mem": "4G",
            "sims_per_task": 60,  # OPTIMIZED: To respect cluster submission limits.
        },
        "PARAM_GRID": {
            "b_m_full": np.linspace(0.5, 1.0, 6).tolist(),
            "phi_full": np.linspace(-1.0, 1.0, 11).tolist(),
            "k_total_full": np.logspace(-2, 2, 25).tolist(),
        },
        "SIM_SETS": {
            "main_scan": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "num_replicates": 16,
                    # Parameters below are derived from diagnostic script `analyze_sampling_precision.py`
                    "warmup_time": 500.0,
                    "num_samples": 225,
                    "sample_interval": 20.0,
                },
                "grid_params": {
                    "b_m": "b_m_full",
                    "phi": "phi_full",
                    "k_total": "k_total_full",
                },
            },
        },
    },
    
    "criticality_v2": {
        "CAMPAIGN_ID": "criticality_v2",
        "run_mode": "correlation_analysis",
        "HPC_PARAMS": {
            "time": "0-04:00:00",
            "mem": "4G",
            "sims_per_task": 50,
        },
        "PARAM_GRID": {
            # [THE FIX] Define the b_m list inside PARAM_GRID with a string key
            "b_m_hires": [s + 1.0 for s in np.linspace(-0.8, 0.0, 17).tolist()],
            "k_total_focused": np.logspace(-1.5, 1.5, 30).tolist(),
        },
        "SIM_SETS": {
            "main_boundary_scan": {
                "base_params": {
                    "width": 256,
                    "length": 50000,
                    "phi": 0.0,
                    "num_replicates": 32,
                    "warmup_time": 800.0,
                    "num_samples": 300,
                    "sample_interval": 15.0,
                },
                "grid_params": {
                    # [THE FIX] Use the string key to refer to the list
                    "b_m": "b_m_hires",
                    "k_total": "k_total_focused",
                },
            },
        },
        "ANALYSIS_PARAMS": {
            "critical_b_m_for_eta": 0.8,
        },
    },
}


# This is a dummy assignment to prevent linting errors if some parts of the
# file are commented out during testing, and for legacy script compatibility.
if "p1_definitive_v2" in EXPERIMENTS:
    ANALYSIS_PARAMS = EXPERIMENTS["p1_definitive_v2"].get("ANALYSIS_PARAMS")
    PARAM_GRID = EXPERIMENTS["p1_definitive_v2"].get("PARAM_GRID")
    CAMPAIGN_ID = EXPERIMENTS["p1_definitive_v2"].get("CAMPAIGN_ID")
