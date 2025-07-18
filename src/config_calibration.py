# In src/config_calibration.py

import numpy as np  # Make sure numpy is imported

# A unique identifier for the definitive, deleterious-focused campaign.
CAMPAIGN_ID = "calibration_v4_deleterious_focus"

# --- Simulation Parameters ---
# Keep the parameters that are working well.
SIM_PARAMS_CALIBRATION = {
    "width": 256,
    "length": 512,
    "initial_patch_size": 80,
    "max_steps": 3_000_000,
    "num_replicates": 500,  # Increased replicates for even smoother data
}

# --- [NEW] Focused Deleterious Parameter Grid ---
PARAM_GRID_CALIBRATION = {
    # We will sample s = b_m - 1 more densely in the deleterious range.
    "b_m": np.unique(
        np.concatenate(
            [
                # High-resolution scan of deleterious mutants (s < 0)
                np.linspace(0, 0.8, 100),
                np.linspace(0.80, 0.98, 20),  # s from -0.20 to -0.02
                np.linspace(0.985, 0.995, 5),  # Very fine scan right near s=0
                # Neutral case
                np.array([1.0]),
            ]
        )
    ).tolist()
}
