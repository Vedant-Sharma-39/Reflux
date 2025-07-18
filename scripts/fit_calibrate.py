# scripts/fit_and_plot_model.py
# [UPGRADED] This script loads the final processed data from a calibration campaign
# and fits THREE physical models (linear, quadratic, cubic) to the v_drift vs. s curve,
# then plots them together for comparison.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Configuration ---
# Add project root to path to import the config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    # Use the CAMPAIGN_ID from your most recent, completed campaign
    from config_calibration import CAMPAIGN_ID
except ImportError:
    print("Error: Could not import from src/config_calibration.py.")
    sys.exit(1)


# --- [NEW] Define all models to fit ---
def linear_model(s, A):
    """Linear response model."""
    return A * s


def quadratic_model(s, A, B):
    """Quadratic model including the first non-linear correction."""
    return A * s + B * s**2


def cubic_model(s, A, B, C):
    """Cubic model for higher-order non-linear effects."""
    return A * s + B * s**2 + C * s**3


def calculate_r_squared(y_true, y_pred):
    """Helper function to calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def main():
    print(f"--- Fitting Models to Calibration Curve for Campaign: {CAMPAIGN_ID} ---")

    # 1. Load the processed data
    analysis_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "analysis")
    data_path = os.path.join(analysis_dir, "calibration_curve_data.csv")

    if not os.path.exists(data_path):
        print(f"Error: Processed data file not found at {data_path}")
        print("Please run 'scripts/analyze_calibration_results.py' first.")
        return

    df = pd.read_csv(data_path)
    s_data = df["s"].values
    v_data = df["v_drift"].values

    # --- [NEW] Perform all three fits ---
    try:
        # Linear Fit
        popt_lin, pcov_lin = curve_fit(linear_model, s_data, v_data, p0=[1.0])
        err_lin = np.sqrt(np.diag(pcov_lin))
        r2_lin = calculate_r_squared(v_data, linear_model(s_data, *popt_lin))

        # Quadratic Fit
        popt_quad, pcov_quad = curve_fit(quadratic_model, s_data, v_data, p0=[1.0, 0.5])
        err_quad = np.sqrt(np.diag(pcov_quad))
        r2_quad = calculate_r_squared(v_data, quadratic_model(s_data, *popt_quad))

        # Cubic Fit
        popt_cub, pcov_cub = curve_fit(cubic_model, s_data, v_data, p0=[1.0, 0.5, 0.1])
        err_cub = np.sqrt(np.diag(pcov_cub))
        r2_cub = calculate_r_squared(v_data, cubic_model(s_data, *popt_cub))
        
        #log-fit
        popt_log, pcov_log = curve_fit()

    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        return

    # --- [NEW] Print a comprehensive report ---
    print("\n--- Model Fit Comparison ---")
    print("-" * 50)
    print(
        f"Linear Fit: v = A*s\n  A = {popt_lin[0]:.4f} ± {err_lin[0]:.4f}\n  R² = {r2_lin:.6f}"
    )
    print("-" * 50)
    print(
        f"Quadratic Fit: v = A*s + B*s²\n  A = {popt_quad[0]:.4f} ± {err_quad[0]:.4f}\n  B = {popt_quad[1]:.4f} ± {err_quad[1]:.4f}\n  R² = {r2_quad:.6f}"
    )
    print("-" * 50)
    print(
        f"Cubic Fit: v = A*s + B*s² + C*s³\n  A = {popt_cub[0]:.4f} ± {err_cub[0]:.4f}\n  B = {popt_cub[1]:.4f} ± {err_cub[1]:.4f}\n  C = {popt_cub[2]:.4f} ± {err_cub[2]:.4f}\n  R² = {r2_cub:.6f}"
    )
    print("-" * 50)

    # --- [NEW] Generate the final, comparative plot ---
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the raw simulation data
    ax.plot(
        s_data,
        v_data,
        "o",
        color="navy",
        markersize=3,
        zorder=10,
        label="Measured $v_{drift}$",
    )

    # Create a smooth x-axis for plotting the curves
    s_smooth = np.linspace(s_data.min(), s_data.max(), 200)

    # Plot the three fitted models
    ax.plot(
        s_smooth,
        linear_model(s_smooth, *popt_lin),
        "g:",
        linewidth=2,
        label=f"Linear Fit (R²={r2_lin:.4f})",
    )
    ax.plot(
        s_smooth,
        quadratic_model(s_smooth, *popt_quad),
        "--",
        color="darkorange",
        linewidth=2.5,
        label=f"Quadratic Fit (R²={r2_quad:.4f})",
    )
    ax.plot(
        s_smooth,
        cubic_model(s_smooth, *popt_cub),
        "-",
        color="crimson",
        linewidth=2.5,
        label=f"Cubic Fit (R²={r2_cub:.4f})",
    )

    ax.set_title("Model Comparison for Effective Drift Velocity", fontsize=16)
    ax.set_xlabel("Selection Coefficient ($s = b_m - 1$)", fontsize=12)
    ax.set_ylabel("Effective Drift Velocity ($v_{drift}$)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.legend(fontsize=11)

    final_plot_path = os.path.join(analysis_dir, "FINAL_MODEL_COMPARISON.png")
    plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
    print(f"\nFinal model comparison plot saved to: {final_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
