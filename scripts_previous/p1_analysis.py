# scripts/p1_final_analysis.py
# The final, consolidated master analysis script for the V2 campaign.
# This version is tightly coupled with src/config.py for robust, data-driven plotting.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter

# --- Robust Path and Config Import ---
# This ensures the script can find the 'src' directory and the config file.
# It handles cases where the script might be run from a different current working directory.
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "src"))
    from config import CAMPAIGN_ID, PARAM_GRID, ANALYSIS_PARAMS
except ImportError:
    # If direct import fails, try to adjust path for alternative run methods
    print("FATAL: Could not import configuration from src/config.py.")
    print("       Attempting alternative path setup...")
    try:
        project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
        sys.path.insert(0, os.path.join(project_root, "src"))
        from config import CAMPAIGN_ID, PARAM_GRID, ANALYSIS_PARAMS
    except Exception as e:
        print(
            f"FATAL: Failed to import config.py even with alternative path setup. Error: {e}"
        )
        print(
            "       Please ensure src/config.py exists and the script is run from the project root or 'scripts' directory."
        )
        sys.exit(1)


# --- Plotting Helpers ---
def prettify_ax(ax):
    """Applies a clean, modern style to a Matplotlib Axes object."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("darkgrey")
    ax.spines["bottom"].set_color("darkgrey")
    ax.tick_params(axis="both", which="major", colors="k", length=6, width=1.2)
    ax.grid(
        True, which="major", linestyle=":", linewidth=0.7, color="lightgrey", zorder=0
    )
    ax.grid(
        True, which="minor", linestyle=":", linewidth=0.5, color="#eeeeee", zorder=0
    )
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.title.set_color("black")


def crossover_function(k, k_c, n):
    """Sigmoidal Hill function for fitting crossover data."""
    return (k**n) / (k_c**n + k**n)


# --- Main Execution ---
def main():
    # --- 0. SETUP ---
    data_file = os.path.join(project_root, "data", f"{CAMPAIGN_ID}_aggregated.csv")
    figures_dir = os.path.join(project_root, "figures", CAMPAIGN_ID)
    os.makedirs(figures_dir, exist_ok=True)

    # Set global plotting parameters here for consistency
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": 14,  # Base font size
        }
    )

    print(f"--- Running Analysis for Campaign: {CAMPAIGN_ID} ---")
    print(f"Figures will be saved to: {figures_dir}")

    # --- 1. LOAD AND PROCESS DATA ---
    print(f"Loading data from: {data_file}")
    if not os.path.exists(data_file):
        print(
            f"FATAL: Data file not found. Please run the aggregation script for campaign '{CAMPAIGN_ID}'."
        )
        sys.exit(1)
    df = pd.read_csv(data_file)

    # Convert all columns except 'task_id' to numeric, coercing errors to NaN
    # This ensures that numerical operations don't fail due to string representations
    cols_to_convert = [col for col in df.columns if col != "task_id"]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

    # Filter out rows where avg_rho_M is negative (indicating an error during simulation)
    df_clean = df[df["avg_rho_M"] >= 0].copy()

    # Calculate f_M, which is used extensively in plotting
    df_clean["f_M"] = (1 - df_clean["phi"]) / 2

    # --- 2. AGGREGATE DATA ---
    # Smart aggregation: Use wider system data for low k_total for higher quality
    # This selection criteria (0.1) should align with the ranges defined in src/config.py
    df_high_k = df_clean[(df_clean["k_total"] > 0.1) & (df_clean["width"] == 128)]
    df_low_k = df_clean[(df_clean["k_total"] <= 0.1) & (df_clean["width"] == 256)]

    # Concatenate the selected dataframes. ignore_index=True ensures a clean index.
    df_best_res = pd.concat([df_low_k, df_high_k], ignore_index=True)

    # Aggregate by the main parameters (b_m, k_total, f_M) to get mean and standard error
    agg_all_bm = (
        df_best_res.groupby(["b_m", "k_total", "f_M"])
        .agg(
            mean_rho_M=("avg_rho_M", "mean"),
            sem_rho_M=("avg_rho_M", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        )
        .reset_index()
    )

    # Process scaling data separately
    df_scaling_raw = df_clean[df_clean["task_id"].str.startswith("scaling")].copy()
    agg_scaling = (
        df_scaling_raw.groupby(["k_total", "f_M", "width"])
        .agg(
            mean_rho_M=("avg_rho_M", "mean"),
            sem_rho_M=("avg_rho_M", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        )
        .reset_index()
    )
    agg_scaling["1/L"] = 1 / agg_scaling["width"]

    # Define color palette for b_m values, ensuring consistent colors across plots
    unique_bm = sorted(agg_all_bm["b_m"].unique())
    B_M_PALETTE = {
        val: color
        for val, color in zip(
            unique_bm, sns.color_palette("PuBuGn", n_colors=len(unique_bm))
        )
    }

    # --- 3. GENERATE FIGURES ---

    # == FIGURE 1: Heatmap (Contour) ==
    print("Generating Figure 1: Contour Heatmap...")
    # Select data for b_m = 0.8 as specified in the config
    agg_heatmap = agg_all_bm[
        np.isclose(agg_all_bm["b_m"], 0.8)
    ].copy()  # .copy() to avoid SettingWithCopyWarning later

    # Pivot data for heatmap, sort index for correct plotting order
    heatmap_data = agg_heatmap.pivot_table(
        index="k_total", columns="f_M", values="mean_rho_M"
    ).sort_index()

    # Create meshgrid for pcolormesh and contour plotting
    X, Y = np.meshgrid(heatmap_data.columns.values, heatmap_data.index.values)
    Z = heatmap_data.values

    fig, ax = plt.subplots(figsize=(16, 12))
    mesh = ax.pcolormesh(X, Y, Z, cmap="summer", shading="gouraud", vmin=0, vmax=1)
    CS = ax.contour(X, Y, Z, levels=[0.25, 0.50, 0.75], colors="white", linewidths=2.5)
    ax.clabel(CS, inline=True, fontsize=14, fmt="%.2f")  # Label contour lines
    ax.set_yscale("log")  # Log scale for k_total

    ax.set_title("Phase Diagram for Mutant Fraction ($b_m = 0.8$)", fontsize=24, pad=20)
    ax.set_xlabel("Bias Towards Mutants ($f_M$)", fontsize=18)
    ax.set_ylabel("Total Switching Rate ($k_{total}$)", fontsize=18)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: f"{val:.0%}")
    )  # Format x-axis as percentage
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: f"{val:g}")
    )  # Format y-axis cleanly

    # Add color bar
    fig.colorbar(mesh, ax=ax).set_label(
        "Mean Mutant Fraction $\\langle\\rho_M\\rangle$", size=18, weight="bold"
    )

    prettify_ax(ax)  # Apply consistent styling
    plt.savefig(
        os.path.join(figures_dir, f"{CAMPAIGN_ID}_fig1_contour_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()  # Close figure to free memory

    # == FIGURE 2: Heatmap (Deviation) ==
    print("Generating Figure 2: Deviation Heatmap...")
    # Fix SettingWithCopyWarning here by explicitly using .loc
    agg_heatmap.loc[:, "deviation"] = agg_heatmap["mean_rho_M"] - agg_heatmap["f_M"]

    deviation_heatmap_data = agg_heatmap.pivot_table(
        index="k_total", columns="f_M", values="deviation"
    )

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        deviation_heatmap_data,
        annot=False,  # Annotate cells with values
        fmt=".2f",  # Format annotations to 2 decimal places
        cmap="Purples",  
        linewidths=1,  # Lines between cells
        cbar_kws={"label": "Deviation from Neutral Expectation ($\\rho_M - f_M$)"},
        ax=ax,
    )
    ax.set_title(
        "Selectional Pull on Mutant Fraction ($b_m = 0.8$)", fontsize=22, pad=20
    )
    ax.set_xlabel("Bias Towards Mutants ($f_M$)", fontsize=18)
    ax.set_ylabel("Total Switching Rate ($k_{total}$)", fontsize=18)

    # Custom formatter for x-axis ticks to match percentages
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: f"{deviation_heatmap_data.columns[int(val)]:.0%}"
        )
    )
    ax.invert_yaxis()  # Invert y-axis to have small k_total at bottom

    prettify_ax(ax)
    plt.savefig(
        os.path.join(figures_dir, f"{CAMPAIGN_ID}_fig2_deviation_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # == FIGURE 3: Slice vs. Bias Subplots ==
    print("Generating Figure 3: Slice vs. Bias Subplots...")
    b_m_to_compare = ANALYSIS_PARAMS["slice_plot_b_m"]
    num_panels = len(b_m_to_compare)
    all_k = PARAM_GRID["k_total_all"]

    # Dynamically select 6 k_total values to plot across the full range
    k_indices = np.round(np.linspace(0, len(all_k) - 1, 6)).astype(int)
    k_to_plot = [all_k[i] for i in k_indices]

    fig, axes = plt.subplots(
        1, num_panels, figsize=(8 * num_panels, 8), sharey=True, constrained_layout=True
    )
    if num_panels == 1:  # Ensure axes is always an array, even for single panel
        axes = [axes]

    fig.suptitle("Mutant Fraction vs. Switching Bias", fontsize=24, y=1.03)
    cmap = sns.color_palette(
        "magma_r", n_colors=len(k_to_plot)
    )  # Colormap for k_total lines

    for i, b_m_val in enumerate(b_m_to_compare):
        ax = axes[i]
        # Select data for the current b_m value
        subset_df = agg_all_bm[np.isclose(agg_all_bm["b_m"], b_m_val)]

        # Plot the theoretical mixing limit (y=x)
        x_theory = np.linspace(0, 1, 100)
        ax.plot(
            x_theory,
            x_theory,
            color="black",
            linestyle="--",
            linewidth=3,
            label="Mixing Limit",
            zorder=2,
        )

        # Plot simulation data slices
        for j, k_val in enumerate(k_to_plot):
            slice_df = subset_df[np.isclose(subset_df["k_total"], k_val)]
            if not slice_df.empty:  # Only plot if data exists for this slice
                ax.errorbar(
                    slice_df["f_M"],
                    slice_df["mean_rho_M"],
                    yerr=slice_df["sem_rho_M"],
                    fmt="-o",
                    capsize=4,
                    markersize=8,
                    linewidth=2.5,
                    label=f"{k_val:g}",
                    color=cmap[j],
                    zorder=3,
                )

        ax.set_title(f"$b_m = {b_m_val}$", fontsize=20)
        ax.set_xlabel("Bias Towards Mutants ($f_M$)", fontsize=18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlim(-0.02, 1.02)
        prettify_ax(ax)

    axes[0].set_ylabel("Mean Mutant Fraction $\\langle\\rho_M\\rangle$", fontsize=18)
    axes[0].set_ylim(-0.02, 1.02)
    # Legend for the first panel (shared y-axis)
    axes[0].legend(
        title="$\\bf{k_{total}}$",
        fontsize=14,
        title_fontsize=16,
        frameon=False,
        loc="upper left",
    )

    plt.savefig(
        os.path.join(figures_dir, f"{CAMPAIGN_ID}_fig3_slice_vs_bias.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # == FIGURE 4: Coexistence Boundary ==
    print("Generating Figure 4: Coexistence Boundary...")
    coexistence_points = []
    
    
    for (b_m, k_total), group in agg_all_bm.groupby(["b_m", "k_total"]):
        group = group.sort_values("mean_rho_M")
        # Check if the mean_rho_M range spans across 0.5 to allow interpolation
        if group["mean_rho_M"].min() < 0.51 and group["mean_rho_M"].max() > 0.49:
            try:
                # Use 'slinear' (spline linear) for interpolation, requiring exact bounds
                interp_func = interp1d(
                    group["mean_rho_M"], group["f_M"], kind="slinear", bounds_error=True
                )
                f_M_coexist = interp_func(0.5)
                coexistence_points.append(
                    {"b_m": b_m, "k_total": k_total, "f_M_coexist": f_M_coexist}
                )
            except ValueError:
                # This can happen if 0.5 is exactly on the boundary or data is too sparse/noisy
                continue

    coexist_df = pd.DataFrame(coexistence_points)
    fig, ax = plt.subplots(figsize=(12, 8))

    for b_m_val, group in coexist_df.groupby("b_m"):
        ax.plot(
            group["k_total"],
            group["f_M_coexist"],
            "-o",
            markersize=10,
            linewidth=3,
            label=f"{b_m_val}",
            color=B_M_PALETTE.get(b_m_val, "k"),
        )

    ax.axhline(
        0.5, color="black", linestyle="--", linewidth=2.5, label="Unbiased Coexistence"
    )

    ax.set_title("Coexistence Boundary", fontsize=22)
    ax.set_xlabel("Total Switching Rate ($k_{total}$)", fontsize=18)
    ax.set_ylabel("Required Bias Towards Mutants ($f_M$)", fontsize=18)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(title="Mutant Fitness ($b_m$)", frameon=False, title_fontsize=16)
    prettify_ax(ax)

    plt.savefig(
        os.path.join(figures_dir, f"{CAMPAIGN_ID}_fig4_coexistence_boundary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # == FIGURE 5: Fitness Cost Comparison Subplots ==
    print("Generating Figure 5: Fitness Cost Comparison...")
    f_M_to_compare = ANALYSIS_PARAMS["fitness_cost_plot_f_M"]
    num_panels = len(f_M_to_compare)

    fig, axes = plt.subplots(
        1, num_panels, figsize=(8 * num_panels, 7), sharey=True, constrained_layout=True
    )
    if num_panels == 1:  # Ensure axes is always an array
        axes = [axes]

    fig.suptitle("Effect of Fitness Cost at Different Bias Levels", fontsize=24, y=1.03)

    for i, f_M_val in enumerate(f_M_to_compare):
        ax = axes[i]
        # Select data for the current f_M value
        subset_df = agg_all_bm[np.isclose(agg_all_bm["f_M"], f_M_val)]

        if subset_df.empty:  # Gracefully handle missing data for a slice
            ax.text(
                0.5,
                0.5,
                f"No data for f_M={f_M_val:.1%}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
                color="red",
            )
            ax.set_title(f"Bias Towards Mutants ($f_M$) = {f_M_val:.1%}", fontsize=20)
            ax.set_xlabel("Total Switching Rate ($k_{total}$)", fontsize=16)
            prettify_ax(ax)
            continue  # Skip plotting for this panel if no data

        for b_m_val, group in subset_df.groupby("b_m"):
            ax.errorbar(
                group["k_total"],
                group["mean_rho_M"],
                yerr=group["sem_rho_M"],
                fmt="-o",
                capsize=5,
                markersize=8,
                linewidth=2.5,
                label=f"{b_m_val}",
                color=B_M_PALETTE.get(b_m_val, "k"),
            )

        ax.set_title(f"Bias Towards Mutants ($f_M$) = {f_M_val:.1%}", fontsize=20)
        ax.set_xlabel("Total Switching Rate ($k_{total}$)", fontsize=16)
        ax.set_xscale("log")
        prettify_ax(ax)

    axes[0].legend(title="Mutant Fitness ($b_m$)", frameon=False, title_fontsize=16)
    axes[0].set_ylabel("Mean Mutant Fraction $\\langle\\rho_M\\rangle$", fontsize=18)
    axes[0].set_ylim(-0.02, 1.02)

    plt.savefig(
        os.path.join(figures_dir, f"{CAMPAIGN_ID}_fig5_fitness_cost.png"), dpi=300
    )
    plt.close()

    # == FIGURE 6: Finite-Size Scaling ==
    print("Generating Figure 6: Finite-Size Scaling...")
    # FIX: Add check for empty agg_scaling DataFrame to prevent UserWarning
    if agg_scaling.empty:
        print(
            "  Skipping Figure 6: No finite-size scaling data found in the aggregated results."
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        for (k_total, f_M), group in agg_scaling.groupby(["k_total", "f_M"]):
            group = group.sort_values("1/L")
            ax.errorbar(
                group["1/L"],
                group["mean_rho_M"],
                yerr=group["sem_rho_M"],
                fmt="-o",
                capsize=5,
                markersize=10,
                linewidth=3,
                label=f"$k_{{total}} = {k_total}$, $f_M = {f_M:.0%}$",
            )

        ax.set_xscale("log")
        ax.set_title("Finite-Size Scaling Analysis", fontsize=22)
        ax.set_xlabel("Inverse System Width (1/L)", fontsize=18)
        ax.set_ylabel("Mean Mutant Fraction ($\\langle\\rho_M\\rangle$)", fontsize=18)
        # FIX: Ensure frameon=False is passed for consistency
        ax.legend(
            fontsize=14,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=2,
            frameon=False,
        )
        prettify_ax(ax)

        plt.savefig(
            os.path.join(figures_dir, f"{CAMPAIGN_ID}_fig6_finite_size_scaling.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # == FIGURE 7: Regime Boundary Analysis ==
    print("Generating Figure 7: Regime Boundary Analysis...")
    k_star_points = []
    f_M_for_k_star = ANALYSIS_PARAMS["crossover_fit_f_M"]
    subset_for_fit = agg_all_bm[np.isclose(agg_all_bm["f_M"], f_M_for_k_star)]

    if not subset_for_fit.empty:  # Only proceed if data exists for fitting
        for b_m_val, group in subset_for_fit.groupby("b_m"):
            group = group.sort_values("k_total")
            x_data = group["k_total"].values
            y_data = group["mean_rho_M"].values / f_M_for_k_star  # Normalized rho_M

            try:
                # FIX: Set lower bound for k_c to a small positive number to avoid 0**n warning
                popt, _ = curve_fit(
                    crossover_function,
                    x_data,
                    y_data,
                    p0=[1.0, 1.0],
                    bounds=([1e-9, 0.1], [np.inf, 10.0]),
                )
                k_star_points.append(
                    {"fitness_cost": 1 - b_m_val, "k_star": popt[0], "hill_n": popt[1]}
                )
            except RuntimeError as e:
                print(
                    f"Warning: Curve fit for crossover failed for b_m = {b_m_val}. Error: {e}"
                )
    else:
        print(
            f"  Skipping Figure 7: No data found for f_M={f_M_for_k_star:.1%} for crossover fitting."
        )

    k_star_df = pd.DataFrame(k_star_points).sort_values("fitness_cost")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)
    fig.suptitle("Analysis of the Fitness-to-Mixing Crossover", fontsize=24)

    if not k_star_df.empty:  # Only plot if fitting was successful for some points
        # Panel 1: Crossover Location
        # Use a specific color from the palette for consistency (e.g., b_m=0.8's color)
        # Ensure B_M_PALETTE.get(0.8) handles cases where 0.8 is not in unique b_m values if it's not a main run.
        plot_color = B_M_PALETTE.get(0.8, sns.color_palette("colorblind")[0])
        ax1.plot(
            k_star_df["fitness_cost"],
            k_star_df["k_star"],
            "-o",
            markersize=12,
            linewidth=3,
            color=plot_color,
        )
        prettify_ax(ax1)
        ax1.set_title("Regime Boundary Location", fontsize=20)
        ax1.set_xlabel("Mutant Fitness Cost ($1 - b_m$)", fontsize=18)
        ax1.set_ylabel("Crossover Switching Rate ($k^*$)", fontsize=18)
        ax1.set_yscale("log")

        # Inset Plot - showing an example fit
        ax_inset = ax1.inset_axes(
            [0.45, 0.1, 0.5, 0.45]
        )  # x, y, width, height in axes coordinates
        example_group = subset_for_fit[np.isclose(subset_for_fit["b_m"], 0.8)]

        if (
            not example_group.empty and len(example_group) > 1
        ):  # Ensure enough data for inset fit
            x_inset = example_group["k_total"].values
            y_inset = example_group["mean_rho_M"].values

            ax_inset.plot(x_inset, y_inset, "o", c=plot_color, label="Data")

            try:
                # Fit for inset, directly on rho_M values
                popt_inset, _ = curve_fit(
                    lambda k, kc, n: f_M_for_k_star * crossover_function(k, kc, n),
                    x_inset,
                    y_inset,
                    p0=[1, 1],
                    bounds=([1e-9, 0.1], [np.inf, 10.0]),
                )
                x_smooth = np.logspace(
                    np.log10(x_inset.min()), np.log10(x_inset.max()), 200
                )
                ax_inset.plot(
                    x_smooth,
                    f_M_for_k_star * crossover_function(x_smooth, *popt_inset),
                    "k--",
                    label="Fit",
                )
                ax_inset.set_xscale("log")
                ax_inset.set_title("Example Fit ($b_m=0.8$)", fontsize=10)
                ax_inset.set_xlabel("$k_{total}$", fontsize=8)
                ax_inset.set_ylabel("$\\langle\\rho_M\\rangle$", fontsize=8)
                ax_inset.tick_params(axis="both", labelsize=8)
                ax_inset.legend(fontsize=8, frameon=False)
            except RuntimeError as e:
                print(f"Warning: Inset curve fit for b_m=0.8 failed. Error: {e}")
                ax_inset.text(
                    0.5,
                    0.5,
                    "Fit Failed",
                    ha="center",
                    va="center",
                    transform=ax_inset.transAxes,
                    fontsize=10,
                    color="red",
                )
        else:
            ax_inset.text(
                0.5,
                0.5,
                "No Inset Data",
                ha="center",
                va="center",
                transform=ax_inset.transAxes,
                fontsize=10,
                color="red",
            )

        # Panel 2: Transition Steepness
        ax2.plot(
            k_star_df["fitness_cost"],
            k_star_df["hill_n"],
            "-o",
            markersize=12,
            linewidth=3,
            color=sns.color_palette("colorblind")[1],
        )
        prettify_ax(ax2)
        ax2.set_title("Transition Steepness", fontsize=20)
        ax2.set_xlabel("Mutant Fitness Cost ($1 - b_m$)", fontsize=18)
        ax2.set_ylabel("Hill Coefficient (n)", fontsize=18)
        ax2.set_ylim(bottom=0)
    else:
        fig.text(
            0.5,
            0.5,
            "Insufficient data to generate Figure 7 plots.",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            color="red",
        )

    plt.savefig(
        os.path.join(figures_dir, f"{CAMPAIGN_ID}_fig7_crossover_analysis.png"), dpi=300
    )
    plt.close()

    print(f"\nDefinitive analysis for campaign '{CAMPAIGN_ID}' is complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)  # Print traceback to stderr
        sys.exit(1)  # Exit with an error code
