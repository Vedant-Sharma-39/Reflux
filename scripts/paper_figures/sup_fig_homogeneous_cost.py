import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Matplotlib and Seaborn Style ---
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
FIG_DPI = 300
COOLWARM = sns.color_palette("coolwarm", as_cmap=True)
MAKO_R = sns.color_palette("mako_r", as_cmap=True)


def get_project_root():
    """Find the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_and_prepare_data(campaign_id: str, project_root: str) -> pd.DataFrame:
    """Loads and prepares the summary data for plotting."""
    summary_path = os.path.join(
        project_root,
        "data",
        campaign_id,
        "analysis",
        f"{campaign_id}_summary_aggregated.csv",
    )
    print(f"Loading data from: {os.path.basename(summary_path)}")
    try:
        df = pd.read_csv(summary_path)
        if df.empty:
            raise FileNotFoundError  # Treat empty file as not found
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(
            f"Error: Data not found or empty for campaign '{campaign_id}'. Cannot generate figure."
        )
        sys.exit(1)

    # --- Data Processing and Normalization ---
    df["s"] = np.round(df["b_m"] - 1.0, 2)

    # Robustly calculate the non-switching (k=0) speed for normalization
    # This is the baseline front speed for each selection coefficient 's'.
    v_k0_map = df[df["k_total"] == 0].set_index("s")["avg_front_speed"]
    if v_k0_map.empty:
        print("Error: No data found for k_total=0. Cannot calculate relative speed.")
        sys.exit(1)

    df["v_k0"] = df["s"].map(v_k0_map)

    # Check for missing baseline data
    if df["v_k0"].isnull().any():
        missing_s = df[df["v_k0"].isnull()]["s"].unique()
        print(
            f"Warning: Missing k_total=0 data for s values: {missing_s}. Rows will be dropped."
        )
        df.dropna(subset=["v_k0"], inplace=True)

    # Calculate the key metric: relative front speed
    df["relative_speed"] = df["avg_front_speed"] / df["v_k0"]

    # Filter out the k=0 data points for plotting, as they are now the baseline (y=1)
    return df[df["k_total"] > 0].copy()


def plot_improved_original(df: pd.DataFrame, output_path: str):
    """
    Visualization A: An improved version of the original concept.
    Facets by selection coefficient 's', highlighting the phi=0 case.
    """
    print("Generating 'Improved Original' plot (faceted by selection)...")

    # Separate the unbiased (phi=0) data to use as a reference line
    df_phi0 = df[df["phi"] == 0.0]
    df_biased = df[df["phi"] != 0.0]

    # Ensure phi=0 data exists to be plotted
    has_phi0 = not df_phi0.empty
    if not has_phi0:
        print("Warning: No data for phi=0. Cannot plot reference line.")

    g = sns.relplot(
        data=df_biased,
        x="k_total",
        y="relative_speed",
        hue="phi",
        col="s",
        kind="line",
        palette=COOLWARM,  # Corrected palette
        marker="o",
        height=4,
        aspect=1,
        col_wrap=5,
        facet_kws={"margin_titles": True},
        legend="full",
    )

    # Add the phi=0 reference line to each facet
    if has_phi0:
        for s_val, ax in g.axes_dict.items():
            data = df_phi0[df_phi0["s"] == s_val]
            if not data.empty:
                ax.plot(
                    data["k_total"],
                    data["relative_speed"],
                    color="black",
                    ls="--",
                    lw=2.5,
                    zorder=10,  # Ensure it's on top
                    label=r"$\phi=0$ (Unbiased)" if ax == g.axes.flatten()[0] else "",
                )

    # --- Aesthetics and Labels ---
    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels(r"Relative Speed, $v / v_{k=0}$")
    g.set_titles(col_template=r"$s = {col_name:.2f}$")
    g.set(xscale="log")
    g.fig.suptitle(
        "Fitness Cost of Switching: Effect of Bias (Faceted by Selection)",
        y=1.05,
        fontsize=20,
    )
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.01, 1))

    for ax in g.axes.flatten():
        ax.axhline(1.0, ls=":", color="gray", zorder=0)

    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


def plot_narrative_overhaul(df: pd.DataFrame, output_path: str):
    """
    Visualization B: A narrative overhaul of the plot.
    Facets by switching bias 'phi', highlighting the effect of selection 's'.
    """
    print("Generating 'Narrative Overhaul' plot (faceted by bias)...")

    phi_order = sorted(df["phi"].unique())

    g = sns.relplot(
        data=df,
        x="k_total",
        y="relative_speed",
        hue="s",
        col="phi",
        col_wrap=5,
        col_order=phi_order,
        kind="line",
        palette=MAKO_R,  # Palette for sequential data 's'
        marker="o",
        height=4,
        aspect=1,
        facet_kws={"margin_titles": True},
        legend="full",
    )

    # --- Aesthetics and Labels ---
    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels(r"Relative Speed, $v / v_{k=0}$")
    g.set_titles(col_template=r"$\phi = {col_name:.2f}$")
    g.set(xscale="log")
    g.fig.suptitle(
        "Fitness Cost of Switching: Effect of Selection (Faceted by Bias)",
        y=1.05,
        fontsize=20,
    )

    # Format the legend for 's'
    g.legend.set_title(r"Selection, $s$")
    # You can customize legend labels if needed, e.g., for better formatting
    # for t in g.legend.texts: t.set_text(f"{float(t.get_text()):.2f}")

    sns.move_legend(g, "upper left", bbox_to_anchor=(1.01, 1))

    for ax in g.axes.flatten():
        ax.axhline(1.0, ls=":", color="gray", zorder=0)

    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure: Fitness Cost in Homogeneous Environments."
    )
    parser.add_argument(
        "campaign_id",
        default="sup_homogeneous_cost",
        nargs="?",
        help="Campaign ID for the homogeneous cost experiment.",
    )
    args = parser.parse_args()
    project_root = get_project_root()

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    df_plot = load_and_prepare_data(args.campaign_id, project_root)

    # --- Generate and save the two figures ---
    path_a = os.path.join(output_dir, "sup_fig_homogeneous_cost_A_by_selection.png")
    plot_improved_original(df_plot, path_a)

    print("-" * 50)

    path_b = os.path.join(output_dir, "sup_fig_homogeneous_cost_B_by_bias.png")
    plot_narrative_overhaul(df_plot, path_b)


if __name__ == "__main__":
    main()
