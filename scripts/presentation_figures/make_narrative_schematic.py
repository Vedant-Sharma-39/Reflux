"""
Generates the definitive introductory narrative schematic for the "Reflux" project.

This is the final version, combining the clear narrative layout and annotations of the
original schematic with the superior hexagonal cell visuals for a publication-quality figure.

Key Design Features:
- Hexagonal Cells: The population front is rendered with distinct hexagons.
- Continuous Environment Sidebar: A single, colored bar on the left clearly
  indicates the environmental state at each time point.
- Explicit Annotations: All titles and descriptive text from the original schematic
  are restored to guide the viewer through the narrative.
- Robust GridSpec Layout: Ensures all elements, including the sidebar and plot matrix,
  are perfectly aligned.
- Professional Typography: Uses a clean, sans-serif font suitable for publications.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import matplotlib

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
]


# --- Configuration for Aesthetics ---
FIG_SIZE = (12, 10)
OUTPUT_DIR = "presentation_figures"
OUTPUT_FILENAME = "narrative_schematic_hex_with_sidebar.png"

# Official project colors (using the more saturated originals as requested)
COLOR_WT = "#02343F"
COLOR_M = "#d35400"
COLOR_ENV_FAVORABLE = "#a9d6e5"
COLOR_ENV_HOSTILE = "#fbc4ab"
COLOR_SUCCESS_TEXT = "green"
COLOR_FAILURE_TEXT = "red"


def _get_hex_corners(center_x, center_y, size=1.0):
    """Calculates the 6 vertices of a flat-topped hexagon."""
    corners = []
    for i in range(6):
        angle_rad = np.pi / 180 * (60 * i + 30)
        corners.append(
            (center_x + size * np.cos(angle_rad), center_y + size * np.sin(angle_rad))
        )
    return np.array(corners)


def draw_hexagon_front(ax, pop_states, hex_size=0.08):
    """Draws a horizontal line of hexagons, adapted for a generic axes."""
    ax.clear()
    ax.set_ylim(0, 1)  # Use a standard 0-1 coordinate system
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.set_aspect("auto")  # Allow axes to fit the GridSpec cell

    num_cells = len(pop_states)
    if num_cells == 0:
        return

    hex_width = hex_size * np.sqrt(3)
    total_width = num_cells * hex_width
    start_x = (1.0 - total_width) / 2.0  # Center the whole front

    pop_colors = {"WT": COLOR_WT, "M": COLOR_M}
    for i, state in enumerate(pop_states):
        center_x = start_x + (i + 0.5) * hex_width
        corners = _get_hex_corners(
            center_x, 0.5, hex_size
        )  # Center vertically in the axes
        polygon = plt.Polygon(
            corners, facecolor=pop_colors[state], edgecolor="black", lw=0.3
        )
        ax.add_patch(polygon)


def draw_continuous_sidebar(ax, env_patches):
    """Draws a single continuous bar for the environment with a time arrow."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)  # Use a 0-3 vertical coordinate system for the 3 patches
    ax.axis("off")

    num_patches = len(env_patches)
    for i, (state, time_label, patch_label) in enumerate(env_patches):
        y_bottom = (num_patches - 1) - i
        color = COLOR_ENV_FAVORABLE if state == "WT_fav" else COLOR_ENV_HOSTILE

        ax.add_patch(
            plt.Rectangle(
                (0.1, y_bottom),
                0.4,
                1,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )
        ax.text(
            0.3,
            y_bottom + 0.75,
            time_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
        )
        ax.text(
            0.3,
            y_bottom + 0.3,
            patch_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
        )

    ax.annotate(
        "Time",
        xy=(0.8, 0.1),
        xytext=(0.8, 2.9),
        arrowprops=dict(arrowstyle="->", facecolor="black", lw=2.5),
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )


def main():
    print(f"Generating final schematic to '{OUTPUT_DIR}/{OUTPUT_FILENAME}'...")

    fig = plt.figure(figsize=FIG_SIZE)
    fig.suptitle(
        "Reversibility Enables Adaptation in a Fluctuating Environment",
        fontsize=22,
        y=0.98,
    )

    # --- Use the robust 3x3 GridSpec layout from the original script ---
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 5, 5], hspace=0.8, wspace=0.3)

    # Create the single, tall axis for the environment sidebar
    ax_env = fig.add_subplot(gs[:, 0])

    # Create the 6 axes for the population fronts
    axes_pop = [[fig.add_subplot(gs[i, j + 1]) for j in range(2)] for i in range(3)]

    # --- Define Population States (using original counts) ---
    np.random.seed(42)
    pop_t1 = ["WT"] * 12 + ["M"] * 8 + ["WT"] * 20
    pop_t2_state = ["M"] * 40
    pop_irreversible_t3 = ["M"] * 40
    pop_reversible_t3 = ["M"] * 40
    num_regenerated = 6
    regenerated_indices = np.random.choice(
        range(len(pop_reversible_t3)), num_regenerated, replace=False
    )
    for idx in regenerated_indices:
        pop_reversible_t3[idx] = "WT"

    pop_states_all = [
        [pop_t1, pop_t1],
        [pop_t2_state, pop_t2_state],
        [pop_irreversible_t3, pop_reversible_t3],
    ]

    # --- Draw all population fronts using HEXAGONS ---
    for i in range(3):
        for j in range(2):
            draw_hexagon_front(axes_pop[i][j], pop_states_all[i][j], hex_size=0.1)

    # --- Draw Environment Sidebar (from original script) ---
    env_patches = [
        ("WT_fav", "Time = $t_1$", "WT-\nFavorable"),
        ("M_fav", "Time = $t_2$", "Hostile\nto WT"),
        ("WT_fav", "Time = $t_3$", "WT-\nFavorable"),
    ]
    draw_continuous_sidebar(ax_env, env_patches)

    # --- Add All Annotations from the Original Script ---
    ax_env.set_title("Environment", fontsize=14, weight="bold", pad=20)
    axes_pop[0][0].set_title(
        "Irreversible Switching ($\phi = -1.0$)", fontsize=16, pad=20
    )
    axes_pop[0][1].set_title("Reversible Switching ($\phi = 0.0$)", fontsize=16, pad=20)

    # Mechanism text (placed above the second row)
    axes_pop[1][0].text(
        0.5,
        1.3,
        "WT $\\longrightarrow$ M",
        ha="center",
        va="center",
        fontsize=18,
        color=COLOR_M,
        transform=axes_pop[1][0].transAxes,
    )
    axes_pop[1][1].text(
        0.5,
        1.3,
        "WT $\\longleftrightarrow$ M",
        ha="center",
        va="center",
        fontsize=18,
        color="black",
        transform=axes_pop[1][1].transAxes,
    )

    # Descriptive text (placed below each plot)
    axes_pop[0][0].text(
        0.5,
        -0.2,
        "Population with WT and Mutant sectors.",
        ha="center",
        va="top",
        fontsize=12,
        transform=axes_pop[0][0].transAxes,
    )
    axes_pop[0][1].text(
        0.5,
        -0.2,
        "Population with WT and Mutant sectors.",
        ha="center",
        va="top",
        fontsize=12,
        transform=axes_pop[0][1].transAxes,
    )
    axes_pop[1][0].text(
        0.5,
        -0.2,
        "Hostile patch eliminates the WT sector.",
        ha="center",
        va="top",
        fontsize=12,
        transform=axes_pop[1][0].transAxes,
    )
    axes_pop[1][1].text(
        0.5,
        -0.2,
        "Hostile patch eliminates the WT sector.",
        ha="center",
        va="top",
        fontsize=12,
        transform=axes_pop[1][1].transAxes,
    )
    axes_pop[2][0].text(
        0.5,
        -0.2,
        "WT is permanently lost.\nNo M→WT pathway exists to recover.",
        ha="center",
        va="top",
        fontsize=12,
        color=COLOR_FAILURE_TEXT,
        transform=axes_pop[2][0].transAxes,
    )
    axes_pop[2][1].text(
        0.5,
        -0.2,
        "M→WT switching regenerates a new WT sector,\nenabling successful re-adaptation.",
        ha="center",
        va="top",
        fontsize=12,
        color=COLOR_SUCCESS_TEXT,
        transform=axes_pop[2][1].transAxes,
    )

    # Adjust main figure layout
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Final schematic with hexagons and sidebar saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
