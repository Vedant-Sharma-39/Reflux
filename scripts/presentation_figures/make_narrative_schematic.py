"""
Generates the definitive introductory narrative schematic for the "Reflux" project.

This is the final, visually enhanced version with key design improvements:
- The environment sidebar is now a single, continuous bar.
- A prominent downward arrow explicitly labels the flow of Time.
- A robust GridSpec layout ensures all elements are perfectly aligned.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# --- Configuration for Aesthetics ---
FIG_SIZE = (12, 10)
OUTPUT_DIR = "presentation_figures"
OUTPUT_FILENAME = "narrative_schematic_final_v4.png"

# Official project colors
COLOR_WT = "#02343F"
COLOR_M = "#d35400"
COLOR_ENV_FAVORABLE = "#a9d6e5"
COLOR_ENV_HOSTILE = "#fbc4ab"


def draw_population_front(ax, pop_states):
    """Draws a series of thin, vertical bars representing the front's composition."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    num_cells = len(pop_states)
    if num_cells == 0:
        return

    # 4:1 ratio for bar to gap width
    total_units = num_cells * 5 - 1
    unit_width = 1.0 / total_units
    bar_width, gap_width = 4 * unit_width, 1 * unit_width
    total_drawing_width = num_cells * bar_width + (num_cells - 1) * gap_width
    start_x = (1.0 - total_drawing_width) / 2.0

    pop_colors = {"WT": COLOR_WT, "M": COLOR_M}
    for i, state in enumerate(pop_states):
        ax.add_patch(
            plt.Rectangle(
                (start_x + i * (bar_width + gap_width), 0.2),
                bar_width,
                0.6,
                facecolor=pop_colors[state],
                edgecolor="none",
            )
        )


def draw_continuous_sidebar(ax, env_patches):
    """Draws a single continuous bar for the environment with a time arrow."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)  # Use a 0-3 vertical coordinate system for the 3 patches
    ax.axis("off")

    num_patches = len(env_patches)
    for i, (state, time_label, patch_label) in enumerate(env_patches):
        y_bottom = (
            num_patches - 1
        ) - i  # Patches are drawn from top (y=2) to bottom (y=0)
        color = COLOR_ENV_FAVORABLE if state == "WT_fav" else COLOR_ENV_HOSTILE

        # Draw the colored rectangle for this segment
        ax.add_patch(
            plt.Rectangle(
                (0.1, y_bottom),
                0.4,
                1,  # x, y, width, height
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )

        # Add text inside the segment
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

    # Add the single downward "Time" arrow
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
    print(
        f"Generating visually enhanced schematic to '{OUTPUT_DIR}/{OUTPUT_FILENAME}'..."
    )

    fig = plt.figure(figsize=FIG_SIZE)
    fig.suptitle(
        "Reversibility Enables Adaptation in a Fluctuating Environment",
        fontsize=22,
        y=0.98,
        family="sans-serif",
    )

    # --- Use GridSpec for a robust, professional layout ---
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 5, 5], hspace=0.8, wspace=0.3)

    # Create the single, tall axis for the environment sidebar
    ax_env = fig.add_subplot(gs[:, 0])

    # Create the 6 axes for the population fronts
    axes_pop = [[fig.add_subplot(gs[i, j + 1]) for j in range(2)] for i in range(3)]

    # --- Define Population States for the Front ---
    np.random.seed(42)  # for reproducible random regeneration
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

    # --- Draw all population fronts ---
    for i in range(3):
        for j in range(2):
            draw_population_front(axes_pop[i][j], pop_states_all[i][j])

    # --- Draw Environment Sidebar ---
    env_patches = [
        ("WT_fav", "Time = $t_1$", "WT-\nFavorable"),
        ("M_fav", "Time = $t_2$", "Hostile\nto WT"),
        ("WT_fav", "Time = $t_3$", "WT-\nFavorable"),
    ]
    draw_continuous_sidebar(ax_env, env_patches)

    # --- Add All Annotations, Anchored to Axes ---
    ax_env.set_title("Environment", fontsize=14, weight="bold", pad=20)
    axes_pop[0][0].set_title(
        "Irreversible Switching ($\phi = -1.0$)", fontsize=16, pad=20
    )
    axes_pop[0][1].set_title("Reversible Switching ($\phi = 0.0$)", fontsize=16, pad=20)

    # Mechanism text
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

    # Descriptive text
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
        color="red",
        transform=axes_pop[2][0].transAxes,
    )
    axes_pop[2][1].text(
        0.5,
        -0.2,
        "M→WT switching regenerates a new WT sector,\nenabling successful re-adaptation.",
        ha="center",
        va="top",
        fontsize=12,
        color="green",
        transform=axes_pop[2][1].transAxes,
    )

    # Adjust main figure layout
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Visually enhanced schematic saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
