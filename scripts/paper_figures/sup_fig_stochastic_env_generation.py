"""
Supplementary Figure: Stochastic Environment Generation Method

This script visualizes the process for generating 1D "scrambled" environments
with stochastically determined patch widths drawn from a Gamma distribution.

The figure illustrates how tuning the Fano factor of the Gamma distribution,
while keeping the mean patch width constant, allows for the creation of
environments with different degrees of spatial heterogeneity. This ranges from
Poisson-like regularity (Fano=1) to highly clustered, heavy-tailed patterns
(large Fano factor).

The three-step process is shown for each example:
1. The underlying probability distribution (Gamma PDF).
2. A single, concrete draw of patch widths from that distribution.
3. The final assembled 1D environment used in simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random


def _generate_stochastic_sequence(
    mean_width: float, fano_factor: float, total_length: int
) -> list[int]:
    """
    Generates a list of patch widths from a Gamma distribution.

    Args:
        mean_width (float): The target mean of the patch widths.
        fano_factor (float): The Fano factor (Variance / Mean). Controls clustering.
        total_length (int): The total length of the sequence to generate.

    Returns:
        list[int]: A list of integer patch widths that sum to at least total_length.
    """
    if mean_width <= 0 or fano_factor <= 0:
        raise ValueError("Mean width and Fano factor must be positive.")

    # Relationship for Gamma distribution:
    # Mean = shape * scale
    # Variance = shape * scale^2
    # Fano Factor = Variance / Mean = scale
    scale = fano_factor
    shape = mean_width / scale

    patch_widths = []
    current_length = 0
    while current_length < total_length:
        # Draw from the Gamma distribution
        width = np.random.gamma(shape, scale)
        # Ensure patch width is at least 1
        int_width = max(1, int(round(width)))
        patch_widths.append(int_width)
        current_length += int_width

    return patch_widths


def create_supplementary_figure(width: int, mean_patch_width: int):
    """
    Creates and saves the exploratory figure detailing the generation process.
    """
    print("Generating supplementary figure for stochastic environment generation...")

    # Define the Fano factors to showcase different levels of clustering
    fano_factors_to_show = [1.0, 10.0, 60.0]
    labels = [
        "Poisson-like (Fano=1.0)",
        "Moderately Clustered (Fano=10.0)",
        "Highly Clustered (Fano=60.0)",
    ]

    # Create figure layout
    fig, axes = plt.subplots(
        len(fano_factors_to_show), 3, figsize=(15, 10), constrained_layout=True
    )
    fig.suptitle(
        "Generation of Stochastically Scrambled Environments", fontsize=22, y=1.05
    )

    for i, (fano, label) in enumerate(zip(fano_factors_to_show, labels)):
        ax_pdf, ax_hist, ax_pattern = axes[i]

        # --- 1. Generate the data for this row ---
        patch_widths = _generate_stochastic_sequence(mean_patch_width, fano, width)

        # --- 2. Plot the theoretical PDF ---
        scale = fano
        shape = mean_patch_width / scale
        x = np.linspace(0, max(patch_widths) * 1.1, 500)
        pdf = stats.gamma.pdf(x, a=shape, scale=scale)

        ax_pdf.plot(x, pdf, color="darkred", lw=2.5)
        ax_pdf.fill_between(x, pdf, color="red", alpha=0.1)
        ax_pdf.set_title(f"(A) Probability Distribution\n{label}", fontsize=14)
        ax_pdf.set_xlabel("Patch Width")
        ax_pdf.set_ylabel("Probability Density")
        ax_pdf.grid(True, ls=":", alpha=0.6)

        # --- 3. Plot the histogram of the actual draw ---
        ax_hist.hist(patch_widths, bins=20, color="navy", alpha=0.7, edgecolor="white")
        ax_hist.axvline(
            mean_patch_width,
            color="darkred",
            ls="--",
            label=f"Target Mean ({mean_patch_width})",
        )
        ax_hist.axvline(
            np.mean(patch_widths),
            color="cyan",
            ls="--",
            label=f"Actual Mean ({np.mean(patch_widths):.1f})",
        )
        ax_hist.set_title(
            f"(B) Single Random Draw\n({len(patch_widths)} patches generated)",
            fontsize=14,
        )
        ax_hist.set_xlabel("Patch Width")
        ax_hist.set_ylabel("Count")
        ax_hist.legend()
        ax_hist.grid(True, ls=":", alpha=0.6)

        # --- 4. Plot the final assembled 1D pattern ---
        pattern = []
        # We have two patch types (e.g., WT-favored, M-favored)
        patch_types = [0, 1]
        for width_val in patch_widths:
            # Alternate between the two types
            patch_type = random.choice(patch_types)
            pattern.extend([patch_type] * width_val)

        # Trim to the exact simulation width
        final_pattern = np.array(pattern[:width]).reshape(1, -1)

        # Use a colormap where 0 is white and 1 is black
        ax_pattern.imshow(
            final_pattern, cmap="binary", interpolation="nearest", aspect="auto"
        )
        ax_pattern.set_title("(C) Resulting 1D Spatial Pattern", fontsize=14)
        ax_pattern.set_yticks([])
        ax_pattern.set_xticks([])

    # Save the figure
    output_filename = "sup_fig_stochastic_env_generation.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Figure saved as '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    # Parameters for the visualization
    WIDTH = 1024  # A representative total length for the environment
    MEAN_PATCH_WIDTH = 60  # The target average patch size

    create_supplementary_figure(width=WIDTH, mean_patch_width=MEAN_PATCH_WIDTH)
