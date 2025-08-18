"""
Supplementary Figure: Initial Patch Generation Method (Polished & Corrected Version)

This script creates a polished, explanatory diagram detailing the method for generating
initial 1D spatial patterns with a fixed number of mutants but variable fragmentation.

This version corrects an IndexError and improves the robustness of the generator.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# --- Add project root to path for consistency ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_grf_threshold_pattern(
    width: int, num_mutants: int, correlation_length: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generates a 1D spatial pattern by thresholding a Correlated Gaussian Random Field.
    This version is more robust, directly selecting the top N indices to avoid
    floating point tie issues at the threshold.
    """
    if correlation_length <= 0:
        raise ValueError("correlation_length must be positive.")
    if num_mutants == 0:
        return np.zeros(width, dtype=int), np.zeros(width), np.inf
    if num_mutants >= width:
        return np.ones(width, dtype=int), np.zeros(width), -np.inf

    # Step 1: Generate the correlated random field
    freqs = np.fft.fftfreq(width)
    power_spectrum = np.exp(-0.5 * (freqs * correlation_length) ** 2)
    noise_freq = np.random.randn(width) + 1j * np.random.randn(width)
    if width > 0:
        noise_freq[0] = 0
        for i in range(1, width // 2 + 1):
            if i < width - i:
                noise_freq[width - i] = np.conj(noise_freq[i])

    field_freq = noise_freq * np.sqrt(power_spectrum)
    grf = np.real(np.fft.ifft(field_freq))

    # --- ROBUSTNESS FIX ---
    # Directly find the indices of the `num_mutants` largest values.
    # This is simpler and avoids all issues with ties at the threshold.
    top_indices = np.argsort(grf)[-num_mutants:]

    # Create the pattern from these indices
    pattern = np.zeros(width, dtype=int)
    pattern[top_indices] = 1

    # The threshold is still useful for visualization
    threshold_value = grf[top_indices[0]] if num_mutants > 0 else np.inf
    # --- END OF ROBUSTNESS FIX ---

    return pattern, grf, threshold_value


def create_exploratory_figure(width: int, num_mutants: int):
    """
    Creates and saves the polished, annotated figure explaining the generation process.
    """
    print("Generating polished exploratory figure for initial patch generation...")

    # --- Aesthetic Choices to Match the Target Image ---
    COLOR_BG = "#f8f9fa"  # Very light gray background
    COLOR_FIELD = "#495057"  # Dark gray for the GRF line
    COLOR_THRESHOLD = "#e63946"  # Bright red for the threshold
    COLOR_FILL = "#6c757d"  # Muted gray for the fill area
    COLOR_PATCH_WT = "#ffffff"  # White for wild-type
    COLOR_PATCH_M = "#212529"  # Near-black for mutant

    scenarios = {"Clumped": 100.0, "Intermediate": 8.0, "Fragmented": 0.5}

    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
    fig.set_facecolor(COLOR_BG)

    gs = gridspec.GridSpec(len(scenarios), 1, figure=fig)

    fig.suptitle(
        "How Initial Patterns are Generated", fontsize=24, weight="bold", y=1.06
    )
    fig.text(
        0.5,
        1.0,
        "Method: Correlated Gaussian Random Field with Rank-Thresholding",
        ha="center",
        va="center",
        fontsize=16,
        style="italic",
        color=COLOR_FIELD,
    )

    all_axes = []
    # This variable will hold the last generated pattern for the final label
    final_pattern_for_label = None

    for i, (label, corr_len) in enumerate(scenarios.items()):
        pattern, grf, threshold = generate_grf_threshold_pattern(
            width, num_mutants, corr_len
        )
        final_pattern_for_label = pattern

        row_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[i], height_ratios=[3, 1], hspace=0.0
        )

        ax_grf = fig.add_subplot(row_gs[0])
        ax_grf.set_facecolor(COLOR_BG)
        ax_grf.plot(grf, color=COLOR_FIELD, linewidth=1.2, alpha=0.9)
        ax_grf.axhline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=2)
        ax_grf.fill_between(
            np.arange(width),
            grf,
            threshold,
            where=grf >= threshold,
            color=COLOR_FILL,
            alpha=0.8,
            interpolate=True,
        )

        ax_grf.set_ylabel("Field Value", fontsize=12)
        ax_grf.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_grf.spines[["top", "right"]].set_visible(False)
        ax_grf.text(
            -0.01,
            1.0,
            f" Correlation Length = {corr_len} ({label})",
            transform=ax_grf.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            weight="bold",
        )

        ax_pattern = fig.add_subplot(row_gs[1], sharex=ax_grf)
        custom_cmap = ListedColormap([COLOR_PATCH_WT, COLOR_PATCH_M])
        ax_pattern.imshow(
            pattern.reshape(1, -1),
            cmap=custom_cmap,
            interpolation="nearest",
            aspect="auto",
        )
        ax_pattern.set_yticks([])
        ax_pattern.spines[:].set_visible(False)

        all_axes.append((ax_grf, ax_pattern))

    # --- Final Touches and Annotations ---
    # --- INDEXERROR FIX IS HERE ---
    # We now use the `final_pattern_for_label` variable which is guaranteed to exist.
    num_fragments = 1 + np.sum(
        final_pattern_for_label[1:] != final_pattern_for_label[:-1]
    )
    all_axes[-1][1].set_xlabel(
        f"Spatial Position (Total Fragments: {num_fragments})", fontsize=12
    )
    # --- END OF FIX ---

    all_axes[-1][1].tick_params(axis="x", bottom=True, labelbottom=True)

    # Add the numbered annotations directly onto the plots
    all_axes[0][0].text(
        0.01,
        0.85,
        "1. Generate a continuous\n    'suitability' field",
        transform=all_axes[0][0].transAxes,
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
        color=COLOR_FIELD,
        bbox=dict(facecolor=COLOR_BG, alpha=0.7, ec="none", pad=0.1),
    )

    all_axes[1][0].text(
        0.99,
        0.95,
        "2. Apply a rank-based\n    threshold to select the\n    top N sites",
        transform=all_axes[1][0].transAxes,
        ha="right",
        va="top",
        fontsize=14,
        weight="bold",
        color=COLOR_THRESHOLD,
        bbox=dict(facecolor=COLOR_BG, alpha=0.7, ec="none", pad=0.1),
    )

    all_axes[2][0].text(
        0.01,
        0.15,
        "3. Create the final\n    binary pattern",
        transform=all_axes[2][0].transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        weight="bold",
        color=COLOR_PATCH_M,
        bbox=dict(facecolor=COLOR_BG, alpha=0.7, ec="none", pad=0.1),
    )

    # Save the figure
    output_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "sup_fig_fragmentation_method.png")

    plt.savefig(
        output_filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    print(f"Polished exploratory figure saved to: '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    WIDTH = 512
    NUM_MUTANTS = 128
    create_exploratory_figure(width=WIDTH, num_mutants=NUM_MUTANTS)
