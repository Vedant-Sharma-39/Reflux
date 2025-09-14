"""
Supplementary Figure: Initial Patch Generation Method (Polished & Corrected Version)

This script creates a polished, explanatory diagram detailing the method for generating
initial 1D spatial patterns with a fixed number of mutants but variable fragmentation.

This version is optimized for publication with improved layout, font sizes, and
vector graphics output (PDF/EPS).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib

# --- Publication Settings ---
# Ensure fonts are embedded in vector graphics files for journal compatibility
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# --- End Publication Settings ---


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_grf_threshold_pattern(
    width: int, num_mutants: int, correlation_length: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generates a 1D spatial pattern by thresholding a Correlated Gaussian Random Field.
    This version robustly selects the top N indices to create the pattern.
    """
    if correlation_length <= 0:
        raise ValueError("correlation_length must be positive.")
    if num_mutants == 0:
        return np.zeros(width, dtype=int), np.zeros(width), np.inf
    if num_mutants >= width:
        return np.ones(width, dtype=int), np.zeros(width), -np.inf

    # Generate the correlated random field in Fourier space
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

    # Robustly find the indices of the `num_mutants` largest values
    top_indices = np.argsort(grf)[-num_mutants:]
    pattern = np.zeros(width, dtype=int)
    pattern[top_indices] = 1

    # The threshold value is still useful for visualization
    threshold_value = grf[top_indices[0]] if num_mutants > 0 else np.inf

    return pattern, grf, threshold_value


def create_exploratory_figure(width: int, num_mutants: int):
    """
    Creates and saves the polished, annotated figure explaining the generation process.
    """
    print("Generating polished exploratory figure for initial patch generation...")

    # --- Aesthetic Choices ---
    COLOR_BG = "#FFFFFF"  # Clean white background
    COLOR_FIELD = "#495057"
    COLOR_THRESHOLD = "#e63946"
    COLOR_FILL = "#6c757d"
    COLOR_PATCH_WT = "#ffffff"
    COLOR_PATCH_M = "#212529"

    scenarios = {"Clumped": 100.0, "Intermediate": 8.0, "Fragmented": 0.5}

    # --- CHANGE: Plotting setup for publication quality ---
    fig = plt.figure(figsize=(12, 8), dpi=300)  # A4 width in inches, high DPI
    fig.set_facecolor(COLOR_BG)
    # --- CHANGE: Added hspace for vertical separation between panels ---
    gs = gridspec.GridSpec(len(scenarios), 1, figure=fig, hspace=0.6)

    fig.suptitle(
        "How Initial Patterns are Generated", fontsize=12, weight="bold", y=1.08
    )
    fig.text(
        0.5,
        1.0,
        "Method: Correlated Gaussian Random Field with Rank-Thresholding",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color=COLOR_FIELD,
    )

    all_axes = []
    final_pattern_for_label = None

    for i, (label, corr_len) in enumerate(scenarios.items()):
        pattern, grf, threshold = generate_grf_threshold_pattern(
            width, num_mutants, corr_len
        )
        final_pattern_for_label = pattern

        # Use a nested GridSpec for each row (GRF plot + binary pattern plot)
        row_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[i], height_ratios=[3, 1], hspace=0.0
        )
        ax_grf = fig.add_subplot(row_gs[0])
        ax_grf.set_facecolor(COLOR_BG)
        ax_grf.plot(grf, color=COLOR_FIELD, linewidth=1.0, alpha=0.9)
        ax_grf.axhline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.5)
        ax_grf.fill_between(
            np.arange(width),
            grf,
            threshold,
            where=grf >= threshold,
            color=COLOR_FILL,
            alpha=0.8,
            interpolate=True,
        )

        # --- CHANGE: Adjusted font and tick sizes for clarity ---
        ax_grf.set_ylabel("Field Value", fontsize=8)
        ax_grf.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_grf.tick_params(axis="y", labelsize=7)
        ax_grf.spines[["top", "right"]].set_visible(False)
        ax_grf.text(
            0.01,
            1.05,  # Positioned slightly above the plot area
            f"Correlation Length = {corr_len} ({label})",
            transform=ax_grf.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
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

    # Calculate number of fragments for the last (most fragmented) pattern
    num_fragments = 1 + np.sum(
        final_pattern_for_label[1:] != final_pattern_for_label[:-1]
    )
    all_axes[-1][1].set_xlabel(
        f"Spatial Position (Total Fragments: {num_fragments})", fontsize=8
    )
    all_axes[-1][1].tick_params(axis="x", bottom=True, labelbottom=True, labelsize=7)

    # --- CHANGE: Adjusted font sizes for annotations ---
    all_axes[0][0].text(
        0.05,
        0.85,
        "1. Generate a continuous\n    'suitability' field",
        transform=all_axes[0][0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        weight="bold",
        color=COLOR_FIELD,
        bbox=dict(facecolor=COLOR_BG, alpha=0.8, ec="none", pad=0.1),
    )
    all_axes[1][0].text(
        0.95,
        0.95,
        "2. Apply a rank-based\n    threshold to select the\n    top N sites",
        transform=all_axes[1][0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        weight="bold",
        color=COLOR_THRESHOLD,
        bbox=dict(facecolor=COLOR_BG, alpha=0.8, ec="none", pad=0.1),
    )
    all_axes[2][0].text(
        0.05,
        0.15,
        "3. Create the final\n    binary pattern",
        transform=all_axes[2][0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        weight="bold",
        color=COLOR_PATCH_M,
        bbox=dict(facecolor=COLOR_BG, alpha=0.8, ec="none", pad=0.1),
    )

    # --- CHANGE: Save to multiple high-quality formats ---
    output_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path_pdf = os.path.join(output_dir, "sup_fig_fragmentation_method.pdf")
    output_path_eps = os.path.join(output_dir, "sup_fig_fragmentation_method.eps")

    plt.savefig(output_path_pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(output_path_eps, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Polished figure saved to: '{output_path_pdf}' and '{output_path_eps}'")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    WIDTH = 512
    NUM_MUTANTS = int(WIDTH * 0.25)  # e.g., 25% mutants
    create_exploratory_figure(width=WIDTH, num_mutants=NUM_MUTANTS)
