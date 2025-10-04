# FILE: scripts/visualize_aif_data.py
#
# A robust, standalone debugging tool to visualize a saved AIF population state
# using the official HexPlotter for a consistent, high-quality visual style.

import sys
import json
import gzip
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Add project root to Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the necessary classes from the project's source code
from src.core.hex_utils import Hex, HexPlotter

# --- Configuration ---
# Point this to the data file you want to inspect
DATA_FILE_PATH = (
    PROJECT_ROOT / "figures" / "debug_runs" / "aif_multisector_final_pop.json.gz"
)
OUTPUT_IMAGE_PATH = (
    PROJECT_ROOT / "figures" / "debug_runs" / "manual_colony_visualization.png"
)

# Define the colormap using the integer types required by HexPlotter
AIF_COLORMAP = {
    1: "#e5b105",  # Susceptible (Orange)
    2: "#e63946",  # Resistant (Red)
    3: "#457b9d",  # Compensated (Blue)
}


def load_population_data_as_hex_dict(file_path: Path) -> dict:
    """
    Loads the saved population data and converts it into the Dict[Hex, int]
    format required by HexPlotter.
    """
    if not file_path.exists():
        sys.exit(
            f"ERROR: Data file not found at '{file_path}'.\n"
            "Please run 'scripts/debug_aif_sector_analysis.py' to generate it first."
        )

    print(f"Loading population data from: {file_path.name}")
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        pop_data_list = json.load(f)

    population_dict = {}
    for cell_data in pop_data_list:
        q, r, cell_type = cell_data["q"], cell_data["r"], cell_data["type"]
        # The third hex coordinate 's' is derived from the invariant q+r+s=0
        h = Hex(q, r, -q - r)
        population_dict[h] = cell_type

    return population_dict


def center_radial_plot(plotter: HexPlotter, population: dict):
    """
    Calculates the colony's bounding box and centers the plot view,
    overriding the default linear-front plotting behavior of HexPlotter.
    """
    if not population or not plotter:
        return

    # Convert all hex coordinates to cartesian points to find the bounds
    points = np.array([plotter._axial_to_cartesian(h) for h in population.keys()])

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Calculate the center and the required radius for the viewport
    center = (min_coords + max_coords) / 2.0
    span = max_coords - min_coords
    radius = max(span) / 2.0 + plotter.size * 4  # Add padding

    plotter.ax.set_xlim(center[0] - radius, center[0] + radius)
    plotter.ax.set_ylim(center[1] - radius, center[1] + radius)


def main():
    """Main visualization pipeline."""
    population = load_population_data_as_hex_dict(DATA_FILE_PATH)

    if not population:
        sys.exit("Loaded data is empty. Nothing to visualize.")

    print(f"Found {len(population)} cells. Generating plot using HexPlotter...")

    # 1. Initialize the official HexPlotter
    plotter = HexPlotter(hex_size=1.0, labels={}, colormap=AIF_COLORMAP)

    # 2. Plot the population. We pass dummy values for front-related parameters
    #    because we will manually set the view limits for our radial colony.
    plotter.plot_population(
        population=population,
        mean_front_q=0,
        sim_width=1,
        title="Manual Visualization of Final Colony State",
    )

    # 3. CRITICAL STEP: Center the plot on the radial colony
    center_radial_plot(plotter, population)

    # 4. Save the final high-quality image
    plotter.save_figure(OUTPUT_IMAGE_PATH, dpi=300)
    plotter.close()

    print(f"\n✅ Visualization saved successfully to: {OUTPUT_IMAGE_PATH}")

    # Optional: Display the plot if running in an interactive session
    # plt.show()


if __name__ == "__main__":
    main()
