# FILE: scripts/generate_diagram_assets_v2.py (Corrected)
#
# This script generates improved SVG assets for a professional diagram
# illustrating the Reflux simulation model.
#
# Assets generated:
# 1. initial_mixed_front.svg: A wide snapshot of the initial population at q=0,
#    showing mixed WT/Mutant cells at the "front" and empty space ahead.
# 2. hex_wt.svg: A single Wild-Type hexagon.
# 3. hex_m.svg: A single Mutant hexagon.
# 4. hex_empty.svg: A single Empty hexagon.
#
# These SVG files can then be imported into a vector graphics editor (e.g., Inkscape)
# for final diagram composition, adding text, arrows, and other conceptual elements.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
import random

# --- Robustly add project root to the Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary classes from your project
from src.core.model import GillespieSimulation, Empty, Wildtype, Mutant
from src.core.hex_utils import Hex, HexPlotter

# --- Configuration for Assets ---
OUTPUT_DIR_NAME = "diagram_assets"
HEX_SIZE = 1.0  # Base size for hexagons in plot
DPI = 300       # Resolution for saving (less critical for SVG, but good practice)

# Your defined project colors (from src/presentation_figures/export_front_images.py)
COLOR_WT = "#02343F"  # Dark blue/teal for Wild-Type
COLOR_M = "#d35400"   # Orange for Mutant
COLOR_EMPTY = "#F0F0F0" # Light gray for Empty cells (for single hex)
COLORMAP_FOR_PLOTTING = {
    Wildtype: COLOR_WT,
    Mutant: COLOR_M,
    Empty: COLOR_EMPTY # Explicitly include empty for plotter
}

# Ensure the output directory exists
output_dir_path = PROJECT_ROOT / OUTPUT_DIR_NAME
output_dir_path.mkdir(exist_ok=True, parents=True)

def _generate_initial_population_for_viz(width: int) -> Dict[Hex, int]:
    """Generates a sample initial population (e.g., mixed) for visualization."""
    pop: Dict[Hex, int] = {}
    
    # Create a mixed population at q=0 for the initial front
    # Let's do a semi-random pattern with distinct blocks for visual clarity
    # These `r_display_idx` values map to the row index on the screen, 0 to `width-1`.
    # The actual Hex.r coordinate depends on `q` and `r_display_idx`.
    
    # Example pattern for width=128: 32 WT, 16 M, 32 WT, 48 M
    block_sizes = [int(width * 0.25), int(width * 0.125), int(width * 0.25), int(width * 0.375)]
    types = [Wildtype, Mutant, Wildtype, Mutant]
    
    current_r_display_idx = 0
    for size, cell_type in zip(block_sizes, types):
        for _ in range(size):
            if current_r_display_idx < width: # Ensure we don't go over width
                # For q=0, Hex.r is directly current_r_display_idx assuming plotter's r-axis alignment
                hex_r = current_r_display_idx 
                h = Hex(0, hex_r, -(0 + hex_r)) # Correct s calculation: s = -(q + r)
                pop[h] = cell_type
            current_r_display_idx += 1
            
    # Pad with Empty cells in higher q (ahead of the front) for visual expansion
    # These q_val and r_display_idx values need to be converted to Hex coordinates
    for q_val in range(1, 4): # Show 3 "empty" rows ahead of the front (to mimic expansion target)
        for r_display_idx in range(width):
            # Hex.r calculation depends on q_val and r_display_idx
            hex_r = r_display_idx - (q_val + (q_val & 1)) // 2 
            h_empty = Hex(q_val, hex_r, -(q_val + hex_r)) # Correct s calculation: s = -(q + r)
            
            # Add empty hexes if they are not already occupied (e.g. from q=0)
            # This check is mostly for robustness, for q > 0 it should always be empty.
            if h_empty not in pop: 
                pop[h_empty] = Empty # Mark these as empty for drawing purposes

    return pop


def _render_and_save_population_snapshot(filename: str, population_to_plot: Dict[Hex, int], sim_width: int):
    """
    Renders a specific population state and saves it as an SVG.
    This version takes the pre-generated population directly.
    """
    print(f"Generating: {filename}...")
    
    plotter = HexPlotter(hex_size=HEX_SIZE, labels={}, colormap=COLORMAP_FOR_PLOTTING)
    
    # Plot the population.
    # Set mean_front_q to 0 to ensure the initial front is at the bottom of the view
    plotter.plot_population(
        population=population_to_plot,
        mean_front_q=0, # Crucial: center view to show initial front + empty space
        sim_width=sim_width,
        title="", # No title for asset
        q_to_patch_index=None # No background patches for a clean asset
    )
    
    # Adjust figure size for a wider view reflecting the "linear range expansion".
    # Ensure a consistent aspect ratio if possible, or make it explicitly wide.
    fig_width_inches = 18 # Wider to show the "linear" aspect
    fig_height_inches = 6 # Taller to show empty rows
    plotter.fig.set_size_inches(fig_width_inches, fig_height_inches)
    
    # Make background transparent for SVG
    plotter.fig.set_facecolor('none')
    plotter.ax.set_facecolor('none')

    # Ensure the view focuses on q=0 and a few rows above/below
    q_min_view = -2 # Show a bit of history (behind q=0)
    q_max_view = 3  # Show empty space ahead (q=1,2,3 for Empty hexes)

    # Correct Hex instantiation for `_axial_to_cartesian` calls by calculating `s`
    x_min_view, _ = plotter._axial_to_cartesian(Hex(q_min_view, 0, -q_min_view)) # Fix: s = -(q+r)
    x_max_view, _ = plotter._axial_to_cartesian(Hex(q_max_view, 0, -q_max_view)) # Fix: s = -(q+r)
    
    # Calculate y-limits based on actual hex `r` range.
    # The `r` coordinate for `r_display_idx`=0 and `width-1` at `q=0` are `0` and `width-1`.
    r_top_hex_coord = 0
    r_bottom_hex_coord = sim_width - 1
    
    _, min_y_val = plotter._axial_to_cartesian(Hex(0, r_top_hex_coord, -r_top_hex_coord)) # Fix: s = -(q+r)
    _, max_y_val = plotter._axial_to_cartesian(Hex(0, r_bottom_hex_coord, -r_bottom_hex_coord)) # Fix: s = -(q+r)
    
    # Adjust y-limits to encompass the visible range of r-coordinates and empty q rows
    y_padding = plotter.size * 2 # Additional padding for hex size
    plotter.ax.set_xlim(x_min_view, x_max_view)
    plotter.ax.set_ylim(min_y_val - y_padding, max_y_val + y_padding)


    # Final aesthetic tweaks for a clean schematic
    plotter.ax.set_xticks([]) # Remove x-axis ticks
    plotter.ax.set_yticks([]) # Remove y-axis ticks
    plotter.ax.set_aspect('equal') # Ensure hexes look correct
    for spine in plotter.ax.spines.values():
        spine.set_visible(False) # Remove border

    plotter.save_figure(output_dir_path / filename, dpi=DPI)
    plotter.close()
    print(f"Saved: {output_dir_path / filename}")


def _render_and_save_single_hexagon(hex_type: int, filename: str):
    """
    Renders a single hexagon of a specified type and saves it as an SVG.
    """
    print(f"Generating: {filename}...")
    
    plotter = HexPlotter(hex_size=HEX_SIZE, labels={}, colormap=COLORMAP_FOR_PLOTTING)
    
    # Create a minimal population dictionary for plotting a single hex
    # Hex(0, 0, 0) is inherently valid (0+0+0=0)
    pop_single_hex = {Hex(0, 0, 0): hex_type} 
    
    # Plot the single hex. No front data, so just pass dummies for view centering.
    plotter.plot_population(
        population=pop_single_hex,
        mean_front_q=0,
        sim_width=1, # Dummy width, doesn't affect single hex
        title="",
        q_to_patch_index=None
    )
    
    # Set tight limits to only show the single hex
    plotter.ax.set_xlim(-HEX_SIZE * 1.5, HEX_SIZE * 1.5)
    plotter.ax.set_ylim(-HEX_SIZE * np.sqrt(3) * 0.75, HEX_SIZE * np.sqrt(3) * 0.75) # Adjust for flat-top hex height
    
    # Make background transparent
    plotter.fig.set_facecolor('none')
    plotter.ax.set_facecolor('none')

    plotter.save_figure(output_dir_path / filename, dpi=DPI)
    plotter.close()
    print(f"Saved: {output_dir_path / filename}")


def main():
    print(f"--- Generating Diagram Assets in '{OUTPUT_DIR_NAME}' ---")

    # Asset 1: Initial Population Snapshot for Panel (a)
    # Use a wider width to better convey "linear front" and show periodicity.
    viz_width = 128 # Example width for the initial mixed front
    
    initial_pop_for_viz = _generate_initial_population_for_viz(viz_width)
    _render_and_save_population_snapshot("initial_mixed_front.svg", initial_pop_for_viz, viz_width)

    # Assets 2, 3, 4: Single Hexagons for Panel (b)
    _render_and_save_single_hexagon(Wildtype, "hex_wt.svg")
    _render_and_save_single_hexagon(Mutant, "hex_m.svg")
    _render_and_save_single_hexagon(Empty, "hex_empty.svg")

    print("\n--- Asset Generation Complete ---")
    print(f"All SVG assets are in: {output_dir_path}")
    print("\nNext steps: Import these SVG files into Inkscape/Illustrator to compose your final diagram.")
    print("You will then manually add text labels, arrows, and arrange them as per your conceptual layout.")

if __name__ == "__main__":
    main()