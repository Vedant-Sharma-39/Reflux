# hex_utils.py
# A robust, self-contained library for hexagonal grid math and high-quality visualization.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


class Hex:
    """A class for a single hexagon using cube coordinates."""

    __slots__ = ("q", "r", "s")

    def __init__(self, q: int, r: int, s: int):
        if round(q + r + s) != 0:
            raise ValueError("Cube coordinates (q, r, s) must sum to 0")
        self.q = q
        self.r = r
        self.s = s

    def __hash__(self):
        return hash((self.q, self.r, self.s))

    def __eq__(self, other):
        return (
            isinstance(other, Hex)
            and self.q == other.q
            and self.r == other.r
            and self.s == other.s
        )

    def __add__(self, other):
        """Vector addition for Hex objects."""
        return Hex(self.q + other.q, self.r + other.r, self.s + other.s)

    def neighbor(self, direction_index: int):
        return self + Hex._directions[direction_index]

    def neighbors(self):
        return [self.neighbor(i) for i in range(6)]


Hex._directions = [
    Hex(1, 0, -1),
    Hex(1, -1, 0),
    Hex(0, -1, 1),
    Hex(-1, 0, 1),
    Hex(-1, 1, 0),
    Hex(0, 1, -1),
]


class HexPlotter:
    """A high-performance plotter for hexagonal grids."""

    def __init__(self, hex_size=1.0, labels=None, colormap=None):
        self.size = hex_size
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.colormap = colormap if colormap is not None else {}
        self.labels = labels if labels is not None else {}
        self.fig.set_facecolor("#f0f0f0")

    def _hex_to_cartesian(self, hex_obj):
        x = self.size * (3.0 / 2.0 * hex_obj.q)
        y = self.size * (np.sqrt(3) / 2.0 * hex_obj.q + np.sqrt(3) * hex_obj.r)
        return x, y

    # In HexPlotter class, replace the plot_population method

    def plot_population(self, population, title="", wt_front=None, m_front=None):
        """
        [MODIFIED] Plots the population, with optional highlighting for front cells.

        Args:
            population (dict): The main dictionary of {Hex: state}.
            title (str): The plot title.
            wt_front (set or dict): A set/dict of wild-type front cells.
            m_front (set or dict): A set/dict of mutant front cells.
        """
        self.ax.clear()
        self.ax.set_facecolor("#f0f0f0")

        # To avoid passing empty lists, default to empty sets
        wt_front = wt_front or set()
        m_front = m_front or set()

        patches, facecolors, edgecolors, linewidths = [], [], [], []

        for hex_obj, value in population.items():
            center_x, center_y = self._hex_to_cartesian(hex_obj)
            hexagon = RegularPolygon(
                (center_x, center_y),
                numVertices=6,
                radius=self.size * (1 / np.sqrt(3)) * 0.98,
                orientation=np.radians(0),
            )
            patches.append(hexagon)
            facecolors.append(self.colormap.get(value, "#FFFFFF"))

            # --- Highlighting Logic ---
            if hex_obj in wt_front or hex_obj in m_front:
                # This is a front cell
                edgecolors.append("#FF0000")  # Bright red edge
                linewidths.append(0.5)  # Make it thicker
            else:
                # This is a bulk cell
                edgecolors.append("#202020")  # Standard dark edge
                linewidths.append(0.15)

        collection = PatchCollection(
            patches, facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths
        )
        self.ax.add_collection(collection)
        self.ax.set_title(title, fontsize=16, pad=20)
        self.ax.set_aspect("equal")
        self.ax.autoscale_view()
        self.ax.axis("off")
        self.add_legend()
        plt.draw()

    def add_legend(self):
        """Creates and adds a legend, compatible with Python < 3.8."""
        legend_patches = []
        for state, label in self.labels.items():
            color = self.colormap.get(state)
            if color is not None:
                patch = mpatches.Patch(color=color, label=label)
                legend_patches.append(patch)

        if legend_patches:
            self.ax.legend(
                handles=legend_patches,
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                frameon=False,
                fontsize=12,
            )

    def save_figure(self, filename, dpi=150):
        self.fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
            facecolor=self.fig.get_facecolor(),
        )
