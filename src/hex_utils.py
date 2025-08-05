# FILE: src/hex_utils.py
# Defines the Hex coordinate system and a Matplotlib-based plotter.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from collections import namedtuple

# Define cell types for clarity
Empty, Wildtype, Mutant = 0, 1, 2

# Define Hex coordinate class using cube coordinates
Hex = namedtuple("Hex", ["q", "r", "s"])


def hex_add(h1, h2):
    return Hex(h1.q + h2.q, h1.r + h2.r, h1.s + h2.s)


# Define the six directions on a hexagonal grid
hex_directions = [
    Hex(1, 0, -1),
    Hex(1, -1, 0),
    Hex(0, -1, 1),
    Hex(-1, 0, 1),
    Hex(-1, 1, 0),
    Hex(0, 1, -1),
]


# Add neighbor and wrap methods to the Hex class
def hex_neighbor(h, direction_idx):
    return hex_add(h, hex_directions[direction_idx])


def hex_neighbors(h):
    return [hex_neighbor(h, i) for i in range(6)]


def hex_wrap(h, width):
    # Apply periodic boundary conditions in the transverse direction (r)
    # The s coordinate is derived from q and r
    wrapped_r = h.r % width
    return Hex(h.q, wrapped_r, -h.q - wrapped_r)


Hex.neighbor = hex_neighbor
Hex.neighbors = hex_neighbors
Hex.wrap = hex_wrap


class HexPlotter:
    def __init__(self, hex_size, labels, colormap, ax=None):
        self.size = hex_size
        self.labels = labels
        self.colormap = colormap
        if ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 15))
        else:
            self.fig, self.ax = ax.get_figure(), ax
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("white")
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])

    def hex_to_cartesian(self, h):
        x = self.size * (3.0 / 2 * h.q)
        y = self.size * (np.sqrt(3) / 2 * h.q + np.sqrt(3) * h.r)
        return x, y

    def plot_population(self, population, title="", wt_front=None, m_front=None):
        self.ax.clear()
        self.ax.set_title(title, fontsize=20)

        all_x, all_y = [], []
        for h, cell_type in population.items():
            x, y = self.hex_to_cartesian(h)
            all_x.append(x)
            all_y.append(y)

            facecolor = self.colormap.get(cell_type, "lightgrey")
            is_wt_front = wt_front is not None and h in wt_front
            is_m_front = m_front is not None and h in m_front

            if is_wt_front or is_m_front:
                edgecolor = "red" if is_m_front else "darkblue"
                linewidth, zorder = 2.0, 5
            else:
                edgecolor, linewidth, zorder = "black", 0.5, 1

            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=self.size,
                orientation=np.radians(30),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            )
            self.ax.add_patch(hexagon)

        if all_x:
            self.ax.set_xlim(min(all_x) - 2 * self.size, max(all_x) + 2 * self.size)
            self.ax.set_ylim(min(all_y) - 2 * self.size, max(all_y) + 2 * self.size)

        self.ax.set_aspect("equal")
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])

    def save_figure(self, filename, dpi=150):
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")
