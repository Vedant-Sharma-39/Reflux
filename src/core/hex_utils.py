# FILE: src/core/hex_utils.py
# Manages hexagonal grid coordinates and plotting. [v6 - Corrected Rotation and Drawing]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import Dict, Set, Optional


class Hex:
    """Represents a hexagon in axial coordinates (q, r)."""

    def __init__(self, q: int, r: int, s: int):
        if round(q + r + s) != 0:
            raise ValueError(
                f"Hex coordinate invariant q+r+s=0 not met: {q}+{r}+{s}={q+r+s}"
            )
        self.q, self.r, self.s = q, r, s

    def __eq__(self, other):
        return self.q == other.q and self.r == other.r

    def __hash__(self):
        return hash((self.q, self.r))

    def __repr__(self):
        return f"Hex({self.q}, {self.r}, {self.s})"

    def neighbors(self) -> list:
        return [
            Hex(self.q + 1, self.r, self.s - 1),
            Hex(self.q - 1, self.r, self.s + 1),
            Hex(self.q, self.r + 1, self.s - 1),
            Hex(self.q, self.r - 1, self.s + 1),
            Hex(self.q + 1, self.r - 1, self.s),
            Hex(self.q - 1, self.r + 1, self.s),
        ]


class HexPlotter:
    """Utility to plot populations on a hexagonal grid."""

    def __init__(self, hex_size: float, labels: Dict, colormap: Dict, ax=None):
        self.size = hex_size
        self.labels = labels
        self.colormap = colormap
        self.fig, self.ax = (plt.gcf(), ax) if ax else plt.subplots(figsize=(12, 12))

        # To make the s-axis vertical, we need to rotate the grid.
        # The unrotated growth direction (increasing -s) is at a +60 degree angle.
        # To make it vertical (+90 degrees), we need a +30 degree rotation.
        rotation_angle_rad = np.deg2rad(30)
        self.cos_a = np.cos(rotation_angle_rad)
        self.sin_a = np.sin(rotation_angle_rad)

    def hex_to_cartesian(self, h: Hex) -> tuple[float, float]:
        """Converts axial hex coordinates to cartesian and rotates for vertical range expansion."""
        # Standard pointy-top conversion
        x_unrotated = self.size * 3 / 2 * h.q
        y_unrotated = self.size * (np.sqrt(3) / 2 * h.q + np.sqrt(3) * h.r)

        # Apply rotation to make the s-axis vertical
        x = x_unrotated * self.cos_a - y_unrotated * self.sin_a
        y = x_unrotated * self.sin_a + y_unrotated * self.cos_a
        return x, y

    def _get_hex_corners(self, center_x: float, center_y: float) -> np.ndarray:
        """
        Calculates corners for a standard pointy-top hexagon.
        The grid rotation is handled by rotating the center point, so the polygon
        itself can remain in a standard, unrotated orientation.
        """
        corners = []
        for i in range(6):
            angle_deg = 60 * i + 30  # +30 degree offset for pointy-top hexagons
            angle_rad = np.pi / 180 * angle_deg
            x_corner = center_x + self.size * np.cos(angle_rad)
            y_corner = center_y + self.size * np.sin(angle_rad)
            corners.append((x_corner, y_corner))
        return np.array(corners)

    def plot_population(
        self,
        population: Dict[Hex, int],
        title: str = "",
        wt_front: Optional[Set[Hex]] = None,
        m_front: Optional[Set[Hex]] = None,
    ):
        self.ax.clear()
        body_patches, body_colors = [], []
        front_patches, front_colors = [], []
        all_fronts = (wt_front or set()) | (m_front or set())

        for h, cell_type in population.items():
            if cell_type == 0:
                continue
            center_x, center_y = self.hex_to_cartesian(h)
            corners = self._get_hex_corners(center_x, center_y)  # Use simplified method
            color = self.colormap.get(cell_type, "gray")
            if h in all_fronts:
                front_patches.append(Polygon(corners))
                front_colors.append(color)
            else:
                body_patches.append(Polygon(corners))
                body_colors.append(color)

        if body_patches:
            self.ax.add_collection(
                PatchCollection(
                    body_patches,
                    facecolor=body_colors,
                    edgecolor="black",
                    lw=0.5,
                    zorder=1,
                )
            )
        if front_patches:
            self.ax.add_collection(
                PatchCollection(
                    front_patches,
                    facecolor=front_colors,
                    edgecolor="red",
                    lw=2.0,
                    zorder=10,
                )
            )
        self.ax.autoscale_view()
        self.ax.set_aspect("equal", "box")
        self.ax.set_title(title, fontsize=16)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.tight_layout()

    def save_figure(self, filename: str, dpi: int = 150):
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")
