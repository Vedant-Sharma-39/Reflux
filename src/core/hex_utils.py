# FILE: src/core/hex_utils.py
# Manages hexagonal grid coordinates and plotting. [v14 - Patch Boundary Visualization]

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
    """Utility to plot populations on a hexagonal grid with a focus on aesthetics."""

    def __init__(self, hex_size: float, labels: Dict, colormap: Dict):
        self.size = hex_size
        self.labels = labels
        self.colormap = colormap
        self.fig = None
        self.ax = None

    def _axial_to_cartesian(self, h: Hex) -> tuple[float, float]:
        """Converts axial hex coordinates to cartesian using a flat-top orientation."""
        x = self.size * (3.0 / 2.0 * h.q)
        y = self.size * (np.sqrt(3) / 2.0 * h.q + np.sqrt(3) * h.r)
        return x, y

    def _get_hex_corners(self, center_x: float, center_y: float) -> np.ndarray:
        """Calculates corners for a 'flat-top' hexagon."""
        corners = []
        for i in range(6):
            angle_deg = 60 * i
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
        q_to_patch_index: Optional[np.ndarray] = None,
    ):
        if not self.fig or not self.ax:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()

        body_patches, body_colors = [], []
        front_patches, front_colors = [], []
        all_fronts = (wt_front or set()) | (m_front or set())
        min_x, max_x, min_y, max_y = (
            float("inf"),
            float("-inf"),
            float("inf"),
            float("-inf"),
        )

        for h, cell_type in population.items():
            if cell_type == 0:
                continue
            center_x, center_y = self._axial_to_cartesian(h)
            min_x, max_x = min(min_x, center_x), max(max_x, center_x)
            min_y, max_y = min(min_y, center_y), max(max_y, center_y)
            corners = self._get_hex_corners(center_x, center_y)
            color = self.colormap.get(cell_type, "gray")
            if h in all_fronts:
                front_patches.append(Polygon(corners))
                front_colors.append(color)
            else:
                body_patches.append(Polygon(corners))
                body_colors.append(color)

        if body_patches:
            body_collection = PatchCollection(
                body_patches,
                facecolor=body_colors,
                edgecolor=body_colors,
                lw=0.1,
                zorder=1,
            )
            self.ax.add_collection(body_collection)
        if front_patches:
            front_collection = PatchCollection(
                front_patches,
                facecolor=front_colors,
                edgecolor="#FFFFFF",
                lw=2.0,
                zorder=10,
            )
            self.ax.add_collection(front_collection)

        # --- NEW: Draw Patch Boundaries ---
        if q_to_patch_index is not None:
            # Find the q-coordinates where the patch index changes
            boundary_q_indices = np.where(np.diff(q_to_patch_index) != 0)[0]
            for q_idx in boundary_q_indices:
                # The boundary is between q_idx and q_idx + 1
                q_boundary = q_idx + 0.5
                x_boundary, _ = self._axial_to_cartesian(
                    Hex(q_boundary, 0, -q_boundary)
                )
                self.ax.axvline(
                    x_boundary,
                    color="cyan",
                    linestyle="--",
                    linewidth=2,
                    zorder=20,
                    label="Patch Boundary",
                )

        self.ax.set_aspect("equal", "box")
        width_range = max_x - min_x
        height_range = max_y - min_y
        if width_range > 0 and height_range > 0:
            base_width_inches = 20
            self.fig.set_size_inches(
                base_width_inches, base_width_inches * (height_range / width_range)
            )
        padding = self.size * 2
        self.ax.set_xlim(min_x - padding, max_x + padding)
        self.ax.set_ylim(min_y - padding, max_y + padding)

        self.ax.set_title(title, fontsize=20, pad=15)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout()

    def save_figure(self, filename: str, dpi: int = 200):
        if self.fig:
            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor="white",
            )

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig, self.ax = None, None
