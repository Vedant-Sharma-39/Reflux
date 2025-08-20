# FILE: src/core/hex_utils.py (FIXED: Aesthetic Overhaul Version)
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
        self.colormap = {int(k): v for k, v in colormap.items()}
        self.fig, self.ax = plt.subplots()
        # --- AESTHETIC FIX 1: Use a clean white background ---
        self.fig.set_facecolor("#FFFFFF")

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
        mean_front_q: float,
        sim_width: int,
        title: str = "",
        q_to_patch_index: Optional[np.ndarray] = None,
    ):
        self.ax.clear()
        self.ax.set_facecolor(self.fig.get_facecolor())

        q_int = int(mean_front_q)
        r_offset_for_y_calc = (q_int + (q_int & 1)) // 2
        r_top = 0 - r_offset_for_y_calc
        r_bottom = (sim_width - 1) - r_offset_for_y_calc
        _, min_y = self._axial_to_cartesian(Hex(q_int, r_top, -q_int - r_top))
        _, max_y = self._axial_to_cartesian(Hex(q_int, r_bottom, -q_int - r_bottom))

        if not population:
            self.ax.set_title(title, fontsize=18, pad=20, color="black", loc="left")
        else:
            if q_to_patch_index is not None:
                # --- AESTHETIC FIX 2: Better background patch colors ---
                patch_colors = {0: "#FFFFFF", 1: "#E0E0E0"}
                boundary_q_indices = np.where(np.diff(q_to_patch_index) != 0)[0]
                regions = np.split(q_to_patch_index, boundary_q_indices + 1)
                q_starts = [0] + (boundary_q_indices + 1).tolist()
                y_padding = self.size * 2
                for i, region in enumerate(regions):
                    patch_id = region[0]
                    q_start, q_end = q_starts[i], q_starts[i] + len(region)
                    x_start, _ = self._axial_to_cartesian(
                        Hex(q_start - 0.5, 0, -q_start + 0.5)
                    )
                    x_end, _ = self._axial_to_cartesian(
                        Hex(q_end - 0.5, 0, -q_end + 0.5)
                    )
                    rect = plt.Rectangle(
                        (x_start, min_y - y_padding),
                        x_end - x_start,
                        (max_y - min_y) + 2 * y_padding,
                        facecolor=patch_colors.get(patch_id, "#FFFFFF"),
                        edgecolor="none",
                        zorder=0,
                    )
                    self.ax.add_patch(rect)
                for boundary_q in boundary_q_indices:
                    q_midpoint = boundary_q + 0.5
                    boundary_x, _ = self._axial_to_cartesian(
                        Hex(q_midpoint, 0, -q_midpoint)
                    )
                    self.ax.axvline(
                        x=boundary_x,
                        color="gray",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                        zorder=0.5,
                    )

            patch_majority_type = {0: 1, 1: 2}
            majority_patches, minority_patches = [], []
            majority_colors, minority_colors = [], []

            for h, cell_type in population.items():
                if cell_type == 0:
                    continue
                patch_idx = (
                    q_to_patch_index[int(h.q)]
                    if q_to_patch_index is not None
                    and 0 <= int(h.q) < len(q_to_patch_index)
                    else 0
                )
                center_x, center_y = self._axial_to_cartesian(h)
                corners = self._get_hex_corners(center_x, center_y)
                color = self.colormap.get(cell_type, "gray")
                if cell_type == patch_majority_type.get(patch_idx, 1):
                    majority_patches.append(Polygon(corners, closed=True))
                    majority_colors.append(color)
                else:
                    minority_patches.append(Polygon(corners, closed=True))
                    minority_colors.append(color)

            # Use a slightly darker shade for edges to create definition
            majority_edge_colors = [
                (r * 0.8, g * 0.8, b * 0.8, a)
                for r, g, b, a in plt.cm.colors.to_rgba_array(majority_colors)
            ]
            self.ax.add_collection(
                PatchCollection(
                    majority_patches,
                    facecolors=majority_colors,
                    edgecolors=majority_edge_colors,
                    lw=0.5,
                    zorder=1,
                )
            )
            self.ax.add_collection(
                PatchCollection(
                    minority_patches,
                    facecolors=minority_colors,
                    edgecolors="white",
                    lw=1.0,
                    zorder=2,
                )
            )

            # --- AESTHETIC FIX 3: Reduced font size ---
            self.ax.set_title(title, fontsize=18, pad=20, color="black", loc="left")

        # Set final view limits
        q_window_width = 100.0
        q_min_view = mean_front_q - q_window_width / 2
        q_max_view = mean_front_q + q_window_width / 2
        x_min_view, _ = self._axial_to_cartesian(
            Hex(int(q_min_view), 0, -int(q_min_view))
        )
        x_max_view, _ = self._axial_to_cartesian(
            Hex(int(q_max_view), 0, -int(q_max_view))
        )

        self.ax.set_aspect("equal", "box")
        padding_y = self.size * 2
        self.ax.set_xlim(x_min_view, x_max_view)
        self.ax.set_ylim(min_y - padding_y, max_y + padding_y)

        # --- AESTHETIC FIX 4: Re-engineered aspect ratio logic ---
        view_width_cartesian = x_max_view - x_min_view
        view_height_cartesian = (max_y - min_y) + 2 * padding_y
        aspect_ratio = (
            view_width_cartesian / view_height_cartesian
            if view_height_cartesian > 0
            else 1.0
        )
        base_height_inches = 16
        self.fig.set_size_inches(base_height_inches * aspect_ratio, base_height_inches)
        # --- END OF FIX ---

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout()

    def save_figure(self, filename: str, dpi: int = 150):
        if self.fig:
            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor=self.fig.get_facecolor(),
            )

    def close(self):
        if self.fig:
            plt.close(self.fig)
