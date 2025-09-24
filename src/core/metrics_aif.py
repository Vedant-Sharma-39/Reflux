# FILE: src/core/metrics_aif.py (Updated)

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from src.core.metrics import MetricTracker
from src.core.hex_utils import Hex


# --- Helper functions for analysis ---
def _calculate_robust_angular_span(angles: np.ndarray) -> float:
    """Calculates the 5th-95th percentile angular span, handling circularity."""
    if len(angles) < 5:
        return np.nan
    p_lower, p_upper = np.percentile(angles, [5, 95])
    return (p_upper - p_lower + 2 * np.pi) % (2 * np.pi)


class AifSectorTrajectoryTracker(MetricTracker):
    """
    Tracks the width of a resistant sector over time (vs. colony radius).
    Records a trajectory and terminates early if the sector goes extinct.
    """

    def __init__(self, sim: "AifModelSimulation", **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.record_radius_interval = kwargs.get("record_radius_interval", 5.0)
        self.next_record_radius = self.record_radius_interval
        self.trajectory_data = []
        self._is_done = False
        self.resistant_types = {2, 3}

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        # Check for extinction first to terminate early
        if self.sim.resistant_cell_count + self.sim.compensated_cell_count == 0:
            self._is_done = True
            return

        current_radius = self.sim.colony_radius
        if current_radius >= self.next_record_radius:
            # Get all resistant/compensated cells
            resistant_cells = [
                h for h, t in self.sim.population.items() if t in self.resistant_types
            ]

            if resistant_cells:
                # Convert to polar coordinates
                points = np.array(
                    [self.sim.plotter._axial_to_cartesian(h) for h in resistant_cells]
                )
                angles = np.arctan2(points[:, 1], points[:, 0])

                # Calculate width and store data
                width = _calculate_robust_angular_span(angles)
                self.trajectory_data.append(
                    {
                        "colony_radius": current_radius,
                        "sector_width_rad": width,
                        "resistant_cell_count": len(resistant_cells),
                    }
                )

            self.next_record_radius += self.record_radius_interval

    def finalize(self) -> Dict[str, Any]:
        return {"sector_trajectory": self.trajectory_data}


# (The existing AifMetricsManager can remain for other analyses)
class AifMetricsManager(MetricTracker):
    def __init__(self, sim: "AifModelSimulation", max_steps: int = 100000, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.max_steps = max_steps
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        if self.sim.step_count >= self.max_steps:
            self._is_done = True

    def finalize(self) -> Dict[str, Any]:
        final_pop_data = [
            {"q": h.q, "r": h.r, "type": t} for h, t in self.sim.population.items()
        ]
        return {"final_population": final_pop_data}
