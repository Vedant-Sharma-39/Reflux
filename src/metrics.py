# src/metrics.py
# Contains the metric collection architecture: Manager and Trackers.

import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING  # <-- IMPORT TYPE_CHECKING
import collections
from hex_utils import Hex
from typing import Tuple

# This is the crucial change to break the circular import.
# The 'if TYPE_CHECKING:' block is only executed by static type checkers (like Mypy),
# not by the Python interpreter at runtime. This allows us to have the type hint
# without creating a runtime import dependency cycle.
if TYPE_CHECKING:
    from linear_model import GillespieSimulation


# Forward declaration for type hinting to avoid circular import at RUNTIME.
# This is now technically redundant if you use the TYPE_CHECKING block, but
# it's good practice to keep it as a clear signal of the dependency.
class GillespieSimulation:
    pass


class MetricTracker(ABC):
    """Abstract base class for a metric plugin."""

    def __init__(self, sim: GillespieSimulation):
        self.sim = sim

    def initialize(self):
        """Called by the MetricsManager at the beginning of a run."""
        pass

    def after_step_hook(self):
        """Called by the MetricsManager after each simulation step."""
        pass

    def finalize(self):
        """Called by the MetricsManager at the end of the simulation."""
        pass


class FrontDynamicsTracker(MetricTracker):
    """Tracks and logs front position, roughness, and composition at time intervals."""

    def __init__(self, sim: "GillespieSimulation", log_interval: float = 1.0):
        super().__init__(sim)
        self.log_interval = log_interval
        self.next_log_time = 0.0
        # Store data in a list of dictionaries for easy conversion to a DataFrame
        self.history = []

    def after_step_hook(self):
        if self.sim.time >= self.next_log_time:
            self.history.append(
                {
                    "time": self.sim.time,
                    "mean_q": self.sim.mean_front_position,
                    "roughness_q": self.sim.front_roughness_q,
                    "mutant_fraction": self.sim.mutant_fraction,
                    "wt_front_cells": len(self.sim.wt_front_cells),
                    "m_front_cells": len(self.sim.m_front_cells),
                }
            )
            self.next_log_time += self.log_interval

    # --- ADD THIS METHOD ---
    def get_dataframe(self) -> pd.DataFrame:
        """Returns the collected history as a Pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()  # Return an empty DataFrame if no data was collected
        return pd.DataFrame(self.history)

    # -----------------------

    # The finalize method is not strictly needed by the runner, but is good to have
    def finalize(self):
        """Optional: Can be used to save the data directly from the tracker."""
        df = self.get_dataframe()
        if not df.empty:
            # You could add logic here to save the df to a file if desired
            print(f"FrontDynamicsTracker finalized with {len(df)} data points.")


class FinalStateTracker(MetricTracker):
    """Saves the final state of the grid and key metadata."""

    def __init__(
        self,
        sim: GillespieSimulation,
        snapshot_filename: str = "final_snapshot.npy",
        metadata_filename: str = "final_metadata.csv",
    ):
        super().__init__(sim)
        self.snapshot_filename = snapshot_filename
        self.metadata_filename = metadata_filename

    def finalize(self):
        # 1. Save the final population grid state as a NumPy array
        grid_array = np.zeros((self.sim.length, self.sim.width), dtype=np.uint8)
        for hex_obj, state in self.sim.population.items():
            col = hex_obj.q
            row = hex_obj.r + (hex_obj.q + (hex_obj.q & 1)) // 2
            if 0 <= col < self.sim.length and 0 <= row < self.sim.width:
                grid_array[col, row] = state

        np.save(self.snapshot_filename, grid_array)

        # 2. Save the final metadata to a CSV
        final_m_frac = len(self.sim.m_front_cells) / (self.sim.front_cell_count + 1e-9)
        metadata = {
            "final_time": [self.sim.time],
            "final_mean_q": [self.sim.mean_front_position],
            "final_mutant_fraction": [final_m_frac],
            "final_wt_front_cells": [len(self.sim.wt_front_cells)],
            "final_m_front_cells": [len(self.sim.m_front_cells)],
        }
        df = pd.DataFrame(metadata)
        df.to_csv(self.metadata_filename, index=False)
        print(f"Final state saved: {self.snapshot_filename}, {self.metadata_filename}")


import collections


class SteadyStateTracker(MetricTracker):
    """
    Tracks a simulation metric to determine its steady-state average.

    This tracker is designed for experiments where we want a single, final
    value from a run, like the average mutant fraction after the system
    has stabilized.
    """

    def __init__(
        self,
        sim: GillespieSimulation,
        warmup_time: float = 200.0,
        sample_interval: float = 5.0,
    ):
        """
        Args:
            sim: The simulation instance.
            warmup_time: How long to run the simulation before starting to
                         collect samples for the steady-state average.
            sample_interval: The time interval between collecting samples.
        """
        super().__init__(sim)
        self.warmup_time = warmup_time
        self.sample_interval = sample_interval
        self.next_sample_time = self.warmup_time

        # Using a deque is efficient for storing a running list of samples
        self.mutant_fraction_samples = collections.deque()
        self.final_result = None

    def after_step_hook(self):
        """Called after each step to check if it's time to sample."""
        if self.sim.time >= self.next_sample_time:
            # We are in the sampling period, so record the current mutant fraction
            self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.next_sample_time += self.sample_interval

    def get_steady_state_mutant_fraction(self) -> float:
        """
        Calculates and returns the steady-state average.
        This should be called *after* the simulation run is complete.
        """
        if not self.mutant_fraction_samples:
            # If the simulation was too short to collect any samples,
            # return the last known value or NaN.
            return np.nan

        # The final result is the mean of all collected samples
        self.final_result = np.mean(self.mutant_fraction_samples)
        return self.final_result

    def finalize(self):
        """
        The finalize step could just print the result for single runs,
        but the main way to get the data will be via the getter method
        from the runner script.
        """
        avg_rho_M = self.get_steady_state_mutant_fraction()
        print(f"SteadyStateTracker: Final avg mutant fraction = {avg_rho_M:.4f}")
        print(
            f"  (Collected {len(self.mutant_fraction_samples)} samples after time {self.warmup_time})"
        )


class SectorWidthTracker(MetricTracker):
    """
    A MetricTracker specifically designed to track the width of a single
    mutant sector as a function of the mean front position (q-axis).

    This tracker is intended for calibration runs where k_total=0 and there is
    an initial mutant patch.
    """

    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 1.0):
        super().__init__(sim)
        self.capture_interval = capture_interval
        # Stores a list of (mean_front_position, sector_width) tuples
        self.width_trajectory = []
        self.next_capture_q = 0.0

    def _get_r_offset(self, cell: Hex) -> int:
        """Helper to convert cube coords back to the transverse r_offset."""
        return cell.r + (cell.q + (cell.q & 1)) // 2

    def after_step_hook(self):
        """
        Called after each step. Records the sector width when the front
        advances past the next capture interval.
        """
        mean_q = self.sim.mean_front_position
        if mean_q < self.next_capture_q:
            return

        self.next_capture_q += self.capture_interval

        mutant_front = self.sim.m_front_cells
        if not mutant_front:
            # The sector is extinct, no more data to record for this run.
            return

        # Find the min and max transverse positions (r_offset) of the mutant front
        r_offsets = sorted([self._get_r_offset(cell) for cell in mutant_front])

        # --- Robust width calculation for periodic boundaries ---
        # Calculate the differences between successive, sorted r_offsets.
        # The largest difference is the "gap" in the periodic boundary.
        diffs = [r_offsets[i + 1] - r_offsets[i] for i in range(len(r_offsets) - 1)]

        # Add the "wrap-around" difference
        wrap_diff = (self.sim.width - r_offsets[-1]) + r_offsets[0]
        diffs.append(wrap_diff)

        max_gap = max(diffs)

        # The width is the total simulation width minus the largest gap.
        # We subtract 1 from the gap because width is inclusive.
        current_width = self.sim.width - (max_gap - 1)

        self.width_trajectory.append((mean_q, current_width))

    def get_trajectory(self) -> List[Tuple[float, int]]:
        """Returns the collected trajectory data."""
        return self.width_trajectory

    def finalize(self):
        """Prints a summary for a single run."""
        if self.width_trajectory:
            final_q, final_w = self.width_trajectory[-1]
            print(
                f"SectorWidthTracker: Captured {len(self.width_trajectory)} points. "
                f"Final state: width={final_w} at q={final_q:.2f}"
            )
        else:
            print(
                "SectorWidthTracker: No data captured (sector may have been extinct initially)."
            )


class SurvivalTracker(MetricTracker):
    """
    A simple tracker for absorbing-state phase transitions.
    It only records whether the wild-type population is extant at the very end.
    """

    def __init__(self, sim: "GillespieSimulation"):
        super().__init__(sim)
        self.wt_survived = 0  # Default to 0 (extinct)

    def finalize(self):
        """Called by the manager at the very end of a run."""
        if len(self.sim.wt_front_cells) > 0:
            self.wt_survived = 1


class MetricsManager:
    """Manages a collection of MetricTracker plugins."""

    def __init__(self):
        self.trackers: List[MetricTracker] = []
        self._sim = None

    def register_simulation(self, sim: "GillespieSimulation"):
        self._sim = sim

    def add_tracker(self, tracker: MetricTracker):
        self.trackers.append(tracker)

    def initialize_all(self):
        for tracker in self.trackers:
            tracker.initialize()

    def after_step(self):
        for tracker in self.trackers:
            tracker.after_step_hook()

    def finalize_all(self):
        """Call to finalize and save data from all trackers."""
        for tracker in self.trackers:
            tracker.finalize()
