# FILE: src/metrics.py
# [DEFINITIVE VERSION]
# This file contains the complete and corrected set of all metric trackers
# required to run every legacy and final experiment in the project.

import numpy as np
import pandas as pd
from abc import ABC
from typing import List, Tuple, TYPE_CHECKING
import collections
from hex_utils import Hex

if TYPE_CHECKING:
    from linear_model import GillespieSimulation, Wildtype, Mutant


class MetricTracker(ABC):
    """Abstract base class for all metric trackers."""

    def __init__(self, sim: "GillespieSimulation"):
        self.sim = sim

    def initialize(self):
        pass

    def after_step_hook(self):
        pass

    def finalize(self):
        pass


class SteadyStateTracker(MetricTracker):
    """For legacy 'steady_state' runs. Measures mean mutant fraction after a warmup."""

    def __init__(
        self, sim: "GillespieSimulation", warmup_time: float, sample_interval: float
    ):
        super().__init__(sim)
        self.warmup_time = warmup_time
        self.sample_interval = sample_interval
        self.next_sample_time = self.warmup_time
        self.mutant_fraction_samples = collections.deque()

    def after_step_hook(self):
        if self.sim.time >= self.next_sample_time:
            self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.next_sample_time += self.sample_interval

    def get_steady_state_mutant_fraction(self) -> float:
        return (
            np.mean(self.mutant_fraction_samples)
            if self.mutant_fraction_samples
            else np.nan
        )


class SectorWidthTracker(MetricTracker):
    """For 'calibration' runs. Measures the width of a single mutant sector over time."""

    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 1.0):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.width_trajectory: List[Tuple[float, int]] = []

    def _get_r_offset(self, cell: Hex) -> int:
        return cell.r + (cell.q + (cell.q & 1)) // 2

    def after_step_hook(self):
        mean_q = self.sim.mean_front_position
        if mean_q < self.next_capture_q:
            return
        self.next_capture_q += self.capture_interval
        mutant_front = self.sim.m_front_cells
        if not mutant_front:
            return
        r_offsets = sorted([self._get_r_offset(cell) for cell in mutant_front])
        diffs = [r_offsets[i + 1] - r_offsets[i] for i in range(len(r_offsets) - 1)]
        diffs.append((self.sim.width - r_offsets[-1]) + r_offsets[0])
        self.width_trajectory.append((mean_q, self.sim.width - (max(diffs) - 1)))

    def get_trajectory(self) -> List[Tuple[float, int]]:
        return self.width_trajectory


class InterfaceRoughnessTracker(MetricTracker):
    """For 'diffusion' runs. Measures W², the squared width of the ENTIRE population front."""

    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 0.5):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.roughness_history = []

    def after_step_hook(self):
        mean_q = self.sim.mean_front_position
        if mean_q < self.next_capture_q:
            return
        self.next_capture_q += self.capture_interval

        # [CORRECTED] Use sim._front_lookup to get ALL front cells (WT and M)
        front_cells = self.sim._front_lookup

        if len(front_cells) < 2:
            return
        q_values = np.array([cell.q for cell in front_cells])
        self.roughness_history.append((mean_q, np.var(q_values)))

    def get_roughness_trajectory(self) -> list[tuple[float, float]]:
        return self.roughness_history


class CorrelationAndStructureTracker(MetricTracker):
    """For legacy 'correlation_analysis' runs. Heavy tracker that calculates g(r)."""

    def __init__(self, sim, warmup_time, num_samples, sample_interval):
        super().__init__(sim)
        self.warmup_time, self.num_samples, self.sample_interval = (
            warmup_time,
            num_samples,
            sample_interval,
        )
        self.next_sample_time = self.warmup_time
        self.samples_taken = 0
        self.g_r_sum = collections.defaultdict(float)
        self.g_r_counts = collections.defaultdict(int)
        self.interface_density_sum = 0.0
        self._adj, self._last_front_set = {}, set()

    def _get_spin(self, cell_type):
        return 1 if cell_type == 1 else -1

    def _update_graph(self):
        current_front_set = self.sim._front_lookup
        if current_front_set == self._last_front_set:
            return
        self._adj = {
            cell: [
                n
                for n in self.sim._get_neighbors_periodic(cell)
                if n in current_front_set
            ]
            for cell in current_front_set
        }
        self._last_front_set = current_front_set.copy()

    def after_step_hook(self):
        if (
            self.samples_taken >= self.num_samples
            or self.sim.time < self.next_sample_time
        ):
            return
        self._update_graph()
        self._sample_structure()
        self.samples_taken += 1
        self.next_sample_time += self.sample_interval

    def _sample_structure(self):
        front_cells = list(self._last_front_set)
        if len(front_cells) < 2:
            return
        spins = {
            cell: self._get_spin(self.sim.population[cell]) for cell in front_cells
        }
        for i in range(len(front_cells)):
            source, q, distances = (
                front_cells[i],
                collections.deque([(front_cells[i], 0)]),
                {front_cells[i]: 0},
            )
            head = 0
            while head < len(q):
                curr, dist = q[head]
                head += 1
                for neighbor in self._adj[curr]:
                    if neighbor not in distances:
                        distances[neighbor] = dist + 1
                        q.append((neighbor, dist + 1))
            for j in range(i + 1, len(front_cells)):
                target = front_cells[j]
                if target in distances:
                    d = distances[target]
                    self.g_r_sum[d] += spins[source] * spins[target]
                    self.g_r_counts[d] += 1

    def get_results(self):
        g_r_list = sorted(
            [
                (r, self.g_r_sum[r] / self.g_r_counts[r])
                for r in self.g_r_counts
                if self.g_r_counts[r] > 0
            ]
        )
        return {"g_r": g_r_list}  # Legacy tracker only returned this


class FrontPropertiesTracker(MetricTracker):
    """For modern 'structure_analysis' runs. Lightweight and fast."""

    def __init__(self, sim, warmup_time, num_samples, sample_interval):
        super().__init__(sim)
        self.warmup_time, self.num_samples, self.sample_interval = (
            warmup_time,
            num_samples,
            sample_interval,
        )
        self.next_sample_time, self.samples_taken = self.warmup_time, 0
        self.interface_density_samples, self.mutant_fraction_samples = [], []
        self.q_start, self.time_start = 0.0, 0.0

    def initialize(self):
        self.q_start, self.time_start = self.sim.mean_front_position, self.sim.time

    def after_step_hook(self):
        if (
            self.samples_taken >= self.num_samples
            or self.sim.time < self.next_sample_time
        ):
            return
        front_cells = self.sim._front_lookup
        if not front_cells:
            return
        total_interfaces, total_neighbors = 0, 0
        for cell in front_cells:
            cell_type = self.sim.population[cell]
            for neighbor in self.sim._get_neighbors_periodic(cell):
                if neighbor in front_cells:
                    total_neighbors += 1
                    if self.sim.population[neighbor] != cell_type:
                        total_interfaces += 1
        if total_neighbors > 0:
            self.interface_density_samples.append(total_interfaces / total_neighbors)
        self.mutant_fraction_samples.append(self.sim.mutant_fraction)
        self.samples_taken += 1
        self.next_sample_time += self.sample_interval

    def finalize(self):
        self.q_end, self.time_end = self.sim.mean_front_position, self.sim.time

    def get_results(self):
        avg_speed = (
            (self.q_end - self.q_start) / (self.time_end - self.time_start)
            if self.time_end > self.time_start
            else 0.0
        )
        avg_interface_density = (
            np.mean(self.interface_density_samples)
            if self.interface_density_samples
            else 0.0
        )
        var_rho_M = (
            np.var(self.mutant_fraction_samples, ddof=1)
            if len(self.mutant_fraction_samples) > 1
            else 0.0
        )
        avg_rho_M = (
            np.mean(self.mutant_fraction_samples)
            if self.mutant_fraction_samples
            else 0.0
        )
        return {
            "avg_front_speed": avg_speed,
            "avg_interface_density": avg_interface_density,
            "var_rho_M": var_rho_M,
            "avg_rho_M": avg_rho_M,
        }


class MetricsManager:
    """A manager to orchestrate the execution of multiple trackers."""

    def __init__(self):
        self.trackers: List[MetricTracker] = []
        self._sim: "GillespieSimulation" = None

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
        for tracker in self.trackers:
            tracker.finalize()
