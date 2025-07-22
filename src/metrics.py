# FILE: src/metrics.py

import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING
import collections
from hex_utils import Hex

if TYPE_CHECKING:
    from linear_model import GillespieSimulation, Wildtype, Mutant


class MetricTracker(ABC):
    def __init__(self, sim: "GillespieSimulation"):
        self.sim = sim

    def initialize(self):
        pass

    def after_step_hook(self):
        pass

    def finalize(self):
        pass


class SteadyStateTracker(MetricTracker):
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
    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 1.0):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.width_trajectory: List[Tuple[float, int]] = []
        self.next_capture_q = 0.0

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
        wrap_diff = (self.sim.width - r_offsets[-1]) + r_offsets[0]
        diffs.append(wrap_diff)
        current_width = self.sim.width - (max(diffs) - 1)
        self.width_trajectory.append((mean_q, current_width))

    def get_trajectory(self) -> List[Tuple[float, int]]:
        return self.width_trajectory


class InterfaceRoughnessTracker(MetricTracker):
    """Tracks the full interface profile to calculate roughness (W²)."""

    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 1.0):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.roughness_history = []

    def after_step_hook(self):
        mean_q = self.sim.mean_front_position
        if mean_q < self.next_capture_q:
            return
        self.next_capture_q += self.capture_interval
        front_cells = self.sim.wt_front_cells.keys()
        if len(front_cells) < 2:
            return
        q_values = np.array([cell.q for cell in front_cells])
        w_squared = np.var(q_values)
        self.roughness_history.append((mean_q, w_squared))

    def get_roughness_trajectory(self) -> list[tuple[float, float]]:
        return self.roughness_history


class FixationTimeTracker(MetricTracker):
    """Tracks the time until a species (wild-type) is eliminated from the front."""

    def __init__(self, sim: "GillespieSimulation"):
        super().__init__(sim)
        self.fixation_time = -1.0  # -1 indicates not yet fixed

    def after_step_hook(self):
        if self.fixation_time < 0 and len(self.sim.wt_front_cells) == 0:
            self.fixation_time = self.sim.time

    def get_fixation_time(self) -> float:
        return self.fixation_time


class SpatialCorrelationTracker(MetricTracker):
    def __init__(
        self,
        sim: "GillespieSimulation",
        warmup_time: float,
        num_samples: int,
        sample_interval: float,
    ):
        super().__init__(sim)
        self.warmup_time = warmup_time
        self.num_samples = num_samples
        self.sample_interval = sample_interval
        self.next_sample_time = self.warmup_time
        self.correlation_sum = collections.defaultdict(lambda: [0.0, 0])
        self.samples_taken = 0

    def _get_spin(self, cell_type: int) -> int:
        return 1 if cell_type == 1 else -1

    def after_step_hook(self):
        if (
            self.samples_taken >= self.num_samples
            or self.sim.time < self.next_sample_time
        ):
            return
        self._calculate_and_accumulate_g_r()
        self.samples_taken += 1
        self.next_sample_time += self.sample_interval

    def _calculate_and_accumulate_g_r(self):
        front_cells_set = self.sim._front_lookup
        if len(front_cells_set) < 2:
            return
        adj = {
            cell: [
                n
                for n in self.sim._get_neighbors_periodic(cell)
                if n in front_cells_set
            ]
            for cell in front_cells_set
        }
        spins = {
            cell: self._get_spin(self.sim.population[cell]) for cell in front_cells_set
        }
        front_cells_list = list(front_cells_set)
        for i in range(len(front_cells_list)):
            source_cell = front_cells_list[i]
            q = collections.deque([source_cell])
            distance = {source_cell: 0}
            while q:
                current_cell = q.popleft()
                for neighbor in adj[current_cell]:
                    if neighbor not in distance:
                        distance[neighbor] = distance[current_cell] + 1
                        q.append(neighbor)
            for j in range(i + 1, len(front_cells_list)):
                target_cell = front_cells_list[j]
                if target_cell in distance:
                    dist = distance[target_cell]
                    product = spins[source_cell] * spins[target_cell]
                    self.correlation_sum[dist][0] += product
                    self.correlation_sum[dist][1] += 1

    def get_correlation_function(self) -> List[Tuple[int, float]]:
        if not self.correlation_sum:
            return []
        g_r_list = [
            (r, total_g / count)
            for r, (total_g, count) in self.correlation_sum.items()
            if count > 0
        ]
        g_r_list.sort(key=lambda x: x[0])
        return g_r_list


class MetricsManager:
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
