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
        """Calculates the mean of the collected samples."""
        return (
            np.mean(self.mutant_fraction_samples)
            if self.mutant_fraction_samples
            else np.nan
        )

    def get_steady_state_mutant_variance(self) -> float:
        """Calculates the variance of the collected samples."""
        if len(self.mutant_fraction_samples) < 2:
            return 0.0
        return np.var(self.mutant_fraction_samples, ddof=1)

    def get_steady_state_sample_count(self) -> int:
        """Returns the number of samples collected."""
        return len(self.mutant_fraction_samples)


class SectorWidthTracker(MetricTracker):
    # This tracker is correct and unchanged.
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
    # This tracker is correct and unchanged.
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


# ==============================================================================
# [UPGRADED] The new, high-performance, data-rich tracker
# ==============================================================================
class CorrelationAndStructureTracker(MetricTracker):
    """
    An advanced tracker that efficiently measures the spatial correlation g(r),
    domain size distribution, and interface density of the growing front.
    It uses graph memoization for a significant performance increase.
    """

    def __init__(self, sim, warmup_time, num_samples, sample_interval):
        super().__init__(sim)
        self.warmup_time = warmup_time
        self.num_samples = num_samples
        self.sample_interval = sample_interval
        self.next_sample_time = self.warmup_time
        self.samples_taken = 0

        # Data accumulators
        self.g_r_sum = collections.defaultdict(float)
        self.g_r_counts = collections.defaultdict(int)
        self.domain_size_counts = collections.defaultdict(int)
        self.interface_density_sum = 0.0

        # Performance optimization: memoized graph
        self._adj = {}
        self._last_front_set = set()

    def _get_spin(self, cell_type):
        return 1 if cell_type == 1 else -1

    def _update_graph(self):
        """Only rebuilds the graph if the set of front cells has changed."""
        current_front_set = self.sim._front_lookup
        if current_front_set == self._last_front_set:
            return  # No change, no need to update

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

        self._update_graph()  # Efficiently update the graph representation
        self._sample_structure()  # Perform all measurements in one pass

        self.samples_taken += 1
        self.next_sample_time += self.sample_interval

    def _sample_structure(self):
        """Calculates all structural properties in a single, efficient pass."""
        front_cells = list(self._last_front_set)
        if len(front_cells) < 2:
            return

        spins = {
            cell: self._get_spin(self.sim.population[cell]) for cell in front_cells
        }

        # --- 1. Calculate g(r) using BFS from each node ---
        for i in range(len(front_cells)):
            source = front_cells[i]
            q = collections.deque([(source, 0)])
            distances = {source: 0}

            # BFS to find distances
            head = 0
            while head < len(q):
                curr, dist = q[head]
                head += 1
                for neighbor in self._adj[curr]:
                    if neighbor not in distances:
                        distances[neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

            # Accumulate correlation products
            for j in range(i + 1, len(front_cells)):
                target = front_cells[j]
                if target in distances:
                    d = distances[target]
                    self.g_r_sum[d] += spins[source] * spins[target]
                    self.g_r_counts[d] += 1

        # --- 2. Calculate Domain Sizes and Interface Density (one graph traversal) ---
        visited = set()
        total_interfaces = 0
        total_neighbors = 0
        for cell in front_cells:
            if cell not in visited:
                domain_size = 0
                cell_type = self.sim.population[cell]
                q = collections.deque([cell])
                visited.add(cell)
                while q:
                    curr = q.popleft()
                    domain_size += 1
                    for neighbor in self._adj[curr]:
                        total_neighbors += 1
                        if self.sim.population[neighbor] != cell_type:
                            total_interfaces += 1
                        if (
                            neighbor not in visited
                            and self.sim.population[neighbor] == cell_type
                        ):
                            visited.add(neighbor)
                            q.append(neighbor)
                self.domain_size_counts[domain_size] += 1

        if total_neighbors > 0:
            # Each interface is counted twice, so divide by 2.
            # Each neighbor pair is counted twice, so divide total_neighbors by 2.
            self.interface_density_sum += (total_interfaces / 2) / (total_neighbors / 2)

    def get_results(self):
        """Returns a dictionary of all calculated metrics."""
        g_r_list = sorted(
            [
                (r, self.g_r_sum[r] / self.g_r_counts[r])
                for r in self.g_r_counts
                if self.g_r_counts[r] > 0
            ]
        )

        domain_dist = sorted(self.domain_size_counts.items())

        avg_interface_density = (
            self.interface_density_sum / self.samples_taken
            if self.samples_taken > 0
            else 0.0
        )

        return {
            "g_r": g_r_list,
            "domain_size_distribution": domain_dist,
            "avg_interface_density": avg_interface_density,
        }


class MetricsManager:
    # This manager is correct and unchanged.
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
