# FILE: src/metrics.py
# [DEFINITIVE, CONSOLIDATED VERSION]
# This file contains all metric trackers required for all experiments.
# It supports legacy runs, the primary campaign, and visualization scripts.

import numpy as np
import pandas as pd
from abc import ABC
from typing import List, Tuple, TYPE_CHECKING
import collections

if TYPE_CHECKING:
    # This block is for static type checkers and linters
    from linear_model import GillespieSimulation
    from fluctuating_model import FluctuatingGillespieSimulation


# ==============================================================================
# 1. ABSTRACT BASE CLASS
# ==============================================================================
class MetricTracker(ABC):
    """Abstract base class for all metric trackers."""

    def __init__(self, sim):
        self.sim = sim

    def initialize(self):
        pass

    def after_step_hook(self):
        pass

    def finalize(self):
        pass


# ==============================================================================
# 2. LEGACY TRACKERS (For Reproducibility of Older Experiments)
# ==============================================================================
class SteadyStateTracker(MetricTracker):
    """For legacy 'steady_state' runs. Measures mean mutant fraction after a warmup."""

    def __init__(self, sim, warmup_time: float, sample_interval: float):
        super().__init__(sim)
        self.warmup_time, self.sample_interval = warmup_time, sample_interval
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

    def __init__(self, sim, capture_interval: float = 1.0):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.width_trajectory: List[Tuple[float, int]] = []

    def _get_r_offset(self, cell) -> int:
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

    def __init__(self, sim, capture_interval: float = 0.5):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.roughness_history = []

    def after_step_hook(self):
        mean_q = self.sim.mean_front_position
        if mean_q < self.next_capture_q:
            return
        self.next_capture_q += self.capture_interval
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
        self.next_sample_time, self.samples_taken = self.warmup_time, 0
        self.g_r_sum, self.g_r_counts = collections.defaultdict(
            float
        ), collections.defaultdict(int)
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
        return {"g_r": g_r_list}


# ==============================================================================
# 3. PRIMARY CAMPAIGN TRACKER (For spatial_bet_hedging_v1)
# ==============================================================================
class FrontPropertiesTracker(MetricTracker):
    """
    Lightweight tracker for modern campaign runs. Calculates and returns summary
    statistics (mean and variance) of key metrics after a warmup period.
    """

    def __init__(self, sim, warmup_time, num_samples, sample_interval):
        super().__init__(sim)
        self.warmup_time, self.num_samples, self.sample_interval = (
            warmup_time,
            num_samples,
            sample_interval,
        )
        self.next_sample_time, self.samples_taken = self.warmup_time, 0
        (
            self.interface_density_samples,
            self.mutant_fraction_samples,
            self.front_speed_samples,
        ) = ([], [], [])
        self.last_sample_time, self.last_sample_q = -1.0, -1.0

    def initialize(self):
        self.last_sample_time = self.sim.time
        self.last_sample_q = self.sim.mean_front_position

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
        current_q, current_time = self.sim.mean_front_position, self.sim.time
        delta_q, delta_t = (
            current_q - self.last_sample_q,
            current_time - self.last_sample_time,
        )
        speed = (delta_q / delta_t) if delta_t > 1e-9 else 0.0
        self.front_speed_samples.append(speed)
        self.last_sample_q, self.last_sample_time = current_q, current_time
        self.samples_taken += 1
        self.next_sample_time += self.sample_interval

    def get_results(self):
        return {
            "avg_front_speed": (
                np.mean(self.front_speed_samples) if self.front_speed_samples else 0.0
            ),
            "var_front_speed": (
                np.var(self.front_speed_samples, ddof=1)
                if len(self.front_speed_samples) > 1
                else 0.0
            ),
            "avg_interface_density": (
                np.mean(self.interface_density_samples)
                if self.interface_density_samples
                else 0.0
            ),
            "avg_rho_M": (
                np.mean(self.mutant_fraction_samples)
                if self.mutant_fraction_samples
                else 0.0
            ),
            "var_rho_M": (
                np.var(self.mutant_fraction_samples, ddof=1)
                if len(self.mutant_fraction_samples) > 1
                else 0.0
            ),
        }


# ==============================================================================
# 4. VISUALIZATION & DEBUG TRACKER
# ==============================================================================
class FrontDynamicsTracker(MetricTracker):
    """
    Creates a detailed time-series of front properties by logging at fixed intervals
    of front advancement (delta_q). Used by debug and visualization scripts.
    """

    def __init__(self, sim, log_q_interval: float = 1.0):
        super().__init__(sim)
        self.log_q_interval = log_q_interval
        self.next_log_q = 0.0
        self.history = []
        self.last_log_time = 0.0
        self.last_log_q = 0.0

    def after_step_hook(self):
        current_q = self.sim.mean_front_position
        if current_q >= self.next_log_q:
            current_time = self.sim.time
            delta_q, delta_t = (
                current_q - self.last_log_q,
                current_time - self.last_log_time,
            )
            speed = (delta_q / delta_t) if delta_t > 1e-9 else 0.0
            self.history.append(
                {
                    "time": current_time,
                    "mutant_fraction": self.sim.mutant_fraction,
                    "mean_front_q": current_q,
                    "front_speed": speed,
                }
            )
            self.last_log_q, self.last_log_time = current_q, current_time
            self.next_log_q += self.log_q_interval

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


class TimeSeriesTracker(MetricTracker):
    """
    Creates a detailed time-series of key metrics by logging at fixed time intervals.
    Used for perturbation and relaxation dynamics experiments.
    """

    def __init__(self, sim, log_interval: float = 1.0):
        super().__init__(sim)
        self.log_interval = log_interval
        self.next_log_time = 0.0
        self.history = []

    def after_step_hook(self):
        if self.sim.time >= self.next_log_time:
            self.history.append(
                {
                    "time": self.sim.time,
                    "mutant_fraction": self.sim.mutant_fraction,
                }
            )
            self.next_log_time += self.log_interval

    def get_timeseries(self):
        return self.history


# ==============================================================================
# 5. ORCHESTRATOR CLASS
# ==============================================================================
class MetricsManager:
    """A manager to orchestrate the execution of multiple trackers."""

    def __init__(self):
        self.trackers: List[MetricTracker] = []
        self.sim = None

    def register_simulation(self, sim):
        self.sim = sim

    def add_tracker(self, tracker: MetricTracker):
        self.trackers.append(tracker)

    def initialize_all(self):
        for tracker in self.trackers:
            tracker.initialize()

    def after_step_hook(self):
        for tracker in self.trackers:
            tracker.after_step_hook()

    def finalize_all(self):
        for tracker in self.trackers:
            tracker.finalize()
