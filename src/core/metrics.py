# FILE: src/core/metrics.py
# A collection of modular metric trackers. [v5 - Corrected Stop Condition]

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.model import GillespieSimulation


class MetricTracker:
    """Abstract base class for all metric trackers."""

    def __init__(self, sim: "GillespieSimulation", **kwargs):
        self.sim = sim

    def initialize(self):
        """Called by the manager once the simulation is fully constructed."""
        pass

    def after_step_hook(self):
        """Called by the MetricsManager after each simulation step."""
        pass

    def is_done(self) -> bool:
        """Allows a tracker to signal that the simulation can stop early."""
        return False

    def finalize(self) -> Dict[str, Any]:
        """Called at the end of a simulation run to collect results."""
        return {}


class SectorWidthTracker(MetricTracker):
    """For 'calibration' runs. Measures the width of the mutant sector over front position."""

    def __init__(
        self, sim: "GillespieSimulation", capture_interval: float = 1.0, **kwargs
    ):
        super().__init__(sim=sim, **kwargs)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.width_trajectory: List[Tuple[float, float]] = []
        self._is_finished = False

    def initialize(self):
        self.after_step_hook()

    def is_done(self) -> bool:
        return self._is_finished

    def after_step_hook(self):
        mean_q = self.sim.mean_front_position
        if mean_q >= self.next_capture_q:
            width = self.sim.mutant_sector_width
            self.width_trajectory.append((mean_q, width))

            # [CRITICAL FIX] Correctly check for extinction (width=0) or fixation (width=sim.width).
            # This will now stop the simulation as soon as the sector vanishes.
            if width <= 0 or width >= self.sim.width:
                self._is_finished = True

            self.next_capture_q = (
                np.floor(mean_q / self.capture_interval) + 1
            ) * self.capture_interval

    def finalize(self) -> Dict[str, Any]:
        return {"trajectory": self.width_trajectory}


class InterfaceRoughnessTracker(MetricTracker):
    """For 'diffusion' runs. Measures W², the squared width of the front."""

    def __init__(
        self, sim: "GillespieSimulation", capture_interval: float = 0.5, **kwargs
    ):
        super().__init__(sim=sim, **kwargs)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.roughness_history: List[Tuple[float, float]] = []

    def initialize(self):
        self.after_step_hook()

    def after_step_hook(self):
        mean_q = self.sim.mean_front_position
        if mean_q >= self.next_capture_q:
            self.roughness_history.append((mean_q, self.sim.interface_width_sq))
            self.next_capture_q = (
                np.floor(mean_q / self.capture_interval) + 1
            ) * self.capture_interval

    def finalize(self) -> Dict[str, Any]:
        return {"roughness_trajectory": self.roughness_history}


class SteadyStatePropertiesTracker(MetricTracker):
    """For 'phase_diagram' etc. Measures mean properties after a warmup time."""

    def __init__(
        self,
        sim: "GillespieSimulation",
        warmup_time: float,
        num_samples: int,
        sample_interval: float,
        **kwargs
    ):
        super().__init__(sim=sim, **kwargs)
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")
        self.warmup_time, self.num_samples, self.sample_interval = (
            warmup_time,
            num_samples,
            sample_interval,
        )
        self.next_sample_time = self.warmup_time
        self.samples_taken = 0
        self.mutant_fraction_samples: List[float] = []
        self.interface_density_samples: List[float] = []

    def is_done(self) -> bool:
        return self.samples_taken >= self.num_samples

    def after_step_hook(self):
        if self.sim.time >= self.next_sample_time and not self.is_done():
            self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.interface_density_samples.append(self.sim.interface_density)
            self.next_sample_time += self.sample_interval
            self.samples_taken += 1

    def finalize(self) -> Dict[str, Any]:
        return {
            "avg_rho_M": (
                np.mean(self.mutant_fraction_samples)
                if self.mutant_fraction_samples
                else np.nan
            ),
            "var_rho_M": (
                np.var(self.mutant_fraction_samples, ddof=1)
                if len(self.mutant_fraction_samples) > 1
                else np.nan
            ),
            "avg_interface_density": (
                np.mean(self.interface_density_samples)
                if self.interface_density_samples
                else np.nan
            ),
        }


class TimeSeriesTracker(MetricTracker):
    """For 'relaxation' runs. Logs composition over time."""

    def __init__(self, sim: "GillespieSimulation", log_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        if log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        self.log_interval = log_interval
        self.next_log_time = 0.0
        self.history: List[Dict] = []

    def initialize(self):
        self.after_step_hook()

    def after_step_hook(self):
        while self.sim.time >= self.next_log_time:
            self.history.append(
                {
                    "time": self.next_log_time,
                    "mutant_fraction": self.sim.mutant_fraction,
                }
            )
            self.next_log_time += self.log_interval

    def finalize(self) -> Dict[str, Any]:
        return {"timeseries": self.history}


class FrontDynamicsTracker(MetricTracker):
    """For 'bet_hedging' runs. Logs properties over front position."""

    def __init__(self, sim: "GillespieSimulation", log_q_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        if log_q_interval <= 0:
            raise ValueError("log_q_interval must be positive.")
        self.log_q_interval = log_q_interval
        self.next_log_q = 0.0
        self.history: List[Dict] = []
        self.last_log_time, self.last_log_q = 0.0, 0.0

    def initialize(self):
        self.after_step_hook()

    def after_step_hook(self):
        current_q = self.sim.mean_front_position
        while current_q >= self.next_log_q:
            current_time = self.sim.time
            delta_q = self.next_log_q - self.last_log_q
            delta_t = current_time - self.last_log_time
            speed = (delta_q / delta_t) if delta_t > 1e-9 else 0.0
            self.history.append(
                {
                    "time": current_time,
                    "mean_front_q": self.next_log_q,
                    "mutant_fraction": self.sim.mutant_fraction,
                    "front_speed": speed,
                }
            )
            self.last_log_q, self.last_log_time = self.next_log_q, current_time
            self.next_log_q += self.log_q_interval

    def finalize(self) -> Dict[str, Any]:
        return {"front_dynamics": self.history}


class MetricsManager:
    """Manages a collection of metric trackers for a single simulation run."""

    def __init__(self):
        self._tracker_configs: List[Tuple[Type[MetricTracker], Dict[str, Any]]] = []
        self.trackers: List[MetricTracker] = []
        self._is_initialized = False

    def add_tracker(self, tracker_class: Type[MetricTracker], **kwargs):
        if self._is_initialized:
            raise RuntimeError("Cannot add trackers after simulation registration.")
        self._tracker_configs.append((tracker_class, kwargs))

    def register_simulation(self, sim: "GillespieSimulation"):
        if self._is_initialized:
            raise RuntimeError("Simulation already registered.")
        for tracker_class, kwargs in self._tracker_configs:
            self.trackers.append(tracker_class(sim=sim, **kwargs))
        self._is_initialized = True

    def initialize_all(self):
        if not self._is_initialized:
            return
        for tracker in self.trackers:
            tracker.initialize()

    def after_step_hook(self):
        if not self._is_initialized:
            return
        for tracker in self.trackers:
            tracker.after_step_hook()

    def is_done(self) -> bool:
        if not self._is_initialized:
            return False
        return any(tracker.is_done() for tracker in self.trackers)

    def finalize(self) -> Dict[str, Any]:
        if not self._is_initialized:
            return {}
        all_results = {}
        for tracker in self.trackers:
            all_results.update(tracker.finalize())
        return all_results
