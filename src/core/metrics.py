# FILE: src/core/metrics.py
# A collection of modular metric trackers compatible with the unified GillespieSimulation model. [v2 - Improved Manager Logic & API Consistency]

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.model import GillespieSimulation


class MetricTracker:
    """Abstract base class for all metric trackers.

    Each tracker is designed to be attached to a simulation instance and will
    be notified of each step via the `after_step_hook` method.
    """

    def __init__(self, sim: "GillespieSimulation"):
        self.sim = sim

    def after_step_hook(self):
        """Called by the MetricsManager after each simulation step."""
        pass

    def finalize(self) -> Dict[str, Any]:
        """Called at the end of a simulation run to collect results."""
        return {}


class SectorWidthTracker(MetricTracker):
    """For 'calibration' runs. Measures the width of the mutant sector over front position."""

    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 1.0):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.width_trajectory: List[Tuple[float, float]] = []

    def after_step_hook(self):
        mean_q = self.sim.mean_front_position
        if mean_q >= self.next_capture_q:
            self.width_trajectory.append((mean_q, self.sim.mutant_sector_width))
            # In case of large time steps, advance to the next multiple of the interval
            self.next_capture_q = (
                np.floor(mean_q / self.capture_interval) + 1
            ) * self.capture_interval

    def finalize(self) -> Dict[str, Any]:
        return {"trajectory": self.width_trajectory}


class InterfaceRoughnessTracker(MetricTracker):
    """For 'diffusion' runs. Measures W², the squared width of the front."""

    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 0.5):
        super().__init__(sim)
        self.capture_interval = capture_interval
        self.next_capture_q = 0.0
        self.roughness_history: List[Tuple[float, float]] = []

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
    """For 'structure_analysis'. Measures mean properties after a warmup time."""

    def __init__(
        self,
        sim: "GillespieSimulation",
        warmup_time: float,
        num_samples: int,
        sample_interval: float,
    ):
        super().__init__(sim)
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")
        self.warmup_time = warmup_time
        self.num_samples = num_samples
        self.sample_interval = sample_interval
        self.next_sample_time = self.warmup_time
        self.samples_taken = 0
        self.mutant_fraction_samples: List[float] = []
        self.interface_density_samples: List[float] = []

    def after_step_hook(self):
        if (
            self.samples_taken < self.num_samples
            and self.sim.time >= self.next_sample_time
        ):
            self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.interface_density_samples.append(self.sim.interface_density)
            self.next_sample_time += self.sample_interval
            self.samples_taken += 1

    def finalize(self) -> Dict[str, Any]:
        return {
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
            "avg_interface_density": (
                np.mean(self.interface_density_samples)
                if self.interface_density_samples
                else 0.0
            ),
        }


class TimeSeriesTracker(MetricTracker):
    """For 'relaxation' and 'perturbation' runs. Logs composition over time."""

    def __init__(self, sim: "GillespieSimulation", log_interval: float):
        super().__init__(sim)
        if log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        self.log_interval = log_interval
        self.next_log_time = 0.0
        self.history: List[Dict] = []

    def after_step_hook(self):
        if self.sim.time >= self.next_log_time:
            self.history.append(
                {"time": self.sim.time, "mutant_fraction": self.sim.mutant_fraction}
            )
            self.next_log_time = (
                np.floor(self.sim.time / self.log_interval) + 1
            ) * self.log_interval

    def finalize(self) -> Dict[str, Any]:
        return {"timeseries": self.history}


class FrontDynamicsTracker(MetricTracker):
    """For 'spatial_fluctuation_analysis'. Logs properties over front position."""

    def __init__(self, sim: "GillespieSimulation", log_q_interval: float):
        super().__init__(sim)
        if log_q_interval <= 0:
            raise ValueError("log_q_interval must be positive.")
        self.log_q_interval = log_q_interval
        self.next_log_q = 0.0
        self.history: List[Dict] = []
        self.last_log_time = 0.0
        self.last_log_q = 0.0

    def after_step_hook(self):
        current_q = self.sim.mean_front_position
        if current_q >= self.next_log_q:
            current_time = self.sim.time
            delta_q = current_q - self.last_log_q
            delta_t = current_time - self.last_log_time
            speed = (delta_q / delta_t) if delta_t > 1e-9 else 0.0

            self.history.append(
                {
                    "time": current_time,
                    "mean_front_q": current_q,
                    "mutant_fraction": self.sim.mutant_fraction,
                    "front_speed": speed,
                }
            )

            self.last_log_q, self.last_log_time = current_q, current_time
            self.next_log_q = (
                np.floor(current_q / self.log_q_interval) + 1
            ) * self.log_q_interval

    def finalize(self) -> Dict[str, Any]:
        """Returns the collected data as a dictionary.

        The calling code can easily convert this to a pandas DataFrame via:
        `pd.DataFrame(results['front_dynamics'])`
        """
        return {"front_dynamics": self.history}


class MetricsManager:
    """Manages a collection of metric trackers for a single simulation run."""

    def __init__(self):
        """Initializes the manager.

        The workflow is:
        1. Instantiate MetricsManager.
        2. Use `add_tracker` to register tracker configurations.
        3. In the simulation's __init__, call `register_simulation` to
           instantiate trackers with the simulation instance.
        """
        self._tracker_configs: List[Tuple[Type[MetricTracker], Dict[str, Any]]] = []
        self.trackers: List[MetricTracker] = []
        self._is_initialized = False

    def add_tracker(self, tracker_class: Type[MetricTracker], **kwargs):
        """Registers a tracker's class and its configuration arguments.

        The tracker itself will be instantiated later when register_simulation is called.
        """
        if self._is_initialized:
            raise RuntimeError(
                "Cannot add new trackers after simulation has been registered."
            )
        self._tracker_configs.append((tracker_class, kwargs))

    def register_simulation(self, sim: "GillespieSimulation"):
        """Instantiates all configured trackers with the given simulation instance."""
        if self._is_initialized:
            raise RuntimeError("Simulation has already been registered.")

        for tracker_class, kwargs in self._tracker_configs:
            self.trackers.append(tracker_class(sim=sim, **kwargs))
        self._is_initialized = True

    def after_step_hook(self):
        """Propagates the after-step hook to all registered trackers."""
        if not self._is_initialized:
            return
        for tracker in self.trackers:
            tracker.after_step_hook()

    def finalize(self) -> Dict[str, Any]:
        """Collects and aggregates results from all trackers."""
        if not self._is_initialized:
            return {}

        all_results = {}
        for tracker in self.trackers:
            all_results.update(tracker.finalize())
        return all_results
