# FILE: src/core/metrics.py
# A collection of modular metric trackers. [v11 - Terminology Aligned with Literature]

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.model import GillespieSimulation


class MetricTracker:
    def __init__(self, sim: "GillespieSimulation", **kwargs):
        self.sim = sim

    def initialize(self):
        pass

    def after_step_hook(self):
        pass

    def is_done(self) -> bool:
        return False

    def finalize(self) -> Dict[str, Any]:
        return {}


class SectorWidthTracker(MetricTracker):
    # ... (class is unchanged)
    def __init__(self, sim: "GillespieSimulation", **kwargs):
        super().__init__(sim, **kwargs)
        self.trajectory: List[Tuple[float, float]] = []
        self.started = False
        self._is_done = False
        self.result: Dict[str, Any] = {}

    def after_step_hook(self):
        current_width = self.sim.mutant_sector_width
        self.trajectory.append((self.sim.time, current_width))
        if not self.started:
            self.started = True
        if current_width == 0:
            self.result["outcome"] = "extinction"
            self.result["time_to_outcome"] = self.sim.time
            self._is_done = True
        elif current_width == self.sim.width:
            self.result["outcome"] = "fixation"
            self.result["time_to_outcome"] = self.sim.time
            self._is_done = True

    def is_done(self) -> bool:
        return self.started and self._is_done

    def finalize(self) -> dict:
        final_data = {"trajectory": self.trajectory}
        final_data.update(self.result)
        return final_data


class InterfaceRoughnessTracker(MetricTracker):
    """For 'diffusion' runs. Measures W², the squared roughness of the front."""

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
            self.roughness_history.append((mean_q, self.sim.front_roughness_sq))
            self.next_capture_q = (
                np.floor(mean_q / self.capture_interval) + 1
            ) * self.capture_interval

    def finalize(self) -> Dict[str, Any]:
        return {"roughness_sq_trajectory": self.roughness_history}


class SteadyStatePropertiesTracker(MetricTracker):
    """For 'phase_diagram' etc. Measures mean properties after a warmup time."""

    def __init__(
        self,
        sim: "GillespieSimulation",
        warmup_time: float,
        num_samples: int,
        sample_interval: float,
        **kwargs,
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
        self.front_length_samples: List[float] = []
        self.domain_boundary_samples: List[float] = []

    def is_done(self) -> bool:
        return self.samples_taken >= self.num_samples

    def after_step_hook(self):
        if self.sim.time >= self.next_sample_time and not self.is_done():
            self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.front_length_samples.append(self.sim.expanding_front_length)
            self.domain_boundary_samples.append(self.sim.domain_boundary_length)
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
            "avg_front_length": (
                np.mean(self.front_length_samples)
                if self.front_length_samples
                else np.nan
            ),
            "avg_domain_boundary_length": (
                np.mean(self.domain_boundary_samples)
                if self.domain_boundary_samples
                else np.nan
            ),
        }


class TimeSeriesTracker(MetricTracker):
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

class RecoveryDynamicsTracker(MetricTracker):
    """
    Exhaustive tracker for recovery/relaxation experiments.
    - Logs a detailed timeseries of the relaxation process.
    - After relaxation, samples steady-state properties of the new equilibrium.
    """
    def __init__(
        self, sim: "GillespieSimulation",
        timeseries_interval: float,
        warmup_time_ss: float, # SS = Steady State
        num_samples_ss: int,
        sample_interval_ss: float,
        **kwargs
    ):
        super().__init__(sim=sim, **kwargs)
        self.timeseries_interval = timeseries_interval
        self.warmup_time_ss = warmup_time_ss
        self.num_samples_ss = num_samples_ss
        self.sample_interval_ss = sample_interval_ss
        
        # For timeseries part
        self.next_log_time = 0.0
        self.timeseries_history = []
        
        # For steady-state part
        self.is_warmed_up = False
        self.next_sample_time_ss = 0.0
        self.samples_taken_ss = 0
        self.mutant_fraction_samples_ss = []
        self.front_speed_samples_ss = []
        self.last_sample_q_ss = 0.0
        self.last_sample_time_ss = 0.0

    def after_step_hook(self):
        # --- Part 1: Log the timeseries continuously ---
        while self.sim.time >= self.next_log_time:
            self.timeseries_history.append({
                "time": self.next_log_time,
                "mutant_fraction": self.sim.mutant_fraction,
            })
            self.next_log_time += self.timeseries_interval

        # --- Part 2: Sample for steady-state properties after warmup ---
        if not self.is_warmed_up and self.sim.time >= self.warmup_time_ss:
            self.is_warmed_up = True
            self.last_sample_q_ss = self.sim.mean_front_position
            self.last_sample_time_ss = self.sim.time
            self.next_sample_time_ss = self.sim.time + self.sample_interval_ss

        if self.is_warmed_up and self.sim.time >= self.next_sample_time_ss and self.samples_taken_ss < self.num_samples_ss:
            current_q = self.sim.mean_front_position
            current_time = self.sim.time
            delta_q = current_q - self.last_sample_q_ss
            delta_t = current_time - self.last_sample_time_ss
            
            self.mutant_fraction_samples_ss.append(self.sim.mutant_fraction)
            self.front_speed_samples_ss.append(delta_q / delta_t if delta_t > 1e-9 else 0.0)

            self.last_sample_q_ss = current_q
            self.last_sample_time_ss = current_time
            self.next_sample_time_ss += self.sample_interval_ss
            self.samples_taken_ss += 1

    def finalize(self) -> Dict[str, Any]:
        return {
            "timeseries": self.timeseries_history,
            "avg_rho_M_final": np.mean(self.mutant_fraction_samples_ss) if self.mutant_fraction_samples_ss else np.nan,
            "var_rho_M_final": np.var(self.mutant_fraction_samples_ss, ddof=1) if len(self.mutant_fraction_samples_ss) > 1 else np.nan,
            "avg_front_speed_final": np.mean(self.front_speed_samples_ss) if self.front_speed_samples_ss else np.nan,
        }

class HomogeneousDynamicsTracker(MetricTracker):
    """
    Enhanced version of SteadyStatePropertiesTracker for precise measurements
    in homogeneous environments. Adds front speed calculation.
    """
    def __init__(self, sim: "GillespieSimulation", warmup_time: float, num_samples: int, sample_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.warmup_time = warmup_time
        self.num_samples = num_samples
        self.sample_interval = sample_interval
        
        self.is_warmed_up = False
        self.next_sample_time = 0.0
        self.samples_taken = 0
        
        self.mutant_fraction_samples = []
        self.front_speed_samples = []
        
        self.last_sample_q = 0.0
        self.last_sample_time = 0.0

    def is_done(self) -> bool:
        return self.samples_taken >= self.num_samples

    def after_step_hook(self):
        if not self.is_warmed_up and self.sim.time >= self.warmup_time:
            self.is_warmed_up = True
            self.last_sample_q = self.sim.mean_front_position
            self.last_sample_time = self.sim.time
            self.next_sample_time = self.sim.time + self.sample_interval

        if self.is_warmed_up and self.sim.time >= self.next_sample_time and not self.is_done():
            current_q = self.sim.mean_front_position
            current_time = self.sim.time
            delta_q = current_q - self.last_sample_q
            delta_t = current_time - self.last_sample_time
            
            self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.front_speed_samples.append(delta_q / delta_t if delta_t > 1e-9 else 0.0)

            self.last_sample_q = current_q
            self.last_sample_time = current_time
            self.next_sample_time += self.sample_interval
            self.samples_taken += 1

    def finalize(self) -> Dict[str, Any]:
        return {
            "avg_rho_M": np.mean(self.mutant_fraction_samples) if self.mutant_fraction_samples else np.nan,
            "var_rho_M": np.var(self.mutant_fraction_samples, ddof=1) if len(self.mutant_fraction_samples) > 1 else np.nan,
            "avg_front_speed": np.mean(self.front_speed_samples) if self.front_speed_samples else np.nan,
        }



class MetricsManager:
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
