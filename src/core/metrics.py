# FILE: src/core/metrics.py (DEFINITIVELY CORRECTED AND COMPLETE)

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Type, TYPE_CHECKING, Optional
from collections import deque
from scipy.spatial import distance

if TYPE_CHECKING:
    from src.core.model import GillespieSimulation
    from src.core.model_aif import AifModelSimulation
from src.config import PARAM_GRID


class MetricTracker:
    """Base class for all metric trackers."""
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

# --- ALL ORIGINAL TRACKERS RESTORED, WITH SPEED LOGIC CORRECTED ---

class InvasionOutcomeTracker(MetricTracker):
    """
    A purpose-built, efficient tracker for invasion probability experiments.
    """
    def __init__(
        self,
        sim: "GillespieSimulation",
        progress_threshold_frac: float = 0.25,
        progress_check_time: float = 20000.0,
        **kwargs,
    ):
        super().__init__(sim=sim, **kwargs)
        self.progress_threshold_width = sim.width * progress_threshold_frac
        self.progress_check_time = progress_check_time
        self._outcome: Optional[str] = None
        self._time_to_outcome: Optional[float] = None
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        current_width = self.sim.mutant_sector_width
        if current_width == 0:
            self._outcome, self._time_to_outcome, self._is_done = "extinction", self.sim.time, True
            return
        if current_width >= self.sim.width:
            self._outcome, self._time_to_outcome, self._is_done = "fixation", self.sim.time, True
            return
        if self.sim.time > self.progress_check_time:
            if current_width < self.progress_threshold_width:
                self._outcome, self._time_to_outcome, self._is_done = "extinction_by_timeout", self.sim.time, True
                return

    def finalize(self) -> Dict[str, Any]:
        if self._outcome is None:
            self._outcome, self._time_to_outcome = "undecided_timeout", self.sim.time
        return {"outcome": self._outcome, "time_to_outcome": self._time_to_outcome}


# FILE: src/core/metrics.py (The Corrected Tracker)

class BoundaryDynamicsTracker(MetricTracker):
    """
    For 'boundary_analysis' (calibration) runs.

    CORRECTED to measure mutant width ONLY on the expanding front and to
    terminate the simulation when the mutant lineage is lost from the front.
    The unnecessary 'num_fragments' field has been removed.
    """
    def __init__(self, sim: "GillespieSimulation", **kwargs):
        super().__init__(sim, **kwargs)
        self.trajectory: List[Tuple[float, float]] = []
        self._is_done = False
        self.result: Dict[str, Any] = {}
        self.outcome_recorded = False

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        # --- FIX 1: Measure the width based on the front cells only ---
        if not self.sim.m_front_cells:
            current_front_width = 0
        else:
            # The width is the number of unique 'r' coordinates on the front
            front_r_coords = {h.r for h in self.sim.m_front_cells.keys()}
            current_front_width = len(front_r_coords)

        # Append this correct measurement to the trajectory.
        self.trajectory.append((self.sim.time, current_front_width))

        # --- FIX 2: Check for termination based on the state of the front ---
        if self.outcome_recorded:
            return

        front_has_mutants = bool(self.sim.m_front_cells)
        front_has_wildtype = bool(self.sim.wt_front_cells)

        # Only check after the simulation has properly started
        if self.sim.step_count > 0:
            if not front_has_mutants:
                self.result["outcome"] = "mutant_front_extinction"
                self.result["time_to_outcome"] = self.sim.time
                self._is_done = True
                self.outcome_recorded = True
            elif not front_has_wildtype:
                self.result["outcome"] = "mutant_front_fixation"
                self.result["time_to_outcome"] = self.sim.time
                self._is_done = True
                self.outcome_recorded = True

    def finalize(self) -> dict:
        # Record a timeout if no other outcome was reached
        if not self.outcome_recorded:
            self.result["outcome"] = "undecided_timeout"
            self.result["time_to_outcome"] = self.sim.time
        
        final_data = {"trajectory": self.trajectory}
        final_data.update(self.result)
        
        # --- FIX 3: 'num_fragments' has been removed ---
        
        return final_data

class FixationTimeTracker(MetricTracker):
    """
    For experiments that should terminate upon mutant fixation or extinction.
    """
    def __init__(self, sim: "GillespieSimulation", **kwargs):
        super().__init__(sim, **kwargs)
        self._is_done = False
        self.result: Dict[str, Any] = {}
        self.outcome_recorded = False
        self.max_mutant_q_reached = 0.0

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        if self.sim.m_front_cells:
            current_max_mutant_q = max(h.q for h in self.sim.m_front_cells)
            self.max_mutant_q_reached = max(self.max_mutant_q_reached, current_max_mutant_q)
        if self.outcome_recorded:
            return
        front_has_wt = len(self.sim.wt_front_cells) > 0
        front_has_m = len(self.sim.m_front_cells) > 0
        if not (front_has_wt and front_has_m) and self.sim.step_count > 0:
            self.result["outcome"] = "fixation" if front_has_m else "extinction"
            self.result["time_to_outcome"], self.result["q_at_outcome"] = self.sim.time, self.max_mutant_q_reached
            self._is_done, self.outcome_recorded = True, True

    def finalize(self) -> Dict[str, Any]:
        if not self.outcome_recorded:
            self.result["outcome"], self.result["time_to_outcome"], self.result["q_at_outcome"] = "boundary_hit", self.sim.time, self.max_mutant_q_reached
        self.result["num_fragments"] = self.sim.initial_num_fragments
        return self.result


class RelaxationConvergenceTracker(MetricTracker):
    """
    For 'relaxation_converged' runs.
    """
    def __init__(self, sim: "GillespieSimulation", sample_interval: float, convergence_window: int, convergence_threshold: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.sample_interval, self.convergence_window, self.convergence_threshold = sample_interval, convergence_window, convergence_threshold
        self.next_log_time = 0.0
        self.history: List[Dict] = []
        self.rho_m_deque = deque(maxlen=convergence_window)
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        while self.sim.time >= self.next_log_time:
            current_rho_m = self.sim.mutant_fraction
            self.history.append({"time": self.next_log_time, "mutant_fraction": current_rho_m})
            self.rho_m_deque.append(current_rho_m)
            self.next_log_time += self.sample_interval
            if len(self.rho_m_deque) == self.convergence_window:
                mean_rho = np.mean(self.rho_m_deque)
                if (mean_rho > 1e-6 and (np.std(self.rho_m_deque, ddof=1) / mean_rho) < self.convergence_threshold) or \
                   (mean_rho <= 1e-6 and np.std(self.rho_m_deque, ddof=1) < self.convergence_threshold):
                    self._is_done = True
                    break

    def finalize(self) -> Dict[str, Any]:
        return {"timeseries": self.history}


class InterfaceRoughnessTracker(MetricTracker):
    """
    RESTORED. For 'diffusion' runs. Measures W², the squared roughness of the front.
    """
    def __init__(self, sim: "GillespieSimulation", capture_interval: float = 0.5, **kwargs):
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
            self.next_capture_q = (np.floor(mean_q / self.capture_interval) + 1) * self.capture_interval

    def finalize(self) -> Dict[str, Any]:
        return {"roughness_sq_trajectory": self.roughness_history}


class SteadyStatePropertiesTracker(MetricTracker):
    """
    For 'phase_diagram' etc. Measures mean properties after a warmup time.
    """
    def __init__(self, sim: "GillespieSimulation", warmup_time: float, num_samples: int, sample_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.warmup_time, self.num_samples, self.sample_interval = warmup_time, num_samples, sample_interval
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
            self.domain_boundary_samples.append(self.sim.expanding_front_length)
            self.next_sample_time += self.sample_interval
            self.samples_taken += 1

    def finalize(self) -> Dict[str, Any]:
        return {
            "avg_rho_M": np.mean(self.mutant_fraction_samples) if self.mutant_fraction_samples else np.nan,
            "var_rho_M": np.var(self.mutant_fraction_samples, ddof=1) if len(self.mutant_fraction_samples) > 1 else np.nan,
            "avg_front_length": np.mean(self.front_length_samples) if self.front_length_samples else np.nan,
            "avg_domain_boundary_length": np.mean(self.domain_boundary_samples) if self.domain_boundary_samples else np.nan,
        }


class TimeSeriesTracker(MetricTracker):
    """RESTORED. Generic timeseries logger."""
    def __init__(self, sim: "GillespieSimulation", log_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.log_interval = log_interval
        self.next_log_time = 0.0
        self.history: List[Dict] = []
    def initialize(self):
        self.after_step_hook()
    def after_step_hook(self):
        while self.sim.time >= self.next_log_time:
            self.history.append({"time": self.next_log_time, "mutant_fraction": self.sim.mutant_fraction})
            self.next_log_time += self.log_interval
    def finalize(self) -> Dict[str, Any]:
        return {"timeseries": self.history}


class FrontDynamicsTracker(MetricTracker):
    """CORRECTED speed logic."""
    def __init__(self, sim: "GillespieSimulation", log_q_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.log_q_interval = log_q_interval
        self.next_log_q = 0.0
        self.history: List[Dict] = []
        self.last_pos, self.last_time, self.last_log_time, self.last_log_q = 0.0, 0.0, 0.0, 0.0

    def initialize(self):
        self.last_pos = self.last_log_q = self.sim.mean_front_position
        self.last_time = self.last_log_time = self.sim.time
        self.next_log_q = self.sim.mean_front_position + self.log_q_interval

    def after_step_hook(self):
        current_pos, current_time = self.sim.mean_front_position, self.sim.time
        while current_pos >= self.next_log_q:
            step_dist, step_time = current_pos - self.last_pos, current_time - self.last_time
            fraction_of_step = (self.next_log_q - self.last_pos) / step_dist if step_dist > 1e-9 else 1.0
            time_at_crossing = self.last_time + fraction_of_step * step_time
            delta_q, delta_t = self.next_log_q - self.last_log_q, time_at_crossing - self.last_log_time
            speed = (delta_q / delta_t) if delta_t > 1e-9 else 0.0
            self.history.append({"time": time_at_crossing, "mean_front_q": self.next_log_q, "mutant_fraction": self.sim.mutant_fraction, "front_speed": speed})
            self.last_log_q, self.last_log_time = self.next_log_q, time_at_crossing
            self.next_log_q += self.log_q_interval
        self.last_pos, self.last_time = current_pos, current_time

    def finalize(self) -> Dict[str, Any]:
        return {"front_dynamics": self.history}


class RecoveryDynamicsTracker(MetricTracker):
    """CORRECTED speed logic."""
    def __init__(self, sim: "GillespieSimulation", timeseries_interval: float, warmup_time_ss: float, num_samples_ss: int, sample_interval_ss: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.timeseries_interval, self.warmup_time_ss, self.num_samples_ss, self.sample_interval_ss = timeseries_interval, warmup_time_ss, num_samples_ss, sample_interval_ss
        self.next_log_time = 0.0
        self.timeseries_history = []
        self.is_warmed_up, self.next_sample_time_ss, self.samples_taken_ss = False, 0.0, 0
        self.mutant_fraction_samples_ss, self.front_speed_samples_ss = [], []
        self.last_pos, self.last_time, self.last_sample_q_ss, self.last_sample_time_ss = 0.0, 0.0, 0.0, 0.0

    def after_step_hook(self):
        current_pos, current_time = self.sim.mean_front_position, self.sim.time
        while current_time >= self.next_log_time:
            self.timeseries_history.append({"time": self.next_log_time, "mutant_fraction": self.sim.mutant_fraction})
            self.next_log_time += self.timeseries_interval
        if not self.is_warmed_up and current_time >= self.warmup_time_ss:
            self.is_warmed_up = True
            self.last_sample_q_ss, self.last_sample_time_ss = current_pos, current_time
            self.next_sample_time_ss = current_time + self.sample_interval_ss
        while self.is_warmed_up and current_time >= self.next_sample_time_ss and self.samples_taken_ss < self.num_samples_ss:
            step_time, step_dist = current_time - self.last_time, current_pos - self.last_pos
            fraction_of_step = (self.next_sample_time_ss - self.last_time) / step_time if step_time > 1e-9 else 1.0
            pos_at_sample_time = self.last_pos + fraction_of_step * step_dist
            dist_this_interval, time_this_interval = pos_at_sample_time - self.last_sample_q_ss, self.next_sample_time_ss - self.last_sample_time_ss
            if time_this_interval > 1e-9:
                self.front_speed_samples_ss.append(dist_this_interval / time_this_interval)
                self.mutant_fraction_samples_ss.append(self.sim.mutant_fraction)
            self.last_sample_q_ss, self.last_sample_time_ss = pos_at_sample_time, self.next_sample_time_ss
            self.next_sample_time_ss += self.sample_interval_ss
            self.samples_taken_ss += 1
        self.last_pos, self.last_time = current_pos, current_time

    def finalize(self) -> Dict[str, Any]:
        return {
            "timeseries": self.timeseries_history,
            "avg_rho_M_final": np.mean(self.mutant_fraction_samples_ss) if self.mutant_fraction_samples_ss else np.nan,
            "var_rho_M_final": np.var(self.mutant_fraction_samples_ss, ddof=1) if len(self.mutant_fraction_samples_ss) > 1 else np.nan,
            "avg_front_speed_final": np.mean(self.front_speed_samples_ss) if self.front_speed_samples_ss else np.nan,
        }


class HomogeneousDynamicsTracker(MetricTracker):
    """CORRECTED speed logic."""
    def __init__(self, sim: "GillespieSimulation", warmup_time: float, num_samples: int, sample_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.warmup_time, self.num_samples, self.sample_interval = warmup_time, num_samples, sample_interval
        self.is_warmed_up, self.next_sample_time, self.samples_taken = False, 0.0, 0
        self.mutant_fraction_samples, self.front_speed_samples = [], []
        self.last_pos, self.last_time, self.last_sample_q, self.last_sample_time = 0.0, 0.0, 0.0, 0.0

    def is_done(self) -> bool:
        return self.samples_taken >= self.num_samples

    def after_step_hook(self):
        current_pos, current_time = self.sim.mean_front_position, self.sim.time
        if not self.is_warmed_up and current_time >= self.warmup_time:
            self.is_warmed_up = True
            self.last_sample_q, self.last_sample_time = current_pos, current_time
            self.next_sample_time = current_time + self.sample_interval
        while self.is_warmed_up and current_time >= self.next_sample_time and not self.is_done():
            step_time, step_dist = current_time - self.last_time, current_pos - self.last_pos
            fraction_of_step = (self.next_sample_time - self.last_time) / step_time if step_time > 1e-9 else 1.0
            pos_at_sample_time = self.last_pos + fraction_of_step * step_dist
            dist_this_interval, time_this_interval = pos_at_sample_time - self.last_sample_q, self.next_sample_time - self.last_sample_time
            if time_this_interval > 1e-9:
                self.front_speed_samples.append(dist_this_interval / time_this_interval)
                self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.last_sample_q, self.last_sample_time = pos_at_sample_time, self.next_sample_time
            self.next_sample_time += self.sample_interval
            self.samples_taken += 1
        self.last_pos, self.last_time = current_pos, current_time

    def finalize(self) -> Dict[str, Any]:
        return {
            "avg_rho_M": np.mean(self.mutant_fraction_samples) if self.mutant_fraction_samples else np.nan,
            "var_rho_M": np.var(self.mutant_fraction_samples, ddof=1) if len(self.mutant_fraction_samples) > 1 else np.nan,
            "avg_front_speed": np.mean(self.front_speed_samples) if self.front_speed_samples else np.nan,
        }


class CyclicTimeSeriesTracker(MetricTracker):
    """RESTORED. For 'tracking_analysis' runs."""
    def __init__(self, sim: "GillespieSimulation", warmup_cycles: int, measure_cycles: int, sample_interval: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.warmup_cycles, self.measure_cycles, self.sample_interval = warmup_cycles, measure_cycles, sample_interval
        self.cycle_q = self.sim.patch_sequence[0][1] * len(set(p[0] for p in self.sim.patch_sequence))
        if self.cycle_q <= 0: raise ValueError("CyclicTimeSeriesTracker requires a defined positive cycle length.")
        self.state, self.cycles_completed, self.total_cycles = "warming_up", 0, warmup_cycles + measure_cycles
        self.history, self.next_log_time, self.measurement_start_time = [], -1.0, -1.0

    def is_done(self) -> bool:
        return self.state == "done"

    def after_step_hook(self):
        if self.state == "done": return
        while self.sim.mean_front_position >= (self.cycles_completed + 1) * self.cycle_q:
            self.cycles_completed += 1
            if self.state == "warming_up" and self.cycles_completed >= self.warmup_cycles:
                self.state, self.measurement_start_time, self.next_log_time = "measuring", self.sim.time, self.sim.time
            if self.cycles_completed >= self.total_cycles:
                self.state = "done"; break
        if self.state == "measuring":
            while self.sim.time >= self.next_log_time:
                if self.state != "done":
                    self.history.append({"time": self.next_log_time - self.measurement_start_time, "mutant_fraction": self.sim.mutant_fraction})
                self.next_log_time += self.sample_interval

    def finalize(self) -> Dict[str, Any]:
        return {"timeseries": self.history}


class FrontConvergenceTracker(MetricTracker):
    """
    Tracks the convergence of the front speed over discrete spatial cycles.
    The speed calculation is robust to time-delayed events like switching lags.
    """
    def __init__(self, sim: "GillespieSimulation", max_cycles: int, convergence_window_cycles: int, convergence_threshold: float, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.max_duration = max_cycles
        self.duration_unit = "cycles"
        self.convergence_window = convergence_window_cycles
        self.convergence_threshold = convergence_threshold

        self.speeds = deque(maxlen=self.convergence_window)
        self.rho_m_samples = deque(maxlen=self.convergence_window)
        
        self.termination_reason = f"max_{self.duration_unit}_reached"
        self.durations_completed = 0
        self._is_done = False

        env_def = kwargs.get("env_definition", {})
        if isinstance(env_def, str):
            env_def = PARAM_GRID.get("env_definitions", {}).get(env_def, {})
        
        cycle_len = 0.0
        if env_def:
            if not env_def.get("scrambled"):
                cycle_len = sum(p.get("width", 0) for p in env_def.get("patches", []))
            else:
                cycle_len = env_def.get("cycle_length", 0)

        self.interval_length = cycle_len
        if self.interval_length <= 0:
            raise ValueError("FrontConvergenceTracker requires a positive cycle length in the environment definition.")

        self.last_crossing_time = 0.0

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        current_pos = self.sim.mean_front_position
        current_time = self.sim.time
        target_pos = (self.durations_completed + 1) * self.interval_length

        while current_pos >= target_pos:
            time_at_crossing = current_time
            cycle_duration = time_at_crossing - self.last_crossing_time
            
            if cycle_duration > 1e-9:
                speed = self.interval_length / cycle_duration
                self.speeds.append(speed)
                self.rho_m_samples.append(self.sim.mutant_fraction)

            self.last_crossing_time = time_at_crossing
            self.durations_completed += 1

            if len(self.speeds) == self.convergence_window:
                mean_speed = np.mean(self.speeds)
                if mean_speed > 1e-9 and (np.std(self.speeds, ddof=1) / mean_speed) < self.convergence_threshold:
                    self.termination_reason = "converged"
                    self._is_done = True
                    break

            if self.durations_completed >= self.max_duration:
                self._is_done = True
                break
            
            target_pos = (self.durations_completed + 1) * self.interval_length

        if self._is_done:
            return

    def finalize(self) -> Dict[str, Any]:
        return {
            "avg_front_speed": np.mean(self.speeds) if self.speeds else 0.0,
            "var_front_speed": np.var(self.speeds, ddof=1) if len(self.speeds) > 1 else 0.0,
            "avg_rho_M": np.mean(self.rho_m_samples) if self.rho_m_samples else 0.0,
            f"num_{self.duration_unit}_completed": self.durations_completed,
            "termination_reason": self.termination_reason,
        }


class MetricsManager:
    """Orchestrates the creation and execution of metric trackers."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self._tracker_configs: List[Tuple[Type[MetricTracker], Dict[str, Any]]] = []
        self.trackers: List[MetricTracker] = []
        self._is_initialized = False

    def add_tracker(self, tracker_class: Type[MetricTracker], tracker_param_map: Dict[str, str]):
        if self._is_initialized: raise RuntimeError("Cannot add trackers after simulation registration.")
        self._tracker_configs.append((tracker_class, tracker_param_map))

    def register_simulation(self, sim: "GillespieSimulation"):
        if self._is_initialized: raise RuntimeError("Simulation already registered.")
        for tracker_class, tracker_param_map in self._tracker_configs:
            tracker_kwargs = {}
            for key, val_str in tracker_param_map.items():
                if val_str.startswith("'") and val_str.endswith("'"):
                    tracker_kwargs[key] = val_str.strip("'")
                elif val_str in self.params:
                    tracker_kwargs[key] = self.params[val_str]
            tracker_kwargs.update(self.params)
            self.trackers.append(tracker_class(sim=sim, **tracker_kwargs))
        sim.metrics_manager, self._is_initialized = self, True
        self.initialize_all()

    def initialize_all(self):
        if not self._is_initialized: return
        for tracker in self.trackers: tracker.initialize()

    def after_step_hook(self):
        if not self._is_initialized: return
        for tracker in self.trackers: tracker.after_step_hook()

    def is_done(self) -> bool:
        if not self._is_initialized: return False
        return any(tracker.is_done() for tracker in self.trackers)

    def finalize(self) -> Dict[str, Any]:
        if not self._is_initialized: return {}
        all_results = {}
        for tracker in self.trackers: all_results.update(tracker.finalize())
        return all_results