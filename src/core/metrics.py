# FILE: src/core/metrics.py (Corrected with Historical Max Q and num_fragments)

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Type, TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from src.core.model import GillespieSimulation
# This import is now needed to resolve the env_definition string
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


class InvasionOutcomeTracker(MetricTracker):
    """
    A purpose-built, efficient tracker for invasion probability experiments.
    - Stops immediately upon true fixation or extinction.
    - Implements an intelligent timeout to terminate long-running, failed invasions early.
    - Does not record bulky trajectory data, only the final outcome.
    """

    def __init__(
        self,
        sim: "GillespieSimulation",
        progress_threshold_frac: float = 0.25,
        progress_check_time: float = 20000.0,
        **kwargs,
    ):
        super().__init__(sim=sim, **kwargs)
        # The width the mutant sector must reach to be considered a "successful" invasion
        self.progress_threshold_width = sim.width * progress_threshold_frac
        # The time limit by which this progress must be made
        self.progress_check_time = progress_check_time

        self._outcome: Optional[str] = None
        self._time_to_outcome: Optional[float] = None
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        # 1. Check for the two definitive, absorbing-state outcomes first
        current_width = self.sim.mutant_sector_width
        if current_width == 0:
            self._outcome = "extinction"
            self._time_to_outcome = self.sim.time
            self._is_done = True
            return

        if current_width >= self.sim.width:
            self._outcome = "fixation"
            self._time_to_outcome = self.sim.time
            self._is_done = True
            return

        # 2. Check for the intelligent "failed invasion" timeout
        if self.sim.time > self.progress_check_time:
            if current_width < self.progress_threshold_width:
                # The patch failed to establish itself after a very long time.
                # We classify this as an effective extinction to save compute time.
                self._outcome = "extinction_by_timeout"
                self._time_to_outcome = self.sim.time
                self._is_done = True
                return

    def finalize(self) -> Dict[str, Any]:
        # 3. Handle the case where the simulation was stopped by an external limit
        if self._outcome is None:
            self._outcome = "undecided_timeout"
            self._time_to_outcome = self.sim.time

        return {
            "outcome": self._outcome,
            "time_to_outcome": self._time_to_outcome,
        }


class BoundaryDynamicsTracker(MetricTracker):
    """
    For 'boundary_analysis' (calibration) runs. Tracks the boundary between
    two competing populations, recording the width of the mutant sector
    and the time until either extinction or fixation.
    """

    def __init__(self, sim: "GillespieSimulation", **kwargs):
        super().__init__(sim, **kwargs)
        self.trajectory: List[Tuple[float, float]] = []
        self.started = False
        self._is_done = False  # This flag is TRUE only on true fixation/extinction
        self.result: Dict[str, Any] = {}

    def after_step_hook(self):
        # This part is fine, it just records the trajectory
        current_width = self.sim.mutant_sector_width
        self.trajectory.append((self.sim.time, current_width))
        if not self.started:
            self.started = True

        # This part correctly identifies a true outcome and stops the simulation
        if current_width == 0:
            self.result["outcome"] = "extinction"
            self.result["time_to_outcome"] = self.sim.time
            self._is_done = True
        elif current_width == self.sim.width:
            self.result["outcome"] = "fixation"
            self.result["time_to_outcome"] = self.sim.time
            self._is_done = True

    def is_done(self) -> bool:
        # The tracker signals to the main loop to stop once an outcome is reached
        return self.started and self._is_done

    def finalize(self) -> dict:

        if not self._is_done:
            self.result["outcome"] = "undecided_timeout"
            self.result["time_to_outcome"] = self.sim.time

        final_data = {"trajectory": self.trajectory}
        final_data.update(self.result)
        final_data["num_fragments"] = self.sim.initial_num_fragments

        return final_data


class FixationTimeTracker(MetricTracker):
    """
    For experiments that should terminate upon mutant fixation or extinction.

    This tracker monitors the composition of the expanding front. It signals
    completion as soon as the front contains only one cell type (WT or Mutant).
    It records the outcome, the time it occurred, and the max front position reached.
    If the boundary is hit before an outcome is decided, it records that instead.
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
            self.max_mutant_q_reached = max(
                self.max_mutant_q_reached, current_max_mutant_q
            )

        if self.outcome_recorded:
            return

        front_has_wt = len(self.sim.wt_front_cells) > 0
        front_has_m = len(self.sim.m_front_cells) > 0

        if not (front_has_wt and front_has_m) and self.sim.step_count > 0:
            if front_has_m:
                self.result["outcome"] = "fixation"
            else:
                self.result["outcome"] = "extinction"

            self.result["time_to_outcome"] = self.sim.time
            self.result["q_at_outcome"] = self.max_mutant_q_reached
            self._is_done = True
            self.outcome_recorded = True

    def finalize(self) -> Dict[str, Any]:
        if not self.outcome_recorded:
            self.result["outcome"] = "boundary_hit"
            self.result["time_to_outcome"] = self.sim.time
            self.result["q_at_outcome"] = self.max_mutant_q_reached

        # --- THIS IS THE FIX ---
        # Ensure 'num_fragments' is always included in the output.
        self.result["num_fragments"] = self.sim.initial_num_fragments
        # --- END OF FIX ---

        return self.result


class RelaxationConvergenceTracker(MetricTracker):
    """
    For 'relaxation_converged' runs. Records a high-resolution time series
    and stops the simulation once the mutant fraction has converged to a
    steady state using a robust, relative convergence criterion.
    """

    def __init__(
        self,
        sim: "GillespieSimulation",
        sample_interval: float,
        convergence_window: int,
        convergence_threshold: float,
        **kwargs,
    ):
        super().__init__(sim=sim, **kwargs)
        self.sample_interval = sample_interval
        self.next_log_time = 0.0
        self.history: List[Dict] = []
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.rho_m_deque = deque(maxlen=convergence_window)
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        while self.sim.time >= self.next_log_time:
            current_rho_m = self.sim.mutant_fraction
            self.history.append(
                {"time": self.next_log_time, "mutant_fraction": current_rho_m}
            )
            self.rho_m_deque.append(current_rho_m)
            self.next_log_time += self.sample_interval

            if len(self.rho_m_deque) == self.convergence_window:
                mean_rho = np.mean(self.rho_m_deque)
                if mean_rho > 1e-6:
                    relative_std_dev = np.std(self.rho_m_deque, ddof=1) / mean_rho
                    if relative_std_dev < self.convergence_threshold:
                        self._is_done = True
                        break
                else:
                    if np.std(self.rho_m_deque, ddof=1) < self.convergence_threshold:
                        self._is_done = True
                        break

    def finalize(self) -> Dict[str, Any]:
        return {"timeseries": self.history}


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
        self,
        sim: "GillespieSimulation",
        timeseries_interval: float,
        warmup_time_ss: float,  # SS = Steady State
        num_samples_ss: int,
        sample_interval_ss: float,
        **kwargs,
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
            self.timeseries_history.append(
                {
                    "time": self.next_log_time,
                    "mutant_fraction": self.sim.mutant_fraction,
                }
            )
            self.next_log_time += self.timeseries_interval

        # --- Part 2: Sample for steady-state properties after warmup ---
        if not self.is_warmed_up and self.sim.time >= self.warmup_time_ss:
            self.is_warmed_up = True
            self.last_sample_q_ss = self.sim.mean_front_position
            self.last_sample_time_ss = self.sim.time
            self.next_sample_time_ss = self.sim.time + self.sample_interval_ss

        if (
            self.is_warmed_up
            and self.sim.time >= self.next_sample_time_ss
            and self.samples_taken_ss < self.num_samples_ss
        ):
            current_q = self.sim.mean_front_position
            current_time = self.sim.time
            delta_q = current_q - self.last_sample_q_ss
            delta_t = current_time - self.last_sample_time_ss

            self.mutant_fraction_samples_ss.append(self.sim.mutant_fraction)
            self.front_speed_samples_ss.append(
                delta_q / delta_t if delta_t > 1e-9 else 0.0
            )

            self.last_sample_q_ss = current_q
            self.last_sample_time_ss = current_time
            self.next_sample_time_ss += self.sample_interval_ss
            self.samples_taken_ss += 1

    def finalize(self) -> Dict[str, Any]:
        return {
            "timeseries": self.timeseries_history,
            "avg_rho_M_final": (
                np.mean(self.mutant_fraction_samples_ss)
                if self.mutant_fraction_samples_ss
                else np.nan
            ),
            "var_rho_M_final": (
                np.var(self.mutant_fraction_samples_ss, ddof=1)
                if len(self.mutant_fraction_samples_ss) > 1
                else np.nan
            ),
            "avg_front_speed_final": (
                np.mean(self.front_speed_samples_ss)
                if self.front_speed_samples_ss
                else np.nan
            ),
        }


class HomogeneousDynamicsTracker(MetricTracker):
    """
    Enhanced version of SteadyStatePropertiesTracker for precise measurements
    in homogeneous environments. Adds front speed calculation.
    """

    def __init__(
        self,
        sim: "GillespieSimulation",
        warmup_time: float,
        num_samples: int,
        sample_interval: float,
        **kwargs,
    ):
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

        if (
            self.is_warmed_up
            and self.sim.time >= self.next_sample_time
            and not self.is_done()
        ):
            current_q = self.sim.mean_front_position
            current_time = self.sim.time
            delta_q = current_q - self.last_sample_q
            delta_t = current_time - self.last_sample_time

            self.mutant_fraction_samples.append(self.sim.mutant_fraction)
            self.front_speed_samples.append(
                delta_q / delta_t if delta_t > 1e-9 else 0.0
            )

            self.last_sample_q = current_q
            self.last_sample_time = current_time
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
            "avg_front_speed": (
                np.mean(self.front_speed_samples)
                if self.front_speed_samples
                else np.nan
            ),
        }


class CyclicTimeSeriesTracker(MetricTracker):
    """
    For 'tracking_analysis' runs. Warms up the simulation for a number of
    environmental cycles and then records a high-resolution time series
    of the population dynamics over subsequent measurement cycles.
    """

    def __init__(
        self,
        sim: "GillespieSimulation",
        warmup_cycles: int,
        measure_cycles: int,
        sample_interval: float,
        **kwargs,
    ):
        super().__init__(sim=sim, **kwargs)
        self.warmup_cycles = warmup_cycles
        self.measure_cycles = measure_cycles
        self.sample_interval = sample_interval

        # Calculate environmental cycle length in q-space
        self.cycle_q = self.sim.patch_sequence[0][1] * len(
            set(p[0] for p in self.sim.patch_sequence)
        )
        if self.cycle_q <= 0:
            raise ValueError(
                "CyclicTimeSeriesTracker requires a defined positive cycle length."
            )

        self.state = "warming_up"  # States: warming_up, measuring, done
        self.cycles_completed = 0
        self.total_cycles = warmup_cycles + measure_cycles

        self.history: List[Dict] = []
        self.next_log_time = -1.0
        self.measurement_start_time = -1.0

    def is_done(self) -> bool:
        return self.state == "done"

    def after_step_hook(self):
        if self.state == "done":
            return

        # Use a while loop to robustly "catch up" on all cycles completed in this step
        while (
            self.sim.mean_front_position >= (self.cycles_completed + 1) * self.cycle_q
        ):
            self.cycles_completed += 1

            # Check for state transitions *inside* the loop
            if (
                self.state == "warming_up"
                and self.cycles_completed >= self.warmup_cycles
            ):
                self.state = "measuring"
                self.measurement_start_time = self.sim.time
                self.next_log_time = self.sim.time

            # Check for the end condition
            if self.cycles_completed >= self.total_cycles:
                self.state = "done"
                # Break from this loop; no need to check further cycles
                break

        # After the cycle counter is fully updated, perform logging if we are in the measuring state
        if self.state == "measuring":
            # Use a while loop here too, in case sim.time advanced significantly
            while self.sim.time >= self.next_log_time:
                # We only log if the simulation isn't already done
                if self.state != "done":
                    self.history.append(
                        {
                            "time": self.next_log_time - self.measurement_start_time,
                            "mutant_fraction": self.sim.mutant_fraction,
                        }
                    )
                self.next_log_time += self.sample_interval

    def finalize(self) -> Dict[str, Any]:
        # The main output is the bulky timeseries data
        return {"timeseries": self.history}


class FrontConvergenceTracker(MetricTracker):

    def __init__(
        self,
        sim: "GillespieSimulation",
        max_duration,
        duration_unit,
        convergence_window,
        convergence_threshold,
        **kwargs,
    ):
        super().__init__(sim=sim, **kwargs)
        self.max_duration = max_duration
        self.duration_unit = duration_unit
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

        self.speeds = deque(maxlen=self.convergence_window)
        self.rho_m_samples = deque(maxlen=self.convergence_window)
        self.termination_reason = f"max_{self.duration_unit}_reached"
        self.durations_completed = 0
        self._is_done = False

        if self.duration_unit == "cycles":
            env_map = kwargs.get("environment_map", {})
            patch_width = kwargs.get("patch_width", 0)

            # --- THIS IS THE FIX for Problem 1 ---
            env_def = kwargs.get("env_definition", {})
            # Resolve the env_def if it's a string key
            if isinstance(env_def, str):
                env_def = PARAM_GRID.get("env_definitions", {}).get(env_def, {})
            # --- END FIX ---

            cycle_q = 0.0
            if patch_width > 0 and env_map:
                cycle_q = patch_width * len(env_map)
            elif env_def:
                cycle_q = (
                    sum(p["width"] for p in env_def.get("patches", []))
                    if not env_def.get("scrambled")
                    else env_def.get("cycle_length", 0)
                )

            self.interval_length = cycle_q
            if self.interval_length <= 0:
                raise ValueError(
                    "Cycle-based convergence requires a defined cycle length."
                )
        else:  # 'time'
            self.interval_length = kwargs.get("convergence_check_interval", 100.0)

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        if not hasattr(self, "last_pos"):
            self.last_pos = self.sim.mean_front_position
            self.last_time = self.sim.time

        current_progress = (
            self.sim.mean_front_position
            if self.duration_unit == "cycles"
            else self.sim.time
        )
        target_progress = (self.durations_completed + 1) * self.interval_length

        if current_progress >= target_progress:
            end_pos, end_time = self.sim.mean_front_position, self.sim.time
            delta_dist, delta_time = end_pos - self.last_pos, end_time - self.last_time

            self.speeds.append(delta_dist / delta_time if delta_time > 1e-9 else 0.0)
            self.rho_m_samples.append(self.sim.mutant_fraction)
            self.durations_completed += 1

            self.last_pos, self.last_time = end_pos, end_time

            if len(self.speeds) == self.convergence_window:
                mean_speed = np.mean(self.speeds)
                if (
                    mean_speed > 1e-9
                    and (np.std(self.speeds, ddof=1) / mean_speed)
                    < self.convergence_threshold
                ):
                    self.termination_reason = "converged"
                    self._is_done = True

            if self.durations_completed >= self.max_duration:
                self._is_done = True

    def finalize(self) -> Dict[str, Any]:
        return {
            "avg_front_speed": np.mean(self.speeds) if self.speeds else 0.0,
            "var_front_speed": (
                np.var(self.speeds, ddof=1) if len(self.speeds) > 1 else 0.0
            ),
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

    def add_tracker(
        self, tracker_class: Type[MetricTracker], tracker_param_map: Dict[str, str]
    ):
        """Adds a tracker to be initialized when the simulation is registered."""
        if self._is_initialized:
            raise RuntimeError("Cannot add trackers after simulation registration.")
        self._tracker_configs.append((tracker_class, tracker_param_map))

    def register_simulation(self, sim: "GillespieSimulation"):
        """Creates tracker instances and links them to the simulation."""
        if self._is_initialized:
            raise RuntimeError("Simulation already registered.")

        for tracker_class, tracker_param_map in self._tracker_configs:
            # --- Dependency Injection Happens Here ---
            # Build the kwargs dict for the tracker from the master params dict.
            tracker_kwargs = {}
            for key, val_str in tracker_param_map.items():
                if val_str.startswith("'") and val_str.endswith("'"):
                    tracker_kwargs[key] = val_str.strip("'")
                elif val_str in self.params:
                    tracker_kwargs[key] = self.params[val_str]

            # Also pass ALL original params to kwargs for full, failsafe access
            tracker_kwargs.update(self.params)

            self.trackers.append(tracker_class(sim=sim, **tracker_kwargs))

        sim.metrics_manager = self
        self._is_initialized = True
        self.initialize_all()

    def initialize_all(self):
        """Initializes all registered trackers."""
        if not self._is_initialized:
            return
        for tracker in self.trackers:
            tracker.initialize()

    def after_step_hook(self):
        """Hook called after each simulation step."""
        if not self._is_initialized:
            return
        for tracker in self.trackers:
            tracker.after_step_hook()

    def is_done(self) -> bool:
        """Checks if any tracker has signaled completion."""
        if not self._is_initialized:
            return False
        return any(tracker.is_done() for tracker in self.trackers)

    def finalize(self) -> Dict[str, Any]:
        """Gathers results from all trackers."""
        if not self._is_initialized:
            return {}
        all_results = {}
        for tracker in self.trackers:
            all_results.update(tracker.finalize())
        return all_results
