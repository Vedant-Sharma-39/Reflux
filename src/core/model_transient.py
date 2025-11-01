import heapq
from typing import Dict, Set, Tuple

from src.core.model import GillespieSimulation, Wildtype, Mutant

# --- Define new transient states ---
MutantSwitchingToWT = 3
WildtypeSwitchingToMutant = 4


class GillespieTransientStateSimulation(GillespieSimulation):
    """
    An extension of the GillespieSimulation that correctly models a transient,
    "stuck" state during phenotype switching.

    This version uses a simple, robust, one-event-per-step loop that is
    guaranteed to be correct by always comparing the absolute times of the
    next possible stochastic and deterministic events.
    """

    def __init__(self, **params):
        self.switching_lag_duration = params.get("switching_lag_duration", 0.0)
        self.delayed_events = []
        self.event_counter = 0  # To maintain event order in the priority queue
        super().__init__(**params)

        if self.plotter:
            self.plotter.colormap[MutantSwitchingToWT] = "#f9a68d"
            self.plotter.colormap[WildtypeSwitchingToMutant] = "#79bde1"

    def _update_single_cell_events(self, h: "Hex"):
        for event in list(self.cell_to_events.get(h, set())):
            self._remove_event(event)

        cell_type = self.population.get(h)
        if cell_type in (MutantSwitchingToWT, WildtypeSwitchingToMutant):
            if h in self._front_lookup:
                self._remove_from_front(h)
            return

        super()._update_single_cell_events(h)

    def _execute_event(self, event_type: str, parent: "Hex", target):
        if event_type == "switch":
            old_type = self.population[parent]
            final_type = target

            if old_type == Mutant and final_type == Wildtype:
                transient_state = MutantSwitchingToWT
            elif old_type == Wildtype and final_type == Mutant:
                transient_state = WildtypeSwitchingToMutant
            else:
                return

            self.population[parent] = transient_state
            completion_time = self.time + self.switching_lag_duration
            self.event_counter += 1
            event_tuple = (completion_time, self.event_counter, parent, final_type)
            heapq.heappush(self.delayed_events, event_tuple)
            self._update_cell_and_neighbors(parent)
        else:
            super()._execute_event(event_type, parent, target)

    def _complete_delayed_event(self):
        completion_time, _, cell, final_type = heapq.heappop(self.delayed_events)
        self.time = completion_time

        current_type = self.population.get(cell)
        if current_type in (MutantSwitchingToWT, WildtypeSwitchingToMutant):
            self.population[cell] = final_type
            if final_type == Wildtype and current_type == MutantSwitchingToWT:
                self.mutant_cell_count -= 1
                self.mutant_r_counts[cell.r] -= 1
            elif final_type == Mutant and current_type == WildtypeSwitchingToMutant:
                self.mutant_cell_count += 1
                self.mutant_r_counts[cell.r] += 1
            
            self._update_cell_and_neighbors(cell)

    def step(self):
        """
        A robust step method that correctly handles both stochastic and deterministic
        events using the Next Reaction Method. It processes exactly one event per call,
        advancing the clock tod the time of the earliest possible event.
        """
        if self.step_count > 0 and self.step_count % self.cache_prune_step_interval == 0:
            self._prune_neighbor_cache()
        self.step_count += 1

        # 1. Determine the time of the next potential stochastic event
        current_total_rate = self.total_rate
        t_next_stochastic = float('inf')
        if current_total_rate > 1e-9:
            dt_stochastic = -_np.log(_random.random()) / current_total_rate
            t_next_stochastic = self.time + dt_stochastic

        # 2. Determine the time of the next scheduled deterministic event
        t_next_deterministic = float('inf')
        if self.delayed_events:
            t_next_deterministic = self.delayed_events[0][0]

        # Check if there are no more events possible in the simulation
        if t_next_stochastic == float('inf') and t_next_deterministic == float('inf'):
            return False, False

        # 3. Compare times and execute whichever single event is next
        if t_next_stochastic < t_next_deterministic:
            # The next event is stochastic. Advance time and execute it.
            self.time = t_next_stochastic
            
            rand_val = _random.random() * current_total_rate
            event_idx = self.tree.find_event(rand_val)
            if event_idx in self.idx_to_event:
                event_type, parent, target = self.idx_to_event[event_idx]
                self._execute_event(event_type, parent, target)
        else:
            # The next event is deterministic. Let the handler advance the time.
            self._complete_delayed_event()

        boundary_hit = self.mean_front_position >= self.length - 2
        return True, boundary_hit

import numpy as _np
import random as _random