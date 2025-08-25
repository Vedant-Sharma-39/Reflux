# FILE: src/core/model_transient.py

import heapq
from typing import Dict, Set, Tuple

from src.core.model import GillespieSimulation, Wildtype, Mutant

# --- Define new transient states ---
MutantSwitchingToWT = 3
WildtypeSwitchingToMutant = 4


class GillespieTransientStateSimulation(GillespieSimulation):
    """
    An extension of the GillespieSimulation that models a transient, "stuck"
    state during phenotype switching.

    When a switching event is chosen by the Gillespie algorithm, the cell does not
    change its type instantly. Instead, it enters a transient state (e.g.,
    MutantSwitchingToWT) for a fixed, deterministic `switching_lag_duration`.
    During this time, the cell is inert: it cannot grow or initiate new switches.

    The main simulation loop is modified to handle both stochastic events from the
    rate tree and the next scheduled completion of a transient state, executing
    whichever comes first.
    """

    def __init__(self, **params):
        # The duration a cell remains in the transient state.
        self.switching_lag_duration = params.get("switching_lag_duration", 0.0)

        # A priority queue to store future, deterministic completion events.
        # Format: (completion_time, cell_hex, final_cell_type)
        self.delayed_events = []

        # Call parent constructor. It sets up the grid and initial population.
        super().__init__(**params)

        # --- IMPORTANT: Update the plotter's colormap to handle new states ---
        if self.plotter:
            # Assign distinct colors for visualization
            self.plotter.colormap[MutantSwitchingToWT] = "#f9a68d"  # Lighter orange
            self.plotter.colormap[WildtypeSwitchingToMutant] = "#79bde1"  # Lighter blue

    def _update_single_cell_events(self, h: "Hex"):
        """
        Overrides the parent method to make transient cells inert.
        """
        # First, remove any existing events for this cell.
        for event in list(self.cell_to_events.get(h, set())):
            self._remove_event(event)

        cell_type = self.population.get(h)
        # --- MODIFICATION: If cell is in a transient state, it has NO events. ---
        if cell_type in (MutantSwitchingToWT, WildtypeSwitchingToMutant):
            if h in self._front_lookup:
                self._remove_from_front(h)
            return

        # If not transient, proceed with the original logic from the parent.
        super()._update_single_cell_events(h)

    def _execute_event(self, event_type: str, parent: "Hex", target):
        """
        Overrides the parent method to intercept 'switch' events and turn them
        into delayed, transient states.
        """
        # --- MODIFICATION: Intercept switch events ---
        if event_type == "switch":
            old_type = self.population[parent]
            final_type = target  # The intended final cell type

            # Determine the new transient state
            if old_type == Mutant and final_type == Wildtype:
                transient_state = MutantSwitchingToWT
            elif old_type == Wildtype and final_type == Mutant:
                transient_state = WildtypeSwitchingToMutant
            else:
                return # Should not happen

            # 1. Put the cell into the transient state immediately.
            self.population[parent] = transient_state

            # 2. Schedule the completion of the switch for a future time.
            completion_time = self.time + self.switching_lag_duration
            heapq.heappush(self.delayed_events, (completion_time, parent, final_type))

            # 3. Update the cell and its neighbors. This will effectively remove
            #    all events for the parent cell, making it inert.
            self._update_cell_and_neighbors(parent)

        # If it's not a switch event, let the parent handle it (i.e., growth).
        else:
            super()._execute_event(event_type, parent, target)

    def _complete_delayed_event(self):
        """
        Processes the next event from the delayed_events queue.
        """
        completion_time, cell, final_type = heapq.heappop(self.delayed_events)

        # This should always be true due to the logic in step()
        self.time = completion_time

        # Only complete the switch if the cell is still in the correct transient state.
        # (It's possible a simulation ends or something else happens, though unlikely here)
        current_type = self.population.get(cell)
        if current_type in (MutantSwitchingToWT, WildtypeSwitchingToMutant):
            # 1. Set the cell to its final type.
            self.population[cell] = final_type

            # 2. Update population counts.
            if final_type == Wildtype and current_type == MutantSwitchingToWT:
                self.mutant_cell_count -= 1
                self.mutant_r_counts[cell.r] -= 1
            elif final_type == Mutant and current_type == WildtypeSwitchingToMutant:
                self.mutant_cell_count += 1
                self.mutant_r_counts[cell.r] += 1
            
            # 3. The cell is no longer inert. Update it to add its new
            #    possible events (growth, switching back) to the rate tree.
            self._update_cell_and_neighbors(cell)


    def step(self):
        """
        A completely new step method that chooses between the next stochastic
        event and the next scheduled deterministic event.
        """
        if self.step_count > 0 and self.step_count % self.cache_prune_step_interval == 0:
            self._prune_neighbor_cache()
        self.step_count += 1

        # --- Determine time to next stochastic event ---
        total_rate = self.total_rate
        dt_stochastic = float('inf')
        if total_rate > 1e-9:
            dt_stochastic = -_np.log(_random.random()) / total_rate

        # --- Determine time to next deterministic (delayed) event ---
        dt_deterministic = float('inf')
        next_delayed_time = float('inf')
        if self.delayed_events:
            next_delayed_time = self.delayed_events[0][0]
            dt_deterministic = next_delayed_time - self.time
        
        # If no events of any kind are possible, stop.
        if dt_stochastic == float('inf') and dt_deterministic == float('inf'):
            return False, False
        
        # --- Choose whichever event happens first ---
        if dt_stochastic < dt_deterministic:
            # A stochastic event (growth or switch INITIATION) is next.
            self.time += dt_stochastic
            rand_val = _random.random() * total_rate
            event_idx = self.tree.find_event(rand_val)
            if event_idx not in self.idx_to_event:
                return False, False # Should not happen if rate > 0
            event_type, parent, target = self.idx_to_event[event_idx]
            self._execute_event(event_type, parent, target)
        else:
            # A deterministic event (switch COMPLETION) is next.
            self._complete_delayed_event()
            # Note: time is set inside _complete_delayed_event

        boundary_hit = self.mean_front_position >= self.length - 2
        return True, boundary_hit

# We need to import these here to avoid circular dependency issues at the top level
# but make them available inside the jit-compiled functions if needed.
import numpy as _np
import random as _random