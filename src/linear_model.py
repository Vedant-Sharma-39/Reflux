# FILE: src/linear_model.py
#
# The original, simpler Gillespie simulation model for a uniform environment.
# This is primarily used for calibration, KPZ analysis, and as a baseline.

import numpy as np
import random
from typing import Dict, List, Set, Tuple, Optional
from hex_utils import Hex
from metrics import MetricsManager

Empty, Wildtype, Mutant = 0, 1, 2


class GillespieSimulation:
    def __init__(
        self,
        width: int,
        length: int,
        b_m: float,
        k_total: float,
        phi: float,
        initial_condition_type: str = "mixed",
        initial_mutant_patch_size: int = 0,
        b_wt: float = 1.0,
        metrics_manager: Optional[MetricsManager] = None,
    ):
        self.width = width
        self.length = length
        self.b_wt = b_wt
        self.b_m = b_m
        self.k_total_base = k_total  # Store base for perturbations
        self.phi_base = phi
        self.time = 0.0

        self.population: Dict[Hex, int] = self._initialize_population(
            initial_condition_type, initial_mutant_patch_size
        )
        self.wt_front_cells: Dict[Hex, List[Hex]] = {}
        self.m_front_cells: Dict[Hex, List[Hex]] = {}
        self._front_lookup: Set[Hex] = set()

        self.sum_q: float = 0.0
        self.sum_q_sq: float = 0.0
        self.front_cell_count: int = 0
        self.mean_front_position: float = 0.0

        self.rates: Dict[str, float] = {}
        self.total_rate = 0.0

        self.set_switching_rate(k_total, phi)  # Initialize rates
        self._find_initial_front()
        self._update_rates()

        self.metrics_manager = metrics_manager
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)

    def set_switching_rate(self, new_k_total: float, new_phi: float):
        """Allows for dynamically changing the global switching parameters."""
        self.k_total = new_k_total
        self.phi = np.clip(new_phi, -1, 1)
        self.k_wt_to_m = (self.k_total / 2.0) * (1.0 - self.phi)
        self.k_m_to_wt = (self.k_total / 2.0) * (1.0 + self.phi)
        self._update_rates()  # Immediately update total rate

    def _initialize_population(self, ic_type, patch_size):
        pop = {}
        if ic_type == "patch":
            start_r = (self.width - patch_size) // 2
            for r in range(start_r, start_r + patch_size):
                pop[Hex(0, r, -r)] = Mutant
            for r in range(self.width):
                if not (start_r <= r < start_r + patch_size):
                    pop[Hex(0, r, -r)] = Wildtype
        else:  # Default to 'mixed'
            for r in range(self.width):
                pop[Hex(0, r, -r)] = random.choice([Wildtype, Mutant])
        return pop

    def _find_initial_front(self):
        for cell, cell_type in self.population.items():
            empty_neighbors = [
                n
                for n in cell.neighbors()
                if self.population.get(n.wrap(self.width), Empty) == Empty
            ]
            if empty_neighbors:
                front_dict = (
                    self.wt_front_cells if cell_type == Wildtype else self.m_front_cells
                )
                front_dict[cell] = empty_neighbors
                self._front_lookup.add(cell)
                self._update_front_stats_add(cell)

    def _update_rates(self):
        self.rates["wt_growth"] = len(self.wt_front_cells) * self.b_wt
        self.rates["m_growth"] = len(self.m_front_cells) * self.b_m
        self.rates["wt_to_m_switching"] = len(self.wt_front_cells) * self.k_wt_to_m
        self.rates["m_to_wt_switching"] = len(self.m_front_cells) * self.k_m_to_wt
        self.total_rate = sum(self.rates.values())
        return self.total_rate > 1e-9

    def step(self):
        if not self._update_rates():
            return False, False

        dt = -np.log(random.random()) / self.total_rate
        self.time += dt

        rand_val = random.random() * self.total_rate
        boundary_hit = False

        cumulative_rate = 0.0
        for event_type, rate in self.rates.items():
            cumulative_rate += rate
            if rand_val < cumulative_rate:
                boundary_hit = self._execute_event(event_type)
                break

        if self.metrics_manager:
            self.metrics_manager.after_step()

        return True, boundary_hit

    def _execute_event(self, event_type: str) -> bool:
        """Executes event. Returns True if boundary is hit."""
        if event_type == "wt_growth":
            if not self.wt_front_cells:
                return False
            parent_cell = random.choice(list(self.wt_front_cells.keys()))
            empty_neighbor = random.choice(self.wt_front_cells[parent_cell])
            self._add_cell(empty_neighbor, Wildtype)
            return empty_neighbor.q >= self.length - 1

        elif event_type == "m_growth":
            if not self.m_front_cells:
                return False
            parent_cell = random.choice(list(self.m_front_cells.keys()))
            empty_neighbor = random.choice(self.m_front_cells[parent_cell])
            self._add_cell(empty_neighbor, Mutant)
            return empty_neighbor.q >= self.length - 1

        elif event_type == "wt_to_m_switching":
            if not self.wt_front_cells:
                return False
            cell_to_switch = random.choice(list(self.wt_front_cells.keys()))
            self._update_cell_type(cell_to_switch, Wildtype, Mutant)

        elif event_type == "m_to_wt_switching":
            if not self.m_front_cells:
                return False
            cell_to_switch = random.choice(list(self.m_front_cells.keys()))
            self._update_cell_type(cell_to_switch, Mutant, Wildtype)

        return False

    def _add_cell(self, cell: Hex, cell_type: int):
        self.population[cell] = cell_type
        self._update_front_after_addition(cell, cell_type)

    def _update_cell_type(self, cell: Hex, old_type: int, new_type: int):
        self.population[cell] = new_type
        if old_type == Wildtype:
            empty_neighbors = self.wt_front_cells.pop(cell)
        else:
            empty_neighbors = self.m_front_cells.pop(cell)

        if new_type == Wildtype:
            self.wt_front_cells[cell] = empty_neighbors
        else:
            self.m_front_cells[cell] = empty_neighbors

    def _update_front_after_addition(self, new_cell: Hex, new_cell_type: int):
        for neighbor in new_cell.neighbors():
            parent_cell = neighbor.wrap(self.width)
            if parent_cell in self._front_lookup:
                front_dict = (
                    self.wt_front_cells
                    if self.population.get(parent_cell) == Wildtype
                    else self.m_front_cells
                )
                if new_cell in front_dict.get(parent_cell, []):
                    front_dict[parent_cell].remove(new_cell)
                    if not front_dict[parent_cell]:
                        del front_dict[parent_cell]
                        self._front_lookup.remove(parent_cell)
                        self._update_front_stats_remove(parent_cell)

        empty_neighbors = [
            n
            for n in new_cell.neighbors()
            if self.population.get(n.wrap(self.width), Empty) == Empty
        ]
        if empty_neighbors:
            front_dict = (
                self.wt_front_cells if new_cell_type == Wildtype else self.m_front_cells
            )
            front_dict[new_cell] = empty_neighbors
            self._front_lookup.add(new_cell)
            self._update_front_stats_add(new_cell)

    def _update_front_stats_add(self, cell: Hex):
        self.sum_q += cell.q
        self.sum_q_sq += cell.q**2
        self.front_cell_count += 1
        self._recalculate_mean_front()

    def _update_front_stats_remove(self, cell: Hex):
        self.sum_q -= cell.q
        self.sum_q_sq -= cell.q**2
        self.front_cell_count -= 1
        self._recalculate_mean_front()

    def _recalculate_mean_front(self):
        self.mean_front_position = (
            self.sum_q / self.front_cell_count if self.front_cell_count > 0 else 0.0
        )
