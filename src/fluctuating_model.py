# FILE: src/fluctuating_model.py
#
# Definitive, vectorized model supporting environments with patches of
# both symmetric and asymmetric widths.

import numpy as np
import random
from typing import Dict, Set, Tuple, List, Optional
from hex_utils import Hex
from metrics import MetricsManager

Empty, Wildtype, Mutant = 0, 1, 2


class FluctuatingGillespieSimulation:
    def __init__(
        self,
        width: int,
        length: int,
        k_total: float,
        phi: float,
        b_m: float,
        environment_map: Dict,
        environment_patch_sequence: Optional[List[Tuple[int, int]]] = None,
        patch_width: Optional[int] = None,
        initial_condition_type: str = "mixed",
        initial_mutant_patch_size: int = 0,
        metrics_manager: Optional[MetricsManager] = None,
    ):
        self.width, self.length = width, length
        self.environment_map = environment_map

        # --- Backward compatibility for old symmetric patch_width param ---
        if environment_patch_sequence is None and patch_width is not None:
            self.patch_sequence = [(0, patch_width), (1, patch_width)]
        elif environment_patch_sequence is not None:
            self.patch_sequence = environment_patch_sequence
        else:
            # Default to a single, uniform environment if no patch info given
            self.patch_sequence = [(0, self.length)]

        self.num_patches = len(environment_map)

        self.global_k_total = k_total
        self.global_phi = phi
        self.global_b_m = b_m
        self.time = 0.0
        self.population: Dict[Hex, int] = self._initialize_population(
            initial_condition_type, initial_mutant_patch_size
        )
        self.wt_front_cells: Dict[Hex, List[Hex]] = {}
        self.m_front_cells: Dict[Hex, List[Hex]] = {}
        self._front_lookup: Set[Hex] = set()
        self.sum_q, self.sum_q_sq, self.front_cell_count = 0.0, 0.0, 0
        self.rates: Dict[str, float] = {}
        self.total_rate = 0.0
        self.mean_front_position = 0.0

        # --- Pre-calculation for environment lookup ---
        self.q_to_patch_index = self._build_q_to_patch_map()
        self.patch_params = self._build_patch_params_array()

        self._find_initial_front()
        self._update_rates()
        self.metrics_manager = metrics_manager
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)

    def _build_q_to_patch_map(self) -> np.ndarray:
        """Builds a fast lookup array mapping a q-coordinate to a patch type index."""
        q_map = np.zeros(self.length, dtype=int)
        cycle_len = sum(width for _, width in self.patch_sequence)
        if cycle_len <= 0:
            if self.patch_sequence:
                q_map[:] = self.patch_sequence[0][0]
            return q_map

        current_q = 0
        while current_q < self.length:
            q_pos_in_cycle = 0
            for patch_type, patch_width in self.patch_sequence:
                start_q, end_q = (
                    current_q + q_pos_in_cycle,
                    current_q + q_pos_in_cycle + patch_width,
                )
                if start_q < self.length:
                    q_map[start_q : min(end_q, self.length)] = patch_type
                q_pos_in_cycle += patch_width
            current_q += cycle_len
        return q_map

    def _build_patch_params_array(self) -> np.ndarray:
        """Builds a NumPy array of parameters for each patch type for fast vectorized access."""
        k_wt_m_global, k_m_wt_global = self._calculate_asymmetric_rates(
            self.global_k_total, self.global_phi
        )
        return np.array(
            [
                [
                    self.environment_map.get(i, {}).get("b_wt", 1.0),
                    self.environment_map.get(i, {}).get("b_m", self.global_b_m),
                    k_wt_m_global,
                    k_m_wt_global,
                ]
                for i in range(self.num_patches)
            ]
        )

    @staticmethod
    def _calculate_asymmetric_rates(k_total, phi):
        phi = np.clip(phi, -1, 1)
        return (k_total / 2.0) * (1.0 - phi), (k_total / 2.0) * (1.0 + phi)

    def _update_rates(self):
        """Vectorized rate calculation for all front cells."""
        self.rates.clear()

        if self.wt_front_cells:
            wt_q_coords = np.array([cell.q for cell in self.wt_front_cells], dtype=int)
            wt_patch_indices = self.q_to_patch_index[
                np.clip(wt_q_coords, 0, self.length - 1)
            ]
            self.rates["wt_growth"] = self.patch_params[wt_patch_indices, 0].sum()
        else:
            self.rates["wt_growth"] = 0.0

        if self.m_front_cells:
            m_q_coords = np.array([cell.q for cell in self.m_front_cells], dtype=int)
            m_patch_indices = self.q_to_patch_index[
                np.clip(m_q_coords, 0, self.length - 1)
            ]
            self.rates["m_growth"] = self.patch_params[m_patch_indices, 1].sum()
        else:
            self.rates["m_growth"] = 0.0

        self.rates["wt_to_m_switching"] = (
            self.global_k_total
            * (1.0 - self.global_phi)
            / 2.0
            * len(self.wt_front_cells)
        )
        self.rates["m_to_wt_switching"] = (
            self.global_k_total
            * (1.0 + self.global_phi)
            / 2.0
            * len(self.m_front_cells)
        )

        self.total_rate = sum(self.rates.values())
        return self.total_rate > 1e-9

    def step(self):
        if not self._update_rates():
            return False, False  # No events can occur

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

        return True, boundary_hit

    def _execute_event(self, event_type: str) -> bool:
        """Executes the chosen Gillespie event."""
        if event_type == "wt_growth":
            parent_cell = random.choice(list(self.wt_front_cells.keys()))
            empty_neighbor = random.choice(self.wt_front_cells[parent_cell])
            self._add_cell(empty_neighbor, Wildtype)
            return empty_neighbor.q >= self.length - 1

        elif event_type == "m_growth":
            parent_cell = random.choice(list(self.m_front_cells.keys()))
            empty_neighbor = random.choice(self.m_front_cells[parent_cell])
            self._add_cell(empty_neighbor, Mutant)
            return empty_neighbor.q >= self.length - 1

        elif event_type == "wt_to_m_switching":
            cell_to_switch = random.choice(list(self.wt_front_cells.keys()))
            self._update_cell_type(cell_to_switch, Wildtype, Mutant)

        elif event_type == "m_to_wt_switching":
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

    def _initialize_population(self, ic_type, patch_size):
        pop = {}
        if ic_type == "patch":
            start_r = (self.width - patch_size) // 2
            for r in range(start_r, start_r + patch_size):
                pop[Hex(0, r, -r)] = Mutant
            for r in range(self.width):
                if not (start_r <= r < start_r + patch_size):
                    pop[Hex(0, r, -r)] = Wildtype
        else:  # mixed
            for r in range(self.width):
                pop[Hex(0, r, -r)] = random.choice([Wildtype, Mutant])
        return pop

    def _find_initial_front(self):
        self.sum_q, self.sum_q_sq, self.front_cell_count = 0.0, 0.0, 0
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
