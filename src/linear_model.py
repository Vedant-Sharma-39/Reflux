# src/linear_model.py
# The core Gillespie simulation engine for a linear range expansion on a hexagonal grid.
# [REFACTORED] This version models growth and type-switching as four distinct,
# independent, rate-based events.

import numpy as np
import random
from typing import Dict, Set, List, Tuple
from hex_utils import Hex
from metrics import MetricsManager

# --- State Definitions ---
Empty, Wildtype, Mutant = 0, 1, 2


class GillespieSimulation:
    def __init__(
        self,
        width: int,
        length: int,
        b_m: float,
        k_total: float,
        phi: float,
        initial_mutant_patch_size: int = 0,
        metrics_manager: MetricsManager = None,
    ):
        self.width = width
        self.length = length
        self.b_wt = 1.0
        self.b_m = b_m
        # k_wt_m and k_m_wt are now treated as the base rates for switching events
        self.k_wt_m, self.k_m_wt = self._calculate_asymmetric_rates(k_total, phi)
        self.initial_mutant_patch_size = initial_mutant_patch_size

        self.time = 0.0
        self.population: Dict[Hex, int] = self._initialize_population()

        # Using dictionaries for O(1) addition/removal of front cells
        self.wt_front_cells: Dict[Hex, bool] = {}
        self.m_front_cells: Dict[Hex, bool] = {}
        self._front_lookup: Set[Hex] = set()

        # Metric state variables (for roughness, etc.)
        self.sum_q = 0.0
        self.sum_q_sq = 0.0
        self.front_cell_count = 0

        self.rates: Dict[str, float] = {}
        self.total_rate = 0.0

        self._find_initial_front()
        self._update_rates()

        self.metrics_manager = metrics_manager
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)

    # --- Public Properties ---
    @property
    def mutant_fraction(self) -> float:
        if self.front_cell_count == 0:
            return 0.0
        return len(self.m_front_cells) / self.front_cell_count

    @property
    def mean_front_position(self) -> float:
        if self.front_cell_count == 0:
            return float(self.length)
        return self.sum_q / self.front_cell_count

    @property
    def front_roughness_q(self) -> float:
        if self.front_cell_count < 2:
            return 0.0
        mean_q = self.mean_front_position
        variance_q = (self.sum_q_sq / self.front_cell_count) - (mean_q**2)
        return np.sqrt(max(0, variance_q))

    # --- Initialization and Helpers ---
    def _initialize_population(self) -> Dict[Hex, int]:
        pop = {}
        if self.initial_mutant_patch_size > 0:
            patch_start = (self.width - self.initial_mutant_patch_size) // 2
            for row in range(self.width):
                cell = self._axial_to_cube(0, row)
                if patch_start <= row < patch_start + self.initial_mutant_patch_size:
                    pop[cell] = Mutant
                else:
                    pop[cell] = Wildtype
        else:
            pop = {self._axial_to_cube(0, row): Wildtype for row in range(self.width)}
        return pop

    @staticmethod
    def _axial_to_cube(q: int, r_offset: int) -> Hex:
        # Using axial coordinates (q, r_offset) for the grid
        r = r_offset - (q + (q & 1)) // 2
        return Hex(q, r, -q - r)

    def _get_neighbors_periodic(self, cell: Hex) -> List[Hex]:
        neighbors = []
        for neighbor_base in cell.neighbors():
            col = neighbor_base.q
            row_offset = (
                neighbor_base.r + (neighbor_base.q + (neighbor_base.q & 1)) // 2
            )
            row_offset %= self.width
            neighbors.append(self._axial_to_cube(col, row_offset))
        return neighbors

    def _is_valid_growth_site(self, cell: Hex) -> bool:
        return 0 <= cell.q < self.length and cell not in self.population

    @staticmethod
    def _calculate_asymmetric_rates(k_total: float, phi: float) -> Tuple[float, float]:
        phi = np.clip(phi, -1, 1)
        # phi > 0 biases towards WT, phi < 0 biases towards M
        k_wt_m = (k_total / 2.0) * (1.0 - phi)
        k_m_wt = (k_total / 2.0) * (1.0 + phi)
        return k_wt_m, k_m_wt

    def _find_initial_front(self):
        for cell in self.population:
            self._check_and_update_front_status(cell)

    def _check_and_update_front_status(self, cell: Hex):
        is_on_front = any(
            self._is_valid_growth_site(n) for n in self._get_neighbors_periodic(cell)
        )
        is_in_lookup = cell in self._front_lookup

        if is_on_front and not is_in_lookup:
            self._front_lookup.add(cell)
            cell_type = self.population.get(cell)
            (self.wt_front_cells if cell_type == Wildtype else self.m_front_cells)[
                cell
            ] = True
            self.front_cell_count += 1
            self.sum_q += cell.q
            self.sum_q_sq += cell.q**2
        elif not is_on_front and is_in_lookup:
            self._front_lookup.remove(cell)
            cell_type = self.population.get(cell)
            if cell_type == Wildtype:
                del self.wt_front_cells[cell]
            else:
                del self.m_front_cells[cell]
            self.front_cell_count -= 1
            self.sum_q -= cell.q
            self.sum_q_sq -= cell.q**2

    # --- Core Simulation Logic (Refactored) ---
    def _update_rates(self):
        """[REFACTORED] Calculates rates for four independent event types."""
        count_wt = len(self.wt_front_cells)
        count_m = len(self.m_front_cells)

        self.rates = {
            "WT_grows": count_wt * self.b_wt,
            "M_grows": count_m * self.b_m,
            "WT_switches_to_M": count_wt * self.k_wt_m,
            "M_switches_to_WT": count_m * self.k_m_wt,
        }
        self.total_rate = sum(self.rates.values())

    def _execute_growth_event(self, parent_type: int) -> bool:
        """[REFACTORED] Executes growth. Offspring type matches parent."""
        parent_pool_dict = (
            self.wt_front_cells if parent_type == Wildtype else self.m_front_cells
        )
        if not parent_pool_dict:
            return False

        parent_cell = random.choice(list(parent_pool_dict.keys()))
        empty_neighbors = [
            n
            for n in self._get_neighbors_periodic(parent_cell)
            if self._is_valid_growth_site(n)
        ]

        if not empty_neighbors:
            self._check_and_update_front_status(parent_cell)
            self._update_rates()
            return False

        growth_site = random.choice(empty_neighbors)
        self.population[growth_site] = parent_type

        cells_to_recheck = self._get_neighbors_periodic(growth_site) + [
            growth_site,
            parent_cell,
        ]
        for cell in cells_to_recheck:
            if cell in self.population:
                self._check_and_update_front_status(cell)

        self._update_rates()
        return growth_site.q >= self.length - 1

    def _execute_switch_event(self, original_type: int, new_type: int):
        """[NEW] Executes a type-switching event for a front cell."""
        source_pool_dict = (
            self.wt_front_cells if original_type == Wildtype else self.m_front_cells
        )
        target_pool_dict = (
            self.m_front_cells if new_type == Mutant else self.wt_front_cells
        )

        if not source_pool_dict:
            return

        cell_to_switch = random.choice(list(source_pool_dict.keys()))
        self.population[cell_to_switch] = new_type

        del source_pool_dict[cell_to_switch]
        target_pool_dict[cell_to_switch] = True

        self._update_rates()

    def step(self) -> Tuple[bool, bool]:
        """[REFACTORED] Performs one Gillespie step, dispatching one of four events."""
        if self.total_rate <= 1e-9:
            return (False, False)

        delta_t = -np.log(random.uniform(0, 1)) / self.total_rate
        self.time += delta_t
        u = random.uniform(0, 1) * self.total_rate
        boundary_was_hit = False

        cumulative_rate = self.rates["WT_grows"]
        if u < cumulative_rate:
            boundary_was_hit = self._execute_growth_event(Wildtype)
            if self.metrics_manager:
                self.metrics_manager.after_step()
            return (True, boundary_was_hit)

        cumulative_rate += self.rates["M_grows"]
        if u < cumulative_rate:
            boundary_was_hit = self._execute_growth_event(Mutant)
            if self.metrics_manager:
                self.metrics_manager.after_step()
            return (True, boundary_was_hit)

        cumulative_rate += self.rates["WT_switches_to_M"]
        if u < cumulative_rate:
            self._execute_switch_event(Wildtype, Mutant)
            if self.metrics_manager:
                self.metrics_manager.after_step()
            return (True, False)

        # The last possible event is M_switches_to_WT
        self._execute_switch_event(Mutant, Wildtype)
        if self.metrics_manager:
            self.metrics_manager.after_step()
        return (True, False)
