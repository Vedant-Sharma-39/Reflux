# FILE: src/linear_model.py
# [DEFINITIVELY CORRECTED & ROBUST - WITH PERTURBATION METHOD]
# This version fixes the AttributeError by storing `phi` and adds a
# dedicated `set_switching_rate` method for robustly handling perturbations.

import numpy as np
import random
from typing import Dict, Set, List, Tuple
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
        initial_condition_type: str = "patch",
        initial_mutant_patch_size: int = 0,
        metrics_manager: MetricsManager = None,
    ):
        # --- Store all core parameters ---
        self.width, self.length = width, length
        self.b_wt, self.b_m = 1.0, b_m
        self.k_total = k_total
        self.phi = phi  # <-- [FIX 1/2] STORE PHI AS AN ATTRIBUTE
        self.k_total_base = k_total 
        self.phi_base = phi      
        # --- Initialize state variables ---
        self.k_wt_m, self.k_m_wt = self._calculate_asymmetric_rates(
            self.k_total, self.phi
        )
        self.initial_condition_type = initial_condition_type
        self.initial_mutant_patch_size = initial_mutant_patch_size
        self.time = 0.0
        self.population: Dict[Hex, int] = self._initialize_population()
        self.wt_front_cells, self.m_front_cells, self._front_lookup = {}, {}, set()
        self.sum_q, self.sum_q_sq, self.front_cell_count = 0.0, 0.0, 0
        self.rates, self.total_rate = {}, 0.0
        self._find_initial_front()
        self._update_rates()
        self.metrics_manager = metrics_manager
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)

    def set_switching_rate(self, new_k_total: float, new_phi: float = None):
        """
        Allows for dynamically changing the global switching
        parameters during a simulation, as required for perturbation experiments.
        """
        if new_phi is None:
            new_phi = self.phi_base  

        # Update the component rates
        self.k_wt_m, self.k_m_wt = self._calculate_asymmetric_rates(
            new_k_total, new_phi
        )

        # This is crucial: immediately update the total event rate in the simulation
        self._update_rates()

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

    def _initialize_population(self) -> Dict[Hex, int]:
        if self.initial_condition_type == "mixed":
            return {
                self._axial_to_cube(0, r): random.choice([Wildtype, Mutant])
                for r in range(self.width)
            }
        elif self.initial_condition_type == "patch":
            if self.initial_mutant_patch_size > 0:
                start = (self.width - self.initial_mutant_patch_size) // 2
                end = start + self.initial_mutant_patch_size
                return {
                    self._axial_to_cube(0, r): Mutant if start <= r < end else Wildtype
                    for r in range(self.width)
                }
            else:
                return {self._axial_to_cube(0, r): Wildtype for r in range(self.width)}
        else:
            raise ValueError(
                f"Unknown initial_condition_type: {self.initial_condition_type}"
            )

    @staticmethod
    def _axial_to_cube(q, r_offset):
        r = r_offset - (q + (q & 1)) // 2
        return Hex(q, r, -q - r)

    def _get_neighbors_periodic(self, cell):
        return [
            self._axial_to_cube(n.q, (n.r + (n.q + (n.q & 1)) // 2) % self.width)
            for n in cell.neighbors()
        ]

    def _is_valid_growth_site(self, cell):
        return 0 <= cell.q < self.length and cell not in self.population

    @staticmethod
    def _calculate_asymmetric_rates(k_total, phi):
        phi = np.clip(phi, -1, 1)
        return (k_total / 2.0) * (1.0 - phi), (k_total / 2.0) * (1.0 + phi)

    def _find_initial_front(self):
        for cell in list(self.population):
            self._check_and_update_front_status(cell)

    def _check_and_update_front_status(self, cell):
        is_on_front = any(
            self._is_valid_growth_site(n) for n in self._get_neighbors_periodic(cell)
        )
        is_in_lookup = cell in self._front_lookup
        if is_on_front and not is_in_lookup:
            self._front_lookup.add(cell)
            c_type = self.population.get(cell)
            (self.wt_front_cells if c_type == Wildtype else self.m_front_cells)[
                cell
            ] = True
            self.front_cell_count += 1
            self.sum_q += cell.q
            self.sum_q_sq += cell.q**2
        elif not is_on_front and is_in_lookup:
            self._front_lookup.remove(cell)
            c_type = self.population.get(cell)
            if c_type == Wildtype:
                del self.wt_front_cells[cell]
            else:
                del self.m_front_cells[cell]
            self.front_cell_count -= 1
            self.sum_q -= cell.q
            self.sum_q_sq -= cell.q**2

    def _update_rates(self):
        c_wt, c_m = len(self.wt_front_cells), len(self.m_front_cells)
        self.rates = {
            "WT_grows": c_wt * self.b_wt,
            "M_grows": c_m * self.b_m,
            "WT_switches_to_M": c_wt * self.k_wt_m,
            "M_switches_to_WT": c_m * self.k_m_wt,
        }
        self.total_rate = sum(self.rates.values())

    def _execute_growth_event(self, parent_type: int):
        parent_pool = (
            self.wt_front_cells if parent_type == Wildtype else self.m_front_cells
        )
        if not parent_pool:
            return False, False

        parent_cell = random.choice(list(parent_pool.keys()))
        empty_neighbors = [
            n
            for n in self._get_neighbors_periodic(parent_cell)
            if self._is_valid_growth_site(n)
        ]

        if not empty_neighbors:
            return True, False

        growth_site = random.choice(empty_neighbors)
        self.population[growth_site] = parent_type

        for c in self._get_neighbors_periodic(growth_site) + [growth_site, parent_cell]:
            if c in self.population:
                self._check_and_update_front_status(c)

        return True, growth_site.q >= self.length - 1

    def _execute_switch_event(self, original_type: int):
        source_pool = (
            self.wt_front_cells if original_type == Wildtype else self.m_front_cells
        )
        target_pool = (
            self.m_front_cells if original_type == Wildtype else self.wt_front_cells
        )
        if not source_pool:
            return

        cell_to_switch = random.choice(list(source_pool.keys()))
        self.population[cell_to_switch] = (
            Mutant if original_type == Wildtype else Wildtype
        )

        del source_pool[cell_to_switch]
        target_pool[cell_to_switch] = True

    def step(self) -> Tuple[bool, bool]:
        if self.total_rate <= 1e-9:
            return (False, False)

        dt = -np.log(random.uniform(0, 1)) / self.total_rate
        self.time += dt
        u = random.uniform(0, 1) * self.total_rate

        did_step, boundary_hit = True, False

        cumulative_rate = self.rates["WT_grows"]
        if u < cumulative_rate:
            did_step, boundary_hit = self._execute_growth_event(Wildtype)
        else:
            cumulative_rate += self.rates["M_grows"]
            if u < cumulative_rate:
                did_step, boundary_hit = self._execute_growth_event(Mutant)
            else:
                cumulative_rate += self.rates["WT_switches_to_M"]
                if u < cumulative_rate:
                    self._execute_switch_event(Wildtype)
                else:
                    self._execute_switch_event(Mutant)

        self._update_rates()  # This is called here for non-perturbation runs
        if self.metrics_manager:
            self.metrics_manager.after_step_hook()
        return did_step, boundary_hit
