# FILE: src/fluctuating_model.py
# [DEFINITIVE, VECTORIZED & OPTIMIZED VERSION]
# This version achieves a massive speedup by replacing slow Python loops in the
# rate calculation with fast, vectorized NumPy operations. It preserves the
# exact physics of the independent switching model (Model 1) while being
# computationally efficient.

import numpy as np
import random
from typing import Dict, Set, Tuple
from hex_utils import Hex
from metrics import MetricsManager

Empty, Wildtype, Mutant = 0, 1, 2


class FluctuatingGillespieSimulation:
    def __init__(
        self,
        width: int,
        length: int,
        environment_map: Dict,
        patch_width: int,
        k_total: float,
        phi: float,
        b_m: float,
        initial_condition_type: str = "mixed",
        initial_mutant_patch_size: int = 0,
        metrics_manager: MetricsManager = None,
    ):
        self.width, self.length = width, length
        self.environment_map = environment_map
        self.patch_width = patch_width
        self.num_patches = len(environment_map)
        self.global_k_total = k_total
        self.global_phi = phi
        self.global_b_m = b_m
        self.time = 0.0
        self.population: Dict[Hex, int] = self._initialize_population(
            initial_condition_type, initial_mutant_patch_size
        )
        self.wt_front_cells, self.m_front_cells, self._front_lookup = {}, {}, set()
        self.sum_q, self.sum_q_sq, self.front_cell_count = 0.0, 0.0, 0
        self.rates = {}
        self.total_rate = 0.0

        # --- NEW OPTIMIZATIONS ---
        # Pre-calculate patch indices for all possible q-values to avoid division in the loop.
        self.q_to_patch_index = (
            np.floor(np.arange(length) / patch_width).astype(int) % self.num_patches
        )

        # Pre-calculate parameters for each patch into a NumPy array for fast lookups.
        # Columns: [b_wt, b_m, k_wt_m, k_m_wt]
        k_wt_m_global, k_m_wt_global = self._calculate_asymmetric_rates(
            self.global_k_total, self.global_phi
        )
        self.patch_params = np.array(
            [
                [
                    environment_map.get(i, {}).get("b_wt", 1.0),
                    environment_map.get(i, {}).get("b_m", self.global_b_m),
                    k_wt_m_global,
                    k_m_wt_global,
                ]
                for i in range(self.num_patches)
            ]
        )

        self._find_initial_front()
        self._update_rates()
        self.metrics_manager = metrics_manager
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)

    @staticmethod
    def _calculate_asymmetric_rates(k_total, phi):
        phi = np.clip(phi, -1, 1)
        return (k_total / 2.0) * (1.0 - phi), (k_total / 2.0) * (1.0 + phi)

    def _update_rates(self):
        """
        [VECTORIZED & OPTIMIZED] Calculates total rates for each event class using
        fast NumPy operations, eliminating the slow Python for-loop.
        """
        # --- WT Cells Rate Calculation ---
        if self.wt_front_cells:
            wt_q_coords = np.array([cell.q for cell in self.wt_front_cells], dtype=int)
            # Clip q_coords to be within bounds of the pre-calculated array
            wt_q_coords = np.clip(wt_q_coords, 0, self.length - 1)
            wt_patch_indices = self.q_to_patch_index[wt_q_coords]
            wt_params_per_cell = self.patch_params[wt_patch_indices]

            # Sum up rates for each event type across all WT front cells
            total_wt_growth_rate = np.sum(wt_params_per_cell[:, 0])  # Sum of b_wt
            total_wt_switch_rate = np.sum(wt_params_per_cell[:, 2])  # Sum of k_wt_m

            self.rates["WT_grows"] = total_wt_growth_rate
            self.rates["WT_switches_to_M"] = total_wt_switch_rate
        else:
            self.rates["WT_grows"] = 0.0
            self.rates["WT_switches_to_M"] = 0.0

        # --- Mutant Cells Rate Calculation ---
        if self.m_front_cells:
            m_q_coords = np.array([cell.q for cell in self.m_front_cells], dtype=int)
            m_q_coords = np.clip(m_q_coords, 0, self.length - 1)
            m_patch_indices = self.q_to_patch_index[m_q_coords]
            m_params_per_cell = self.patch_params[m_patch_indices]

            # Sum up rates for each event type across all M front cells
            total_m_growth_rate = np.sum(m_params_per_cell[:, 1])  # Sum of b_m
            total_m_switch_rate = np.sum(m_params_per_cell[:, 3])  # Sum of k_m_wt

            self.rates["M_grows"] = total_m_growth_rate
            self.rates["M_switches_to_WT"] = total_m_switch_rate
        else:
            self.rates["M_grows"] = 0.0
            self.rates["M_switches_to_WT"] = 0.0

        self.total_rate = sum(self.rates.values())

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

    def _initialize_population(
        self, initial_condition_type, initial_mutant_patch_size
    ) -> Dict[Hex, int]:
        if initial_condition_type == "mixed":
            return {
                self._axial_to_cube(0, r): random.choice([Wildtype, Mutant])
                for r in range(self.width)
            }
        elif initial_condition_type == "patch":
            if initial_mutant_patch_size > 0:
                start = (self.width - initial_mutant_patch_size) // 2
                end = start + initial_mutant_patch_size
                return {
                    self._axial_to_cube(0, r): Mutant if start <= r < end else Wildtype
                    for r in range(self.width)
                }
            else:
                return {self._axial_to_cube(0, r): Wildtype for r in range(self.width)}
        raise ValueError(f"Unknown initial_condition_type: {initial_condition_type}")

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

    def _is_valid_growth_neighbor(self, cell):
        return self._is_valid_growth_site(cell)

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

    def _execute_growth_event(self, parent_type: int):
        """Selects a random parent weighted by its local growth rate."""
        parent_pool = (
            self.wt_front_cells if parent_type == Wildtype else self.m_front_cells
        )
        if not parent_pool:
            return False, False

        parents = list(parent_pool.keys())
        q_coords = np.array([p.q for p in parents], dtype=int)
        q_coords = np.clip(q_coords, 0, self.length - 1)
        patch_indices = self.q_to_patch_index[q_coords]

        rate_idx = 0 if parent_type == Wildtype else 1
        weights = self.patch_params[patch_indices, rate_idx]

        if np.sum(weights) <= 1e-9:
            return True, False

        parent_cell = random.choices(parents, weights=weights, k=1)[0]

        empty_neighbors = [
            n
            for n in self._get_neighbors_periodic(parent_cell)
            if self._is_valid_growth_neighbor(n)
        ]
        if not empty_neighbors:
            return True, False

        growth_site = random.choice(empty_neighbors)
        self.population[growth_site] = parent_type

        for c in self._get_neighbors_periodic(growth_site) + [growth_site, parent_cell]:
            if c in self.population:
                self._check_and_update_front_status(c)

        boundary_hit = growth_site.q >= self.length - 1
        return True, boundary_hit

    def _execute_switch_event(self, original_type: int):
        source_pool = (
            self.wt_front_cells if original_type == Wildtype else self.m_front_cells
        )
        if not source_pool:
            return

        parents = list(source_pool.keys())
        q_coords = np.array([p.q for p in parents], dtype=int)
        q_coords = np.clip(q_coords, 0, self.length - 1)
        patch_indices = self.q_to_patch_index[q_coords]

        rate_idx = 2 if original_type == Wildtype else 3
        weights = self.patch_params[patch_indices, rate_idx]

        if np.sum(weights) <= 1e-9:
            return

        cell_to_switch = random.choices(parents, weights=weights, k=1)[0]

        new_type = Mutant if original_type == Wildtype else Wildtype
        self.population[cell_to_switch] = new_type

        target_pool = (
            self.m_front_cells if original_type == Wildtype else self.wt_front_cells
        )
        del source_pool[cell_to_switch]
        target_pool[cell_to_switch] = True

    def step(self) -> Tuple[bool, bool]:
        if self.total_rate <= 1e-9:
            return False, False
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

        self._update_rates()
        if self.metrics_manager:
            self.metrics_manager.after_step_hook()
        return did_step, boundary_hit
