# FILE: src/core/model.py
# The unified Gillespie simulation model. [v24 - Terminology Aligned with Literature]

import numpy as np
import random
from typing import Dict, Set, Tuple, List, Optional

from src.core.hex_utils import Hex
from src.core.metrics import MetricsManager

Empty, Wildtype, Mutant = 0, 1, 2


class SummedRateTree:
    # ... (class is unchanged)
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)

    def _update_path(self, index: int, delta: float):
        index //= 2
        while index > 0:
            self.tree[index] += delta
            index //= 2

    def update(self, index: int, rate: float):
        if not (0 <= index < self.capacity):
            raise IndexError("Event index out of bounds for SummedRateTree.")
        leaf_index = index + self.capacity
        delta = rate - self.tree[leaf_index]
        if abs(delta) < 1e-12:
            return
        self.tree[leaf_index] = rate
        self._update_path(leaf_index, delta)

    def find_event(self, value: float) -> int:
        index = 1
        while index < self.capacity:
            left_child = 2 * index
            if value < self.tree[left_child]:
                index = left_child
            else:
                value -= self.tree[left_child]
                index = left_child + 1
        return index - self.capacity

    def get_total_rate(self) -> float:
        return self.tree[1]


class GillespieSimulation:
    # ... (__init__ and other methods are mostly unchanged) ...
    def __init__(
        self, width: int, length: int, b_m: float, k_total: float, phi: float, **kwargs
    ):
        self.width, self.length = width, length
        self.global_b_m, self.global_k_total, self.global_phi = b_m, k_total, phi
        self.time, self.step_count = 0.0, 0
        MAX_EVENTS = width * length * 7
        self.tree = SummedRateTree(MAX_EVENTS)
        self.event_to_idx: Dict[Tuple, int] = {}
        self.idx_to_event: Dict[int, Tuple] = {}
        self.free_indices = list(range(MAX_EVENTS - 1, -1, -1))
        self.mutant_cell_count, self.total_cell_count = 0, 0
        ic_type = kwargs.get("initial_condition_type", "mixed")
        ic_patch_size = kwargs.get("initial_mutant_patch_size", 0)
        self.is_radial_growth = ic_type == "single_cell"
        self.population: Dict[Hex, int] = self._initialize_population(
            ic_type, ic_patch_size
        )
        self.wt_front_cells: Dict[Hex, List[Hex]] = {}
        self.m_front_cells: Dict[Hex, List[Hex]] = {}
        self._front_lookup: Set[Hex] = set()
        self._wt_m_interface_bonds = 0
        self._precompute_env_params(**kwargs)
        self._find_initial_front()
        self.metrics_manager = kwargs.get("metrics_manager")
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)
            self.metrics_manager.initialize_all()

    @property
    def total_rate(self) -> float:
        return self.tree.get_total_rate()

    @property
    def mean_front_position(self) -> float:
        """The average position of the expanding front along the growth axis (q)."""
        return (
            np.mean([-h.s for h in self._front_lookup]) if self._front_lookup else 0.0
        )

    @property
    def mutant_fraction(self) -> float:
        """The fraction of mutant cells in the entire population (ρ_M)."""
        return (
            self.mutant_cell_count / self.total_cell_count
            if self.total_cell_count > 0
            else 0.0
        )

    @property
    def front_roughness_sq(self) -> float:
        """
        The squared width of the expanding front (W²), a measure of geometric
        roughness calculated as the variance of front cell positions.
        """
        return np.var([-h.s for h in self._front_lookup]) if self._front_lookup else 0.0

    @property
    def expanding_front_length(self) -> float:
        """
        The number of cells at the population-empty space interface. This serves as a
        proxy for the effective population size (N_e) at the frontier.
        """
        return float(len(self._front_lookup))

    @property
    def domain_boundary_length(self) -> float:
        """
        The total length of the interface between Wild-Type and Mutant domains,
        a measure of genetic demixing/segregation.
        """
        return self._wt_m_interface_bonds / 2.0

    @property
    def mutant_sector_width(self) -> float:
        """The number of mutant cells at the expanding front."""
        return float(len(self.m_front_cells))

    # ... (The rest of the file, including the optimized _execute_event, is unchanged from the previous step) ...
    def _initialize_population(self, ic_type, patch_size):
        pop = {}
        if ic_type == "single_cell":
            pop[Hex(0, 0, 0)] = Wildtype
            self.total_cell_count = 1
        else:
            start_idx = (self.width - patch_size) // 2
            for i in range(self.width):
                h = Hex(i, -i, 0)
                cell_type = (
                    Mutant
                    if (ic_type == "patch" and start_idx <= i < start_idx + patch_size)
                    else Wildtype
                )
                pop[h] = cell_type
                if cell_type == Mutant:
                    self.mutant_cell_count += 1
                self.total_cell_count += 1
        return pop

    def _precompute_env_params(self, **kwargs):
        env_map = kwargs.get("environment_map")
        patch_width = kwargs.get("patch_width")
        if env_map is None:
            self.environment_map = {0: {"b_wt": 1.0, "b_m": self.global_b_m}}
            self.patch_sequence = [(0, patch_width or self.length)]
        else:
            self.environment_map = env_map
            self.patch_sequence = (
                [(i % len(env_map), patch_width) for i in range(len(env_map))]
                if patch_width
                else []
            )

        k_wt_m, k_m_wt = self._calculate_asymmetric_rates(
            self.global_k_total, self.global_phi
        )
        self.num_patches = len(self.environment_map)
        self.patch_params = np.array(
            [
                [
                    self.environment_map.get(i, {}).get("b_wt", 1.0),
                    self.environment_map.get(i, {}).get("b_m", self.global_b_m),
                    k_wt_m,
                    k_m_wt,
                ]
                for i in range(self.num_patches)
            ]
        )
        self._precompute_patch_indices()

    def _precompute_patch_indices(self):
        self.q_to_patch_index = np.zeros(self.length, dtype=int)
        current_q = 0
        if not self.patch_sequence:
            self.patch_sequence = [(0, self.length)]
        cycle_len = sum(width for _, width in self.patch_sequence)
        if cycle_len > 0:
            while current_q < self.length:
                q_in_cycle = 0
                for patch_type, width in self.patch_sequence:
                    start, end = current_q + q_in_cycle, current_q + q_in_cycle + width
                    if start < self.length:
                        self.q_to_patch_index[start : min(end, self.length)] = (
                            patch_type
                        )
                    q_in_cycle += width
                current_q += cycle_len

    def _calculate_initial_wt_mutant_interface(self) -> int:
        bond_count = 0
        for h, cell_type in self.population.items():
            if cell_type == Empty:
                continue
            for neighbor_hex in self._get_neighbors_periodic(h):
                neighbor_type = self.population.get(neighbor_hex)
                if (
                    neighbor_type is not None
                    and neighbor_type != Empty
                    and neighbor_type != cell_type
                ):
                    bond_count += 1
        return bond_count

    def _find_initial_front(self):
        self.population_snapshot = self.population.copy()
        for h in list(self.population.keys()):
            self._update_events_for_cell(h)
        self._wt_m_interface_bonds = self._calculate_initial_wt_mutant_interface()

    def _get_neighbors_periodic(self, h: Hex) -> list:
        unwrapped = h.neighbors()
        wrapped = []
        for n in unwrapped:
            q, r = n.q, n.r
            if q < 0:
                q += self.width
                r -= self.width
            elif q >= self.width:
                q -= self.width
                r += self.width
            s = -q - r
            wrapped.append(Hex(q, r, s))
        return wrapped

    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool:
        if neighbor in self.population:
            return False
        if self.is_radial_growth:
            return True
        return neighbor.s <= parent.s

    def _add_event(self, event: Tuple, rate: float):
        if not self.free_indices:
            raise MemoryError("SummedRateTree is full.")
        idx = self.free_indices.pop()
        self.event_to_idx[event], self.idx_to_event[idx] = idx, event
        self.tree.update(idx, rate)

    def _remove_event(self, event: Tuple):
        if event not in self.event_to_idx:
            return
        idx = self.event_to_idx.pop(event)
        self.free_indices.append(idx)
        del self.idx_to_event[idx]
        self.tree.update(idx, 0.0)

    def _get_rate_params_for_cell(self, h: Hex):
        q_idx = np.clip(int(-h.s), 0, self.length - 1)
        patch_idx = self.q_to_patch_index[q_idx]
        return self.patch_params[patch_idx]

    def _update_events_for_cell(self, h: Hex):
        cell_type = self.population.get(h)
        if h in self._front_lookup:
            old_cell_type = self.population_snapshot.get(h, cell_type)
            front_dict = (
                self.wt_front_cells if old_cell_type == Wildtype else self.m_front_cells
            )
            for neighbor in front_dict.get(h, []):
                self._remove_event(("grow", h, neighbor))
            self._remove_event(("switch", h, None))
            self._front_lookup.remove(h)
            if h in self.wt_front_cells:
                del self.wt_front_cells[h]
            if h in self.m_front_cells:
                del self.m_front_cells[h]
        if cell_type is None or cell_type == Empty:
            return
        empty_neighbors = [
            n
            for n in self._get_neighbors_periodic(h)
            if self._is_valid_growth_neighbor(n, h)
        ]
        if empty_neighbors:
            self._front_lookup.add(h)
            front_dict = (
                self.wt_front_cells if cell_type == Wildtype else self.m_front_cells
            )
            front_dict[h] = empty_neighbors
            b_wt, b_m, k_wt_m, k_m_wt = self._get_rate_params_for_cell(h)
            growth_rate = b_wt if cell_type == Wildtype else b_m
            for n in empty_neighbors:
                self._add_event(("grow", h, n), growth_rate)
            switch_rate = k_wt_m if cell_type == Wildtype else k_m_wt
            if switch_rate > 0:
                self._add_event(("switch", h, None), switch_rate)

    @staticmethod
    def _calculate_asymmetric_rates(k_total, phi):
        phi = np.clip(phi, -1, 1)
        return (k_total / 2.0) * (1.0 - phi), (k_total / 2.0) * (1.0 + phi)

    def _execute_event(self, event_type: str, parent: Hex, target: Optional[Hex]):
        affected_hexes = {parent}
        for n in self._get_neighbors_periodic(parent):
            if n in self.population:
                affected_hexes.add(n)
        if target:
            affected_hexes.add(target)

        interface_delta = 0
        hex_to_check = target if event_type == "grow" else parent
        old_type = self.population.get(hex_to_check, Empty)

        for n in self._get_neighbors_periodic(hex_to_check):
            n_type = self.population.get(n)
            if n_type is not None and n_type != Empty:
                if n_type != old_type and old_type != Empty:
                    interface_delta -= 2

        self.population_snapshot = {h: self.population.get(h) for h in affected_hexes}

        if event_type == "grow":
            new_type = self.population[parent]
            self.population[target] = new_type
            if new_type == Mutant:
                self.mutant_cell_count += 1
            self.total_cell_count += 1
            for n in self._get_neighbors_periodic(target):
                if n in self.population:
                    affected_hexes.add(n)
        elif event_type == "switch":
            old_cell_type = self.population[parent]
            new_type = Mutant if old_cell_type == Wildtype else Wildtype
            self.population[parent] = new_type
            if new_type == Mutant:
                self.mutant_cell_count += 1
            else:
                self.mutant_cell_count -= 1

        new_type = self.population[hex_to_check]
        for n in self._get_neighbors_periodic(hex_to_check):
            n_type = self.population.get(n)
            if n_type is not None and n_type != Empty:
                if n_type != new_type:
                    interface_delta += 2

        self._wt_m_interface_bonds += interface_delta

        for h in affected_hexes:
            self._update_events_for_cell(h)

    def step(self):
        self.step_count += 1
        current_total_rate = self.total_rate
        if current_total_rate <= 1e-9:
            return False, False
        dt = -np.log(random.random()) / current_total_rate
        self.time += dt
        rand_val = random.random() * current_total_rate
        event_idx = self.tree.find_event(rand_val)
        if event_idx not in self.idx_to_event:
            return True, False
        event_type, parent, target = self.idx_to_event[event_idx]
        self._execute_event(event_type, parent, target)
        boundary_hit = self.mean_front_position >= self.length - 2
        return True, boundary_hit
