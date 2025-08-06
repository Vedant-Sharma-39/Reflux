# FILE: src/core/model.py
# The unified Gillespie simulation model. [v18 - Corrected Hex Invariant]

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
        return (
            np.mean([-h.s for h in self._front_lookup]) if self._front_lookup else 0.0
        )

    @property
    def mutant_fraction(self) -> float:
        return (
            self.mutant_cell_count / self.total_cell_count
            if self.total_cell_count > 0
            else 0.0
        )

    @property
    def interface_width_sq(self) -> float:
        return np.var([-h.s for h in self._front_lookup]) if self._front_lookup else 0.0

    @property
    def interface_density(self) -> float:
        return len(self._front_lookup)

    @property
    def mutant_sector_width(self) -> float:
        return float(len(self.m_front_cells))

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
        env_seq = kwargs.get("environment_patch_sequence")
        if env_map is None or env_seq is None:
            self.environment_map = {0: {"b_wt": 1.0, "b_m": self.global_b_m}}
            self.patch_sequence = [(0, kwargs.get("patch_width", self.length))]
        else:
            self.environment_map = env_map
            self.patch_sequence = (
                env_seq if isinstance(env_seq, list) else kwargs.get(env_seq, [])
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

    def _find_initial_front(self):
        self.population_snapshot = self.population.copy()
        for h in list(self.population.keys()):
            self._update_events_for_cell(h)

    def _get_neighbors_periodic(self, h: Hex) -> list:
        """[FIX] Correctly recalculate 's' to maintain the q+r+s=0 invariant."""
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
            # The original `n.s` is now incorrect. Recalculate it.
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
        if cell_type is None:
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
        self.population_snapshot = {h: self.population.get(h) for h in affected_hexes}
        if event_type == "grow":
            cell_type = self.population[parent]
            self.population[target] = cell_type
            if cell_type == Mutant:
                self.mutant_cell_count += 1
            self.total_cell_count += 1
            for n in self._get_neighbors_periodic(target):
                if n in self.population:
                    affected_hexes.add(n)
        elif event_type == "switch":
            new_type = Mutant if self.population[parent] == Wildtype else Wildtype
            self.population[parent] = new_type
            if new_type == Mutant:
                self.mutant_cell_count += 1
            else:
                self.mutant_cell_count -= 1
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

    def get_correlation_function(self):
        return []
