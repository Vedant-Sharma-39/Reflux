# FILE: src/core/model.py
# The unified Gillespie simulation model. [v14 - Corrected Boundaries & Slant]

import numpy as np
import random
from typing import Dict, Set, Tuple, List, Optional

from src.core.hex_utils import Hex
from src.core.metrics import MetricsManager

Empty, Wildtype, Mutant = 0, 1, 2


class SummedRateTree:
    """A binary tree for O(log M) event selection and rate updates."""

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
        self.global_b_m = b_m
        self.global_k_total = k_total
        self.global_phi = phi
        self.time = 0.0

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
        self.sum_q, self.sum_q_sq, self.front_cell_count = 0.0, 0.0, 0

        self._precompute_env_params(**kwargs)
        self._find_initial_front()

        self.metrics_manager = kwargs.get("metrics_manager")
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)

    # --- Properties ---
    @property
    def total_rate(self) -> float:
        return self.tree.get_total_rate()

    @property
    def mean_front_position(self) -> float:
        s_coords = [-h.s for h in self._front_lookup]
        return np.mean(s_coords) if s_coords else 0.0

    @property
    def mutant_fraction(self) -> float:
        return (
            self.mutant_cell_count / self.total_cell_count
            if self.total_cell_count > 0
            else 0.0
        )

    @property
    def interface_width_sq(self) -> float:
        if not self._front_lookup:
            return 0.0
        s_coords = np.array([-h.s for h in self._front_lookup])
        return np.var(s_coords)

    @property
    def interface_density(self) -> float:
        return len(self._front_lookup)

    @property
    def mutant_sector_width(self) -> float:
        if not self.m_front_cells:
            return 0.0

        coords = sorted([h.q - h.r for h in self.m_front_cells])
        if not coords:
            return 0.0
        if len(coords) <= 1:
            return float(len(coords))

        # Calculate gaps between consecutive points, including the wrap-around gap
        gaps = [coords[i + 1] - coords[i] for i in range(len(coords) - 1)]
        circumference = 2 * self.width
        wrap_around_gap = (coords[0] + circumference) - coords[-1]
        gaps.append(wrap_around_gap)

        # The span of the sector is the circumference minus the largest gap.
        # We add 1 to count the number of discrete sites, matching original intent.
        span = circumference - max(gaps)
        return span + 1

    # --- Initialization Methods ---
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
                    or (ic_type == "mixed" and random.random() < 0.5)
                    else Wildtype
                )
                pop[h] = cell_type
                if cell_type == Mutant:
                    self.mutant_cell_count += 1
                self.total_cell_count += 1
        return pop

    def _precompute_env_params(self, **kwargs):
        environment_map = kwargs.get("environment_map", None)
        environment_patch_sequence = kwargs.get("environment_patch_sequence", None)

        if environment_map is None or environment_patch_sequence is None:
            self.environment_map = {0: {"b_wt": 1.0, "b_m": self.global_b_m}}
            patch_width = kwargs.get("patch_width", self.length)
            self.patch_sequence = [(0, patch_width)]
        else:
            self.environment_map = environment_map
            if isinstance(environment_patch_sequence, str) and kwargs.get(
                environment_patch_sequence
            ):
                self.patch_sequence = kwargs.get(environment_patch_sequence)
            else:
                self.patch_sequence = environment_patch_sequence

        k_wt_m_global, k_m_wt_global = self._calculate_asymmetric_rates(
            self.global_k_total, self.global_phi
        )
        self.num_patches = len(self.environment_map)
        self.patch_params = np.array(
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
        self._precompute_patch_indices()

    def _precompute_patch_indices(self):
        self.q_to_patch_index = np.zeros(self.length, dtype=int)
        current_q = 0
        cycle_len = sum(width for _, width in self.patch_sequence)
        if cycle_len > 0:
            while current_q < self.length:
                q_pos_in_cycle = 0
                for patch_type, patch_width in self.patch_sequence:
                    start_q = current_q + q_pos_in_cycle
                    end_q = start_q + patch_width
                    if start_q < self.length:
                        self.q_to_patch_index[start_q : min(end_q, self.length)] = (
                            patch_type
                        )
                    q_pos_in_cycle += patch_width
                current_q += cycle_len

    def _find_initial_front(self):
        self.population_snapshot = self.population.copy()
        for h in list(self.population.keys()):
            self._update_events_for_cell(h)

    # --- Core State & Event Management ---
    def _get_neighbors_periodic(self, h: Hex) -> list:
        unwrapped = h.neighbors()
        wrapped = []
        for n in unwrapped:
            q, r = n.q, n.r
            transverse_coord = q - r

            # The transverse direction is q-r, with a period of 2*self.width.
            # The wrapping vector must preserve the s-coordinate, which defines the forward direction.
            # The correct wrapping vector in (q,r) is (+width, -width) or (-width, +width).
            if transverse_coord < 0:
                # Wrap from "left" to "right"
                q += self.width
                r -= self.width
            elif transverse_coord >= 2 * self.width:
                # Wrap from "right" to "left"
                q -= self.width
                r += self.width

            wrapped.append(Hex(q, r, -q - r))
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
        self.event_to_idx[event] = idx
        self.idx_to_event[idx] = event
        self.tree.update(idx, rate)

    def _remove_event(self, event: Tuple):
        if event not in self.event_to_idx:
            return
        idx = self.event_to_idx.pop(event)
        self.free_indices.append(idx)
        del self.idx_to_event[idx]
        self.tree.update(idx, 0.0)

    def _get_rate_params_for_cell(self, h: Hex):
        front_pos = -h.s
        q_idx = np.clip(int(front_pos), 0, self.length - 1)
        patch_idx = self.q_to_patch_index[q_idx]
        return self.patch_params[patch_idx]

    def _update_events_for_cell(self, h: Hex):
        cell_type = self.population.get(h)
        was_in_front = h in self._front_lookup

        if was_in_front:
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
            for neighbor in empty_neighbors:
                self._add_event(("grow", h, neighbor), growth_rate)
            switch_rate = k_wt_m if cell_type == Wildtype else k_m_wt
            self._add_event(("switch", h, None), switch_rate)

    @staticmethod
    def _calculate_asymmetric_rates(k_total, phi):
        phi = np.clip(phi, -1, 1)
        return (k_total / 2.0) * (1.0 - phi), (k_total / 2.0) * (1.0 + phi)

    # --- Event Execution ---
    def _execute_event(self, event_type: str, parent: Hex, target: Optional[Hex]):
        self.population_snapshot = {
            h: self.population.get(h)
            for h in [parent] + self._get_neighbors_periodic(parent)
            if h is not None and h in self.population
        }
        if event_type == "grow":
            cell_type = self.population[parent]
            self.population[target] = cell_type
            if cell_type == Mutant:
                self.mutant_cell_count += 1
            self.total_cell_count += 1
            affected_hexes = {parent, target}
            for n in self._get_neighbors_periodic(target):
                if n in self.population:
                    affected_hexes.add(n)
            for h in affected_hexes:
                self._update_events_for_cell(h)
        elif event_type == "switch":
            new_type = Mutant if self.population[parent] == Wildtype else Wildtype
            self.population[parent] = new_type
            if new_type == Mutant:
                self.mutant_cell_count += 1
            else:
                self.mutant_cell_count -= 1
            affected_hexes = {parent}
            for n in self._get_neighbors_periodic(parent):
                if n in self.population:
                    affected_hexes.add(n)
            for h in affected_hexes:
                self._update_events_for_cell(h)

    # --- Main Step Function ---
    def step(self):
        current_total_rate = self.tree.get_total_rate()
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
        # Placeholder for future implementation
        return []
