# FILE: src/core/model.py (Final Version with Asymmetric Patch Logic)

import numpy as np
import random
from typing import Dict, Set, Tuple, List, Optional

from src.core.hex_utils import Hex
from src.core.metrics import MetricsManager
from pathlib import Path

Empty, Wildtype, Mutant = 0, 1, 2


class SummedRateTree:
    # ... (SummedRateTree class is unchanged) ...
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
            raise IndexError("Index out of bounds.")
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
        self.population: Dict[Hex, int] = self._initialize_population_pointytop(
            ic_type, ic_patch_size
        )
        self.wt_front_cells: Dict[Hex, List[Hex]] = {}
        self.m_front_cells: Dict[Hex, List[Hex]] = {}
        self._front_lookup: Set[Hex] = set()

        self._precompute_env_params(**kwargs)  # <-- This function is now updated
        self.plotter = None

        # Check for the run_mode passed in through kwargs
        if kwargs.get("run_mode") == "visualization":
            from src.core.hex_utils import HexPlotter
            self.plotter = HexPlotter(hex_size=1.0, labels={}, colormap={1: "#0c2c5c", 2: "#d6a000"})

            # Create a unique directory for this task's images
            self.snapshot_dir = Path(kwargs.get("output_dir_viz", ".")) / kwargs.get("task_id", "viz_task")
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

            self.snapshot_interval_cycles = kwargs.get("snapshot_interval_cycles", 1)
            # Calculate the length of one environmental cycle to trigger snapshots
            self.cycle_q_viz = self.patch_sequence[0][1] * len(self.patch_sequence) if self.patch_sequence else 0
            self.next_snapshot_q = self.cycle_q_viz

        self._wt_m_interface_bonds = 0
        self._find_initial_front()
        self.metrics_manager = kwargs.get("metrics_manager")
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)
            self.metrics_manager.initialize_all()

    def _precompute_env_params(self, **kwargs):
        # --- NEW: Handles both old and new environment definitions ---
        env_def = kwargs.get("env_definition")
        k_wt_m, k_m_wt = self._calculate_asymmetric_rates(
            self.global_k_total, self.global_phi
        )

        if env_def and isinstance(env_def, dict):
            # Logic for new, structured environments (like asymmetric_patches)
            patch_param_list = []
            sorted_patches = sorted(env_def.get("patches", []), key=lambda p: p["id"])
            for patch in sorted_patches:
                p_params = patch.get("params", {})
                patch_param_list.append(
                    [
                        p_params.get("b_wt", 1.0),
                        p_params.get("b_m", self.global_b_m),
                        k_wt_m,
                        k_m_wt,
                    ]
                )
            self.patch_params = np.array(patch_param_list)
            self.num_patches = len(self.patch_params)

            if env_def.get("scrambled"):
                avg_width = env_def.get("avg_patch_width", 50)
                patch_types = [p["id"] for p in sorted_patches]
                proportions = [p["proportion"] for p in sorted_patches]
                self.patch_sequence = []
                total_len = 0
                while total_len < self.length:
                    width = int(np.random.exponential(scale=avg_width))
                    if width == 0:
                        continue
                    patch_type = np.random.choice(patch_types, p=proportions)
                    self.patch_sequence.append((patch_type, width))
                    total_len += width
            else:
                self.patch_sequence = [
                    (p["id"], p["width"]) for p in env_def.get("patches", [])
                ]
        else:
            # Fallback to original logic for symmetric patches
            env_map = kwargs.get("environment_map", {})
            patch_width = kwargs.get("patch_width", 0)
            self.patch_sequence = (
                [(i % len(env_map), patch_width) for i in range(len(env_map))]
                if patch_width > 0 and env_map
                else [(0, self.length)]
            )
            self.num_patches = len(env_map) if env_map else 1
            self.patch_params = np.array(
                [
                    [
                        env_map.get(i, {}).get("b_wt", 1.0),
                        env_map.get(i, {}).get("b_m", self.global_b_m),
                        k_wt_m,
                        k_m_wt,
                    ]
                    for i in range(self.num_patches)
                ]
                if env_map
                else [[1.0, self.global_b_m, k_wt_m, k_m_wt]]
            )

        self._precompute_patch_indices()

    # ... (The rest of the GillespieSimulation class is unchanged) ...
    @property
    def total_rate(self) -> float:
        return self.tree.get_total_rate()

    @property
    def mean_front_position(self) -> float:
        return np.mean([h.q for h in self._front_lookup]) if self._front_lookup else 0.0

    @property
    def mutant_fraction(self) -> float:
        return (
            self.mutant_cell_count / self.total_cell_count
            if self.total_cell_count > 0
            else 0.0
        )

    @property
    def front_roughness_sq(self) -> float:
        return np.var([h.q for h in self._front_lookup]) if self._front_lookup else 0.0

    @property
    def expanding_front_length(self) -> float:
        return float(len(self._front_lookup))

    @property
    def domain_boundary_length(self) -> float:
        return self._wt_m_interface_bonds / 2.0

    @property
    def mutant_sector_width(self) -> float:
        return float(len(self.m_front_cells))

    def _axial_to_cube(self, q: int, r_offset: int) -> Hex:
        """[PROVEN LOGIC] Helper to convert from the working offset coordinate system."""
        r = r_offset - (q + (q & 1)) // 2
        return Hex(q, r, -q - r)

    def _initialize_population_pointytop(self, ic_type, patch_size):
        """[PROVEN LOGIC] Initializes a vertical line of cells."""
        pop = {}
        if ic_type == "single_cell":
            pop[Hex(0, 0, 0)] = Wildtype
            self.total_cell_count = 1
        else:
            start_idx = (self.width - patch_size) // 2
            for i in range(self.width):
                h = self._axial_to_cube(0, i)
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

    def _precompute_patch_indices(self):
        self.q_to_patch_index = np.zeros(self.length, dtype=int)
        current_q = 0
        if not self.patch_sequence or self.patch_sequence[0][1] == 0:
            return
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
        for h in list(self.population.keys()):
            self._update_events_for_cell(h)
        self._wt_m_interface_bonds = self._calculate_initial_wt_mutant_interface()

    def _get_neighbors_periodic(self, h: Hex) -> list:
        """[PROVEN LOGIC] Correct periodic boundaries for pointy-top, ported from linear_model."""
        unwrapped = h.neighbors()
        wrapped = []
        for n in unwrapped:
            offset_r = n.r + (n.q + (n.q & 1)) // 2
            offset_r %= self.width
            wrapped.append(self._axial_to_cube(n.q, offset_r))
        return wrapped

    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool:
        if neighbor in self.population:
            return False
        if self.is_radial_growth:
            return True
        if not (0 <= neighbor.q < self.length):
            return False
        return neighbor.q >= parent.q

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
        q_idx = np.clip(int(h.q), 0, self.length - 1)
        patch_idx = self.q_to_patch_index[q_idx]
        return self.patch_params[patch_idx]

    def _update_events_for_cell(self, h: Hex):
        cell_type = self.population.get(h)
        if h in self._front_lookup:
            for neighbor in self.wt_front_cells.get(h, []) + self.m_front_cells.get(
                h, []
            ):
                self._remove_event(("grow", h, neighbor))
            self._remove_event(("switch", h, None))
            self._front_lookup.discard(h)
            self.wt_front_cells.pop(h, None)
            self.m_front_cells.pop(h, None)
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
        affected_hexes = {parent, target} if target else {parent}
        for n in self._get_neighbors_periodic(parent):
            if n in self.population:
                affected_hexes.add(n)
        if target:
            for n in self._get_neighbors_periodic(target):
                if n in self.population:
                    affected_hexes.add(n)
        interface_delta = 0
        hex_to_check = target if event_type == "grow" else parent
        old_type = self.population.get(hex_to_check, Empty)
        for n in self._get_neighbors_periodic(hex_to_check):
            n_type = self.population.get(n)
            if (
                n_type is not None
                and n_type != Empty
                and old_type != Empty
                and n_type != old_type
            ):
                interface_delta -= 2
        if event_type == "grow":
            new_type = self.population[parent]
            self.population[target] = new_type
            if new_type == Mutant:
                self.mutant_cell_count += 1
            self.total_cell_count += 1
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
            if n_type is not None and n_type != Empty and n_type != new_type:
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

        if self.plotter and self.cycle_q_viz > 0 and self.mean_front_position >= self.next_snapshot_q:
            cycle_num = int(round(self.next_snapshot_q / self.cycle_q_viz))
            title = f"Task {self.snapshot_dir.name}\nCycle {cycle_num}, Time: {self.time:.1f}"
            self.plotter.plot_population(self.population, title=title)
            snapshot_path = self.snapshot_dir / f"snap_cycle_{cycle_num:03d}.png"
            self.plotter.save_figure(snapshot_path)
            # Set the q-position for the next snapshot
            self.next_snapshot_q += self.cycle_q_viz * self.snapshot_interval_cycles

        boundary_hit = self.mean_front_position >= self.length - 2
        return True, boundary_hit
