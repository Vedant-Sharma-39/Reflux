# FILE: src/core/model.py (This is the definitive, fully-corrected version)

import numpy as np
import random
from typing import Dict, Set, Tuple, List, Optional

from src.core.hex_utils import Hex, HexPlotter
from src.core.metrics import MetricsManager
from pathlib import Path

Empty, Wildtype, Mutant = 0, 1, 2


class SummedRateTree:
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
        self._precompute_env_params(**kwargs)
        self.plotter: Optional[HexPlotter] = None
        if kwargs.get("run_mode") == "visualization":
            self.plotter = HexPlotter(
                hex_size=1.0, labels={}, colormap={1: "#0c2c5c", 2: "#d6a000"}
            )
            self.snapshot_dir = Path(kwargs.get("output_dir_viz", ".")) / kwargs.get(
                "task_id", "viz_task"
            )
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

            # --- NEW DYNAMIC SNAPSHOT LOGIC ---
            self.max_snapshots = kwargs.get("max_snapshots", 5)
            self.snapshot_q_offset = kwargs.get("snapshot_q_offset", 2.0)
            self.snapshots_taken = 0

            boundary_indices = np.where(np.diff(self.q_to_patch_index) != 0)[0]
            # The boundary is at q_idx + 0.5. The trigger is offset units before that.
            self.snapshot_q_triggers = [
                (q_idx + 0.5) - self.snapshot_q_offset for q_idx in boundary_indices
            ]
            self.next_snapshot_trigger_index = 0
            # --- END NEW LOGIC ---

        self._wt_m_interface_bonds = 0
        self._find_initial_front()
        self.metrics_manager = kwargs.get("metrics_manager")
        if self.metrics_manager:
            self.metrics_manager.register_simulation(self)
            self.metrics_manager.initialize_all()

    def _precompute_env_params(self, **kwargs):
        env_def = kwargs.get("env_definition")
        self.k_wt_m, self.k_m_wt = self._calculate_asymmetric_rates(
            self.global_k_total, self.global_phi
        )
        if env_def and isinstance(env_def, dict):
            patch_param_list = [
                [
                    p.get("params", {}).get("b_wt", 1.0),
                    p.get("params", {}).get("b_m", self.global_b_m),
                ]
                for p in sorted(env_def.get("patches", []), key=lambda p: p["id"])
            ]
            self.patch_params = np.array(patch_param_list)
            if env_def.get("scrambled"):
                avg_width = env_def.get("avg_patch_width", 50)
                patch_types = [
                    p["id"]
                    for p in sorted(env_def.get("patches", []), key=lambda p: p["id"])
                ]
                proportions = [
                    p["proportion"]
                    for p in sorted(env_def.get("patches", []), key=lambda p: p["id"])
                ]
                self.patch_sequence = []
                total_len = 0
                while total_len < self.length:
                    width = int(np.random.exponential(scale=avg_width))
                    if width == 0:
                        continue
                    self.patch_sequence.append(
                        (np.random.choice(patch_types, p=proportions), width)
                    )
                    total_len += width
            else:
                self.patch_sequence = [
                    (p["id"], p["width"]) for p in env_def.get("patches", [])
                ]
        else:
            env_map = kwargs.get("environment_map", {})
            patch_width = kwargs.get("patch_width", 0)
            if patch_width > 0 and env_map:
                # --- FIX: Generate the full, explicit patch sequence ---
                num_patches = (
                    self.length + patch_width - 1
                ) // patch_width  # Ceiling division
                self.patch_sequence = [
                    (i % len(env_map), patch_width) for i in range(num_patches)
                ]
                # --- END FIX ---
            else:
                self.patch_sequence = [(0, self.length)]

            self.patch_params = np.array(
                [
                    [
                        env_map.get(str(i), {}).get("b_wt", 1.0),
                        env_map.get(str(i), {}).get("b_m", self.global_b_m),
                    ]
                    for i in range(len(env_map) if env_map else 1)
                ]
            )
        self._precompute_patch_indices()

    def _initialize_population_pointytop(
        self, ic_type: str, patch_size: int
    ) -> Dict[Hex, int]:
        pop: Dict[Hex, int] = {}
        if ic_type == "single_cell":
            pop[Hex(0, 0, 0)] = Wildtype
            self.total_cell_count = 1
            return pop
        for i in range(self.width):
            h = self._axial_to_cube(0, i)
            if ic_type == "mixed":
                cell_type = random.choice([Wildtype, Mutant])
            elif ic_type == "patch":
                cell_type = (
                    Mutant
                    if (self.width - patch_size) // 2
                    <= i
                    < (self.width - patch_size) // 2 + patch_size
                    else Wildtype
                )
            else:
                cell_type = Wildtype
            pop[h] = cell_type
            if cell_type == Mutant:
                self.mutant_cell_count += 1
            self.total_cell_count += 1
        return pop

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

    def _update_events_for_cell(self, h: Hex):
        cell_type = self.population.get(h)
        if h in self._front_lookup:
            for neighbor in self.wt_front_cells.pop(h, []) + self.m_front_cells.pop(
                h, []
            ):
                self._remove_event(("grow", h, neighbor))
            self._front_lookup.discard(h)
        self._remove_event(("switch", h, Wildtype))
        self._remove_event(("switch", h, Mutant))
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
            if cell_type == Wildtype and self.k_wt_m > 0:
                self._add_event(("switch", h, Mutant), self.k_wt_m)
            elif cell_type == Mutant and self.k_m_wt > 0:
                self._add_event(("switch", h, Wildtype), self.k_m_wt)
            for n in empty_neighbors:
                b_wt_target, b_m_target = self._get_params_for_q(n.q)
                growth_rate = b_wt_target if cell_type == Wildtype else b_m_target
                if growth_rate > 0:
                    self._add_event(("grow", h, n), growth_rate)

    def _execute_event(self, event_type: str, parent: Hex, target: Optional[Hex]):
        affected_hexes = {parent}
        if event_type == "grow":
            affected_hexes.add(target)
            new_type = self.population[parent]
            self.population[target] = new_type
            if new_type == Mutant:
                self.mutant_cell_count += 1
            self.total_cell_count += 1
        elif event_type == "switch":
            old_type = self.population[parent]
            new_type = target
            if new_type == old_type:
                return
            self.population[parent] = new_type
            if new_type == Mutant:
                self.mutant_cell_count += 1
            else:
                self.mutant_cell_count -= 1
        else:
            return
        final_affected_to_update = set(affected_hexes)
        for h in affected_hexes:
            for neighbor in self._get_neighbors_periodic(h):
                final_affected_to_update.add(neighbor)
        for h in final_affected_to_update:
            if h in self.population:
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

        # --- REPLACED DYNAMIC SNAPSHOT LOGIC ---
        if self.plotter and self.snapshots_taken < self.max_snapshots:
            if self.next_snapshot_trigger_index < len(self.snapshot_q_triggers):
                next_trigger_q = self.snapshot_q_triggers[
                    self.next_snapshot_trigger_index
                ]
                if self.mean_front_position >= next_trigger_q:
                    # Take snapshot
                    boundary_q = next_trigger_q + self.snapshot_q_offset
                    title = (
                        f"Task {self.snapshot_dir.name[-12:]}\n"
                        f"Snapshot {self.snapshots_taken + 1}/{self.max_snapshots} (q ≈ {self.mean_front_position:.1f} approaching {boundary_q:.1f})"
                    )
                    self.plotter.plot_population(
                        self.population,
                        title=title,
                        q_to_patch_index=self.q_to_patch_index,
                    )
                    snapshot_path = (
                        self.snapshot_dir / f"snap_{self.snapshots_taken + 1:02d}.png"
                    )
                    self.plotter.save_figure(snapshot_path)

                    # Update state
                    self.snapshots_taken += 1
                    self.next_snapshot_trigger_index += 1
        # --- END REPLACED LOGIC ---

        boundary_hit = self.mean_front_position >= self.length - 2
        return True, boundary_hit

    def _axial_to_cube(self, q: int, r_offset: int) -> Hex:
        r = r_offset - (q + (q & 1)) // 2
        return Hex(q, r, -q - r)

    def _precompute_patch_indices(self):
        self.q_to_patch_index = np.zeros(self.length, dtype=int)
        current_q = 0
        if not self.patch_sequence or self.patch_sequence[0][1] == 0:
            return

        # This function now correctly iterates over the full, explicit patch sequence
        for patch_type, width in self.patch_sequence:
            start = current_q
            end = current_q + width
            if start < self.length:
                self.q_to_patch_index[start : min(end, self.length)] = patch_type
            current_q += width
            if current_q >= self.length:
                break

    def _find_initial_front(self):
        for h in list(self.population.keys()):
            self._update_events_for_cell(h)

    def _get_neighbors_periodic(self, h: Hex) -> list:
        unwrapped = h.neighbors()
        wrapped = []
        for n in unwrapped:
            offset_r = n.r + (n.q + (n.q & 1)) // 2
            offset_r %= self.width
            wrapped.append(self._axial_to_cube(n.q, offset_r))
        return wrapped

    def _get_params_for_q(self, q: int):
        """Gets environmental parameters for a given q-coordinate."""
        q_idx = np.clip(int(q), 0, self.length - 1)
        patch_idx = self.q_to_patch_index[q_idx]
        return self.patch_params[patch_idx]

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

    @staticmethod
    def _calculate_asymmetric_rates(k_total, phi):
        phi = np.clip(phi, -1, 1)
        return (k_total / 2.0) * (1.0 - phi), (k_total / 2.0) * (1.0 + phi)
