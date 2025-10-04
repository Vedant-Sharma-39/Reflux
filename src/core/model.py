# FILE: src/core/model.py (CORRECTED with Normalized Growth Rate)

import numpy as np
import random
from typing import Dict, Set, Tuple, List, Optional
from pathlib import Path
from collections import Counter, defaultdict
from itertools import groupby

from numba import jit

from src.core.hex_utils import Hex, HexPlotter
from src.core.metrics import MetricsManager
from src.config import PARAM_GRID

Empty, Wildtype, Mutant = 0, 1, 2


@jit(nopython=True)
def _update_tree_path_numba(tree, index, delta):
    index //= 2
    while index > 0:
        tree[index] += delta
        index //= 2


@jit(nopython=True)
def _find_event_numba(tree, capacity, value):
    index = 1
    while index < capacity:
        left_child = 2 * index
        if value < tree[left_child]:
            index = left_child
        else:
            value -= tree[left_child]
            index = left_child + 1
    return index - capacity


def _generate_grf_initial_condition(
    width: int, num_mutants: int, correlation_length: float
) -> np.ndarray:
    if correlation_length <= 0:
        pattern = np.zeros(width, dtype=int)
        indices = np.random.choice(np.arange(width), size=num_mutants, replace=False)
        pattern[indices] = 1
        return pattern
    freqs = np.fft.fftfreq(width)
    power_spectrum = np.exp(-0.5 * (freqs * correlation_length) ** 2)
    noise_freq = np.random.randn(width) + 1j * np.random.randn(width)
    if width > 0:
        noise_freq[0] = 0
        for i in range(1, width // 2 + 1):
            if i < width - i:
                noise_freq[width - i] = np.conj(noise_freq[i])
    field_freq = noise_freq * np.sqrt(power_spectrum)
    grf = np.real(np.fft.ifft(field_freq))
    top_indices = np.argsort(grf)[-num_mutants:]
    pattern = np.zeros(width, dtype=int)
    pattern[top_indices] = 1
    return pattern


class SummedRateTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
    def update(self, index: int, rate: float):
        leaf_index = index + self.capacity
        delta = rate - self.tree[leaf_index]
        if abs(delta) < 1e-12: return
        self.tree[leaf_index] = rate
        _update_tree_path_numba(self.tree, leaf_index, delta)
    def find_event(self, value: float) -> int:
        return _find_event_numba(self.tree, self.capacity, value)
    def get_total_rate(self) -> float:
        return self.tree[1]


class GillespieSimulation:
    def __init__(self, **params):
        # ... (constructor is unchanged) ...
        self.all_params = params
        self.metrics_manager: Optional[MetricsManager] = None
        width = params.get("width", 256)
        length = params.get("length", 4096)
        b_m = params.get("b_m", 1.0)
        k_total = params.get("k_total", 0.0)
        phi = params.get("phi", 0.0)
        self.width, self.length = width, length
        self.global_b_m, self.global_k_total, self.global_phi = b_m, k_total, phi
        self.time, self.step_count = 0.0, 0
        capacity = params.get("event_tree_capacity", width * 100)
        self.MAX_EVENTS = int(capacity)
        self.tree = SummedRateTree(self.MAX_EVENTS)
        self.event_to_idx: Dict[Tuple, int] = {}
        self.idx_to_event: Dict[int, Tuple] = {}
        self.cell_to_events: Dict[Hex, Set[Tuple]] = defaultdict(set)
        self.free_indices = []
        self.next_event_idx = 0
        self.mutant_cell_count, self.total_cell_count = 0, 0
        ic_type = params.get("initial_condition_type", "mixed")
        ic_patch_size = params.get("initial_mutant_patch_size", 0)
        self._precompute_env_params(**params)
        self.wt_front_cells: Dict[Hex, List[Hex]] = {}
        self.m_front_cells: Dict[Hex, List[Hex]] = {}
        self.mutant_r_counts: Counter = Counter()
        self.population: Dict[Hex, int] = self._initialize_population_pointytop(
            ic_type, ic_patch_size
        )
        self._front_lookup: Set[Hex] = set()
        self._front_q_sum = 0.0
        self._front_q_sum_sq = 0.0
        self.neighbor_cache: Dict[Hex, List[Hex]] = {}
        self.cache_prune_step_interval = params.get("cache_prune_step_interval", 5000)
        self.cache_prune_q_distance = params.get("cache_prune_q_distance", 200)
        self._setup_visualization(**params)
        self._find_initial_front()

    def _update_single_cell_events(self, h: Hex):
        """
        CORRECTED to remove the "land grab" bias by normalizing the growth rate.
        This fix applies to both the linear and radial models.
        """
        for event in list(self.cell_to_events.get(h, set())):
            self._remove_event(event)
        if h in self._front_lookup:
            self._remove_from_front(h)
            self.wt_front_cells.pop(h, None)
            self.m_front_cells.pop(h, None)
        
        cell_type = self.population.get(h)
        if cell_type is None or cell_type == Empty:
            return

        empty_neighbors = [n for n in self._get_neighbors_periodic(h) if self._is_valid_growth_neighbor(n, h)]
        
        if empty_neighbors:
            self._add_to_front(h)
            if cell_type == Wildtype:
                self.wt_front_cells[h] = empty_neighbors
            elif cell_type == Mutant:
                self.m_front_cells[h] = empty_neighbors

            # Handle switching events (rate is independent of geometry)
            if cell_type == Wildtype and self.k_wt_m > 0:
                self._add_event(("switch", h, Mutant), self.k_wt_m)
            elif cell_type == Mutant and self.k_m_wt > 0:
                self._add_event(("switch", h, Wildtype), self.k_m_wt)

            # --- START OF CRITICAL FIX ---
            # Handle growth events with normalized rate
            b_wt, b_m = self._get_params_for_q(h.q) # Get fitness for the parent cell's location
            growth_rate = b_wt if cell_type == Wildtype else b_m
            
            if growth_rate > 0:
                # The total growth potential of a cell is its birth_rate.
                # This potential is divided equally among all possible growth directions.
                num_empty = len(empty_neighbors)
                normalized_rate = growth_rate / num_empty
                
                for neighbor in empty_neighbors:
                    self._add_event(("grow", h, neighbor), normalized_rate)
            # --- END OF CRITICAL FIX ---

    # ... (The rest of the file is unchanged) ...
    def _precompute_env_params(self, **kwargs):
        self.k_wt_m, self.k_m_wt = self._calculate_asymmetric_rates(self.global_k_total, self.global_phi)
        self.is_radial_growth = kwargs.get("initial_condition_type") == "single_cell"
        env_def = kwargs.get("env_definition");
        if isinstance(env_def, str): env_def = PARAM_GRID.get("env_definitions", {}).get(env_def)
        self.patch_sequence = []
        if env_def and isinstance(env_def, dict):
            if env_def.get("scrambled"):
                patches = env_def.get("patches", [{"id": 0, "params": {}}]); patch_types = [p["id"] for p in sorted(patches, key=lambda p: p["id"])]
                proportions = [p["proportion"] for p in sorted(patches, key=lambda p: p["id"])]; dist = env_def.get("patch_width_distribution", "gamma")
                mean_width = env_def.get("mean_patch_width", 60); fano = env_def.get("fano_factor", 1); scale = fano; shape = mean_width / fano if fano > 0 else 0
                target_len = self.length + 500; total_len = 0
                while total_len < target_len:
                    width = int(np.random.gamma(shape, scale)) if dist == "gamma" else int(mean_width)
                    if width >= 1: self.patch_sequence.append((np.random.choice(patch_types, p=proportions), width)); total_len += width
            else:
                base_pattern = [(p["id"], p["width"]) for p in env_def.get("patches", [])]
                if base_pattern:
                    cycle_q = sum(w for _, w in base_pattern)
                    if cycle_q > 0:
                        total_len = 0
                        while total_len < self.length: self.patch_sequence.extend(base_pattern); total_len += cycle_q
        self.q_to_patch_index = np.zeros(self.length, dtype=int); current_q = 0
        if self.patch_sequence:
            for patch_type, width in self.patch_sequence:
                start, end = current_q, current_q + width
                if start < self.length: self.q_to_patch_index[start : min(end, self.length)] = patch_type
                current_q = end
                if current_q >= self.length: break
        base_patches = (env_def or {}).get("patches", [{"id": 0, "params": {}}]); param_map = {p["id"]: p.get("params", {}) for p in base_patches}
        patch_ids = sorted(list(set(self.q_to_patch_index))); self.patch_params = []
        for i in range(max(patch_ids) + 1 if patch_ids else 1):
            params = param_map.get(i, {}); self.patch_params.append((params.get("b_wt", 1.0), params.get("b_m", self.global_b_m)))
    def _initialize_population_pointytop(self, ic_type: str, patch_size: int) -> Dict[Hex, int]:
        pop: Dict[Hex, int] = {};
        if ic_type == "single_cell":
            pop[Hex(0, 0, 0)] = Wildtype; self.total_cell_count = 1; self.initial_num_fragments = 1; return pop
        initial_pattern = np.full(self.width, Wildtype, dtype=int)
        if ic_type == "mixed": initial_pattern = np.random.choice([Wildtype, Mutant], size=self.width)
        elif ic_type == "patch":
            start_idx = (self.width - patch_size) // 2; end_idx = start_idx + patch_size
            initial_pattern[start_idx:end_idx] = Mutant
        elif ic_type == "grf_threshold":
            num_mutants = patch_size; correlation_length = self.all_params.get("correlation_length", 1.0)
            mutant_locations = _generate_grf_initial_condition(self.width, num_mutants, correlation_length)
            initial_pattern[mutant_locations == 1] = Mutant
        self.initial_num_fragments = int(1 + np.sum(initial_pattern[1:] != initial_pattern[:-1]))
        for i, cell_type in enumerate(initial_pattern):
            h = self._axial_to_cube(0, i); pop[h] = cell_type
            if cell_type == Mutant: self.mutant_cell_count += 1; self.mutant_r_counts[h.r] += 1
            self.total_cell_count += 1
        return pop
    def _setup_visualization(self, **params):
        self.plotter: Optional[HexPlotter] = None
        if params.get("run_mode") == "visualization":
            self.plotter = HexPlotter(hex_size=1.0, labels={}, colormap={1: "#003f5c", 2: "#ff7c43"})
            self.snapshot_dir = Path("figures/debug_runs") / params.get("campaign_id", "viz"); self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    @property
    def total_rate(self) -> float: return self.tree.get_total_rate()
    @property
    def mean_front_position(self) -> float: count = len(self._front_lookup); return self._front_q_sum / count if count > 0 else 0.0
    @property
    def mutant_fraction(self) -> float: return self.mutant_cell_count / self.total_cell_count if self.total_cell_count > 0 else 0.0
    @property
    def front_roughness_sq(self) -> float:
        count = len(self._front_lookup)
        if count < 2: return 0.0
        mean_q = self._front_q_sum / count; mean_q_sq = self._front_q_sum_sq / count
        return max(0.0, mean_q_sq - mean_q**2)
    @property
    def expanding_front_length(self) -> float: return float(len(self._front_lookup))
    @property
    def mutant_sector_width(self) -> int: return len([r for r, count in self.mutant_r_counts.items() if count > 0])
    def _add_to_front(self, h: Hex): self._front_lookup.add(h); self._front_q_sum += h.q; self._front_q_sum_sq += h.q**2
    def _remove_from_front(self, h: Hex): self._front_lookup.discard(h); self._front_q_sum -= h.q; self._front_q_sum_sq -= h.q**2
    def _add_event(self, event: Tuple, rate: float):
        parent_cell = event[1]
        if event in self.event_to_idx: idx = self.event_to_idx[event]
        else:
            if self.free_indices: idx = self.free_indices.pop()
            else:
                idx = self.next_event_idx
                if idx >= self.MAX_EVENTS: raise MemoryError(f"SummedRateTree capacity ({self.MAX_EVENTS}) exceeded.")
                self.next_event_idx += 1
            self.event_to_idx[event] = idx; self.idx_to_event[idx] = event
        self.tree.update(idx, rate); self.cell_to_events[parent_cell].add(event)
    def _remove_event(self, event: Tuple):
        if event not in self.event_to_idx: return
        parent_cell = event[1]; idx = self.event_to_idx.pop(event); self.free_indices.append(idx)
        del self.idx_to_event[idx]; self.tree.update(idx, 0.0)
        self.cell_to_events[parent_cell].discard(event)
        if not self.cell_to_events[parent_cell]: del self.cell_to_events[parent_cell]
    def _execute_event(self, event_type: str, parent: Hex, target: Optional[Hex]):
        if event_type == "grow":
            parent_type = self.population[parent]; self.population[target] = parent_type
            if parent_type == Mutant: self.mutant_cell_count += 1; self.mutant_r_counts[target.r] += 1
            self.total_cell_count += 1; self._update_cell_and_neighbors(parent); self._update_cell_and_neighbors(target)
        elif event_type == "switch":
            old_type = self.population[parent]; new_type = target
            self.population[parent] = new_type
            if new_type == Mutant and old_type == Wildtype: self.mutant_cell_count += 1; self.mutant_r_counts[parent.r] += 1
            elif new_type == Wildtype and old_type == Mutant: self.mutant_cell_count -= 1; self.mutant_r_counts[parent.r] -= 1
            self._update_cell_and_neighbors(parent)
    def _update_cell_and_neighbors(self, h: Hex):
        neighbors = self._get_neighbors_periodic(h); self._update_single_cell_events(h)
        for neighbor in neighbors:
            if neighbor in self.population: self._update_single_cell_events(neighbor)
    def _find_initial_front(self):
        self.mutant_r_counts = Counter(h.r for h, type in self.population.items() if type == Mutant)
        for h in list(self.population.keys()): self._update_single_cell_events(h)
    def step(self):
        if self.step_count > 0 and self.step_count % self.cache_prune_step_interval == 0: self._prune_neighbor_cache()
        self.step_count += 1; current_total_rate = self.total_rate
        if current_total_rate <= 1e-9: return False, False
        dt = -np.log(random.random()) / current_total_rate; self.time += dt
        rand_val = random.random() * current_total_rate; event_idx = self.tree.find_event(rand_val)
        if event_idx not in self.idx_to_event: return False, False
        event_type, parent, target = self.idx_to_event[event_idx]; self._execute_event(event_type, parent, target)
        boundary_hit = self.mean_front_position >= self.length - 2
        return True, boundary_hit
    def _axial_to_cube(self, q: int, r_offset: int) -> Hex: return Hex(q, r_offset - (q + (q & 1)) // 2, -q - (r_offset - (q + (q & 1)) // 2))
    def _get_neighbors_periodic(self, h: Hex) -> list:
        if h in self.neighbor_cache: return self.neighbor_cache[h]
        neighbors = [self._axial_to_cube(n.q, (n.r + (n.q + (n.q & 1)) // 2) % self.width) for n in h.neighbors()]
        self.neighbor_cache[h] = neighbors; return neighbors
    def _prune_neighbor_cache(self):
        if not self.neighbor_cache: return
        prune_q = self.mean_front_position - self.cache_prune_q_distance
        for h in [h for h in self.neighbor_cache if h.q < prune_q]: del self.neighbor_cache[h]
    def _get_params_for_q(self, q: int) -> Tuple[float, float]:
        q_idx = max(0, min(q, self.length - 1)); return self.patch_params[self.q_to_patch_index[q_idx]]
    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool:
        if neighbor in self.population: return False
        if self.is_radial_growth: return True
        if not (0 <= neighbor.q < self.length): return False
        return neighbor.q >= parent.q
    @staticmethod
    def _calculate_asymmetric_rates(k_total, phi):
        phi = np.clip(phi, -1, 1); return (k_total / 2.0) * (1.0 - phi), (k_total / 2.0) * (1.0 + phi)