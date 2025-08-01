# FILE: src/fluctuating_model.py
# [DEFINITIVE, VECTORIZED & OPTIMIZED VERSION]
# This version achieves a massive speedup by replacing slow Python loops in the
# rate calculation with fast, vectorized NumPy operations. It preserves the
# exact physics of the independent switching model (Model 1) while being
# computationally efficient.

import numpy as np
import random
from typing import Dict, Set
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
        self.population: Dict[Hex, int] = self._initialize_population(initial_condition_type, initial_mutant_patch_size)
        self.wt_front_cells, self.m_front_cells, self._front_lookup = {}, {}, set()
        self.sum_q, self.sum_q_sq, self.front_cell_count = 0.0, 0.0, 0
        self.rates = {}
        self.total_rate = 0.0
        
        # --- NEW OPTIMIZATIONS ---
        # Pre-calculate patch indices for all possible q-values to avoid division in the loop.
        self.q_to_patch_index = np.floor(np.arange(length) / patch_width).astype(int) % self.num_patches
        
        # Pre-calculate parameters for each patch into a NumPy array for fast lookups.
        # Columns: [b_wt, b_m, k_wt_m, k_m_wt]
        k_wt_m_global, k_m_wt_global = self._calculate_asymmetric_rates(self.global_k_total, self.global_phi)
        self.patch_params = np.array([
            [
                environment_map.get(i, {}).get('b_wt', 1.0),
                environment_map.get(i, {}).get('b_m', self.global_b_m),
                k_wt_m_global,
                k_m_wt_global
            ] for i in range(self.num_patches)
        ])
        
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
            total_wt_rates = np.sum(wt_params_per_cell, axis=0)
            self.rates["WT_grows"] = total_wt_rates[0]
            self.rates["WT_switches_to_M"] = total_wt_rates[2]
        else:
            self.rates["WT_grows"] = 0.0
            self.rates["WT_switches_to_M"] = 0.0

        # --- Mutant Cells Rate Calculation ---
        if self.m_front_cells:
            m_q_coords = np.array([cell.q for cell in self.m_front_cells], dtype=int)
            m_q_coords = np.clip(m_q_coords, 0, self.length - 1)
            m_patch_indices = self.q_to_patch_index[m_q_coords]
            m_params_per_cell = self.patch_params[m_patch_indices]
            total_m_rates = np.sum(m_params_per_cell, axis=0)
            self.rates["M_grows"] = total_m_rates[1]
            self.rates["M_switches_to_WT"] = total_m_rates[3]
        else:
            self.rates["M_grows"] = 0.0
            self.rates["M_switches_to_WT"] = 0.0
            
        self.total_rate = sum(self.rates.values())

    def _execute_growth_event(self, parent_type: int):
        """Selects a random parent weighted by its local growth rate."""
        parent_pool = self.wt_front_cells if parent_type == Wildtype else self.m_front_cells
        if not parent_pool: return False, False
        
        parents = list(parent_pool.keys())
        q_coords = np.array([p.q for p in parents], dtype=int)
        q_coords = np.clip(q_coords, 0, self.length - 1)
        patch_indices = self.q_to_patch_index[q_coords]
        params_per_cell = self.patch_params[patch_indices]
        
        rate_idx = 0 if parent_type == Wildtype else 1
        weights = params_per_cell[:, rate_idx]
        
        if np.sum(weights) <= 1e-9: return True, False
        
        parent_cell = random.choices(parents, weights=weights, k=1)[0]
        
        empty_neighbors = [n for n in self._get_neighbors_periodic(parent_cell) if self._is_valid_growth_neighbor(n)]
        if not empty_neighbors: return True, False  