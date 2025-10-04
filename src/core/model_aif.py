# FILE: src/core/model_aif.py (CORRECTED with Normalized Growth Rate)

import numpy as np
from typing import Dict, List
from collections import Counter
from scipy.ndimage import gaussian_filter

from src.core.model import GillespieSimulation, Empty
from src.core.hex_utils import Hex

Susceptible = 1
Resistant = 2
Compensated = 3

class AifModelSimulation(GillespieSimulation):
    def __init__(self, **params):
        # ... (all parameters in __init__ are correct and remain the same) ...
        self.b_sus = params.get("b_sus", 1.0)
        self.b_res = params.get("b_res", 1.0 - 0.013)
        self.b_comp = params.get("b_comp", 1.0)
        self.k_res_comp = params.get("k_res_comp", 1e-4)
        self.initial_resistant_fraction = params.get("initial_resistant_fraction", 0.1)
        self.initial_droplet_radius = params.get("initial_droplet_radius", 15)
        self.sector_width_initial = params.get("sector_width_initial", 20)
        self.num_sectors = params.get("num_sectors", 1)
        self.correlation_length = params.get("correlation_length", 3.0)
        self.susceptible_cell_count = 0
        self.resistant_cell_count = 0
        self.compensated_cell_count = 0
        params.setdefault("width", self.initial_droplet_radius * 4)
        length = params.setdefault("length", self.initial_droplet_radius * 10)
        radial_capacity = (6 * length) * 7 * 2.0
        params["event_tree_capacity"] = max(20000, radial_capacity)
        super().__init__(**params)
        self.mutant_cell_count = self.resistant_cell_count + self.compensated_cell_count
        if self.plotter:
            self.plotter.colormap = {
                Susceptible: "#6c757d", Resistant: "#e63946", Compensated: "#457b9d"
            }

    def _initialize_population_pointytop(
        self, ic_type: str, patch_size: int
    ) -> Dict[Hex, int]:
        # ... (The GRF initial condition method is correct and remains unchanged) ...
        def _axial_to_cartesian_static(h_obj: Hex, size: float = 1.0):
            x = size * (3.0 / 2.0 * h_obj.q); y = size * (np.sqrt(3) / 2.0 * h_obj.q + np.sqrt(3) * h_obj.r)
            return x, y
        pop: Dict[Hex, int] = {}; radius_in_hex_units = self.initial_droplet_radius
        cartesian_radius = float(radius_in_hex_units); droplet_cells: List[Hex] = []
        for q in range(-radius_in_hex_units, radius_in_hex_units + 1):
            r_min = max(-radius_in_hex_units, -q - radius_in_hex_units); r_max = min(radius_in_hex_units, -q + radius_in_hex_units)
            for r in range(r_min, r_max + 1):
                h = Hex(q, r, -q - r); x, y = _axial_to_cartesian_static(h)
                if np.sqrt(x**2 + y**2) <= cartesian_radius: droplet_cells.append(h)
        for h in droplet_cells: pop[h] = Susceptible
        if ic_type == "aif_droplet":
            num_to_make_resistant = int(len(droplet_cells) * self.initial_resistant_fraction)
            resistant_indices = np.random.choice(len(droplet_cells), size=num_to_make_resistant, replace=False)
            for i in resistant_indices: pop[droplet_cells[i]] = Resistant
        elif ic_type == "aif_droplet_grf":
            q_coords = [h.q for h in droplet_cells]; r_coords = [h.r for h in droplet_cells]
            q_min, q_max = min(q_coords), max(q_coords); r_min, r_max = min(r_coords), max(r_coords)
            grid_shape = (q_max - q_min + 1, r_max - r_min + 1)
            correlated_field = gaussian_filter(np.random.randn(*grid_shape), sigma=self.correlation_length)
            cell_grf_values = [correlated_field[h.q - q_min, h.r - r_min] for h in droplet_cells]
            num_to_make_resistant = int(len(droplet_cells) * self.initial_resistant_fraction)
            highest_value_indices = np.argsort(cell_grf_values)[-num_to_make_resistant:]
            for i in highest_value_indices: pop[droplet_cells[i]] = Resistant
        elif ic_type in ("sector", "multi_sector"):
            sector_angle_width = self.sector_width_initial / radius_in_hex_units
            sector_angles = [i * (2 * np.pi / self.num_sectors) for i in range(self.num_sectors)]
            for h in droplet_cells:
                x, y = _axial_to_cartesian_static(h); angle = np.arctan2(y, x)
                for sector_angle in sector_angles:
                    if abs((angle - sector_angle + np.pi) % (2 * np.pi) - np.pi) < sector_angle_width / 2.0: pop[h] = Resistant; break
        else: return super()._initialize_population_pointytop(ic_type, patch_size)
        counts = Counter(pop.values())
        self.susceptible_cell_count = counts.get(Susceptible, 0); self.resistant_cell_count = counts.get(Resistant, 0)
        self.compensated_cell_count = counts.get(Compensated, 0); self.total_cell_count = len(pop)
        if ic_type in ("sector", "multi_sector"): self.initial_num_fragments = self.num_sectors
        else: self.initial_num_fragments = self.resistant_cell_count
        return pop

    def _update_single_cell_events(self, h: Hex):
        """
        CORRECTED to remove the "land grab" bias by normalizing the growth rate.
        A cell's total growth potential is now divided among available empty sites.
        """
        for event in list(self.cell_to_events.get(h, set())):
            self._remove_event(event)
        cell_type = self.population.get(h)
        if cell_type in (None, Empty):
            return

        empty_neighbors = [n for n in self._get_neighbors_periodic(h) if self._is_valid_growth_neighbor(n, h)]
        
        if empty_neighbors:
            self._add_to_front(h)
            base_birth_rate = 0.0
            if cell_type == Susceptible: base_birth_rate = self.b_sus
            elif cell_type == Resistant: base_birth_rate = self.b_res
            elif cell_type == Compensated: base_birth_rate = self.b_comp
            
            # --- START OF CRITICAL FIX ---
            if base_birth_rate > 0:
                # The total growth potential of a cell is its birth_rate.
                # This potential is divided equally among all possible growth directions.
                num_empty = len(empty_neighbors)
                normalized_rate = base_birth_rate / num_empty
                
                for neighbor in empty_neighbors:
                    self._add_event(("grow", h, neighbor), normalized_rate)
            # --- END OF CRITICAL FIX ---
            
            if cell_type == Resistant and self.k_res_comp > 0:
                self._add_event(("switch", h, Compensated), self.k_res_comp)
        else:
            self._remove_from_front(h)

    # --- All other methods are unchanged ---
    def _execute_event(self, event_type: str, parent: Hex, target):
        if event_type == "grow":
            parent_type = self.population[parent]
            self.population[target] = parent_type
            if parent_type == Susceptible: self.susceptible_cell_count += 1
            elif parent_type == Resistant: self.resistant_cell_count += 1
            elif parent_type == Compensated: self.compensated_cell_count += 1
            self.total_cell_count += 1
            self._update_cell_and_neighbors(parent)
            self._update_cell_and_neighbors(target)
        elif event_type == "switch":
            self.population[parent] = Compensated
            self.resistant_cell_count -= 1
            self.compensated_cell_count += 1
            self._update_cell_and_neighbors(parent)
        self.mutant_cell_count = self.resistant_cell_count + self.compensated_cell_count
    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool: return neighbor not in self.population
    def _get_neighbors_periodic(self, h: Hex) -> list:
        if h in self.neighbor_cache: return self.neighbor_cache[h]
        neighbors = h.neighbors()
        self.neighbor_cache[h] = neighbors
        return neighbors
    @property
    def colony_radius(self) -> float:
        if not self._front_lookup: return 0.0
        distances = [(abs(h.q) + abs(h.r) + abs(h.s)) / 2.0 for h in self._front_lookup]
        return np.median(distances) if distances else 0.0