# FILE: src/core/model_aif.py (MODIFIED with Band-Based Seeding)

import numpy as np
from typing import Dict, List
from collections import Counter
from scipy.ndimage import gaussian_filter1d

from src.core.model import GillespieSimulation, Empty
from src.core.hex_utils import Hex

Susceptible = 1
Resistant = 2
Compensated = 3

class AifModelSimulation(GillespieSimulation):
    def __init__(self, **params):
        # All existing parameters remain
        self.b_sus = params.get("b_sus", 1.0); self.b_res = params.get("b_res", 1.0 - 0.013)
        self.b_comp = params.get("b_comp", 1.0); self.k_res_comp = params.get("k_res_comp", 1e-4)
        self.initial_resistant_fraction = params.get("initial_resistant_fraction", 0.1)
        self.initial_droplet_radius = params.get("initial_droplet_radius", 15)
        self.correlation_length = params.get("correlation_length", 3.0)
        self.num_initial_resistant_cells = params.get("num_initial_resistant_cells", 0)

        # --- NEW PARAMETERS FOR BAND-BASED SEEDING ---
        self.band_width = params.get("band_width", 3)
        self.num_bands = params.get("num_bands", 5)

        # ... (rest of __init__ is unchanged) ...
        self.susceptible_cell_count = 0; self.resistant_cell_count = 0; self.compensated_cell_count = 0
        params.setdefault("width", self.initial_droplet_radius * 4)
        length = params.setdefault("length", self.initial_droplet_radius * 10)
        radial_capacity = (6 * length) * 7 * 2.0
        params["event_tree_capacity"] = max(20000, radial_capacity)
        super().__init__(**params)
        self.mutant_cell_count = self.resistant_cell_count + self.compensated_cell_count
        if self.plotter:
            self.plotter.colormap = {Susceptible: "#6c757d", Resistant: "#e63946", Compensated: "#457b9d"}

    def _initialize_population_pointytop(
        self, ic_type: str, patch_size: int
    ) -> Dict[Hex, int]:
        """
        Includes 'aif_front_bands' IC to seed bands of a fixed width.
        """
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

        droplet_set = set(droplet_cells); front_cells = []
        for cell in droplet_cells:
            for neighbor in cell.neighbors():
                if neighbor not in droplet_set: front_cells.append(cell); break
        
        # Sort front cells by angle to create a continuous 1D front
        front_angles = [np.arctan2(_axial_to_cartesian_static(h)[1], _axial_to_cartesian_static(h)[0]) for h in front_cells]
        sorted_indices = np.argsort(front_angles)
        sorted_front_cells = [front_cells[i] for i in sorted_indices]
        n_front = len(sorted_front_cells)

        if ic_type == "aif_front_seeded":
            num_to_make_resistant = int(n_front * self.initial_resistant_fraction)
            resistant_indices = np.random.choice(n_front, size=num_to_make_resistant, replace=False)
            for i in resistant_indices: pop[sorted_front_cells[i]] = Resistant
            print(f"Seeded {num_to_make_resistant} resistant cells randomly onto the front.")

        elif ic_type == "aif_front_bands":
            # --- NEW, SIMPLER BAND-BASED LOGIC ---
            num_to_make_resistant = self.band_width * self.num_bands
            if num_to_make_resistant > n_front:
                print(f"Warning: Requested {self.num_bands} bands of width {self.band_width} ({num_to_make_resistant} cells), but only {n_front} front cells exist. Aborting seeding.")
            else:
                print(f"Seeding {self.num_bands} resistant bands of width {self.band_width} randomly on the front.")
                start_indices = np.random.choice(n_front, size=self.num_bands, replace=False)
                resistant_indices = set()
                for start in start_indices:
                    for i in range(self.band_width):
                        index_to_mark = (start + i) % n_front # Use modulo to wrap around
                        resistant_indices.add(index_to_mark)
                
                for i in resistant_indices:
                    pop[sorted_front_cells[i]] = Resistant
                print(f"Seeded a total of {len(resistant_indices)} resistant cells.")
            # --- END OF NEW LOGIC ---

        elif ic_type == "aif_front_grf":
            num_to_make_resistant = self.num_initial_resistant_cells
            if num_to_make_resistant == 0:
                print("Warning: num_initial_resistant_cells is 0. No resistant cells will be seeded.")
            elif num_to_make_resistant > n_front:
                print(f"Warning: Requested {num_to_make_resistant} resistant cells, but only {n_front} front cells exist. Capping at {n_front}.")
                num_to_make_resistant = n_front
            
            print(f"Seeding a fixed number of {num_to_make_resistant} resistant cells in correlated clusters on the front.")
            random_line = np.random.randn(n_front)
            correlated_line = gaussian_filter1d(random_line, sigma=self.correlation_length, mode='wrap')
            highest_value_indices = np.argsort(correlated_line)[-num_to_make_resistant:]
            for i in highest_value_indices: pop[sorted_front_cells[i]] = Resistant

        counts = Counter(pop.values())
        self.susceptible_cell_count=counts.get(Susceptible,0); self.resistant_cell_count=counts.get(Resistant,0)
        self.compensated_cell_count=counts.get(Compensated,0); self.total_cell_count=len(pop)
        self.initial_num_fragments=self.resistant_cell_count
        return pop

    # ... (the rest of the file is unchanged) ...
    def _update_single_cell_events(self, h: Hex):
        for event in list(self.cell_to_events.get(h, set())): self._remove_event(event)
        cell_type = self.population.get(h)
        if cell_type in (None, Empty): return
        empty_neighbors = [n for n in self._get_neighbors_periodic(h) if self._is_valid_growth_neighbor(n, h)]
        if empty_neighbors:
            self._add_to_front(h)
            base_birth_rate = 0.0
            if cell_type == Susceptible: base_birth_rate = self.b_sus
            elif cell_type == Resistant: base_birth_rate = self.b_res
            elif cell_type == Compensated: base_birth_rate = self.b_comp
            if base_birth_rate > 0:
                num_empty = len(empty_neighbors)
                normalized_rate = base_birth_rate / num_empty
                for neighbor in empty_neighbors: self._add_event(("grow", h, neighbor), normalized_rate)
            if cell_type == Resistant and self.k_res_comp > 0: self._add_event(("switch", h, Compensated), self.k_res_comp)
        else: self._remove_from_front(h)
        
    def _execute_event(self, event_type: str, parent: Hex, target):
        if event_type == "grow":
            parent_type = self.population[parent]; self.population[target] = parent_type
            if parent_type == Susceptible: self.susceptible_cell_count += 1
            elif parent_type == Resistant: self.resistant_cell_count += 1
            elif parent_type == Compensated: self.compensated_cell_count += 1
            self.total_cell_count += 1; self._update_cell_and_neighbors(parent); self._update_cell_and_neighbors(target)
        elif event_type == "switch":
            self.population[parent] = Compensated; self.resistant_cell_count -= 1
            self.compensated_cell_count += 1; self._update_cell_and_neighbors(parent)
        self.mutant_cell_count = self.resistant_cell_count + self.compensated_cell_count
    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool: return neighbor not in self.population
    def _get_neighbors_periodic(self, h: Hex) -> list:
        if h in self.neighbor_cache: return self.neighbor_cache[h]
        neighbors = h.neighbors(); self.neighbor_cache[h] = neighbors; return neighbors
    @property
    def colony_radius(self) -> float:
        if not self._front_lookup: return 0.0
        distances = [(abs(h.q) + abs(h.r) + abs(h.s)) / 2.0 for h in self._front_lookup]
        return np.median(distances) if distances else 0.0