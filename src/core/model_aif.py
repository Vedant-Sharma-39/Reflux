# FILE: src/core/model_aif.py (Updated with Multi-Sector Initial Condition)

import numpy as np
from typing import Dict

from src.core.model import GillespieSimulation, Empty
from src.core.hex_utils import Hex

Susceptible = 1
Resistant = 2
Compensated = 3


class AifModelSimulation(GillespieSimulation):
    def __init__(self, **params):
        self.b_sus = params.get("b_sus", 1.0)
        self.b_res = params.get("b_res", 1.0 - 0.013)
        self.b_comp = params.get("b_comp", 1.0)
        self.k_res_comp = params.get("k_res_comp", 1e-4)
        self.initial_resistant_fraction = params.get("initial_resistant_fraction", 0.1)
        self.initial_droplet_radius = params.get("initial_droplet_radius", 15)
        self.sector_width_initial = params.get("sector_width_initial", 20)
        self.num_sectors = params.get("num_sectors", 1)

        self.susceptible_cell_count = 0
        self.resistant_cell_count = 0
        self.compensated_cell_count = 0

        params.setdefault("width", self.initial_droplet_radius * 4)
        length = params.setdefault("length", self.initial_droplet_radius * 10)

        max_front_cells = 6 * length
        events_per_cell = 7
        safety_factor = 2.0
        radial_capacity = max_front_cells * events_per_cell * safety_factor
        params["event_tree_capacity"] = max(20000, radial_capacity)

        super().__init__(**params)

        self.mutant_cell_count = self.resistant_cell_count + self.compensated_cell_count

        if self.plotter:
            self.plotter.colormap = {
                Susceptible: "#6c757d",
                Resistant: "#e63946",
                Compensated: "#457b9d",
            }

    def _initialize_population_pointytop(
        self, ic_type: str, patch_size: int
    ) -> Dict[Hex, int]:

        def _axial_to_cartesian_static(h_obj: Hex, size: float = 1.0):
            x = size * (3.0 / 2.0 * h_obj.q)
            y = size * (np.sqrt(3) / 2.0 * h_obj.q + np.sqrt(3) * h_obj.r)
            return x, y

        if ic_type == "aif_droplet":
            pop: Dict[Hex, int] = {}
            radius = self.initial_droplet_radius
            for q in range(-radius, radius + 1):
                r_min, r_max = max(-radius, -q - radius), min(radius, -q + radius)
                for r in range(r_min, r_max + 1):
                    h = Hex(q, r, -q - r)
                    cell_type = np.random.choice(
                        [Susceptible, Resistant],
                        p=[
                            1.0 - self.initial_resistant_fraction,
                            self.initial_resistant_fraction,
                        ],
                    )
                    pop[h] = cell_type
                    if cell_type == Susceptible:
                        self.susceptible_cell_count += 1
                    elif cell_type == Resistant:
                        self.resistant_cell_count += 1
            self.total_cell_count, self.initial_num_fragments = (
                len(pop),
                self.resistant_cell_count,
            )
            return pop

        elif ic_type == "sector" or ic_type == "multi_sector":
            pop: Dict[Hex, int] = {}
            radius = self.initial_droplet_radius
            for q in range(-radius, radius + 1):
                r_min, r_max = max(-radius, -q - radius), min(radius, -q + radius)
                for r in range(r_min, r_max + 1):
                    pop[Hex(q, r, -q - r)] = Susceptible

            sector_angle_width = self.sector_width_initial / radius
            sector_angles = [
                i * (2 * np.pi / self.num_sectors) for i in range(self.num_sectors)
            ]

            for h in list(pop.keys()):
                x, y = _axial_to_cartesian_static(h)
                angle = np.arctan2(y, x)
                for sector_angle in sector_angles:
                    # Check if cell angle is within the desired sector wedge
                    if (
                        abs((angle - sector_angle + np.pi) % (2 * np.pi) - np.pi)
                        < sector_angle_width / 2.0
                    ):
                        pop[h] = Resistant
                        break

            counts = {c: list(pop.values()).count(c) for c in [Susceptible, Resistant]}
            self.susceptible_cell_count = counts.get(Susceptible, 0)
            self.resistant_cell_count = counts.get(Resistant, 0)
            self.total_cell_count, self.initial_num_fragments = (
                len(pop),
                self.num_sectors,
            )
            return pop

        else:
            return super()._initialize_population_pointytop(ic_type, patch_size)

    def _update_single_cell_events(self, h: Hex):
        for event in list(self.cell_to_events.get(h, set())):
            self._remove_event(event)
        cell_type = self.population.get(h)
        if cell_type in (None, Empty):
            return

        empty_neighbors = [
            n
            for n in self._get_neighbors_periodic(h)
            if self._is_valid_growth_neighbor(n, h)
        ]
        if empty_neighbors:
            self._add_to_front(h)
            birth_rate = 0.0
            if cell_type == Susceptible:
                birth_rate = self.b_sus
            elif cell_type == Resistant:
                birth_rate = self.b_res
            elif cell_type == Compensated:
                birth_rate = self.b_comp
            if birth_rate > 0:
                for neighbor in empty_neighbors:
                    self._add_event(("grow", h, neighbor), birth_rate)
            if cell_type == Resistant and self.k_res_comp > 0:
                self._add_event(("switch", h, Compensated), self.k_res_comp)
        else:
            self._remove_from_front(h)

    def _execute_event(self, event_type: str, parent: Hex, target):
        if event_type == "grow":
            parent_type = self.population[parent]
            self.population[target] = parent_type
            if parent_type == Susceptible:
                self.susceptible_cell_count += 1
            elif parent_type == Resistant:
                self.resistant_cell_count += 1
            elif parent_type == Compensated:
                self.compensated_cell_count += 1
            self.total_cell_count += 1
            self._update_cell_and_neighbors(parent)
            self._update_cell_and_neighbors(target)
        elif event_type == "switch":
            self.population[parent] = Compensated
            self.resistant_cell_count -= 1
            self.compensated_cell_count += 1
            self._update_cell_and_neighbors(parent)
        self.mutant_cell_count = self.resistant_cell_count + self.compensated_cell_count

    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool:
        return neighbor not in self.population

    def _get_neighbors_periodic(self, h: Hex) -> list:
        if h in self.neighbor_cache:
            return self.neighbor_cache[h]
        neighbors = h.neighbors()
        self.neighbor_cache[h] = neighbors
        return neighbors

    @property
    def colony_radius(self) -> float:
        if not self._front_lookup:
            return 0.0
        distances = [(abs(h.q) + abs(h.r) + abs(h.s)) / 2.0 for h in self._front_lookup]
        return np.median(distances) if distances else 0.0
