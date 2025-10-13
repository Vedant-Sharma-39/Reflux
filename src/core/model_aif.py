# FILE: src/core/model_aif.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

from src.core.model import GillespieSimulation, Empty
from src.core.hex_utils import Hex

Susceptible = 1
Resistant   = 2
Compensated = 3


# ---------- lightweight cached geometry ----------
def axial_cartesian_fast(q: int, r: int) -> Tuple[float, float]:
    """Pointy-top axial -> 2D Cartesian (no scale)."""
    x = 1.5 * q
    y = (np.sqrt(3) / 2.0) * q + np.sqrt(3) * r
    return x, y


class AifModelSimulation(GillespieSimulation):
    """
    Radial colony with two types: Susceptible (grey) and Resistant (red).
    Switching/compensation is OFF by default (k_res_comp = 0).

    Logs sector widths on the *true* front with persistent lineage IDs (root_sid),
    supports radius-aligned logging (ΔR), and optional width-dependent selection.
    """

    # ------------------------------ init ------------------------------
    def __init__(self, **params):
        # Fitness & IC
        self.b_sus  = float(params.get("b_sus", 1.0))
        self.b_res  = float(params.get("b_res", 1.0 - 0.013))  # constant selection default
        self.b_comp = float(params.get("b_comp", 1.0))
        self.k_res_comp = float(params.get("k_res_comp", 0.0))  # OFF for now

        self.initial_droplet_radius     = int(params.get("initial_droplet_radius", 15))
        self.initial_resistant_fraction = float(params.get("initial_resistant_fraction", 0.10))
        self.num_initial_resistant_cells = int(params.get("num_initial_resistant_cells", 0))
        self.correlation_length = float(params.get("correlation_length", 3.0))
        self.band_width = int(params.get("band_width", 3))
        self.num_bands  = int(params.get("num_bands", 5))

        # Optional width-dependent selection (disabled by default)
        wd = params.get("width_selection", {}) or {}
        self.widthsel_mode     = (wd.get("mode") or "constant").lower()  # "constant" | "logistic"
        self.widthsel_s0       = float(wd.get("s0", 0.013))
        self.widthsel_wc_cells = float(wd.get("wc_cells", 56.0))
        self._s_eff_max_w      = int(params.get("widthsel_lookup_max", 1024))
        self._s_eff_table      = self._build_s_eff_table(self._s_eff_max_w, self.widthsel_s0, self.widthsel_wc_cells)

        # Caches for geometry and front radius
        self._xy_cache: Dict[Hex, Tuple[float, float]] = {}
        self._r2_cache: Dict[Hex, float] = {}
        self._front_r_sum: float = 0.0
        self._front_count: int = 0

        # Sector width cache (for width-dep selection)
        self._last_sector_width_by_hex: Dict[Hex, int] = {}

        # Radial event capacity (bigger than linear)
        params.setdefault("width",  self.initial_droplet_radius * 4)
        params.setdefault("length", self.initial_droplet_radius * 10)
        radial_capacity = int((6 * params["length"]) * 8 * 2.0)
        params["event_tree_capacity"] = max(params.get("event_tree_capacity", 20000), radial_capacity)

        # Parent init (builds initial population & front bookkeeping)
        super().__init__(**params)

        if self.plotter:
            self.plotter.colormap = {Susceptible: "#6c757d", Resistant: "#e63946", Compensated: "#457b9d"}

        # Counters
        counts = Counter(self.population.values())
        self.susceptible_cell_count = counts.get(Susceptible, 0)
        self.resistant_cell_count   = counts.get(Resistant, 0)
        self.compensated_cell_count = counts.get(Compensated, 0)
        self.total_cell_count       = len(self.population)
        self.mutant_cell_count      = self.resistant_cell_count

        # Sector logging & persistence
        self.sector_metrics_interval = int(params.get("sector_metrics_interval", 0)) or 0   # by step
        self.sector_metrics_dr       = float(params.get("sector_metrics_dr", 0.0))          # by ΔR
        self.radius_check_interval   = int(params.get("radius_check_interval", 64))         # throttle ΔR checks
        self.front_denoise_window    = int(params.get("front_denoise_window", 0))
        self.min_island_len          = int(params.get("min_island_len", 0))
        self.sid_iou_thresh          = float(params.get("sid_iou_thresh", 0.30))
        self.sid_center_delta        = float(params.get("sid_center_delta", 0.05))          # radians
        self.death_hysteresis        = int(params.get("death_hysteresis", 2))

        self.sector_traj_log: List[dict] = []
        self._sid_next = 1
        self._sid_root_map: Dict[int, int] = {}                             # ephemeral sid -> persistent root
        self._prev_sectors: List[Tuple[int, int, float, float, int]] = []   # [(sid,t,sθ,eθ,w)]
        self._death_streaks: Dict[int, int] = {}

        self._last_logged_step_for_sectors = -1
        self._last_radius_check_step = -1
        self._next_radius_log: Optional[float] = None

        # Initialize ΔR trigger after initial front exists
        if self.sector_metrics_dr and self.sector_metrics_dr > 0.0:
            r0 = self._mean_front_radius_fast()
            self._next_radius_log = r0 + self.sector_metrics_dr

        # Warm the sector-width cache once
        self._refresh_sector_width_cache()

    # --------------------- geometry & helpers ---------------------
    def _xy_r2(self, h: Hex) -> Tuple[float, float, float]:
        """Return (x, y, r^2) with caching."""
        if h in self._r2_cache:
            x, y = self._xy_cache[h]
            return x, y, self._r2_cache[h]
        x, y = axial_cartesian_fast(h.q, h.r)
        r2 = x * x + y * y
        self._xy_cache[h] = (x, y)
        self._r2_cache[h] = r2
        return x, y, r2

    def _radius(self, h: Hex) -> float:
        return float(np.sqrt(self._xy_r2(h)[2]))

    def _mean_front_radius_fast(self) -> float:
        return (self._front_r_sum / self._front_count) if self._front_count > 0 else 0.0

    # extend base front add/remove to maintain radius sum/count
    def _add_to_front(self, h: Hex):
        if h not in self._front_lookup:
            super()._add_to_front(h)
            self._front_count += 1
            self._front_r_sum += self._radius(h)

    def _remove_from_front(self, h: Hex):
        if h in self._front_lookup:
            super()._remove_from_front(h)
            self._front_count -= 1
            self._front_r_sum -= self._radius(h)

    # ---------------- population initialization ----------------
    def _initialize_population_pointytop(self, ic_type: str, patch_size: int) -> Dict[Hex, int]:
        """
        Build a true *circular* initial droplet in Euclidean metric, then seed
        resistant bands on the rim if requested.
        """
        pop: Dict[Hex, int] = {}
        R = int(self.initial_droplet_radius)
        droplet_cells: List[Hex] = []

        # Euclidean circle test in axial-cartesian coords
        cart_R2 = float(R) * float(R)
        for q in range(-R, R + 1):
            for r in range(-2 * R, 2 * R + 1):  # generous axial window
                h = Hex(q, r, -q - r)
                x, y = axial_cartesian_fast(h.q, h.r)
                if (x * x + y * y) <= cart_R2:
                    droplet_cells.append(h)

        for h in droplet_cells:
            pop[h] = Susceptible

        # rim (front) cells of the droplet
        droplet_set = set(droplet_cells)
        front_cells = []
        for cell in droplet_cells:
            for n in cell.neighbors():
                if n not in droplet_set:
                    front_cells.append(cell)
                    break

        # order rim by angle
        def ang(h: Hex) -> float:
            x, y, _ = self._xy_r2(h)
            return float(np.arctan2(y, x))

        sorted_front = sorted(front_cells, key=ang)
        n_front = len(sorted_front)

        if ic_type == "aif_front_bands":
            starts = np.random.choice(n_front, size=min(self.num_bands, n_front), replace=False)
            chosen = set()
            for s in starts:
                for i in range(self.band_width):
                    chosen.add((s + i) % n_front)
            for i in chosen:
                pop[sorted_front[i]] = Resistant
            print(f"Seeding {len(starts)} resistant bands; each {self.band_width} cells wide (total {len(chosen)}).")
        elif ic_type == "aif_front_seeded":
            k = max(1, int(n_front * self.initial_resistant_fraction))
            idxs = np.random.choice(n_front, size=min(k, n_front), replace=False)
            for i in idxs:
                pop[sorted_front[i]] = Resistant
            print(f"Seeded {len(idxs)} resistant front cells (random).")

        return pop

    # ---------------------------- dynamics ----------------------------
    def _build_s_eff_table(self, max_w: int, s0: float, wc_cells: float):
        """Lookup for logistic width-dependent selection (if enabled)."""
        if self.widthsel_mode != "logistic":
            return None
        ws = np.arange(1, max_w + 1, dtype=float)
        s_eff = (2.0 * s0) / (1.0 + np.exp(wc_cells / ws))
        return s_eff

    def _s_eff_fast(self, w: int) -> float:
        if self.widthsel_mode != "logistic":
            return 0.0
        if w <= 1:
            w = 1
        if w <= self._s_eff_max_w:
            return float(self._s_eff_table[w - 1])
        return float((2.0 * self.widthsel_s0) / (1.0 + np.exp(self.widthsel_wc_cells / float(w))))

    def _update_single_cell_events(self, h: Hex):
        # purge old events for this cell
        for event in list(self.cell_to_events.get(h, set())):
            self._remove_event(event)

        cell_type = self.population.get(h)
        if cell_type in (None, Empty):
            self._remove_from_front(h)
            return

        empties = [n for n in self._get_neighbors_periodic(h) if self._is_valid_growth_neighbor(n, h)]
        if empties:
            self._add_to_front(h)

            # birth rate (constant or width-dependent for red)
            if cell_type == Susceptible:
                base_birth_rate = self.b_sus
            elif cell_type == Resistant:
                if self.widthsel_mode == "logistic":
                    w = int(self._last_sector_width_by_hex.get(h, 1))
                    s_eff = self._s_eff_fast(w)
                    base_birth_rate = self.b_sus * (1.0 - s_eff)
                else:
                    base_birth_rate = self.b_res
            else:
                base_birth_rate = self.b_comp

            if base_birth_rate > 0.0:
                rate = base_birth_rate / len(empties)
                for nei in empties:
                    self._add_event(("grow", h, nei), rate)

            if cell_type == Resistant and self.k_res_comp > 0.0:
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

    # --------------------- front -> sectors ---------------------
    def _compute_front_chain(self) -> List[Tuple[Hex, int, float]]:
        """
        Angle-ordered *true* front. We rely on _front_lookup (cells with at least
        one empty neighbor; outwardness handled statistically for radial colonies).
        """
        if not self._front_lookup:
            return []
        items = []
        for h in self._front_lookup:
            t = self.population.get(h)
            if t is None:
                continue
            x, y, _ = self._xy_r2(h)
            theta = float(np.arctan2(y, x))
            items.append((h, t, theta))
        items.sort(key=lambda z: z[2])
        return items

    @staticmethod
    def _circular_majority(labels: List[int], w: int) -> List[int]:
        if not labels or w is None or w <= 1:
            return labels
        n = len(labels)
        out = labels[:]
        half = w // 2
        for i in range(n):
            idxs = [(i + k) % n for k in range(-half, half + 1)]
            votes = sum(1 for j in idxs if labels[j] == Resistant)
            out[i] = Resistant if votes > len(idxs) // 2 else Susceptible
        return out

    @staticmethod
    def _compress_runs(types: List[int], thetas: List[float]) -> List[Tuple[int, int, float, float]]:
        n = len(types)
        if n == 0:
            return []
        out: List[Tuple[int, int, float, float]] = []
        i = 0
        while i < n:
            tcur = types[i]
            j = i + 1
            while j < n and types[j] == tcur:
                j += 1
            out.append((tcur, j - i, thetas[i], thetas[j - 1]))
            i = j
        # wrap merge if first and last are same type
        if len(out) > 1 and out[0][0] == out[-1][0]:
            t0, w0, s0, e0 = out[0]
            tL, wL, sL, eL = out[-1]
            out = [(t0, w0 + wL, sL, eL)] + out[1:-1]
        return out

    @staticmethod
    def _merge_min_islands(sectors: List[Tuple[int, int, float, float]], min_len: int) -> List[Tuple[int, int, float, float]]:
        if min_len is None or min_len <= 0 or len(sectors) < 3:
            return sectors
        out = sectors[:]
        changed = True
        while changed:
            changed = False
            n = len(out)
            if n < 3:
                break
            i = 1
            while i < n - 1:
                tP, wP, sP, eP = out[i - 1]
                tC, wC, sC, eC = out[i]
                tN, wN, sN, eN = out[i + 1]
                if wC < min_len and tP == tN and tP != tC:
                    out[i - 1] = (tP, wP + wC + wN, sP, eN)
                    del out[i:i + 2]
                    n -= 2
                    changed = True
                else:
                    i += 1
        if len(out) > 1 and out[0][0] == out[-1][0] and out[0][1] < min_len:
            t0, w0, s0, e0 = out[0]
            tL, wL, sL, eL = out[-1]
            out = [(t0, w0 + wL, s0, eL)] + out[1:-1]
        return out

    def _sectorize_front(self) -> List[Tuple[int, int, float, float]]:
        chain = self._compute_front_chain()
        if not chain:
            return []
        types = [t for (_, t, _) in chain]
        thetas = [th for (_, _, th) in chain]
        if self.front_denoise_window and self.front_denoise_window > 1:
            types = self._circular_majority(types, self.front_denoise_window)
        sectors = self._compress_runs(types, thetas)
        if self.min_island_len and self.min_island_len > 0:
            sectors = self._merge_min_islands(sectors, self.min_island_len)
        return sectors

    @staticmethod
    def _circ_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        import math
        def unwrap(s, e): return (s, e + (2 * math.pi if e < s else 0.0))
        a0, a1 = unwrap(a[0], a[1]); b0, b1 = unwrap(b[0], b[1])
        L = max(a0, b0); R = min(a1, b1)
        ov = max(0.0, R - L)
        un = (a1 - a0) + (b1 - b0) - ov
        return (ov / un) if un > 0 else 0.0

    def _assign_persistent_ids(self, sectors_now: List[Tuple[int, int, float, float]]
                               ) -> List[Tuple[int, int, float, float, int, int]]:
        """
        Match sectors to previous snapshot and return
        [(type, width, startθ, endθ, sid, root_sid)].
        New sectors get a new sid and a new root_sid (itself).
        """
        if not sectors_now:
            self._prev_sectors = []
            return []

        assigned: List[Tuple[int, int, float, float, int, int]] = []
        used_prev: Set[int] = set()

        for (t_now, w_now, s_now, e_now) in sectors_now:
            best_sid, best_iou, best_gap = None, -1.0, 1e9
            c_now = 0.5 * (s_now + (e_now + (2 * np.pi if e_now < s_now else 0.0)))

            for (sid_prev, t_prev, s_prev, e_prev, w_prev) in self._prev_sectors:
                if t_prev != t_now:
                    continue
                iou = self._circ_iou((s_now, e_now), (s_prev, e_prev))
                c_prev = 0.5 * (s_prev + (e_prev + (2 * np.pi if e_prev < s_prev else 0.0)))
                gap = abs(c_now - c_prev)
                if gap > np.pi:
                    gap = 2 * np.pi - gap
                better = (iou > best_iou) or (abs(iou - best_iou) < 1e-9 and gap < best_gap)
                if (sid_prev not in used_prev) and better:
                    best_sid, best_iou, best_gap = sid_prev, iou, gap

            if best_sid is not None and (best_iou >= self.sid_iou_thresh or best_gap <= self.sid_center_delta):
                root = self._sid_root_map.get(best_sid, best_sid)
                sid = best_sid
                assigned.append((t_now, w_now, s_now, e_now, sid, root))
                used_prev.add(best_sid)
            else:
                sid = self._sid_next
                self._sid_next += 1
                root = sid
                assigned.append((t_now, w_now, s_now, e_now, sid, root))

        # update previous snapshot & root map
        self._prev_sectors = [(sid, t, s, e, w) for (t, w, s, e, sid, _) in assigned]
        for (_, _, _, _, sid, root) in assigned:
            self._sid_root_map[sid] = root

        return assigned

    def _refresh_sector_width_cache(self):
        """Map each front Hex to its sector width (used if width-dependent selection is on)."""
        self._last_sector_width_by_hex.clear()
        chain = self._compute_front_chain()
        if not chain:
            return
        types = [t for (_, t, _) in chain]
        if self.front_denoise_window and self.front_denoise_window > 1:
            types = self._circular_majority(types, self.front_denoise_window)
        n = len(types)
        i = 0
        while i < n:
            tcur = types[i]
            j = i + 1
            while j < n and types[j] == tcur:
                j += 1
            width = j - i
            if i == 0 and j == n and n > 1 and types[0] == types[-1]:
                width = n
            for k in range(i, j):
                h = chain[k][0]
                self._last_sector_width_by_hex[h] = width
            i = j

    def _log_sector_widths_snapshot(self):
        sectors = self._sectorize_front()
        sectors_sid = self._assign_persistent_ids(sectors)
        self._refresh_sector_width_cache()  # keep cache synced at logging moments

        r_mean = self._mean_front_radius_fast()
        step = self.step_count

        for (t, w, s, e, sid, root) in sectors_sid:
            self.sector_traj_log.append(
                dict(step=int(step), radius=float(r_mean),
                     sid=int(sid), root_sid=int(root), type=int(t),
                     width_cells=int(w), start_theta=float(s), end_theta=float(e))
            )

    # ------------------------------ stepping ------------------------------
    def step(self):
        progressed, boundary_hit = super().step()
        if not progressed:
            return False, boundary_hit

        do_log = False

        # (1) step-based cadence
        if self.sector_metrics_interval > 0:
            if (self.step_count - self._last_logged_step_for_sectors) >= self.sector_metrics_interval:
                do_log = True

        # (2) radius-based cadence (throttled)
        if self.sector_metrics_dr and self.sector_metrics_dr > 0.0:
            if (self.step_count - self._last_radius_check_step) >= self.radius_check_interval:
                self._last_radius_check_step = self.step_count
                if self._next_radius_log is None:
                    self._next_radius_log = self._mean_front_radius_fast() + self.sector_metrics_dr
                else:
                    current_r = self._mean_front_radius_fast()
                    if current_r >= self._next_radius_log:
                        do_log = True
                        jumps = max(1, int((current_r - self._next_radius_log) // self.sector_metrics_dr) + 1)
                        self._next_radius_log += jumps * self.sector_metrics_dr

        if do_log:
            self._log_sector_widths_snapshot()
            self._last_logged_step_for_sectors = self.step_count

        return True, boundary_hit

    # ------------------------ engine overrides ------------------------
    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool:
        # radial colony: any empty neighbor is valid
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
