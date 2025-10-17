# FILE: src/core/model_aif.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

from src.core.model import GillespieSimulation, Empty
from src.core.hex_utils import Hex

Susceptible = 1
Resistant   = 2
Compensated = 3


# -------- axial -> Cartesian (pointy-top), cached in class ----------
def axial_cartesian_fast(q: int, r: int) -> Tuple[float, float]:
    x = 1.5 * q
    y = (np.sqrt(3) / 2.0) * q + np.sqrt(3) * r
    return x, y


class AifModelSimulation(GillespieSimulation):
    """
    Minimal AIF radial colony model with two main types (S=grey, R=red).
    Optional R->Compensated switching (off by default).

    Provides:
      - Circular droplet IC and resistant rim bands ("aif_front_bands")
      - True-front sectorization with circular denoise + min-island merge
      - Persistent sector IDs and root IDs
      - Sector width logging by ΔR and/or step interval

    Kept small to support scripts/debug_aif_streaks.py.
    """

    # ------------------------------ init ------------------------------
    def __init__(self, **params):
        # birth/switch
        self.b_sus  = float(params.get("b_sus", 1.0))
        self.b_res  = float(params.get("b_res", 1.0))
        self.b_comp = float(params.get("b_comp", 1.0))
        self.k_res_comp = float(params.get("k_res_comp", 0.0))  # usually 0 in debug script

        # initial condition controls
        self.initial_droplet_radius     = int(params.get("initial_droplet_radius", 15))
        self.initial_resistant_fraction = float(params.get("initial_resistant_fraction", 0.10))
        self.num_initial_resistant_cells = int(params.get("num_initial_resistant_cells", 0))  # unused here but harmless
        self.band_width = int(params.get("band_width", 3))
        self.num_bands  = int(params.get("num_bands", 5))

        # sector logging & persistence
        self.sector_metrics_interval = int(params.get("sector_metrics_interval", 0)) or 0  # by step
        self.sector_metrics_dr       = float(params.get("sector_metrics_dr", 0.0))          # by ΔR
        self.radius_check_interval   = int(params.get("radius_check_interval", 64))         # throttle ΔR checks
        self.front_denoise_window    = int(params.get("front_denoise_window", 0))           # circular majority window
        self.min_island_len          = int(params.get("min_island_len", 0))                 # merge tiny sectors
        self.sid_iou_thresh          = float(params.get("sid_iou_thresh", 0.30))            # match persistence
        self.sid_center_delta        = float(params.get("sid_center_delta", 0.05))          # radians
        self.death_hysteresis        = int(params.get("death_hysteresis", 2))               # (kept for compatibility)
        self.sector_width_initial = int(params.get("sector_width_initial", 24))
        self.sector_theta_center  = float(params.get("sector_theta_center", 0.0))

        # --- NEW: extinction-aware controls / status (safe-by-default) ---
        self.stop_on_front_extinction: bool = bool(params.get("stop_on_front_extinction", False))
        self.tracked_root_sid: Optional[int] = params.get("tracked_root_sid", None)
        # For backward-compatibility, we do NOT auto-select a lineage unless you opt in.
        self.infer_tracked_root: bool = bool(params.get("infer_tracked_root", False))
        self.termination_reason: Optional[str] = None
        self.extinction_scope: Optional[str] = None  # "tracked_lineage" | "single_mutant" | None

        # Track how many distinct mutant roots ever appeared on the front
        self._mutant_roots_ever_seen: Set[int] = set()

        # lightweight caches
        self._xy_cache: Dict[Hex, Tuple[float, float]] = {}
        self._r2_cache: Dict[Hex, float] = {}
        self._front_r_sum: float = 0.0
        self._front_count: int = 0

        # persistent sector bookkeeping
        self.sector_traj_log: List[dict] = []
        self._sid_next = 1
        self._sid_root_map: Dict[int, int] = {}                             # ephemeral sid -> persistent root
        self._prev_sectors: List[Tuple[int, int, float, float, int]] = []   # [(sid,t,sθ,eθ,w)]

        self._last_logged_step_for_sectors = -1
        self._last_radius_check_step = -1
        self._next_radius_log: Optional[float] = None

        # make the event tree big enough for radial growth
        params.setdefault("width",  self.initial_droplet_radius * 4)
        params.setdefault("length", self.initial_droplet_radius * 10)
        radial_capacity = int((6 * params["length"]) * 8 * 2.0)
        params["event_tree_capacity"] = max(params.get("event_tree_capacity", 20000), radial_capacity)

        super().__init__(**params)

        if self.plotter:
            self.plotter.colormap = {
                Susceptible: "#6c757d",
                Resistant:   "#e63946",
                Compensated: "#457b9d",
            }

        # counts (handy but not strictly needed by debug script)
        counts = Counter(self.population.values())
        self.susceptible_cell_count = counts.get(Susceptible, 0)
        self.resistant_cell_count   = counts.get(Resistant, 0)
        self.compensated_cell_count = counts.get(Compensated, 0)
        self.total_cell_count       = len(self.population)
        self.mutant_cell_count      = self.resistant_cell_count + self.compensated_cell_count

        # initialize ΔR trigger after first front exists
        if self.sector_metrics_dr and self.sector_metrics_dr > 0.0:
            r0 = self._mean_front_radius_fast()
            self._next_radius_log = r0 + self.sector_metrics_dr

    # --------------------- geometry & helpers ---------------------
    def _xy_r2(self, h: Hex) -> Tuple[float, float, float]:
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

    # --- replace/patch _initialize_population_pointytop with this version ---
    def _initialize_population_pointytop(self, ic_type: str, patch_size: int) -> Dict[Hex, int]:
        """
        Build a circular initial droplet and seed resistant cells on the rim
        according to `ic_type`:
        - "sector": one contiguous resistant block of given width (in rim cells),
                    centered at sector_theta_center (radians).
        - "aif_front_bands": multiple bands of width `band_width` repeated `num_bands`.
        - "aif_front_seeded": random single-cell seeds along the rim.
        """
        pop: Dict[Hex, int] = {}
        R = int(self.initial_droplet_radius)

        # 1) circular droplet (Euclidean in axial-cartesian coords)
        droplet_cells: List[Hex] = []
        cart_R2 = float(R) * float(R)
        for q in range(-R, R + 1):
            for r in range(-2 * R, 2 * R + 1):
                h = Hex(q, r, -q - r)
                x, y = axial_cartesian_fast(h.q, h.r)
                if (x * x + y * y) <= cart_R2:
                    droplet_cells.append(h)
        for h in droplet_cells:
            pop[h] = Susceptible

        # 2) rim cells
        droplet_set = set(droplet_cells)
        front_cells = [cell for cell in droplet_cells if any(n not in droplet_set for n in cell.neighbors())]
        if not front_cells:
            return pop

        # 3) angle-sorted rim
        def ang(h: Hex) -> float:
            x, y, _ = self._xy_r2(h)
            return float(np.arctan2(y, x))
        sorted_front = sorted(front_cells, key=ang)
        thetas = [ang(h) for h in sorted_front]
        n_front = len(sorted_front)
        if n_front == 0:
            return pop

        # 4) seed by type
        if ic_type == "sector":
            theta_c = float(getattr(self, "sector_theta_center", 0.0))
            w_req   = int(getattr(self, "sector_width_initial", 24))
            w = max(1, min(w_req, n_front))

            def wrap_delta(a, c):
                d = abs(a - c)
                return min(d, 2 * np.pi - d)
            center_idx = int(np.argmin([wrap_delta(t, theta_c) for t in thetas]))

            half = w // 2
            start = (center_idx - half) % n_front
            for k in range(w):
                idx = (start + k) % n_front
                pop[sorted_front[idx]] = Resistant

        elif ic_type == "aif_front_bands":
            starts = np.random.choice(n_front, size=min(self.num_bands, n_front), replace=False)
            for s in starts:
                for i in range(self.band_width):
                    pop[sorted_front[(s + i) % n_front]] = Resistant

        elif ic_type == "aif_front_seeded":
            k = max(1, int(n_front * self.initial_resistant_fraction))
            idxs = np.random.choice(n_front, size=min(k, n_front), replace=False)
            for i in idxs:
                pop[sorted_front[i]] = Resistant

        # else: leave all susceptible
        return pop

    # ---------------------------- dynamics ----------------------------
    def _update_single_cell_events(self, h: Hex):
        # purge old events for this cell
        for event in list(self.cell_to_events.get(h, set())):
            self._remove_event(event)

        t = self.population.get(h)
        if t in (None, Empty):
            self._remove_from_front(h)
            return

        empties = [n for n in h.neighbors() if n not in self.population]
        if empties:
            self._add_to_front(h)

            if t == Susceptible:
                base_birth_rate = self.b_sus
            elif t == Resistant:
                base_birth_rate = self.b_res
            else:
                base_birth_rate = self.b_comp

            if base_birth_rate > 0.0:
                rate = base_birth_rate / len(empties)
                for nei in empties:
                    self._add_event(("grow", h, nei), rate)

            if t == Resistant and self.k_res_comp > 0.0:
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
        Angle-ordered true front: cells with ≥1 empty neighbor (outwardness
        handled statistically in radial growth).
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
        """
        Binary denoise over circular list: treat {Resistant,Compensated} as "mutant".
        Outputs labels as either Resistant (mutant) or Susceptible (non-mutant).
        """
        if not labels or w is None or w <= 1:
            return labels
        n = len(labels)
        out = labels[:]
        half = w // 2
        def is_mutant(t): return t in (Resistant, Compensated)
        for i in range(n):
            idxs = [(i + k) % n for k in range(-half, half + 1)]
            votes = sum(1 for j in idxs if is_mutant(labels[j]))
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
        # symmetric wrap merge if the last tiny sector matches first type
        if len(out) > 1 and out[0][0] == out[-1][0] and out[-1][1] < min_len:
            t0, w0, s0, e0 = out[0]
            tL, wL, sL, eL = out[-1]
            out = [(t0, w0 + wL, sL, e0)] + out[1:-1]
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
        Return [(type, width, startθ, endθ, sid, root_sid)].
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

    def _log_sector_widths_snapshot(self) -> List[Tuple[int, int, float, float, int, int]]:
        """
        Take a sector snapshot, append to trajectory log, and return enriched sectors:
        [(type, width_cells, start_theta, end_theta, sid, root_sid)]
        Also:
          - if infer_tracked_root=True and tracked_root_sid is unset, auto-select first resistant root.
          - update self._mutant_roots_ever_seen with any mutant roots present at the front.
        """
        sectors = self._sectorize_front()
        sectors_sid = self._assign_persistent_ids(sectors)

        r_mean = self._mean_front_radius_fast()
        step = self.step_count

        mutant_roots_now: Set[int] = set()
        for (t, w, s, e, sid, root) in sectors_sid:
            self.sector_traj_log.append(
                dict(step=int(step), radius=float(r_mean),
                     sid=int(sid), root_sid=int(root), type=int(t),
                     width_cells=int(w), start_theta=float(s), end_theta=float(e))
            )
            if t in (Resistant, Compensated):
                mutant_roots_now.add(int(root))

        # Update 'ever seen' set
        self._mutant_roots_ever_seen.update(mutant_roots_now)

        # (Optional) auto-seed tracked lineage, but only if explicitly enabled
        if self.infer_tracked_root and self.tracked_root_sid is None:
            for root in mutant_roots_now:
                self.tracked_root_sid = int(root)
                break

        return sectors_sid

    def _front_mutant_roots_now(self, sectors_sid: List[Tuple[int, int, float, float, int, int]]) -> Set[int]:
        roots: Set[int] = set()
        for (t, _w, _s, _e, _sid, root) in sectors_sid:
            if t in (Resistant, Compensated):
                roots.add(int(root))
        return roots

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

        # (2) radius-based cadence (throttled checks)
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
            sectors_sid = self._log_sector_widths_snapshot()
            self._last_logged_step_for_sectors = self.step_count

            # Extinction-aware early stopping (safe & unambiguous)
            if self.stop_on_front_extinction:
                roots_now = self._front_mutant_roots_now(sectors_sid)

                if self.tracked_root_sid is not None:
                    # strict, lineage-specific extinction
                    if int(self.tracked_root_sid) not in roots_now:
                        self.termination_reason = "tracked_lineage_extinct"
                        self.extinction_scope = "tracked_lineage"
                        return False, boundary_hit
                else:
                    # Only valid to call "sector extinction" if the system has ever been single-lineage
                    # AND now there are no mutants at the front.
                    ever = len(self._mutant_roots_ever_seen)
                    if ever == 1 and len(roots_now) == 0:
                        self.termination_reason = "single_mutant_extinct"
                        self.extinction_scope = "single_mutant"
                        return False, boundary_hit
                    # Otherwise ambiguous (multi-lineage): keep running (no misleading reason).

        # If we ever hit the boundary, record it unless a more specific reason already exists.
        if boundary_hit and self.termination_reason is None:
            self.termination_reason = "boundary_hit"

        return True, boundary_hit

    # ------------------------ engine hooks ------------------------
    def _is_valid_growth_neighbor(self, neighbor: Hex, parent: Hex) -> bool:
        # radial colony: any empty neighbor is valid
        return neighbor not in self.population

    def _get_neighbors_periodic(self, h: Hex) -> list:
        # simple cached neighbors (kept name for compatibility)
        if h in self.neighbor_cache:
            return self.neighbor_cache[h]
        neighbors = h.neighbors()
        self.neighbor_cache[h] = neighbors
        return neighbors

    @property
    def colony_radius(self) -> float:
        # Euclidean median of front radii (consistent with ΔR)
        if not self._front_lookup:
            return 0.0
        rs = [np.sqrt(self._xy_r2(h)[2]) for h in self._front_lookup]
        return float(np.median(rs)) if rs else 0.0
