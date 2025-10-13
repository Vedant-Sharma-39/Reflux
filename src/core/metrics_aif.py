# FILE: src/core/metrics_aif.py
from typing import Dict, Any, List
from src.core.metrics import MetricTracker

try:
    from src.core.model_aif import AifModelSimulation  # type: ignore
except Exception:
    AifModelSimulation = object  # typing fallback

class AifMetricsManager(MetricTracker):
    """
    Tracker for 'aif_width_analysis'.
    - Lets the engine do ΔR logging into sim.sector_traj_log.
    - On finalize(), returns:
        sector_trajectory (list[dict]),
        num_sector_snapshots (int),
        final_population (optional),
        a few convenience scalars.
    """
    def __init__(self, sim: "AifModelSimulation", include_final_population: bool = True, **kwargs: Any):
        super().__init__(sim=sim, **kwargs)
        self.include_final_population = bool(include_final_population)

    def is_done(self) -> bool:
        return False  # worker/engine decide termination

    def after_step_hook(self):
        pass  # ΔR logging happens inside the engine

    def finalize(self) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = list(self.sim.sector_traj_log)
        out: Dict[str, Any] = {
            "sector_trajectory": rows,
            "num_sector_snapshots": len(rows),
            "final_radius": float(getattr(self.sim, "_mean_front_radius_fast")()),
            "final_step": int(self.sim.step_count),
            "final_front_length": int(getattr(self.sim, "_front_count", 0)),
        }
        if self.include_final_population:
            out["final_population"] = [
                {"q": int(h.q), "r": int(h.r), "type": int(t)}
                for h, t in self.sim.population.items()
            ]
        return out
