# FILE: src/core/metrics_aif.py (Definitive, Final Version)

from typing import Dict, Any
if 'AifModelSimulation' not in globals():
    from src.core.model_aif import AifModelSimulation
from src.core.metrics import MetricTracker

class AifMetricsManager(MetricTracker):
    """
    PURPOSE: This is the designated tracker for HPC campaigns that require a full
    spatial analysis of the final colony state. Its sole function is to run
    a simulation for a fixed duration and then save the ENTIRE final population.
    
    This provides the "fossil record" that an analysis script can later process
    to measure properties like sector width vs. radius.
    
    RUN_MODE in config.py: "aif_width_analysis"
    OUTPUT: A "final_population" data file (e.g., pop_*.json.gz).
    """
    def __init__(self, sim: "AifModelSimulation", max_steps: int = 200000, **kwargs):
        super().__init__(sim=sim, **kwargs)
        self.max_steps = max_steps
        self._is_done = False

    def is_done(self) -> bool:
        return self._is_done

    def after_step_hook(self):
        # The tracker's only job is to stop the simulation at the right time.
        if self.sim.step_count >= self.max_steps:
            self._is_done = True
        
        # Also stop if the resistant population dies out completely.
        if self.sim.resistant_cell_count + self.sim.compensated_cell_count == 0:
            self._is_done = True

    def finalize(self) -> Dict[str, Any]:
        # Package the entire final population dictionary. The worker will see the
        # "final_population" key and save this data to a separate file.
        final_pop_data = [
            {"q": h.q, "r": h.r, "type": t} for h, t in self.sim.population.items()
        ]
        return {"final_population": final_pop_data}