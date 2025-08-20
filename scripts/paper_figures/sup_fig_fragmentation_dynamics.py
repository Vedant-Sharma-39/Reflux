"""
Supplementary Figure: Direct Visualization of Invasion Depth at Extinction

This script generates a static, side-by-side comparison of the final state for
a clumped vs. a fragmented initial population under strong negative selection.

It runs two dedicated simulations until extinction and plots a single snapshot
of the population state just before the final extinction event. The annotations
provide the key data: time to extinction and maximum invasion depth, offering
direct visual evidence that clustering leads to deeper invasions before collapse.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Add project root to path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model import GillespieSimulation
from src.core.metrics import MetricsManager, FixationTimeTracker


def get_snapshot_data(sim: "GillespieSimulation") -> dict:
    """Captures the current state of the simulation for plotting."""
    return {
        "population": sim.population.copy(),
        "mean_front_q": sim.mean_front_position,
        "width": sim.width,
    }


def get_final_state(sim_params: dict) -> dict:
    """
    Runs a simulation until extinction and returns the state just before the
    final extinction event, along with the parameters used.
    """
    print(f"--- Running simulation for: {sim_params['name'].upper()} ---")
    np.random.seed(42)
    sim = GillespieSimulation(**sim_params["params"])
    manager = MetricsManager(sim_params["params"])
    manager.add_tracker(FixationTimeTracker, {})
    manager.register_simulation(sim)

    last_valid_state = None

    while not manager.is_done():
        last_valid_state = get_snapshot_data(sim)
        active, boundary_hit = sim.step()
        manager.after_step_hook()
        if not active or boundary_hit:
            break

    final_results = manager.finalize()
    print(
        f"  -> Finished. Outcome: {final_results.get('outcome', 'N/A')}, "
        f"Time: {final_results.get('time_to_outcome', 0.0):.1f}, "
        f"Max q: {final_results.get('q_at_outcome', 0.0):.1f}"
    )

    if last_valid_state is None:
        last_valid_state = get_snapshot_data(sim)

    # --- FIX 2: Return the original parameters for plotter configuration ---
    return {
        "snapshot": last_valid_state,
        "result": final_results,
        "params": sim_params["params"],
    }
    # --- END OF FIX 2 ---


def plot_final_state(ax, final_state_data: dict, title: str):
    """Plots a single final state snapshot onto a matplotlib axis."""
    snapshot = final_state_data["snapshot"]
    result = final_state_data["result"]
    params = final_state_data["params"]  # Get the original sim params

    time_extinct = result.get("time_to_outcome", 0.0)
    max_q = result.get("q_at_outcome", 0.0)

    # --- FIX 2: Create a plotter from a sim with the correct parameters ---
    # This ensures the plotter is configured for the correct grid size, etc.
    plotter = GillespieSimulation(**params).plotter
    plotter.ax = ax
    plotter.fig = ax.figure
    # --- END OF FIX 2 ---

    plotter.plot_population(
        population=snapshot["population"],
        mean_front_q=max_q,  # Center view on the max q reached
        sim_width=snapshot["width"],
    )

    ax.set_title(
        f"{title}\n$t_{{extinct}}={time_extinct:.1f}$, Max $q={max_q:.1f}$", fontsize=16
    )


def main():
    # --- 1. Define the two scenarios to compare ---
    base_params = {
        "run_mode": "visualization",
        "width": 256,
        "length": 4096,
        "k_total": 0.0,
        "phi": 0.0,
        "b_m": 0.7,
        "initial_mutant_patch_size": 64,
        "initial_condition_type": "grf_threshold",
    }
    scenarios = {
        "Clumped": {
            "name": "clumped",
            "params": {**base_params, "correlation_length": 100.0},
        },
        "Fragmented": {
            "name": "fragmented",
            "params": {**base_params, "correlation_length": 1.0},
        },
    }

    # --- 2. Run simulations and get the final state for each ---
    final_states = {}
    for name, scenario_data in scenarios.items():
        final_states[name] = get_final_state(scenario_data)

    # --- 3. Create the final 1x2 static figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        "Clustered Populations Invade Deeper Before Extinction", fontsize=20, y=1.05
    )

    plot_final_state(axes[0], final_states["Clumped"], "(A) Initial State: Clumped")
    plot_final_state(
        axes[1], final_states["Fragmented"], "(B) Initial State: Fragmented"
    )

    # --- 4. Save the figure ---
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sup_fig_fragmentation_dynamics.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Final static figure saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
