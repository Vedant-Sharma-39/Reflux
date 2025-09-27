# FILE: scripts/debug_aif_sector_analysis.py
#
# A standalone script to generate the raw data needed for multi-sector analysis.
# It runs a single AIF simulation with a multi-sector initial condition and saves
# the final population state to a compressed JSON file.

import sys
import json
import gzip
from pathlib import Path
from tqdm import tqdm

# --- Add project root to Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model_aif import AifModelSimulation


def main():
    """Configures, runs, and saves the result of a multi-sector simulation."""
    print("--- Running AIF Multi-Sector Simulation for Data Generation ---")

    params = {
        # --- Simulation Control ---
        "campaign_id": "debug_aif_multisector",
        "max_steps": 200_000,
        # --- Initial Conditions ---
        "initial_condition_type": "sector",
        "initial_droplet_radius": 40,
        "num_sectors": 3,
        "sector_width_initial": 30,  # Controls the angular width of each initial sector
        # --- AIF Model Physics ---
        "b_sus": 1.0,
        "b_res": 0.7,  # Resistant cells grow slightly slower
        "b_comp": 1.0,  # Compensated cells recover full fitness
        "k_res_comp": 0,  # A slow rate of compensation
    }

    print("\nUsing parameters:")
    print(json.dumps(params, indent=2))

    # --- Setup output path ---
    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data_path = output_dir / "aif_multisector_final_pop.json.gz"

    # --- Initialize and run the simulation ---
    sim = AifModelSimulation(**params)

    with tqdm(total=params["max_steps"], desc="Simulating") as pbar:
        while sim.step_count < params["max_steps"]:
            active, _ = sim.step()
            pbar.update(1)
            if not active:
                print("\nSimulation ended (population extinction).")
                break

    print(f"\nSimulation finished at step {sim.step_count} and time {sim.time:.2f}.")

    # --- Save the final population data ---
    final_pop_data = [
        {"q": h.q, "r": h.r, "type": t} for h, t in sim.population.items()
    ]

    print(f"Saving final population data for {len(final_pop_data)} cells...")
    with gzip.open(output_data_path, "wt", encoding="utf-8") as f:
        json.dump(final_pop_data, f)

    print(f"✅ Data saved successfully to: {output_data_path}")
    print("\nYou can now run 'scripts/analyze_aif_sectors.py' to analyze this file.")


if __name__ == "__main__":
    main()
