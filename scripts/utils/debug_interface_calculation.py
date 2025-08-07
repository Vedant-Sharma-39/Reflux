# FILE: scripts/utils/debug_interface_calculation.py
# A dedicated script to rigorously verify the accuracy of the optimized,
# incremental WT-Mutant interface calculation against the brute-force method.

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the simulation model (which uses the FAST, optimized calculation)
from src.core.model import GillespieSimulation, Empty, Wildtype, Mutant
from src.core.hex_utils import Hex

# ==============================================================================
# "GROUND TRUTH" - The Slow, Brute-Force Calculation
# We copy this logic here to have a separate, trusted reference.
# ==============================================================================


def _get_neighbors_periodic_bruteforce(h: Hex, width: int) -> list:
    """A self-contained neighbor function for the brute-force calculation."""
    unwrapped = h.neighbors()
    wrapped = []
    for n in unwrapped:
        q, r = n.q, n.r
        if q < 0:
            q += width
            r -= width
        elif q >= width:
            q -= width
            r += width
        s = -q - r
        wrapped.append(Hex(q, r, s))
    return wrapped


def calculate_brute_force_interface(population: Dict[Hex, int], width: int) -> float:
    """Calculates the WT-Mutant interface by iterating through all cells."""
    shared_boundaries = 0
    for h, cell_type in population.items():
        if cell_type == Empty:
            continue

        for neighbor_hex in _get_neighbors_periodic_bruteforce(h, width):
            neighbor_type = population.get(neighbor_hex)

            if (
                neighbor_type is not None
                and neighbor_type != Empty
                and neighbor_type != cell_type
            ):
                shared_boundaries += 1

    return shared_boundaries / 2.0


# ==============================================================================
# MAIN VERIFICATION SCRIPT
# ==============================================================================


def main():
    print("--- Verifying Optimized Interface Calculation ---")

    # Use parameters that will generate a complex interface
    TEST_PARAMS = {
        "width": 64,
        "length": 128,
        "b_m": 0.9,
        "k_total": 0.5,
        "phi": 0.0,
        "initial_condition_type": "mixed",
    }

    DEBUG_CONFIG = {"num_steps": 20000, "log_interval": 100}  # Log data every 100 steps

    # 1. Instantiate the simulation with the OPTIMIZED code
    sim = GillespieSimulation(**TEST_PARAMS)

    results = []

    # 2. Run the simulation step-by-step
    for step in tqdm(range(DEBUG_CONFIG["num_steps"]), desc="Running verification"):
        # At each logging interval, record both values
        if step % DEBUG_CONFIG["log_interval"] == 0:

            # Get the FAST value from the optimized model's property
            optimized_val = sim.wt_mutant_interface_length

            # Calculate the SLOW value using the ground-truth function
            ground_truth_val = calculate_brute_force_interface(
                sim.population, sim.width
            )

            results.append(
                {
                    "step": step,
                    "optimized": optimized_val,
                    "ground_truth": ground_truth_val,
                }
            )

        # Execute one step of the simulation
        active, _ = sim.step()
        if not active:
            print("Simulation stalled.")
            break

    # 3. Analyze and Plot the results
    df = pd.DataFrame(results)
    df["difference"] = (df["optimized"] - df["ground_truth"]).abs()

    max_error = df["difference"].max()

    print("\n--- Verification Results ---")
    if max_error < 1e-9:
        print(f"\n[PASS] The optimized calculation is ACCURATE.")
        print(f"Maximum absolute error was {max_error:.2e}, which is effectively zero.")
    else:
        print(f"\n[FAIL] The optimized calculation is INACCURATE.")
        print(f"Maximum absolute error was {max_error:.4f}.")
        print("Discrepancies found. Check the plot and the logic.")

    # 4. Create a comparison plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("Optimized vs. Brute-Force Interface Calculation", fontsize=20)

    # Top panel: Both calculations overlaid
    ax1.plot(df["step"], df["optimized"], "r-", lw=2, label="Optimized (Incremental)")
    ax1.plot(
        df["step"],
        df["ground_truth"],
        "ko",
        markersize=4,
        mfc="none",
        label="Ground Truth (Brute-Force)",
    )
    ax1.set_ylabel("Interface Length")
    ax1.legend()
    ax1.grid(True, linestyle=":")
    ax1.set_title("Direct Comparison")

    # Bottom panel: The difference between the two
    ax2.plot(df["step"], df["difference"], "b-")
    ax2.axhline(0, color="k", linestyle="--", lw=1)
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Absolute Difference")
    ax2.set_title("Error (Difference between methods)")
    ax2.set_ylim(bottom=-0.1, top=max(1.0, max_error * 1.2))  # Set a sensible y-limit
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.join(project_root, "debug_images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "verification_interface_calculation.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nVerification plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Ensure your model.py has the new optimized code before running this
    main()
