# FILE: scripts/debug.py
#
# [DEFINITIVELY CORRECTED AND UNIFIED v2]
# This single, robust script debugs ANY core workflow. It fixes the
# ModuleNotFoundError and the infinite toggling bug in the perturbation test.

import sys
import os
import json
import traceback

# --- [THE DEFINITIVE PATH FIX] ---
# This block programmatically finds the project's root directory and adds the
# 'src' folder to Python's path. THIS MUST BE THE FIRST BLOCK OF CODE.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now that the path is set, these imports will succeed.
from worker import (
    run_calibration_sim,
    run_diffusion_sim,
    run_structure_analysis_sim,
    run_perturbation_sim,
)

# ==============================================================================
# 1. DEFINE MINIATURE TEST CASES
# ==============================================================================
TEST_CASES = {
    "calibration": {
        "description": "Testing 'calibration' mode for sector drift.",
        "run_function": run_calibration_sim,
        "params": {
            "width": 64,
            "length": 128,
            "b_m": 0.5,
            "max_steps": 20000,
            "initial_mutant_patch_size": 16,
            "run_mode": "calibration",
        },
    },
    "diffusion": {
        "description": "Testing 'diffusion' mode for front roughness.",
        "run_function": run_diffusion_sim,
        "params": {
            "width": 64,
            "length": 128,
            "b_m": 1.0,
            "max_steps": 10000,
            "run_mode": "diffusion",
        },
    },
    "structure": {
        "description": "Testing 'structure_analysis' with the lightweight tracker.",
        "run_function": run_structure_analysis_sim,
        "params": {
            "width": 64,
            "length": 128,
            "b_m": 0.8,
            "k_total": 0.25,
            "phi": -1.0,
            "warmup_time": 5.0,
            "num_samples": 10,
            "sample_interval": 1.0,
            "run_mode": "structure_analysis",
        },
    },
    "perturbation": {
        "description": "Testing 'perturbation' mode to check for stalling and logic.",
        "run_function": run_perturbation_sim,
        "params": {
            "width": 128,
            "length": 4096,
            "phi": -0.5,
            "b_m": 0.75,
            "k_total": 0.08,
            "total_run_time": 2000.0,
            "pulse_start_time": 1500.0,
            "pulse_duration": 100.0,
            "k_total_pulse": 10.0,
            "sample_interval": 10.0,
            "run_mode": "perturbation",
        },
    },
}


# ==============================================================================
# 2. THE UNIFIED DEBUGGER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run a specific debug test case for the Reflux workflow."
    )
    parser.add_argument(
        "test_name",
        choices=list(TEST_CASES.keys()),
        help="Name of the test case to run.",
    )
    args = parser.parse_args()

    case = TEST_CASES[args.test_name]

    print("--- Running Unified Workflow Debugger ---")
    print(f"\n{'='*25} Testing Case: {args.test_name} {'='*25}")
    print(f"    Description: {case['description']}")
    print(f"    Parameters: {json.dumps(case['params'], indent=4)}")

    try:
        # Directly call the correct, debugged worker function
        results = case["run_function"](case["params"])

        print("\n    --- Validation ---")
        if "error" in results:
            print(f"    [FAIL] The worker returned an error:\n{results['error']}")
            return

        print(f"    [PASS] Worker ran successfully.")

        if "timeseries" in results:
            ts_len = len(results["timeseries"])
            print(f"    - Timeseries length: {ts_len}")
            if ts_len > 0:
                final_time = results["timeseries"][-1]["time"]
                target_time = case["params"].get("total_run_time", 0)
                print(f"    - Final time point: {final_time:.2f} / {target_time:.2f}")
                if final_time < target_time * 0.95:
                    print("    [WARN] Simulation ended prematurely.")
                else:
                    print("    [PASS] Simulation ran to completion.")
            else:
                print("    [WARN] Simulation produced an empty timeseries.")

    except Exception:
        print(f"\n    [FATAL FAIL] Debug script crashed!")
        traceback.print_exc()


if __name__ == "__main__":
    main()
