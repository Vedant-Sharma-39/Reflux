# FILE: scripts/debug_workflow.py
#
# [DEFINITIVELY CORRECTED] This version fixes the ModuleNotFoundError by
# correctly adding the 'src' directory to the system path BEFORE any
# imports from the local project.

import sys
import os
import json

# --- [THE DEFINITIVE FIX] ---
# This block programmatically finds the project's root directory and adds the
# 'src' folder to Python's path. THIS MUST BE THE FIRST THING THE SCRIPT DOES.
try:
    # This works when the script is run from the project root (e.g., python scripts/debug.py)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
except NameError:
    # This works when the script is run from the scripts directory (e.g., cd scripts; python debug.py)
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# Now that the path is set, this import will succeed.
from worker import run_calibration_sim, run_diffusion_sim, run_structure_analysis_sim

# ==============================================================================
# 1. DEFINE THE MINIATURE TEST CASES
# ==============================================================================
TEST_CASES = {
    "calibration_boundary_dynamics": {
        "description": "Testing 'calibration' mode with a 'patch' initial condition.",
        "run_function": run_calibration_sim,
        "expected_key": "trajectory",
        "params": {
            "width": 64,
            "length": 128,
            "b_m": 0.5,
            "k_total": 0.0,
            "phi": 0.0,
            "initial_condition_type": "patch",
            "initial_mutant_patch_size": 16,
            "max_steps": 20000,
        },
    },
    "calibration_front_morphology": {
        "description": "Testing 'diffusion' mode with a 'mixed' initial condition.",
        "run_function": run_diffusion_sim,
        "expected_key": "roughness_trajectory",
        "params": {
            "width": 64,
            "length": 128,
            "b_m": 0.5,
            "k_total": 0.0,
            "phi": 0.0,
            "initial_condition_type": "mixed",
            "max_steps": 10000,
        },
    },
    "structure_analysis_criticality": {
        "description": "Testing 'structure_analysis' mode with the lightweight tracker.",
        "run_function": run_structure_analysis_sim,
        "expected_key": "avg_interface_density",
        "params": {
            "width": 64,
            "length": 128,
            "b_m": 0.8,
            "k_total": 0.25,
            "phi": -1.0,
            "initial_condition_type": "mixed",
            "warmup_time": 5.0,
            "num_samples": 10,
            "sample_interval": 1.0,
        },
    },
}


# ==============================================================================
# 2. THE DEBUGGER
# ==============================================================================
def main():
    print("--- Running Definitive Workflow Debugger ---")
    all_tests_passed = True
    for name, case in TEST_CASES.items():
        print(f"\n{'='*25} Testing Case: {name} {'='*25}")
        print(f"    Description: {case['description']}")
        print(f"    Parameters: {json.dumps(case['params'], indent=4)}")
        try:
            results = case["run_function"](case["params"])
            print("\n    --- Validation ---")
            if "error" in results:
                print(f"    [FAIL] The worker returned an error:\n{results['error']}")
                all_tests_passed = False
                continue
            expected_key = case["expected_key"]
            if expected_key not in results:
                print(f"    [FAIL] Output is missing expected key: '{expected_key}'")
                all_tests_passed = False
                continue
            output_data = results[expected_key]
            if output_data is None or (
                isinstance(output_data, list) and not output_data
            ):
                print(f"    [FAIL] Data for key '{expected_key}' is empty.")
                all_tests_passed = False
                continue
            print(
                f"    [PASS] Worker ran successfully. Found non-empty data for key '{expected_key}'."
            )
        except Exception as e:
            print(f"\n    [FATAL FAIL] Debug script crashed!")
            import traceback

            traceback.print_exc()
            all_tests_passed = False
    print(f"\n{'='*60}")
    if all_tests_passed:
        print("\n✅ All core workflow tests passed! You are ready to launch.")
    else:
        print("\n❌ One or more workflow tests failed. Review [FAIL] messages.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
