# FILE: scripts/run_integration_tests.py (Data-Only Version)
#
# A fast, data-focused integration test suite for the Reflux simulation framework.
# This script does NOT run any plotting scripts.
#
# USAGE:
#   python3 scripts/run_integration_tests.py
#   python3 scripts/run_integration_tests.py --length 256  (Faster testing)
#   python3 scripts/run_integration_tests.py --exp phase_diagram
#
# This script will:
#  1. Create a temporary 'test_output' directory.
#  2. For each experiment (or the one specified):
#     a. Generate a small sample of tasks.
#     b. Run the simulation worker for each task.
#     c. Consolidate the raw results into a summary CSV.
#     d. Verify that the CSV was created and contains data.
#  3. Report a PASS/FAIL status for each experiment.
#  4. Clean up all temporary files.

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

# --- Robustly add project root to the Python path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports from the project source ---
try:
    from src.config import EXPERIMENTS
    from src.worker import run_simulation, BULKY_DATA_KEYS
    from scripts.utils.generate_tasks import generate_tasks_for_experiment
except ImportError as e:
    print(f"FATAL: Could not import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure you are running this script from the project root directory.", file=sys.stderr)
    sys.exit(1)


# --- ANSI Color Codes for Nice Output ---
class Colors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(message: str, color: str, bold: bool = False):
    """Prints a colored status message."""
    bold_code = Colors.BOLD if bold else ""
    print(f"{bold_code}{color}{message}{Colors.ENDC}")


# --- Test Logic ---

def select_sample_tasks(all_tasks: List[Dict[str, Any]], num_samples: int = 3) -> List[Dict[str, Any]]:
    """Selects a representative sample of tasks to run."""
    if not all_tasks:
        return []
    if len(all_tasks) <= num_samples:
        return all_tasks
    
    indices = sorted(list(set([0, len(all_tasks) // 2, len(all_tasks) - 1])))
    return [all_tasks[i] for i in indices]


def run_worker_stage(tasks: List[Dict[str, Any]], test_root: Path, override_length: Optional[int] = None) -> bool:
    """Runs the simulation worker for a list of sample tasks within the test environment."""
    print_status("  [1/3] Running simulation worker for sample tasks...", Colors.OKCYAN)

    if override_length:
        print_status(f"      -> Overriding simulation length to {override_length} for all tasks.", Colors.WARNING)
    
    for task_params in tqdm(tasks, desc="Worker Tasks", ncols=80):
        if override_length:
            task_params['length'] = override_length

        campaign_id = task_params["campaign_id"]
        task_id = task_params["task_id"]
        
        raw_dir = test_root / "data" / campaign_id / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        summary_output_file = raw_dir / f"task_{task_id}.jsonl"

        try:
            result_data = run_simulation(task_params)
            # Remove bulky data to mimic real worker behavior, but don't save it.
            for key in BULKY_DATA_KEYS:
                if key in result_data:
                    result_data.pop(key)
            
            with open(summary_output_file, 'w') as f:
                f.write(json.dumps(result_data) + '\n')
        except Exception:
            print_status(f"\n      ERROR: Worker failed for task {task_id}", Colors.FAIL)
            print(f"      PARAMS: {json.dumps(task_params, indent=2)}")
            print(traceback.format_exc())
            return False
            
    return True


def run_consolidation_stage(campaign_id: str, test_root: Path) -> bool:
    """Consolidates raw data and verifies the output CSV."""
    print_status("  [2/3] Consolidating raw data and verifying CSV...", Colors.OKCYAN)
    raw_dir = test_root / "data" / campaign_id / "raw"
    analysis_dir = test_root / "data" / campaign_id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = analysis_dir / f"{campaign_id}_summary_aggregated.csv"

    all_summaries = []
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print_status(f"      WARNING: No raw data was produced for {campaign_id}. Skipping consolidation.", Colors.WARNING)
        return True

    for jsonl_file in raw_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                all_summaries.append(json.loads(line))
    
    if not all_summaries:
        print_status(f"      ERROR: Raw files were found but contained no valid JSONL data for {campaign_id}", Colors.FAIL)
        return False

    try:
        df = pd.DataFrame(all_summaries)
        df.to_csv(summary_csv_path, index=False)
        print(f"      -> Consolidated {len(df)} records into summary CSV.")

        # --- Verification Step ---
        if not summary_csv_path.exists():
            print_status("      ERROR: Summary CSV file was NOT created.", Colors.FAIL)
            return False
        
        # Check if the CSV has more than just a header
        with open(summary_csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                print_status("      ERROR: Summary CSV was created but is EMPTY (contains header only).", Colors.FAIL)
                return False
        
        print_status("      -> Verification PASSED: CSV is valid and contains data.", Colors.OKGREEN)
        return True
    except Exception:
        print_status("\n      ERROR: Failed to consolidate data into DataFrame or verify CSV.", Colors.FAIL)
        print(traceback.format_exc())
        return False


def test_experiment(experiment_name: str, test_root: Path, override_length: Optional[int] = None) -> bool:
    """Runs a full data-pipeline integration test for a single experiment."""
    print_status(f"\n--- TESTING EXPERIMENT: {experiment_name} ---", Colors.HEADER, bold=True)
    start_time = time.time()
    
    print_status("  [PREP] Generating task list...", Colors.OKCYAN)
    try:
        all_tasks = generate_tasks_for_experiment(experiment_name)
        if not all_tasks:
            print_status("  -> No tasks generated for this experiment. Test PASSED (vacuously).", Colors.OKGREEN)
            return True
        sample_tasks = select_sample_tasks(all_tasks)
        campaign_id = sample_tasks[0]['campaign_id']
        print(f"      -> Generated {len(all_tasks)} total tasks. Sampling {len(sample_tasks)}.")
    except Exception:
        print_status("\n      ERROR: Task generation failed.", Colors.FAIL)
        print(traceback.format_exc())
        return False

    if not run_worker_stage(sample_tasks, test_root, override_length):
        return False
    
    if not run_consolidation_stage(campaign_id, test_root):
        return False

    duration = time.time() - start_time
    print_status(f"  [3/3] All data stages completed successfully in {duration:.2f}s.", Colors.OKCYAN)
    return True

def main():
    """Main function to set up the environment and run tests."""
    parser = argparse.ArgumentParser(description="Run data-only integration tests for the Reflux framework.")
    parser.add_argument("--exp", help="Run tests for a single experiment by name (e.g., 'phase_diagram').")
    parser.add_argument("--length", type=int, default=None, help="Override simulation length for all tests.")
    args = parser.parse_args()

    test_root = PROJECT_ROOT / "test_output"
    
    print(f"Setting up clean test environment at: {test_root}")
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir()

    try:
        experiments_to_run = [args.exp] if args.exp else sorted(EXPERIMENTS.keys())
        results = {}
        
        for exp_name in experiments_to_run:
            if exp_name not in EXPERIMENTS:
                print_status(f"ERROR: Experiment '{exp_name}' not found in config.", Colors.FAIL, bold=True)
                continue
            
            is_success = test_experiment(exp_name, test_root, override_length=args.length)
            results[exp_name] = is_success
            if is_success:
                print_status(f"--- RESULT: {exp_name} PASSED ---", Colors.OKGREEN, bold=True)
            else:
                print_status(f"--- RESULT: {exp_name} FAILED ---", Colors.FAIL, bold=True)

        # --- Final Report ---
        print_status("\n" + "="*50, Colors.HEADER, bold=True)
        print_status("          DATA PIPELINE TEST SUMMARY", Colors.HEADER, bold=True)
        print_status("="*50, Colors.HEADER, bold=True)
        
        passed = [k for k, v in results.items() if v]
        failed = [k for k, v in results.items() if not v]
        
        for p in passed:
            print_status(f"  [PASS] {p}", Colors.OKGREEN)
        for f in failed:
            print_status(f"  [FAIL] {f}", Colors.FAIL)
            
        print("-" * 50)
        print_status(f"TOTAL: {len(results)} | PASSED: {len(passed)} | FAILED: {len(failed)}", Colors.HEADER, bold=True)
        
        if failed:
            sys.exit(1)

    finally:
        # --- Cleanup ---
        print("\nCleaning up test environment...")
        shutil.rmtree(test_root, ignore_errors=True)
        print("Cleanup complete.")

if __name__ == "__main__":
    main()