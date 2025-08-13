# FILE: debug_recovery.py
# A standalone script to debug a single task from the 'recovery_timescale' experiment.
# Place this file in the root of your project directory and run it.

import json
import os
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# You can change this number to debug a different parameter set.
# Line 1 is the first task defined in the experiment config.
TASK_LINE_TO_DEBUG = 1
EXPERIMENT_NAME = "recovery_timescale"


def main():
    """
    Main function to set up paths, select a task, and run the worker.
    """
    # --- 1. Set up project paths ---
    # This is crucial so the worker can find its imports (e.g., src.core.model)
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import project-specific modules AFTER setting the path
    from src.config import EXPERIMENTS
    from scripts.utils.generate_tasks import generate_tasks_for_experiment

    print(f"--- 🕵️  Debugging Experiment: {EXPERIMENT_NAME} ---")

    # --- 2. Generate and select the task parameters ---
    print("Generating all tasks for the experiment to select one...")
    all_tasks = generate_tasks_for_experiment(EXPERIMENT_NAME)

    if not all_tasks or TASK_LINE_TO_DEBUG > len(all_tasks):
        print(
            f"Error: Task line {TASK_LINE_TO_DEBUG} is out of bounds. The experiment has {len(all_tasks)} tasks."
        )
        sys.exit(1)

    # Select the specific task parameters based on the line number
    params_to_run = all_tasks[TASK_LINE_TO_DEBUG - 1]
    task_id = params_to_run.get("task_id", "unknown_task")
    campaign_id = params_to_run.get("campaign_id")
    params_json = json.dumps(params_to_run)

    print(f"\nSelected Task ID: {task_id} (from line {TASK_LINE_TO_DEBUG})")
    print("Parameters:")
    print(json.dumps(params_to_run, indent=2))

    # --- 3. Prepare for the simulation run ---
    # The worker script needs an output directory to save raw files.
    # We will mimic the structure used by manage.py.
    output_dir = project_root / "data" / campaign_id / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRaw output will be directed to: {output_dir}")

    worker_script = project_root / "src" / "worker.py"

    # The worker needs the PROJECT_ROOT environment variable to be set.
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(project_root)

    # --- 4. Run the worker as a subprocess ---
    print("\n--- Running Worker ---")
    cmd = [
        sys.executable,
        str(worker_script),
        "--params",
        params_json,
        "--output-dir",
        str(output_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # --- 5. Process and display the results ---
    if proc.returncode == 0:
        print("\n--- ✅ Worker Finished Successfully ---")
        try:
            # The worker prints the lightweight JSON summary to stdout
            summary_results = json.loads(proc.stdout)
            print("Summary Results:")
            print(json.dumps(summary_results, indent=2))

            # The worker saves the bulky timeseries data to a separate file.
            # Let's find and report its location.
            timeseries_dir = project_root / "data" / campaign_id / "timeseries_raw"
            timeseries_file = timeseries_dir / f"ts_{task_id}.json.gz"

            if timeseries_file.exists():
                print(f"\nDetailed timeseries data saved to:\n{timeseries_file}")
            else:
                print("\nWarning: Timeseries file was not created.")

        except json.JSONDecodeError:
            print("\n❌ FAILED: Worker ran but its output was not valid JSON.")
            print("--- Worker STDOUT ---")
            print(proc.stdout)

    else:
        print(f"\n--- ❌ Worker FAILED with exit code {proc.returncode} ---")
        print("--- Worker STDERR ---")
        print(proc.stderr or "No stderr output.")


if __name__ == "__main__":
    main()
