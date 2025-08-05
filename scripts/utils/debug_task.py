# FILE: scripts/utils/debug_task.py
# A single, robust script to run ANY simulation task locally for debugging.
# It calls the unified worker and pretty-prints the summary and timeseries output.

import argparse
import json
import sys
import os
import subprocess

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single simulation task locally for debugging."
    )
    parser.add_argument(
        "experiment_name",
        help="The name of the experiment, as defined in src/config.py.",
    )
    parser.add_argument(
        "task_line_num",
        type=int,
        help="The line number from the campaign's _task_list.txt file to run.",
    )
    args = parser.parse_args()

    # --- Find the task file ---
    campaign_id = None
    try:
        from src.config import EXPERIMENTS

        campaign_id = EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"]
    except (ImportError, KeyError):
        sys.exit(
            f"Error: Could not find experiment '{args.experiment_name}' in src/config.py"
        )

    task_file_path = os.path.join(project_root, "data", f"{campaign_id}_task_list.txt")

    if not os.path.exists(task_file_path):
        print(f"Task file not found at {task_file_path}")
        print("Attempting to generate it now...")
        gen_script = os.path.join(project_root, "scripts/utils/generate_tasks.py")
        subprocess.run(["python3", gen_script, args.experiment_name], check=True)
        if not os.path.exists(task_file_path):
            sys.exit("Failed to generate task file. Exiting.")

    # --- Read the specified task ---
    try:
        with open(task_file_path, "r") as f:
            for i, line in enumerate(f):
                if i + 1 == args.task_line_num:
                    params_json = line.strip()
                    break
            else:
                raise IndexError
    except IndexError:
        sys.exit(
            f"Error: Line number {args.task_line_num} is out of bounds for the task file."
        )

    # --- Run the worker as a subprocess ---
    print("--- Running Worker in Debug Mode ---")
    print("\nParameters:")
    print(json.dumps(json.loads(params_json), indent=2))
    print("-" * 35)

    worker_script = os.path.join(project_root, "src/worker.py")
    # Use subprocess to perfectly replicate how run_chunk.sh calls the worker
    process = subprocess.run(
        ["python3", worker_script, "--params", params_json],
        capture_output=True,
        text=True,
    )

    # --- Parse and display the output ---
    DELIMITER = "---WORKER_PAYLOAD_SEPARATOR---"
    stdout = process.stdout
    stderr = process.stderr

    if stderr:
        print("\n--- STDERR (Worker Errors) ---")
        try:
            error_data = json.loads(stderr)
            print(json.dumps(error_data, indent=2))
        except json.JSONDecodeError:
            print(stderr)
        print("-" * 35)

    if stdout and DELIMITER in stdout:
        summary_str, timeseries_str = stdout.split(DELIMITER, 1)

        print("\n--- Summary Output (stdout) ---")
        try:
            summary_data = json.loads(summary_str)
            print(json.dumps(summary_data, indent=2))
        except json.JSONDecodeError:
            print("Could not parse summary JSON:")
            print(summary_str)
        print("-" * 35)

        print("\n--- Timeseries Output (stdout) ---")
        try:
            ts_data = json.loads(timeseries_str)
            ts_list = ts_data.get("timeseries", [])
            print(f"Task ID: {ts_data.get('task_id')}")
            print(f"Timeseries contains {len(ts_list)} data points.")
            if ts_list:
                print("First 3 points:", json.dumps(ts_list[:3], indent=2))
                print("...")
                print("Last 3 points:", json.dumps(ts_list[-3:], indent=2))
        except json.JSONDecodeError:
            print("Could not parse timeseries JSON:")
            print(timeseries_str)
        print("-" * 35)
    else:
        print("\n--- Raw STDOUT (Could not find delimiter) ---")
        print(stdout)


if __name__ == "__main__":
    main()
