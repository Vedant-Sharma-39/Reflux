# FILE: debug_workflow.py (Improved Error Reporting)

import subprocess
import sys
import time
from pathlib import Path
import shutil
import json
import pandas as pd

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.resolve()
MANAGE_PY = PROJECT_ROOT / "manage.py"
DEBUG_EXP_NAME = "workflow_debug_test"
DEBUG_CAMPAIGN_ID = "debug_workflow"
DATA_DIR = PROJECT_ROOT / "data" / DEBUG_CAMPAIGN_ID
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"
LOG_DIR = PROJECT_ROOT / "slurm_logs"


# --- Helper for styled printing ---
def print_step(msg):
    print(f"\n\033[1;34m===== STEP: {msg} =====\033[0m")


def print_info(msg):
    print(f"\033[96m  [i] {msg}\033[0m")


def print_success(msg):
    print(f"\033[1;32m  [✔] {msg}\033[0m")


def print_fail(msg, stdout=None, stderr=None):
    print(f"\033[1;31m  [✘] {msg}\033[0m")
    if stdout:
        print(f"\033[91m--- STDOUT ---\n{stdout}\033[0m")
    if stderr:
        print(f"\033[91m--- STDERR ---\n{stderr}\033[0m")
    sys.exit(1)


def run_cmd(cmd):
    """Runs a command and handles errors with detailed output."""
    print_info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print_fail(
            f"Command failed with exit code {result.returncode}.",
            result.stdout,
            result.stderr,
        )
    return result


def main():
    """Runs an end-to-end test of the HPC workflow."""
    print("\n\033[1;35m--- Running HPC Workflow End-to-End Test ---\033[0m")

    # 1. CLEANUP
    print_step("Cleanup Previous Test Run")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    for f in LOG_DIR.glob(f"{DEBUG_CAMPAIGN_ID}*"):
        f.unlink()
    print_success("Cleanup complete.")

    # 2. LAUNCH
    print_step("Launch Debug Job")
    cmd = ["python3", str(MANAGE_PY), "launch", DEBUG_EXP_NAME, "--yes"]
    result = run_cmd(cmd)
    try:
        job_id = result.stdout.strip().split(" ")[-1]
        if not job_id.isdigit():
            raise ValueError
        print_success(f"Job submitted successfully with ID: {job_id}")
    except (ValueError, IndexError):
        print_fail(f"Could not parse Job ID from manage.py output:\n{result.stdout}")

    # In debug_workflow.py

    # 3. WAIT: Poll squeue until the job is finished
    print_step(f"Wait for Job {job_id} to Complete")
    timeout_seconds = 300  # 5 minutes
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            run_cmd(["scancel", job_id])
            print_fail(
                "Timeout exceeded while waiting for job to finish. Job cancelled."
            )

        # Check job status using sacct for more robust information
        result = run_cmd(["sacct", "-j", job_id, "-n", "-o", "State"])
        status = result.stdout.strip().split("\n")[0].strip()

        if "COMPLETED" in status:
            print_success("Job completed successfully.")
            break
        if "FAILED" in status or "CANCELLED" in status or "TIMEOUT" in status:
            print_fail(
                f"Job {job_id} did not complete successfully. Final state: {status}"
            )

        print_info(f"Job status is '{status}', waiting 10 seconds...")
        time.sleep(10)

    # 4. VERIFY RAW OUTPUT
    print_step("Verify Raw Data Generation")
    try:
        chunk_files = list(RAW_DIR.glob("chunk_*.jsonl"))
        if not chunk_files:
            print_fail("No chunk file was created in the raw directory.")
        chunk_file = chunk_files[0]
        with open(chunk_file, "r") as f:
            lines = f.readlines()
        if len(lines) != 1:
            print_fail(f"Chunk file should have 1 line, but has {len(lines)}.")
        task_data = json.loads(lines[0])
        task_id = task_data.get("task_id")
        if not task_id:
            print_fail("Result in chunk file is missing a task_id.")
        print_success(f"Raw chunk file '{chunk_file.name}' is valid.")
    except Exception as e:
        print_fail(f"Failed to verify raw output file: {e}")

    # 5. CONSOLIDATE
    print_step("Test Data Consolidation")
    run_cmd(["python3", str(MANAGE_PY), "consolidate", DEBUG_CAMPAIGN_ID])

    # 6. VERIFY FINAL SUMMARY
    print_step("Verify Final Summary File")
    summary_file = ANALYSIS_DIR / f"{DEBUG_CAMPAIGN_ID}_summary_aggregated.csv"
    if not summary_file.exists():
        print_fail("Final summary CSV file was not created.")
    df = pd.read_csv(summary_file)
    if len(df) != 1:
        print_fail(f"Summary CSV should have 1 row, but has {len(df)}.")
    if df.iloc[0]["task_id"] != task_id:
        print_fail("Task ID in summary CSV does not match.")
    print_success("Final summary CSV is correct.")

    print("\n\033[1;32m✅✅✅ Workflow test PASSED! ✅✅✅\033[0m")


if __name__ == "__main__":
    main()
