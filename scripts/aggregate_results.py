# scripts/aggregate_results.py
# Aggregates individual JSON result files into a single CSV file.
# This version uses multiprocessing for speed and is linked to the central config.

import pandas as pd
import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Optional, but nice for showing progress
import sys

# --- Robust Path and Config Import ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(project_root, "src"))
    from config import CAMPAIGN_ID
except ImportError:
    print("FATAL: Could not import configuration from src/config.py.")
    print(
        "       Please ensure the file exists and the script is run from the project root."
    )
    sys.exit(1)
except NameError:  # Handles case where script is run from a different CWD
    project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
    sys.path.insert(0, os.path.join(project_root, "src"))
    from config import CAMPAIGN_ID


def read_json_file(filepath: str):
    """
    A simple worker function to read and parse a single JSON file.
    Includes error handling for corrupted files.
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        # Return None if the file is corrupted or can't be read
        print(
            f"Warning: Could not process file {os.path.basename(filepath)}. Error: {e}",
            file=sys.stderr,
        )
        return None


def main():
    print(f"--- Aggregating Results for Campaign: {CAMPAIGN_ID} ---")

    # --- 1. Setup Paths and Find Files ---
    # Paths are now dynamic based on CAMPAIGN_ID
    results_dir = os.path.join(project_root, "data", CAMPAIGN_ID, "results")
    output_csv = os.path.join(project_root, "data", f"{CAMPAIGN_ID}_aggregated.csv")

    if not os.path.isdir(results_dir):
        print(
            f"Error: Results directory not found at {results_dir}. Has the campaign been run?"
        )
        sys.exit(1)

    # Get the full path for every JSON file to be processed
    file_paths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]

    if not file_paths:
        print(f"Warning: No .json files found in {results_dir}. No data to aggregate.")
        sys.exit(0)

    # --- 2. Parallel Processing ---
    # Use most available CPU cores for maximum speed, leaving a few for system processes.
    num_processes = max(1, cpu_count() - 2)
    print(
        f"Aggregating {len(file_paths)} result files using {num_processes} processes..."
    )

    # The Pool distributes the list of file_paths to the read_json_file worker function
    with Pool(processes=num_processes) as pool:
        # Use tqdm to wrap the imap_unordered iterator for a nice progress bar
        results = list(
            tqdm(pool.imap_unordered(read_json_file, file_paths), total=len(file_paths))
        )

    # --- 3. Final Data Assembly and Saving ---
    # Filter out any 'None' results from failed file reads
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print(
            "Error: No valid JSON files were successfully parsed. Output CSV will not be created."
        )
        sys.exit(1)

    print(f"\nSuccessfully parsed {len(valid_results)} files. Creating DataFrame...")

    df = pd.DataFrame(valid_results)
    df.to_csv(output_csv, index=False)

    print(f"\nAggregation complete. Final data saved to {output_csv}")


if __name__ == "__main__":
    # It's good practice to set the start method for multiprocessing,
    # especially on Linux/macOS, though it's less critical for this simple task.
    # from multiprocessing import set_start_method
    # set_start_method("fork") # Or "spawn" for cross-platform compatibility
    try:
        main()
    except Exception as e:
        print(
            f"\nAn unexpected error occurred during aggregation: {e}", file=sys.stderr
        )
        import traceback

        traceback.print_exc()
        sys.exit(1)
