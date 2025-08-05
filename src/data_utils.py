# FILE: src/data_utils.py
# [v4 - CORRECTED FOR PICKLING]
# The read_json_worker is reverted to a simple function to avoid
# multiprocessing errors with generator objects.

import pandas as pd
import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import gzip


# --- ### THE FIX: Revert to a simple, non-generator function ### ---
def read_json_worker(filepath: str):
    """A simple worker function to read and parse a single JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, UnicodeDecodeError):
        return None


# --- ### END FIX ### ---


def aggregate_data_cached(
    campaign_id: str,
    project_root: str,
    force_reaggregate: bool = False,
    cache_filename_suffix: str = "",
):
    """
    A robust, reusable function to aggregate JSON results into a cached CSV.
    This can be called from any analysis script.
    """
    analysis_dir = os.path.join(project_root, "data", campaign_id, "analysis")
    results_dir = os.path.join(project_root, "data", campaign_id, "results")

    if cache_filename_suffix and not cache_filename_suffix.startswith("_"):
        cache_filename_suffix = "_" + cache_filename_suffix
    cached_csv_path = os.path.join(
        analysis_dir, f"{campaign_id}{cache_filename_suffix}_aggregated.csv"
    )
    os.makedirs(analysis_dir, exist_ok=True)

    if not force_reaggregate and os.path.exists(cached_csv_path):
        print(f"Loading cached aggregated data from: {cached_csv_path}")
        return pd.read_csv(cached_csv_path, low_memory=False)

    print(f"Aggregating raw data for campaign: {campaign_id}")
    if not os.path.isdir(results_dir):
        print(f"Warning: Results directory not found at {results_dir}")
        return None

    filepaths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ]
    if not filepaths:
        print(f"Warning: No .json files found in {results_dir}")
        return pd.DataFrame()

    num_processes = max(1, cpu_count() - 2)

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(read_json_worker, filepaths),
                total=len(filepaths),
                desc="Aggregating JSONs",
            )
        )

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("Error: No valid JSON files were successfully parsed.")
        return pd.DataFrame()

    df = pd.DataFrame(valid_results)
    df.to_csv(cached_csv_path, index=False)
    print(f"\nAggregation complete. Final data saved to {cached_csv_path}")
    return df
