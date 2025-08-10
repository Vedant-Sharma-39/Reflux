# FILE: src/io/data_loader.py (Corrected)
# This file provides a robust, parallel utility for consolidating raw simulation data.
# The original version was non-functional due to incorrect file handling. This
# version correctly processes the .jsonl files produced by HPC jobs.

import os
import sys
import pandas as pd
import json
import gzip
import multiprocessing
from functools import partial
from tqdm import tqdm
from typing import Optional, Dict, List

# --- Caching Mechanism ---
# A simple in-memory cache to store loaded DataFrames.
_cache: Dict[str, pd.DataFrame] = {}


def load_aggregated_data(
    campaign_id: str, project_root: str, force_reload: bool = False
) -> Optional[pd.DataFrame]:
    """
    Loads the master aggregated data for a given campaign with in-memory caching.
    This is the primary function plotting scripts should use to get data.

    Args:
        campaign_id: The ID of the campaign to load data for.
        project_root: The absolute path to the project's root directory.
        force_reload: If True, bypasses the cache and reloads data from disk.

    Returns:
        A pandas DataFrame with the aggregated data, or an empty DataFrame if not found.
    """
    cache_key = campaign_id
    if not force_reload and cache_key in _cache:
        # print(f"Loading cached data for campaign '{campaign_id}'.")
        return _cache[cache_key].copy()

    master_file_path = os.path.join(
        project_root,
        "data",
        campaign_id,
        "analysis",
        f"{campaign_id}_summary_aggregated.csv",
    )

    if not os.path.exists(master_file_path):
        print(
            f"Warning: Aggregated file not found for campaign '{campaign_id}' at {master_file_path}.",
            file=sys.stderr,
        )
        return pd.DataFrame()

    try:
        # print(f"Loading data from: {master_file_path}")
        df = pd.read_csv(master_file_path, low_memory=False)
        _cache[cache_key] = df  # Update cache
        return df.copy()
    except pd.errors.EmptyDataError:
        print(
            f"Warning: Master file for campaign '{campaign_id}' is empty.",
            file=sys.stderr,
        )
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        print(f"Error loading data for campaign '{campaign_id}': {e}", file=sys.stderr)
        return pd.DataFrame()


# --- Worker Function for Parallel Consolidation (Corrected) ---
def _process_chunk_file(
    file_path: str, timeseries_dir: str, bulky_columns: List[str]
) -> List[Dict]:
    """
    Processes a single raw JSONL chunk file.
    - Reads the file line-by-line.
    - Extracts specific bulky data (like timeseries) and saves it separately.
    - Returns a list of the lightweight summary data dictionaries from that file.
    """
    summaries = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "error" in data:
                        continue  # Skip failed tasks

                    task_id = data.get("task_id")
                    # Split out bulky data to keep the main CSV small
                    for col in bulky_columns:
                        bulky_data = data.pop(col, None)
                        if bulky_data and task_id:
                            out_path = os.path.join(
                                timeseries_dir, f"ts_{task_id}.json.gz"
                            )
                            with gzip.open(out_path, "wt", encoding="utf-8") as f_gz:
                                json.dump(bulky_data, f_gz)

                    summaries.append(data)
                except (json.JSONDecodeError, TypeError):
                    continue  # Skip malformed lines
        return summaries
    except IOError:
        return []


def consolidate_raw_data(campaign_id: str, project_root: str):
    """
    (Corrected) Scans the raw data directory for .jsonl files and uses a pool of
    parallel processes to consolidate summary results. It also moves raw timeseries
    files into their final location.
    """
    campaign_dir = os.path.join(project_root, "data", campaign_id)
    raw_dir = os.path.join(campaign_dir, "raw")
    analysis_dir = os.path.join(campaign_dir, "analysis")
    timeseries_final_dir = os.path.join(campaign_dir, "timeseries")  # Final destination

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(timeseries_final_dir, exist_ok=True)

    if not os.path.isdir(raw_dir):
        print(
            f"Info: Raw data directory not found at {raw_dir}. Nothing to consolidate.",
            file=sys.stderr,
        )
        return

    # Correctly look for .jsonl files produced by HPC jobs
    raw_files = [
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".jsonl")
    ]

    if not raw_files:
        print(
            f"Info: No raw .jsonl files found in {raw_dir} to process.", file=sys.stderr
        )
        return

    num_workers = max(1, os.cpu_count() - 2)
    print(
        f"--- Consolidating {len(raw_files)} files for '{campaign_id}' using {num_workers} workers ---",
        file=sys.stderr,
    )

    # This consolidation step handles timeseries splitting because the worker is inconsistent.
    BULKY_COLUMNS_TO_EXTRACT = ["timeseries"]
    # We will write the split-out timeseries directly to the final directory
    worker_func = partial(
        _process_chunk_file,
        timeseries_dir=timeseries_final_dir,
        bulky_columns=BULKY_COLUMNS_TO_EXTRACT,
    )

    all_summaries = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_func, raw_files),
            total=len(raw_files),
            file=sys.stderr,
            ncols=80,
        )
        for result_list in pbar:
            if result_list:
                all_summaries.extend(result_list)  # Correctly extend the list

    if not all_summaries:
        print(
            "Warning: No valid summary data was extracted from raw files.",
            file=sys.stderr,
        )
        return

    new_df = pd.DataFrame(all_summaries)
    summary_output_file = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )

    # Append to existing data if present
    if os.path.exists(summary_output_file):
        try:
            existing_df = pd.read_csv(summary_output_file, low_memory=False)
            new_df = pd.concat([existing_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass  # Existing file is empty, just use the new data

    # Remove duplicates, keeping the most recent run for each task
    final_df = new_df.drop_duplicates(subset=["task_id"], keep="last")
    final_df.to_csv(summary_output_file, index=False)

    print(f"\nConsolidation complete for {campaign_id}.", file=sys.stderr)
    print(
        f"  - Aggregated {len(final_df)} unique results into: {os.path.basename(summary_output_file)}",
        file=sys.stderr,
    )

    # Clean up the processed raw chunk files
    for f in raw_files:
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python3 {sys.argv[0]} <campaign_id> <project_root_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    campaign_id_arg = sys.argv[1]
    project_root_path = sys.argv[2]
    consolidate_raw_data(campaign_id_arg, project_root_path)
