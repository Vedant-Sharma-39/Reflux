import os
import sys
import pandas as pd
from typing import Optional, Dict

# --- Caching Mechanism ---
# A simple in-memory cache to store loaded DataFrames.
_cache: Dict[str, pd.DataFrame] = {}

def load_aggregated_data(
    campaign_id: str, 
    project_root: str, 
    force_reload: bool = False
) -> Optional[pd.DataFrame]:
    """
    Loads the master aggregated data for a given campaign with in-memory caching.

    Args:
        campaign_id: The ID of the campaign to load data for.
        project_root: The absolute path to the project's root directory.
        force_reload: If True, bypasses the cache and reloads data from disk.

    Returns:
        A pandas DataFrame with the aggregated data, or None if the file doesn't exist.
    """
    cache_key = campaign_id
    if not force_reload and cache_key in _cache:
        print(f"Loading cached data for campaign '{campaign_id}'.")
        return _cache[cache_key].copy()

    master_file_path = os.path.join(
        project_root, "data", campaign_id, "analysis", "master_aggregated.csv"
    )

    if not os.path.exists(master_file_path):
        print(
            f"Error: Master aggregated file not found for campaign '{campaign_id}'.\n"
            f"Please run the consolidation script first:\n"
            f"  python scripts/utils/consolidate_data.py {campaign_id}",
            file=sys.stderr,
        )
        return None

    try:
        print(f"Loading data from: {master_file_path}")
        df = pd.read_csv(master_file_path)
        _cache[cache_key] = df  # Update cache
        return df.copy()
    except pd.errors.EmptyDataError:
        print(f"Warning: Master file for campaign '{campaign_id}' is empty.", file=sys.stderr)
        return pd.DataFrame() # Return an empty DataFrame
    except Exception as e:
        print(f"Error loading data for campaign '{campaign_id}': {e}", file=sys.stderr)
        return None

# The single, authoritative utility for consolidating and loading simulation data.
# This version uses multiprocessing and correctly preserves trajectory columns
# needed for specific analyses (like Figure 1) in the main summary file.

import pandas as pd
import os
import sys
import json
from tqdm import tqdm
import gzip
import multiprocessing
from functools import partial


# --- Worker Function for Parallel Processing ---
def _process_file(file_path: str, timeseries_dir: str, bulky_columns: list):
    """
    Processes a single raw JSON file.
    - Extracts specific bulky data (like timeseries) and saves it separately.
    - Returns the summary data dictionary.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        task_id = data.get("task_id")

        for col in bulky_columns:
            bulky_data = data.pop(col, None)
            if bulky_data and task_id and col == "timeseries":
                out_path = os.path.join(timeseries_dir, f"ts_{task_id}.json.gz")
                with gzip.open(out_path, "wt", encoding="utf-8") as f_gz:
                    json.dump(bulky_data, f_gz)

        return data
    except (json.JSONDecodeError, IOError, TypeError):
        return None


def consolidate_raw_data(campaign_id: str, project_root: str):
    """
    Scans the raw data directory and uses a pool of parallel processes to
    consolidate summary results and extract timeseries data.
    """
    campaign_dir = os.path.join(project_root, "data", campaign_id)
    raw_dir = os.path.join(campaign_dir, "raw")
    analysis_dir = os.path.join(campaign_dir, "analysis")
    timeseries_dir = os.path.join(campaign_dir, "timeseries")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(timeseries_dir, exist_ok=True)

    if not os.path.isdir(raw_dir):
        print(
            f"Info: Raw data directory not found at {raw_dir}. Nothing to consolidate.",
            file=sys.stderr,
        )
        return

    raw_files = [
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".json")
    ]

    if not raw_files:
        print(
            f"Info: No raw .json files found in {raw_dir} to process.", file=sys.stderr
        )
        open(os.path.join(analysis_dir, ".consolidated"), "a").close()
        return

    num_workers = max(1, os.cpu_count() - 2)
    print(
        f"--- Consolidating {len(raw_files)} files for '{campaign_id}' using {num_workers} parallel workers ---",
        file=sys.stderr,
    )

 
    BULKY_COLUMNS_TO_EXTRACT = ["timeseries"]

    worker_func = partial(
        _process_file,
        timeseries_dir=timeseries_dir,
        bulky_columns=BULKY_COLUMNS_TO_EXTRACT,
    )

    summary_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_func, raw_files),
            total=len(raw_files),
            file=sys.stderr,
            ncols=80,
        )
        for result in pbar:
            if result is not None:
                summary_results.append(result)

    if not summary_results:
        print(
            "Warning: No valid summary data was extracted from raw files.",
            file=sys.stderr,
        )
        return

    summary_df = pd.DataFrame(summary_results)
    summary_output_file = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )
    summary_df.to_csv(summary_output_file, index=False)

    open(os.path.join(analysis_dir, ".consolidated"), "a").close()

    print(f"\nConsolidation complete for {campaign_id}.", file=sys.stderr)
    print(
        f"  - Aggregated {len(summary_df)} summaries into: {summary_output_file}",
        file=sys.stderr,
    )


def aggregate_data_cached(campaign_id: str, project_root: str):
    analysis_dir = os.path.join(project_root, "data", campaign_id, "analysis")
    master_summary_path = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )

    if not os.path.exists(master_summary_path):
        sys.exit(
            f"Error: Aggregated file not found at {master_summary_path}. Please run './scripts/analyze.sh' first."
        )

    print(f"Loading master aggregated data for '{campaign_id}'...")
    return pd.read_csv(master_summary_path, low_memory=False)


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
