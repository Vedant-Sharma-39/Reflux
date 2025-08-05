# FILE: src/data_utils.py
# [v3 - CONSOLIDATION ENGINE]
# Implements a robust data consolidation pipeline. Raw chunk files are
# appended to master databases and then removed to manage file counts.

import pandas as pd
import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import gzip


def consolidate_raw_data(campaign_id: str, project_root: str):
    """
    Finds all raw chunk files, appends them to master databases,
    and then deletes the raw files.
    """
    campaign_dir = os.path.join(project_root, "data", campaign_id)
    raw_results_dir = os.path.join(campaign_dir, "results_raw")
    raw_timeseries_dir = os.path.join(campaign_dir, "timeseries_raw")
    analysis_dir = os.path.join(campaign_dir, "analysis")

    os.makedirs(analysis_dir, exist_ok=True)

    master_summary_path = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )
    master_ts_db_path = os.path.join(
        analysis_dir, f"{campaign_id}_timeseries_db.jsonl.gz"
    )

    # --- Consolidate Summary Files ---
    if os.path.isdir(raw_results_dir):
        summary_chunks = [
            f for f in os.listdir(raw_results_dir) if f.endswith(".jsonl")
        ]
        if summary_chunks:
            print(f"Consolidating {len(summary_chunks)} summary chunk files...")
            summary_data = []
            for chunk_file in tqdm(summary_chunks, desc="Processing summary chunks"):
                chunk_path = os.path.join(raw_results_dir, chunk_file)
                with open(chunk_path, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "error" not in data:
                                summary_data.append(data)
                        except json.JSONDecodeError:
                            continue

            if summary_data:
                new_df = pd.DataFrame(summary_data)
                if os.path.exists(master_summary_path):
                    # Append to existing master file
                    new_df.to_csv(
                        master_summary_path, mode="a", header=False, index=False
                    )
                else:
                    # Create new master file
                    new_df.to_csv(master_summary_path, index=False)

            # Delete consolidated raw files
            for chunk_file in summary_chunks:
                os.remove(os.path.join(raw_results_dir, chunk_file))
            print("Summary consolidation complete.")

    # --- Consolidate Timeseries Files ---
    if os.path.isdir(raw_timeseries_dir):
        ts_chunks = [
            f for f in os.listdir(raw_timeseries_dir) if f.endswith(".jsonl.gz")
        ]
        if ts_chunks:
            print(f"Consolidating {len(ts_chunks)} timeseries chunk files...")
            with gzip.open(
                master_ts_db_path, "at"
            ) as master_file:  # Append in text mode
                for chunk_file in tqdm(ts_chunks, desc="Processing timeseries chunks"):
                    chunk_path = os.path.join(raw_timeseries_dir, chunk_file)
                    with gzip.open(chunk_path, "rt") as chunk:
                        for line in chunk:
                            # Verify it's valid JSON before writing to prevent corruption
                            try:
                                json.loads(line)
                                master_file.write(line)
                            except json.JSONDecodeError:
                                continue
                    os.remove(chunk_path)
            print("Timeseries consolidation complete.")


def aggregate_data_cached(
    campaign_id: str, project_root: str, force_reaggregate: bool = False
):
    """
    A robust, reusable function to load the master aggregated CSV.
    If the file doesn't exist, it triggers the consolidation process.
    """
    analysis_dir = os.path.join(project_root, "data", campaign_id, "analysis")
    master_summary_path = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )

    # Always check for and consolidate any new raw data first.
    consolidate_raw_data(campaign_id, project_root)

    if os.path.exists(master_summary_path):
        print(f"Loading master aggregated data from: {master_summary_path}")
        return pd.read_csv(master_summary_path)
    else:
        print(
            "No master aggregated file found after consolidation. No data to analyze."
        )
        return pd.DataFrame()
