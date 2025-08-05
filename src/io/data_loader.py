# FILE: src/io/data_loader.py
# The single, authoritative utility for consolidating and loading simulation data.

import pandas as pd
import os
import json
from tqdm import tqdm
import gzip


def consolidate_raw_data(campaign_id: str, project_root: str):
    """
    Finds all raw chunk files from HPC jobs, appends them to master databases,
    and then deletes the raw files to prevent clutter and manage file counts.
    """
    campaign_dir = os.path.join(project_root, "data", campaign_id)
    raw_results_dir = os.path.join(campaign_dir, "results_raw")
    raw_timeseries_dir = os.path.join(campaign_dir, "timeseries_raw")
    analysis_dir = os.path.join(campaign_dir, "analysis")

    os.makedirs(analysis_dir, exist_ok=True)

    master_summary_path = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )
    master_ts_dir = os.path.join(campaign_dir, "timeseries")
    os.makedirs(master_ts_dir, exist_ok=True)

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
                os.remove(chunk_path)

            if summary_data:
                new_df = pd.DataFrame(summary_data)
                if os.path.exists(master_summary_path):
                    new_df.to_csv(
                        master_summary_path, mode="a", header=False, index=False
                    )
                else:
                    new_df.to_csv(master_summary_path, index=False)
            if os.path.exists(raw_results_dir):
                os.rmdir(raw_results_dir)
            print("Summary consolidation complete.")

    # --- Consolidate Timeseries Files ---
    if os.path.isdir(raw_timeseries_dir):
        ts_chunks = [
            f for f in os.listdir(raw_timeseries_dir) if f.endswith(".jsonl.gz")
        ]
        if ts_chunks:
            print(f"Consolidating {len(ts_chunks)} timeseries chunk files...")
            for chunk_file in tqdm(ts_chunks, desc="Processing timeseries chunks"):
                chunk_path = os.path.join(raw_timeseries_dir, chunk_file)
                with gzip.open(chunk_path, "rt") as chunk:
                    for line in chunk:
                        try:
                            data = json.loads(line)
                            task_id = data.get("task_id")
                            if task_id and data.get("timeseries"):
                                out_path = os.path.join(
                                    master_ts_dir, f"ts_{task_id}.json.gz"
                                )
                                with gzip.open(out_path, "wt") as f_out:
                                    json.dump(data["timeseries"], f_out)
                        except (json.JSONDecodeError, KeyError):
                            continue
                os.remove(chunk_path)
            if os.path.exists(raw_timeseries_dir):
                os.rmdir(raw_timeseries_dir)
            print("Timeseries consolidation complete.")


def aggregate_data_cached(
    campaign_id: str, project_root: str, force_reaggregate: bool = False
):
    """
    A robust, reusable function to load the master aggregated CSV.
    It automatically triggers the consolidation of any new raw data first.
    """
    analysis_dir = os.path.join(project_root, "data", campaign_id, "analysis")
    master_summary_path = os.path.join(
        analysis_dir, f"{campaign_id}_summary_aggregated.csv"
    )

    consolidate_raw_data(campaign_id, project_root)

    if os.path.exists(master_summary_path):
        print(f"Loading master aggregated data from: {master_summary_path}")
        return pd.read_csv(master_summary_path, low_memory=False)
    else:
        print(
            "No master aggregated file found after consolidation. No data to analyze."
        )
        return pd.DataFrame()
