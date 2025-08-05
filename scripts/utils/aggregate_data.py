# FILE: scripts/utils/aggregate_data.py

import sys
import os
import argparse
import pandas as pd
import json
from tqdm import tqdm
import gzip
import shutil


def get_chunk_files(directory: str, extension: str) -> list:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(extension)
    ]


def get_existing_task_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        return set()
    try:
        return set(
            pd.read_csv(filepath, usecols=["task_id"])["task_id"].astype(str).unique()
        )
    except (pd.errors.EmptyDataError, FileNotFoundError, ValueError, KeyError):
        return set()


def consolidate_summaries(raw_dir: str, master_path: str):
    print("--- Consolidating Summary Data ---")
    chunk_files = get_chunk_files(raw_dir, ".jsonl")
    if not chunk_files:
        print("No summary chunk files found. Skipping.")
        return
    existing_ids = get_existing_task_ids(master_path)
    print(f"Found {len(existing_ids)} existing task_ids in master summary file.")
    header_written = os.path.exists(master_path)
    new_records_added = 0
    for chunk_path in tqdm(chunk_files, desc="Processing summary chunks"):
        records_to_add = []
        try:
            with open(chunk_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if (
                            data.get("task_id") not in existing_ids
                            and "error" not in data
                        ):
                            records_to_add.append(data)
                            existing_ids.add(data["task_id"])
                    except json.JSONDecodeError:
                        continue
            if records_to_add:
                pd.DataFrame(records_to_add).to_csv(
                    master_path, mode="a", header=not header_written, index=False
                )
                header_written = True
                new_records_added += len(records_to_add)
            os.remove(chunk_path)
        except Exception as e:
            print(f"\nError processing {chunk_path}: {e}. Skipping.")
            continue
    print(f"Summary consolidation complete. Added {new_records_added} new records.")


def consolidate_timeseries(raw_dir: str, master_path: str):
    print("\n--- Consolidating Timeseries Data ---")
    chunk_files = get_chunk_files(raw_dir, ".jsonl.gz")
    if not chunk_files:
        print("No timeseries chunk files found. Skipping.")
        return
    temp_master_path = master_path + ".tmp"
    if os.path.exists(master_path):
        shutil.copy(master_path, temp_master_path)
    try:
        with gzip.open(temp_master_path, "at", encoding="utf-8") as master_file:
            for chunk_path in tqdm(chunk_files, desc="Processing timeseries chunks"):
                try:
                    with gzip.open(chunk_path, "rt", encoding="utf-8") as chunk:
                        for line in chunk:
                            json.loads(line)
                            master_file.write(line)
                    os.remove(chunk_path)
                except Exception as e:
                    print(f"\nError processing {chunk_path}: {e}. Skipping.")
                    continue
        os.replace(temp_master_path, master_path)
        print(f"Timeseries consolidation complete.")
    except Exception as e:
        print(f"\nFATAL error during timeseries consolidation: {e}")
        if os.path.exists(temp_master_path):
            os.remove(temp_master_path)


def main():
    parser = argparse.ArgumentParser(description="Consolidate raw chunk results.")
    parser.add_argument(
        "campaign_id", help="The campaign ID (directory name in data/)."
    )
    args = parser.parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    campaign_dir = os.path.join(project_root, "data", args.campaign_id)
    raw_results_dir = os.path.join(campaign_dir, "results_raw")
    raw_timeseries_dir = os.path.join(campaign_dir, "timeseries_raw")
    analysis_dir = os.path.join(campaign_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    master_summary_path = os.path.join(
        analysis_dir, f"{args.campaign_id}_summary_aggregated.csv"
    )
    master_ts_db_path = os.path.join(
        analysis_dir, f"{args.campaign_id}_timeseries_db.jsonl.gz"
    )
    consolidate_summaries(raw_results_dir, master_summary_path)
    consolidate_timeseries(raw_timeseries_dir, master_ts_db_path)
    print("\nAggregation finished.")


if __name__ == "__main__":
    main()
