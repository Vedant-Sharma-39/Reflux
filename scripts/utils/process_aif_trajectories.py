# FILE: scripts/utils/process_aif_trajectories.py (Using Upgraded Analysis Engine)

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import json
import gzip
import subprocess
import argparse
import math

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.io.data_loader import load_aggregated_data
# --- NEW: Import the full analysis pipeline from our refactored script ---
from scripts.analyze_aif_sectors import run_full_analysis_pipeline

MAX_ARRAY_SIZE = 1000

# --- Core Worker Logic ---

def analyze_single_pop_file(task_info: dict) -> pd.DataFrame | None:
    """
    Worker function that runs on an HPC node.
    Loads a single population file and processes it using the definitive analysis pipeline.
    """
    pop_file_path = Path(task_info["pop_file_path"])
    if not pop_file_path.exists(): return None
    
    # 1. Load the raw population data ("fossil record")
    try:
        with gzip.open(pop_file_path, "rt", encoding="utf-8") as f:
            pop_data_list = json.load(f)
    except (json.JSONDecodeError, EOFError):
        # Handle cases where the file might be empty or corrupted
        return None

    # 2. Run the full, imported analysis pipeline
    df_trajectory = run_full_analysis_pipeline(pop_data_list)
    
    if df_trajectory.empty:
        # Create a placeholder row for extinctions
        df_trajectory = pd.DataFrame([{"mean_radius": 0, "arc_length": 0}])

    # 3. Attach parameters back to the results
    for key, val in task_info.items():
        if key not in ["pop_file_path", "task_id"]:
            df_trajectory[key] = val
            
    # Also attach the original task_id for joining later if needed
    df_trajectory['task_id'] = task_info['task_id']
    
    return df_trajectory

# --- Workflow Functions (Unchanged) ---
def prepare_hpc_submission(campaign_id: str):
    # ... (This function is correct and remains unchanged) ...
    print(f"--- 🚀 Preparing HPC Analysis for Campaign: {campaign_id} ---")
    df_summary = load_aggregated_data(campaign_id, str(PROJECT_ROOT))
    if df_summary.empty: sys.exit(f"Data for '{campaign_id}' is empty.")
    data_dir, pop_dir = PROJECT_ROOT / "data" / campaign_id, PROJECT_ROOT / "data" / campaign_id / "populations"
    analysis_dir, chunk_output_dir = data_dir / "analysis", data_dir / "analysis" / "temp_chunks"
    analysis_dir.mkdir(exist_ok=True); chunk_output_dir.mkdir(exist_ok=True)
    print(f"Scanning for existing population files in: {pop_dir}...")
    all_pop_files = list(pop_dir.glob("pop_*.json.gz"))
    existing_task_ids = {p.name.removeprefix('pop_').removesuffix('.json.gz') for p in tqdm(all_pop_files, desc="Parsing filenames")}
    if not existing_task_ids: sys.exit(f"No population files found in {pop_dir}.")
    print(f"Found {len(existing_task_ids)} population files.")
    print("Cross-referencing with summary data...")
    df_to_process = df_summary[df_summary['task_id'].isin(existing_task_ids)].copy()
    if df_to_process.empty: sys.exit("No matching tasks found in summary file.")
    df_to_process['pop_file_path'] = df_to_process['task_id'].apply(lambda tid: str(pop_dir / f"pop_{tid}.json.gz"))
    cols_for_tasklist = ["pop_file_path", "task_id", "b_res", "sector_width_initial", "replicate"]
    tasks_to_run = df_to_process[cols_for_tasklist].to_dict('records')
    num_tasks, num_batches = len(tasks_to_run), math.ceil(len(tasks_to_run) / MAX_ARRAY_SIZE)
    if num_batches > 1: print(f"\nAnalysis campaign is too large ({num_tasks} tasks). Submitting as {num_batches} separate batches.")
    log_dir = PROJECT_ROOT / "slurm_logs"; log_dir.mkdir(exist_ok=True)
    for i in range(num_batches):
        batch_start_idx, batch_end_idx = i * MAX_ARRAY_SIZE, min((i + 1) * MAX_ARRAY_SIZE, num_tasks)
        batch_tasks = tasks_to_run[batch_start_idx:batch_end_idx]
        batch_task_list_path = analysis_dir / f"{campaign_id}_analysis_tasks_b{i+1}.jsonl"
        with open(batch_task_list_path, 'w') as f:
            for task in batch_tasks:
                task_copy = task.copy(); task_copy['initial_width'] = task_copy.pop('sector_width_initial'); f.write(json.dumps(task_copy) + '\n')
        job_name, array_size = f"{campaign_id}_analysis_b{i+1}", len(batch_tasks)
        sbatch_cmd = ["sbatch", f"--job-name={job_name}", f"--array=1-{array_size}", f"--output={log_dir}/{job_name}_%A_%a.log", "--time=00:10:00", "--mem=1G", str(PROJECT_ROOT / "scripts/utils/run_analysis_worker.sh"), str(PROJECT_ROOT), str(batch_task_list_path), str(chunk_output_dir)]
        print(f"\n--- Submitting Batch {i+1}/{num_batches} (Tasks: {batch_start_idx+1} to {batch_end_idx}) ---")
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        if result.returncode != 0: print("\n❌ sbatch submission FAILED:", file=sys.stderr); print(result.stderr, file=sys.stderr)
        else: print(f"✅ Batch submitted successfully! Job ID: {result.stdout.strip()}")
    print("\n---\nNEXT STEPS:\n1. Wait for ALL HPC jobs to complete.\n2. Run to aggregate results:\n\n   python scripts/utils/process_aif_trajectories.py aggregate --campaign {campaign_id}\n")

def run_worker():
    # ... (This function is correct and remains unchanged) ...
    task_list_path, output_dir = Path(sys.argv[2]), Path(sys.argv[3])
    array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))
    with open(task_list_path, 'r') as f:
        line = f.readlines()[array_task_id - 1]
        task_info = json.loads(line)
    result_df = analyze_single_pop_file(task_info)
    if result_df is not None and not result_df.empty:
        output_file = output_dir / f"chunk_{task_info['task_id']}.csv.gz"
        result_df.to_csv(output_file, index=False, compression='gzip')

def aggregate_chunks(campaign_id: str):
    # ... (This function is correct and remains unchanged) ...
    print(f"--- 📉 Aggregating Processed Chunks for Campaign: {campaign_id} ---")
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    chunk_dir = analysis_dir / "temp_chunks"
    if not chunk_dir.exists(): sys.exit(f"Intermediate chunk directory not found: {chunk_dir}")
    chunk_files = list(chunk_dir.glob("*.csv.gz"))
    if not chunk_files: sys.exit("No processed chunk files found to aggregate.")
    print(f"Found {len(chunk_files)} chunk files to combine.")
    output_path = analysis_dir / "processed_spatial_trajectories.csv.gz"
    try:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
            first_file = chunk_files[0]
            with gzip.open(first_file, 'rt', encoding='utf-8') as f_in:
                for line in f_in: f_out.write(line)
            for file_path in tqdm(chunk_files[1:], desc="Aggregating chunks"):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f_in:
                    next(f_in)
                    for line in f_in: f_out.write(line)
    except Exception as e:
        print(f"\nAn error occurred during aggregation: {e}", file=sys.stderr); sys.exit(1)
    print(f"\n✅ Aggregation complete. Final data saved to: {output_path}")
    print("Cleaning up intermediate chunk files and task lists...")
    for f in chunk_files: f.unlink()
    for f in analysis_dir.glob("*_analysis_tasks_b*.jsonl"): f.unlink()
    try: chunk_dir.rmdir()
    except OSError: print(f"Warning: Could not remove non-empty chunk directory {chunk_dir}.")

def main():
    parser = argparse.ArgumentParser(description="HPC-powered AIF spatial trajectory processing.")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    prepare_parser = subparsers.add_parser('prepare', help="Prepare and submit analysis jobs to HPC.")
    prepare_parser.add_argument('--campaign', default=EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"], help="Campaign ID to process.")
    agg_parser = subparsers.add_parser('aggregate', help="Aggregate results from completed HPC jobs.")
    agg_parser.add_argument('--campaign', default=EXPERIMENTS["aif_definitive_spatial_scan"]["campaign_id"], help="Campaign ID to process.")
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare_hpc_submission(args.campaign)
    elif args.mode == 'aggregate':
        aggregate_chunks(args.campaign)
    else:
        parser.print_help()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        run_worker()
    else:
        main()