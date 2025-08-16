# FILE: manage.py (Data-Centric Version)
# This script is a data-centric tool for managing simulation campaigns.
# All plotting is handled by standalone scripts in the 'scripts/paper_figures' directory.

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from tqdm import tqdm

# --- Add project root to path ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from scripts.utils.generate_tasks import generate_tasks_for_experiment

app = typer.Typer(
    name="reflux-manager",
    help="A data-centric CLI to manage Reflux simulation campaigns.",
    add_completion=False,
    no_args_is_help=True,
)

VALID_EXP_NAMES = list(EXPERIMENTS.keys())


class SbatchSubmissionError(Exception):
    pass


@app.command()
def launch(
    experiment_name: Optional[str] = typer.Argument(
        None, help="Experiment to launch, e.g., 'phase_diagram'"
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically answer yes to all confirmation prompts.",
    ),
):
    """Syncs config.py with the master task list and submits missing jobs, batching if necessary."""
    if not experiment_name:
        typer.echo("Please choose an experiment to launch:")
        experiment_name = _interactive_select(VALID_EXP_NAMES)
    if experiment_name not in VALID_EXP_NAMES:
        typer.secho(
            f"Error: Invalid experiment name '{experiment_name}'.", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    config = EXPERIMENTS[experiment_name]
    campaign_id = config["campaign_id"]
    typer.secho(
        f"🚀 Managing Campaign: {experiment_name} ({campaign_id})", fg=typer.colors.CYAN
    )
    newly_added_count = _update_master_task_list(experiment_name)
    if newly_added_count > 0:
        typer.secho(
            f"Found and added {newly_added_count} new tasks to the master list.",
            fg=typer.colors.GREEN,
        )
    else:
        typer.echo("Master task list is already up-to-date with config.py.")
    data_dir = PROJECT_ROOT / "data" / campaign_id
    master_task_file = data_dir / f"{campaign_id}_master_tasks.jsonl"
    raw_data_dir = data_dir / "raw"
    raw_data_dir.mkdir(exist_ok=True)
    summary_file = data_dir / "analysis" / f"{campaign_id}_summary_aggregated.csv"
    completed_ids = set()
    if summary_file.exists():
        try:
            df_summary = pd.read_csv(summary_file, low_memory=False)
            if "task_id" in df_summary.columns:
                completed_ids = set(df_summary["task_id"].astype(str))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass
    missing_tasks_file = data_dir / f"{campaign_id}_missing_tasks.jsonl"
    missing_count = 0
    with open(master_task_file) as fin, open(missing_tasks_file, "w") as fout:
        for line in fin:
            if json.loads(line).get("task_id") not in completed_ids:
                fout.write(line)
                missing_count += 1
    total_tasks = sum(1 for _ in open(master_task_file))
    completed_jobs = len(completed_ids)
    percentage = (completed_jobs / total_tasks * 100) if total_tasks > 0 else 0
    typer.echo("\n--- Campaign Status ---")
    typer.secho(f"  Completed: {completed_jobs}", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"  Missing:   {missing_count}", fg=typer.colors.YELLOW, bold=True)
    typer.secho(f"  Total:     {total_tasks} ({percentage:.2f}%)", bold=True)
    if missing_count == 0:
        typer.secho("\n🎉 All tasks are complete.", fg=typer.colors.GREEN)
        raise typer.Exit()
    hpc = config.get("hpc_params", {})
    sims_per_chunk = hpc.get("sims_per_task", 50)
    MAX_ARRAY_SIZE = int(os.environ.get("SLURM_MAX_ARRAY_SIZE", 500))
    total_array_tasks = math.ceil(missing_count / sims_per_chunk)
    num_batches = math.ceil(total_array_tasks / MAX_ARRAY_SIZE)
    prompt_msg = f"\nSubmit 1 job array with {total_array_tasks} tasks?"
    if num_batches > 1:
        prompt_msg = f"\nCampaign is too large. Submit in {num_batches} separate job array batches?"
    if not yes and not typer.confirm(prompt_msg):
        raise typer.Abort()
    try:
        for i in range(num_batches):
            start_idx, end_idx = i * MAX_ARRAY_SIZE + 1, min(
                (i + 1) * MAX_ARRAY_SIZE, total_array_tasks
            )
            array_range = f"{start_idx}-{end_idx}"
            if num_batches > 1:
                typer.echo(
                    f"\n--- Submitting Batch {i+1}/{num_batches} (Tasks {array_range}) ---"
                )
            job_name = f"{campaign_id}_b{i+1}" if num_batches > 1 else campaign_id
            _submit_sbatch_job(
                job_name, hpc, array_range, missing_tasks_file, raw_data_dir
            )
    except SbatchSubmissionError:
        typer.secho("\nFATAL: Job submission failed.", fg=typer.colors.RED, bold=True)
        raise typer.Exit(1)


@app.command()
def status(experiment_name: Optional[str] = typer.Argument(None)):
    """Checks the completion status of an experiment's data."""
    if not experiment_name:
        experiment_name = _interactive_select(VALID_EXP_NAMES)
    if experiment_name not in VALID_EXP_NAMES:
        typer.secho(f"Error: Invalid experiment name.", fg=typer.colors.RED)
        raise typer.Exit(1)
    config, campaign_id = (
        EXPERIMENTS[experiment_name],
        EXPERIMENTS[experiment_name]["campaign_id"],
    )
    typer.secho(
        f"📊 Status for Campaign: {experiment_name} ({campaign_id})",
        fg=typer.colors.CYAN,
    )
    data_dir, master_task_file = (
        PROJECT_ROOT / "data" / campaign_id,
        PROJECT_ROOT / "data" / campaign_id / f"{campaign_id}_master_tasks.jsonl",
    )
    raw_dir, summary_file = (
        data_dir / "raw",
        data_dir / "analysis" / f"{campaign_id}_summary_aggregated.csv",
    )
    if not master_task_file.exists():
        typer.secho(
            "  [!] Master task list not found. Run 'launch' first.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit()
    if raw_dir.exists() and any(raw_dir.iterdir()):
        typer.secho("\n[!] Found unconsolidated raw data.", fg=typer.colors.YELLOW)
        typer.echo(
            "      Run consolidation to update the status."
        )
        typer.secho(f"      python3 manage.py consolidate {campaign_id}", bold=True)
    completed_ids = set()
    if summary_file.exists():
        try:
            df_summary = pd.read_csv(summary_file, low_memory=False)
            if "task_id" in df_summary.columns:
                completed_ids = set(df_summary["task_id"].astype(str))
        except:
            pass
    total_tasks, completed_count = sum(1 for _ in open(master_task_file)), len(
        completed_ids
    )
    missing_count, percentage = total_tasks - completed_count, (
        (completed_count / total_tasks * 100) if total_tasks > 0 else 0
    )
    typer.echo("\n--- Campaign Progress ---")
    color = typer.colors.GREEN if percentage == 100 else typer.colors.YELLOW
    typer.secho(
        f"  Progress:  {completed_count} / {total_tasks} ({percentage:.2f}%)",
        fg=color,
        bold=True,
    )
    typer.secho(f"  Completed: {completed_count}", fg=typer.colors.GREEN)
    typer.secho(f"  Missing:   {missing_count}", fg=typer.colors.RED)
    if missing_count == 0:
        typer.secho("\n🎉 Campaign is 100% complete!", fg=typer.colors.GREEN, bold=True)
    else:
        typer.echo("\n   To run missing jobs, use 'launch':")
        typer.secho(f"   python3 manage.py launch {experiment_name}", bold=True)


@app.command()
def consolidate(
    campaign_id: Optional[str] = typer.Argument(
        None, help="Campaign ID to consolidate, e.g., 'fig2_phase_diagram'"
    )
):
    """Consolidates raw JSONL/JSON data into a final summary CSV."""
    if not campaign_id:
        typer.echo("Please choose a campaign to consolidate:")
        all_campaigns = sorted(
            list(set([v["campaign_id"] for v in EXPERIMENTS.values()]))
        )
        campaign_id = _interactive_select(all_campaigns)
    raw_dir = PROJECT_ROOT / "data" / campaign_id / "raw"
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        typer.secho(
            f"No raw data found for '{campaign_id}'. Nothing to consolidate.",
            fg=typer.colors.GREEN,
        )
        _consolidate_and_ensure_file_exists(campaign_id)
        return
    _consolidate_and_ensure_file_exists(campaign_id)
    typer.secho(
        f"✅ Consolidation for '{campaign_id}' is complete.", fg=typer.colors.GREEN
    )


def _submit_sbatch_job(job_name, hpc_params, array_range, task_file, raw_data_dir):
    log_dir = PROJECT_ROOT / "slurm_logs"
    log_dir.mkdir(exist_ok=True)
    sbatch_cmd = [
        "sbatch", "--parsable", "--job-name", job_name, "--array", array_range,
        "--output", str(log_dir / f"{job_name}_%A_%a.log"),
        "--mem", hpc_params.get("mem", "2G"),
        "--time", hpc_params.get("time", "01:00:00"),
    ]
    if hpc_params.get("partition"):
        sbatch_cmd.extend(["--partition", hpc_params["partition"]])
    if hpc_params.get("ntasks"):
        sbatch_cmd.extend(["--ntasks", str(hpc_params["ntasks"])])
    if hpc_params.get("cpus_per_task"):
        sbatch_cmd.extend(["--cpus-per-task", str(hpc_params["cpus_per_task"])])
    sbatch_cmd.extend([
        str(PROJECT_ROOT / "scripts" / "run_chunk.sh"), str(PROJECT_ROOT),
        str(log_dir), str(task_file), str(raw_data_dir),
        str(hpc_params.get("sims_per_task", 50)),
    ])
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.secho(
            f"sbatch command failed for batch '{job_name}':\n{result.stderr}",
            fg=typer.colors.RED,
        )
        raise SbatchSubmissionError()
    else:
        typer.secho(
            f"✅ Batch '{job_name}' submitted successfully! ID: {result.stdout.strip()}",
            fg=typer.colors.GREEN,
        )


@app.command()
def clean(experiment_name: Optional[str] = typer.Argument(None)):
    """Deletes all data and logs for a given experiment."""
    if not experiment_name:
        experiment_name = _interactive_select(VALID_EXP_NAMES)
    if experiment_name not in VALID_EXP_NAMES:
        typer.secho(f"Error: Invalid experiment name.", fg=typer.colors.RED)
        raise typer.Exit(1)
    campaign_id = EXPERIMENTS[experiment_name]["campaign_id"]
    data_dir = PROJECT_ROOT / "data" / campaign_id
    log_dir = PROJECT_ROOT / "slurm_logs"

    typer.secho(
        f"This will DELETE ALL data for '{campaign_id}' and matching logs:",
        fg=typer.colors.RED, bold=True
    )
    if data_dir.exists():
        typer.echo(f"  - {data_dir}")

    # Find logs associated with this campaign
    logs_to_delete = list(log_dir.glob(f"{campaign_id}*.log"))
    for log_file in logs_to_delete:
        typer.echo(f"  - {log_file}")
        
    if not typer.confirm("\nAre you absolutely sure? This cannot be undone."):
        raise typer.Abort()
        
    if data_dir.exists():
        shutil.rmtree(data_dir)
        typer.echo(f"Removed data directory.")
        
    for log_file in logs_to_delete:
        log_file.unlink()
    if logs_to_delete:
        typer.echo(f"Removed {len(logs_to_delete)} log file(s).")
        
    typer.secho("✨ Cleanup complete.", fg=typer.colors.GREEN)


@app.command()
def debug(
    experiment_name: Optional[str] = typer.Argument(None),
    line: Optional[int] = typer.Option(None, "--line", "-l", help="Line number from the master task list to run."),
):
    """Runs a single task from an experiment locally for debugging."""
    if not experiment_name:
        experiment_name = _interactive_select(VALID_EXP_NAMES)
    if not line:
        line = typer.prompt("Which task line number to debug?", type=int)
    campaign_id = EXPERIMENTS[experiment_name]["campaign_id"]
    master_task_file = (
        PROJECT_ROOT / "data" / campaign_id / f"{campaign_id}_master_tasks.jsonl"
    )
    if not master_task_file.exists():
        typer.echo("Master task file not found. Generating it now...")
        _update_master_task_list(experiment_name)
        
    try:
        with open(master_task_file) as f:
            params_json = f.readlines()[line - 1]
    except IndexError:
        typer.secho(f"Error: Line {line} is out of bounds for the task list.", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    params = json.loads(params_json)
    task_id = params["task_id"]
    typer.secho(f"--- 🕵️  Debugging Task {task_id} (Line {line}) ---", bold=True)
    typer.echo(json.dumps(params, indent=2))
    
    output_dir = PROJECT_ROOT / "data" / campaign_id / "raw"
    output_dir.mkdir(exist_ok=True)
    
    worker_script = str(PROJECT_ROOT / "src" / "worker.py")
    cmd = [
        sys.executable, worker_script, "--params", params_json, "--output-dir", str(output_dir),
    ]
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    
    typer.echo("\n--- Running Worker ---")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if proc.returncode == 0:
        try:
            result_data = json.loads(proc.stdout)
            final_output_path = output_dir / f"task_{task_id}.json"
            with open(final_output_path, "w") as f:
                json.dump(result_data, f, indent=2)
            typer.secho("\n✅ Success", fg=typer.colors.GREEN)
            typer.echo(f"Output file created: {final_output_path}")
        except json.JSONDecodeError:
            typer.secho("\n❌ FAILED: Worker output was not valid JSON.", fg=typer.colors.RED)
            typer.echo(proc.stdout)
    else:
        typer.secho(f"\n❌ FAILED", fg=typer.colors.RED)
        typer.secho("--- Worker STDERR ---\n" + (proc.stderr or "No stderr output."), bold=True)


def _update_master_task_list(experiment_name: str) -> int:
    config, campaign_id = (
        EXPERIMENTS[experiment_name], EXPERIMENTS[experiment_name]["campaign_id"],
    )
    data_dir = PROJECT_ROOT / "data" / campaign_id
    data_dir.mkdir(parents=True, exist_ok=True)
    master_task_file = data_dir / f"{campaign_id}_master_tasks.jsonl"
    desired_tasks = generate_tasks_for_experiment(experiment_name)
    desired_task_map = {t["task_id"]: t for t in desired_tasks}
    existing_task_ids = set()
    if master_task_file.exists():
        with open(master_task_file, "r") as f:
            for line in f:
                try:
                    existing_task_ids.add(json.loads(line)["task_id"])
                except:
                    continue
    new_task_ids = set(desired_task_map.keys()) - existing_task_ids
    if not new_task_ids:
        return 0
    new_tasks_to_write = [desired_task_map[tid] for tid in new_task_ids]
    with open(master_task_file, "a") as f:
        for task in sorted(new_tasks_to_write, key=lambda t: t["task_id"]):
            f.write(json.dumps(task) + "\n")
    return len(new_tasks_to_write)


def _consolidate_and_ensure_file_exists(campaign_id: str):
    campaign_dir = PROJECT_ROOT / "data" / campaign_id
    raw_dir, analysis_dir = campaign_dir / "raw", campaign_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_file = analysis_dir / f"{campaign_id}_summary_aggregated.csv"
    
    all_raw_files = list(raw_dir.glob("chunk_*.jsonl")) + list(raw_dir.glob("task_*.json"))
    
    if not all_raw_files:
        if not summary_file.exists():
            pd.DataFrame().to_csv(summary_file, index=False)
        return

    typer.secho(f"Consolidating summary data for: {campaign_id}", fg=typer.colors.BLUE)
    new_results = []
    
    for raw_file in tqdm(all_raw_files, desc=f"Reading raw files"):
        with open(raw_file, "r") as f:
            if raw_file.suffix == ".jsonl":
                for line in f:
                    try:
                        data = json.loads(line)
                        if "error" not in data: new_results.append(data)
                    except json.JSONDecodeError: continue
            elif raw_file.suffix == ".json":
                try:
                    data = json.load(f)
                    if "error" not in data: new_results.append(data)
                except json.JSONDecodeError: continue

    if not new_results and not summary_file.exists():
        pd.DataFrame().to_csv(summary_file, index=False)
    
    if new_results:
        new_df = pd.DataFrame(new_results)
        if summary_file.exists():
            try:
                existing_df = pd.read_csv(summary_file, low_memory=False)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                combined_df = new_df
        else:
            combined_df = new_df
        final_df = combined_df.drop_duplicates(subset=["task_id"], keep="last")
        final_df.to_csv(summary_file, index=False)
        typer.secho(f"Updated summary: {len(final_df)} total unique results.", fg=typer.colors.GREEN)
        
    typer.echo("Cleaning up raw summary files...")
    for raw_file in all_raw_files:
        raw_file.unlink()

    for data_type in ["timeseries", "trajectories"]:
        # This logic is for HPC workers that write to a temp dir first.
        # It's good practice to keep it.
        raw_data_dir = campaign_dir / f"{data_type}_raw"
        final_data_dir = campaign_dir / data_type
        if raw_data_dir.exists():
            raw_files = list(raw_data_dir.glob("*.json.gz"))
            if raw_files:
                typer.secho(f"Moving {len(raw_files)} new {data_type} files...", fg=typer.colors.BLUE)
                final_data_dir.mkdir(exist_ok=True)
                for f in tqdm(raw_files, desc=f"Moving {data_type} data"):
                    shutil.move(str(f), str(final_data_dir / f.name))
                try: os.rmdir(raw_data_dir)
                except OSError: pass


def _interactive_select(options: list) -> str:
    for i, option in enumerate(options, 1):
        typer.echo(f"  [{i}] {option}")
    choice = typer.prompt("Your choice?", type=int)
    if 1 <= choice <= len(options):
        return options[choice - 1]
    typer.secho("Invalid choice.", fg=typer.colors.RED)
    raise typer.Abort()


if __name__ == "__main__":
    app()