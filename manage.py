# FILE: manage.py (Definitive Final Version with Typer Fix)

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
    help="A unified CLI to manage Reflux simulation campaigns.",
    add_completion=False,
    no_args_is_help=True,
)

VALID_EXP_NAMES = list(EXPERIMENTS.keys())

FIG_MAP = {
    # --- Main Manuscript Figures ---
    "fig1": (
        "scripts/paper_figures/fig1_boundary_analysis.py",
        ["fig1_boundary_analysis", "fig1_kpz_scaling"],
    ),
    "fig2": (
        "scripts/paper_figures/fig2_keystone_analysis.py",
        ["fig2_phase_diagram", "fig3_bet_hedging_final"],
    ),
    "fig3": (
        "scripts/paper_figures/fig3_adaptation_analysis.py",
        ["fig5_asymmetric_patches", "fig3_bet_hedging_final"],
    ),
    "fig4": (
        "scripts/paper_figures/fig4_recovery_analysis.py",
        ["fig4_recovery_timescale"],
    ),
    # --- Supplementary Figures ---
    "sup_fig_phase_diagram": (
        "scripts/paper_figures/sup_fig_phase_diagram.py",
        ["fig2_phase_diagram"],
    ),
    "sup_fig_homogeneous_cost": (
        "scripts/paper_figures/sup_fig_homogeneous_cost.py",
        ["sup_homogeneous_cost"],
    ),
    "sup_fig_asymmetric_details": (
        "scripts/paper_figures/sup_fig_asymmetric_details.py",
        ["fig5_asymmetric_patches", "fig3_bet_hedging_final"],
    ),
    "sup_fig_relaxation_ts": (
        "scripts/paper_figures/sup_fig_relaxation_ts.py",
        ["fig4_relaxation"],
    ),
    "sup_fig_optimal_heatmap": (
        "scripts/paper_figures/sup_fig_optimal_strategy_heatmap.py",
        ["fig3_bet_hedging_final"],
    ),
}

VALID_FIG_NAMES = list(FIG_MAP.keys())


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
    if raw_dir.exists() and (
        any(raw_dir.glob("chunk_*.jsonl")) or any(raw_dir.glob("*.json"))
    ):
        typer.secho("\n[!] Found unconsolidated raw data.", fg=typer.colors.YELLOW)
        typer.echo(
            "      Run consolidation and wait for it to finish before checking status."
        )
        typer.secho(f"      python3 manage.py consolidate {campaign_id}", bold=True)
        raise typer.Exit()
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
    if not campaign_id:
        typer.echo("Please choose a campaign to consolidate:")
        all_campaigns = sorted(
            list(set([v["campaign_id"] for v in EXPERIMENTS.values()]))
        )
        campaign_id = _interactive_select(all_campaigns)
    raw_dir = PROJECT_ROOT / "data" / campaign_id / "raw"
    if not raw_dir.exists() or not (
        any(raw_dir.glob("chunk_*.jsonl")) or any(raw_dir.glob("*.json"))
    ):
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


@app.command(name="plot")
def plot_figure(figure_name: Optional[str] = typer.Argument(None)):
    if not figure_name:
        figure_name = _interactive_select(VALID_FIG_NAMES + ["all"])
    figures_to_run = VALID_FIG_NAMES if figure_name == "all" else [figure_name]
    for fig in figures_to_run:
        if fig not in FIG_MAP:
            continue
        script_path, campaigns = FIG_MAP[fig]
        typer.secho(f"\n--- Generating {fig} ---", fg=typer.colors.BLUE, bold=True)
        for campaign in campaigns:
            _consolidate_and_ensure_file_exists(campaign)
        cmd = [sys.executable, str(PROJECT_ROOT / script_path)] + campaigns
        result = subprocess.run(cmd)
        if result.returncode != 0:
            typer.secho(
                f"Error generating {fig}. Plotting script failed.", fg=typer.colors.RED
            )
            typer.secho(
                "--- STDERR from plotting script ---\n"
                + (result.stderr or "No stderr output."),
                bold=True,
            )
            raise typer.Exit(1)
    typer.secho("\n✅ All requested figures generated.", fg=typer.colors.GREEN)


def _submit_sbatch_job(job_name, hpc_params, array_range, task_file, raw_data_dir):
    log_dir = PROJECT_ROOT / "slurm_logs"
    log_dir.mkdir(exist_ok=True)
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--array",
        array_range,
        "--output",
        str(log_dir / f"{job_name}_%A_%a.log"),
        "--mem",
        hpc_params.get("mem", "2G"),
        "--time",
        hpc_params.get("time", "01:00:00"),
    ]
    if hpc_params.get("partition"):
        sbatch_cmd.extend(["--partition", hpc_params["partition"]])
    if hpc_params.get("ntasks"):
        sbatch_cmd.extend(["--ntasks", str(hpc_params["ntasks"])])
    if hpc_params.get("cpus_per_task"):
        sbatch_cmd.extend(["--cpus-per-task", str(hpc_params["cpus_per_task"])])
    sbatch_cmd.extend(
        [
            str(PROJECT_ROOT / "scripts" / "run_chunk.sh"),
            str(PROJECT_ROOT),
            str(log_dir),
            str(task_file),
            str(raw_data_dir),
            str(hpc_params.get("sims_per_task", 50)),
        ]
    )

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
    if not experiment_name:
        experiment_name = _interactive_select(VALID_EXP_NAMES)
    if experiment_name not in VALID_EXP_NAMES:
        typer.secho(f"Error: Invalid experiment name.", fg=typer.colors.RED)
        raise typer.Exit(1)
    campaign_id = EXPERIMENTS[experiment_name]["campaign_id"]
    data_dir, log_dir_campaign = (
        PROJECT_ROOT / "data" / campaign_id,
        PROJECT_ROOT / "slurm_logs" / campaign_id,
    )
    typer.secho(
        f"This will DELETE data and logs for '{campaign_id}':",
        fg=typer.colors.RED,
        bold=True,
    )
    if not typer.confirm("\nAre you absolutely sure?"):
        raise typer.Abort()
    if data_dir.exists():
        shutil.rmtree(data_dir)
        typer.echo(f"Removed {data_dir}")
    if log_dir_campaign.exists():
        shutil.rmtree(log_dir_campaign)
        typer.echo(f"Removed {log_dir_campaign}")
    typer.secho("✨ Cleanup complete.", fg=typer.colors.GREEN)


@app.command()
def debug(
    experiment_name: Optional[str] = typer.Argument(None),
    line: Optional[int] = typer.Option(None, "--line", "-l"),
):
    if not experiment_name:
        experiment_name = _interactive_select(VALID_EXP_NAMES)
    if not line:
        line = typer.prompt("Which task line number to debug?", type=int)
    campaign_id = EXPERIMENTS[experiment_name]["campaign_id"]
    master_task_file = (
        PROJECT_ROOT / "data" / campaign_id / f"{campaign_id}_master_tasks.jsonl"
    )
    if not master_task_file.exists():
        _update_master_task_list(experiment_name)
    try:
        with open(master_task_file) as f:
            params_json = f.readlines()[line - 1]
    except IndexError:
        typer.secho(f"Error: Line {line} is out of bounds.", fg=typer.colors.RED)
        raise typer.Exit(1)
    params, task_id = json.loads(params_json), json.loads(params_json)["task_id"]
    typer.secho(f"--- 🕵️  Debugging Task {task_id} (Line {line}) ---", bold=True)
    typer.echo(json.dumps(params, indent=2))
    output_dir = PROJECT_ROOT / "data" / campaign_id / "raw"
    output_dir.mkdir(exist_ok=True)
    worker_script = str(PROJECT_ROOT / "src" / "worker.py")
    cmd = [
        sys.executable,
        worker_script,
        "--params",
        params_json,
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    typer.echo("\n--- Running Worker ---")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        try:
            result_data, final_output_path = (
                json.loads(proc.stdout),
                output_dir / f"{task_id}.json",
            )
            with open(final_output_path, "w") as f:
                json.dump(result_data, f, indent=2)
            typer.secho("\n✅ Success", fg=typer.colors.GREEN)
            typer.echo(f"Output file created: {final_output_path}")
        except json.JSONDecodeError:
            typer.secho(
                "\n❌ FAILED: Worker output was not valid JSON.", fg=typer.colors.RED
            )
    else:
        typer.secho(f"\n❌ FAILED", fg=typer.colors.RED)
        typer.secho(
            "--- Worker STDERR ---\n" + (proc.stderr or "No stderr output."), bold=True
        )


def _update_master_task_list(experiment_name: str) -> int:
    config, campaign_id = (
        EXPERIMENTS[experiment_name],
        EXPERIMENTS[experiment_name]["campaign_id"],
    )
    data_dir = PROJECT_ROOT / "data" / campaign_id
    data_dir.mkdir(parents=True, exist_ok=True)
    master_task_file = data_dir / f"{campaign_id}_master_tasks.jsonl"
    desired_tasks, desired_task_map = generate_tasks_for_experiment(experiment_name), {}
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
    raw_chunk_files = list(raw_dir.glob("chunk_*.jsonl"))
    raw_debug_files = list(raw_dir.glob("*.json"))
    all_raw_files = raw_chunk_files + raw_debug_files
    if not all_raw_files:
        if not summary_file.exists():
            pd.DataFrame().to_csv(summary_file, index=False)
        return
    typer.secho(f"Consolidating summary data for: {campaign_id}", fg=typer.colors.BLUE)
    new_results = []
    for raw_file in tqdm(all_raw_files, desc=f"Reading raw files for {campaign_id}"):
        with open(raw_file, "r") as f:
            if raw_file.suffix == ".jsonl":
                for line in f:
                    try:
                        data = json.loads(line)
                        if "error" not in data:
                            new_results.append(data)
                    except json.JSONDecodeError:
                        continue
            elif raw_file.suffix == ".json":
                try:
                    data = json.load(f)
                    if "error" not in data:
                        new_results.append(data)
                except json.JSONDecodeError:
                    continue
    if not new_results:
        for raw_file in all_raw_files:
            raw_file.unlink()
        if not summary_file.exists():
            pd.DataFrame().to_csv(summary_file, index=False)
        return
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
    typer.secho(
        f"Updated summary: {len(final_df)} total unique results.", fg=typer.colors.GREEN
    )
    typer.echo("Cleaning up raw summary files...")
    for raw_file in all_raw_files:
        raw_file.unlink()
    for data_type in ["timeseries", "trajectories"]:
        raw_data_dir, final_data_dir = (
            campaign_dir / f"{data_type}_raw",
            campaign_dir / data_type,
        )
        if raw_data_dir.exists():
            raw_files = list(raw_data_dir.glob("*.json.gz"))
            if raw_files:
                typer.secho(
                    f"Moving {len(raw_files)} new {data_type} files...",
                    fg=typer.colors.BLUE,
                )
                final_data_dir.mkdir(exist_ok=True)
                for f in tqdm(raw_files, desc=f"Moving {data_type} data"):
                    shutil.move(str(f), str(final_data_dir / f.name))
                try:
                    os.rmdir(raw_data_dir)
                except OSError:
                    pass


def _is_job_running(job_name_prefix: str) -> bool:
    cmd = ["squeue", "-u", os.environ.get("USER"), "-h", "-o", "%j"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    running_jobs = result.stdout.strip().split("\n")
    for job in running_jobs:
        if job.strip().startswith(job_name_prefix):
            return True
    return False


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
