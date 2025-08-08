# FILE: manage.py (Corrected Consolidation Logic)
# The single, unified command-line interface for the Reflux project.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import pandas as pd
from tqdm import tqdm

# --- Add project root to path to allow imports from src ---
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from scripts.utils.generate_tasks import main as generate_tasks_main

app = typer.Typer(
    name="reflux-manager",
    help="A unified CLI to manage Reflux simulation campaigns.",
    add_completion=False,
    no_args_is_help=True,
)

# --- Valid choices for interactive prompts ---
VALID_EXP_NAMES = list(EXPERIMENTS.keys())
FIG_MAP = {
    "fig1": (
        "scripts/paper_figures/fig1_boundary_analysis.py",
        ["fig1_boundary_analysis", "fig1_kpz_scaling"],
    ),
    "fig2": ("scripts/paper_figures/fig2_phase_diagram.py", ["fig2_phase_diagram"]),
    "fig3": ("scripts/paper_figures/fig3_bet_hedging.py", ["fig3_bet_hedging_final"]),
    "fig4": ("scripts/paper_figures/fig4_relaxation_dynamics.py", ["fig4_relaxation"]),
}
VALID_FIG_NAMES = list(FIG_MAP.keys())


# ==============================================================================
# [UNCHANGED] HPC & CAMPAIGN MANAGEMENT COMMANDS (launch, clean)
# ... (The code for launch and clean from the previous step is correct and goes here)
# ==============================================================================
@app.command()
def launch(
    experiment_name: Optional[str] = typer.Argument(
        None, help="Experiment to launch, e.g., 'phase_diagram'"
    ),
    force_regenerate_tasks: bool = typer.Option(
        False, "--force-regen", "-r", help="Force regeneration of the master task list."
    ),
):
    """Generates tasks, checks status, and submits missing jobs to the HPC."""
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

    data_dir = PROJECT_ROOT / "data" / campaign_id
    data_dir.mkdir(parents=True, exist_ok=True)
    master_task_file = data_dir / f"{campaign_id}_master_tasks.jsonl"

    if force_regenerate_tasks and master_task_file.exists():
        master_task_file.unlink()
        typer.secho("Forced regeneration of task list.", fg=typer.colors.YELLOW)

    if not master_task_file.exists():
        typer.echo("Master task file not found. Generating...")
        _generate_tasks_wrapper(experiment_name)

    total_tasks = sum(1 for _ in open(master_task_file))

    raw_data_dir = data_dir / "raw"
    raw_data_dir.mkdir(exist_ok=True)
    completed_ids = {f.split(".")[0] for f in os.listdir(raw_data_dir)}

    missing_tasks_file = data_dir / f"{campaign_id}_missing_tasks.jsonl"
    missing_count = 0
    with open(master_task_file) as fin, open(missing_tasks_file, "w") as fout:
        for line in fin:
            if json.loads(line).get("task_id") not in completed_ids:
                fout.write(line)
                missing_count += 1

    completed_jobs = total_tasks - missing_count
    percentage = (completed_jobs / total_tasks * 100) if total_tasks > 0 else 0
    typer.echo("--- Campaign Status ---")
    typer.secho(f"  Completed: {completed_jobs}", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"  Missing:   {missing_count}", fg=typer.colors.YELLOW, bold=True)
    typer.secho(f"  Total:     {total_tasks} ({percentage:.2f}%)", bold=True)

    if missing_count == 0:
        typer.secho("\n🎉 All tasks are complete.", fg=typer.colors.GREEN)
        raise typer.Exit()

    if not typer.confirm(f"\nSubmit job array for the {missing_count} missing tasks?"):
        raise typer.Abort()

    hpc = config.get("hpc_params", {})
    sims_per_chunk = hpc.get("sims_per_task", 50)
    num_array_tasks = (missing_count + sims_per_chunk - 1) // sims_per_chunk

    log_dir = PROJECT_ROOT / "slurm_logs"
    log_dir.mkdir(exist_ok=True)

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        campaign_id,
        "--array",
        f"1-{num_array_tasks}%500",
        "--output",
        str(log_dir / f"{campaign_id}_%A_%a.log"),
        "--mem",
        hpc.get("mem", "2G"),
        "--time",
        hpc.get("time", "01:00:00"),
        str(PROJECT_ROOT / "scripts" / "run_chunk.sh"),
        str(PROJECT_ROOT),
        str(log_dir),
        str(missing_tasks_file),
        str(raw_data_dir),
        str(sims_per_chunk),
    ]

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.secho(f"sbatch submission failed:\n{result.stderr}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(
        f"✅ Job submitted successfully! ID: {result.stdout.strip()}",
        fg=typer.colors.GREEN,
    )


@app.command()
def clean(
    experiment_name: Optional[str] = typer.Argument(None, help="Experiment to clean.")
):
    """Deletes all data and logs for a given experiment campaign."""
    if not experiment_name:
        typer.echo("Please choose an experiment to clean:")
        experiment_name = _interactive_select(VALID_EXP_NAMES)

    campaign_id = EXPERIMENTS[experiment_name]["campaign_id"]
    data_dir = PROJECT_ROOT / "data" / campaign_id
    log_dir_campaign = PROJECT_ROOT / "slurm_logs" / campaign_id

    typer.secho(
        f"This will DELETE the following directories for '{campaign_id}':",
        fg=typer.colors.RED,
        bold=True,
    )
    typer.echo(f"  - Data: {data_dir}")
    if log_dir_campaign.exists():
        typer.echo(f"  - Logs: {log_dir_campaign}")

    if not typer.confirm("\nAre you absolutely sure?"):
        raise typer.Abort()

    if data_dir.exists():
        shutil.rmtree(data_dir)
        typer.echo(f"Removed {data_dir}")
    if log_dir_campaign.exists():
        shutil.rmtree(log_dir_campaign)
        typer.echo(f"Removed {log_dir_campaign}")
    typer.secho("✨ Cleanup complete.", fg=typer.colors.GREEN)


# ==============================================================================
# [UNCHANGED] DATA PROCESSING & PLOTTING (plot_figure)
# ... (The code for plot_figure from the previous step is correct and goes here)
# ==============================================================================
@app.command(name="plot")
def plot_figure(
    figure_name: Optional[str] = typer.Argument(
        None, help="Figure to generate ('fig1', 'fig2', 'all', etc.)."
    )
):
    """Generates a paper-ready figure, consolidating data first if necessary."""
    if not figure_name:
        typer.echo("Please choose a figure to generate:")
        # Add 'all' as an option for the interactive menu
        figure_name = _interactive_select(VALID_FIG_NAMES + ["all"])

    figures_to_run = VALID_FIG_NAMES if figure_name == "all" else [figure_name]

    for fig in figures_to_run:
        if fig not in FIG_MAP:
            typer.secho(
                f"Unknown figure '{fig}'. Choices: {VALID_FIG_NAMES}",
                fg=typer.colors.RED,
            )
            continue

        script_path, campaigns = FIG_MAP[fig]
        typer.secho(f"\n--- Generating {fig} ---", fg=typer.colors.BLUE, bold=True)

        # This is the line with the fixed logic
        for campaign in campaigns:
            _consolidate_and_ensure_file_exists(campaign)

        cmd = [sys.executable, str(PROJECT_ROOT / script_path)] + campaigns
        # We use check=False and manually check the return code to provide a better error message
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            typer.secho(
                f"Error generating {fig}. The plotting script failed.",
                fg=typer.colors.RED,
            )
            typer.echo("--- STDOUT from plotting script ---")
            typer.echo(result.stdout)
            typer.echo("--- STDERR from plotting script ---")
            typer.echo(result.stderr)
            typer.secho("Aborting `plot all`.", fg=typer.colors.RED)
            raise typer.Exit(1)

    typer.secho("\n✅ All requested figures generated.", fg=typer.colors.GREEN)


# ==============================================================================
# [UNCHANGED] DEBUGGING TOOLS (debug)
# ... (The code for debug from the previous step is correct and goes here)
# ==============================================================================
@app.command()
def debug(
    experiment_name: Optional[str] = typer.Argument(None, help="Experiment name."),
    line: Optional[int] = typer.Option(
        None, "--line", "-l", help="Line number from the master task file to debug."
    ),
):
    """Runs a single task from a campaign locally for in-depth debugging."""
    if not experiment_name:
        typer.echo("Please choose an experiment to debug:")
        experiment_name = _interactive_select(VALID_EXP_NAMES)

    if not line:
        line = typer.prompt("Which task line number do you want to debug?", type=int)

    campaign_id = EXPERIMENTS[experiment_name]["campaign_id"]
    master_task_file = (
        PROJECT_ROOT / "data" / campaign_id / f"{campaign_id}_master_tasks.jsonl"
    )

    if not master_task_file.exists():
        typer.secho("Master task file not found. Generating...", fg=typer.colors.YELLOW)
        _generate_tasks_wrapper(experiment_name)

    try:
        with open(master_task_file) as f:
            for i, l in enumerate(f):
                if i == line - 1:
                    params_json = l
                    break
            else:
                raise IndexError
    except IndexError:
        typer.secho(
            f"Error: Line {line} is out of bounds for the task file.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    params = json.loads(params_json)
    task_id = params["task_id"]

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
        typer.secho("\n✅ Success", fg=typer.colors.GREEN)
        typer.echo(f"Output file: {output_dir / (task_id + '.json')}")
    else:
        typer.secho(f"\n❌ FAILED", fg=typer.colors.RED)
        typer.echo(f"Error file: {output_dir / (task_id + '.error')}")
        typer.echo("--- Stderr ---")
        typer.echo(proc.stderr)


# ==============================================================================
# INTERNAL HELPER FUNCTIONS
# ==============================================================================


def _consolidate_and_ensure_file_exists(campaign_id: str):
    """
    FIXED: This function now guarantees the summary file exists, even if empty,
    to prevent downstream FileNotFoundError crashes.
    """
    raw_dir = PROJECT_ROOT / "data" / campaign_id / "raw"
    analysis_dir = PROJECT_ROOT / "data" / campaign_id / "analysis"
    summary_file = analysis_dir / f"{campaign_id}_summary_aggregated.csv"

    # Optimization: If summary file exists and is newer than all raw files, do nothing.
    if summary_file.exists() and raw_dir.exists():
        summary_mtime = summary_file.stat().st_mtime
        raw_files_mtimes = [p.stat().st_mtime for p in raw_dir.glob("*.json")]
        if raw_files_mtimes and max(raw_files_mtimes) < summary_mtime:
            typer.echo(
                f"Skipping consolidation for '{campaign_id}', summary file is up-to-date."
            )
            return

    typer.secho(f"Consolidating data for campaign: {campaign_id}", fg=typer.colors.BLUE)
    analysis_dir.mkdir(exist_ok=True)

    all_results = []
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.json"))
        if raw_files:
            all_results = [
                json.load(open(f))
                for f in tqdm(raw_files, desc=f"Reading {campaign_id}")
            ]

    # Create a DataFrame, which will be empty if all_results is empty.
    df = pd.DataFrame(all_results)

    # CRITICAL FIX: Save the DataFrame to CSV. This creates the file even if empty.
    df.to_csv(summary_file, index=False)

    if not all_results:
        typer.secho(
            f"Warning: No raw data found for '{campaign_id}'. Created an empty summary file.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho(
            f"Consolidated {len(df)} results into {summary_file}", fg=typer.colors.GREEN
        )


# [UNCHANGED] Other helpers (_interactive_select, _generate_tasks_wrapper)
def _interactive_select(options: list) -> str:
    """Displays a numbered list and returns the selected item."""
    for i, option in enumerate(options, 1):
        typer.echo(f"  [{i}] {option}")
    choice = typer.prompt("Your choice?", type=int)
    if 1 <= choice <= len(options):
        return options[choice - 1]
    typer.secho("Invalid choice.", fg=typer.colors.RED)
    raise typer.Abort()


def _generate_tasks_wrapper(experiment_name: str):
    original_argv = sys.argv
    sys.argv = ["generate_tasks.py", experiment_name]
    try:
        generate_tasks_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    app()
