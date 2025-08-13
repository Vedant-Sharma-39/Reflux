# FILE: scripts/debug_boundary_analysis.py
# (Definitive Version) A simple, robust script that defaults to a safe, small-scale
# debug experiment to avoid memory issues on login nodes.

import argparse
import os
import sys
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

# --- Robust Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS
from src.core.model import GillespieSimulation


def get_params(campaign_id, task_id=None):
    """Gets parameters either from a specific task_id or from the default config."""
    if task_id:
        # Load a specific, potentially large task from a master file
        master_file = (
            PROJECT_ROOT / "data" / campaign_id / f"{campaign_id}_master_tasks.jsonl"
        )
        if not master_file.exists():
            raise FileNotFoundError(
                f"Master task file not found for campaign '{campaign_id}'"
            )
        with open(master_file) as f:
            for line in f:
                params = json.loads(line)
                if params.get("task_id") == task_id:
                    params["s"] = params.get("b_m", 1.0) - 1.0
                    return params
        raise ValueError(f"Task ID '{task_id}' not found.")
    else:
        # Construct a default, small-scale parameter set from config
        if campaign_id not in EXPERIMENTS:
            raise ValueError(
                f"Default experiment '{campaign_id}' not found in src/config.py."
            )
        exp_config = EXPERIMENTS[campaign_id]
        sim_set = next(iter(exp_config["sim_sets"].values()))
        params = sim_set.get("base_params", {}).copy()
        params["s"] = params.get("b_m", 1.0) - 1.0
        params["campaign_id"] = campaign_id
        params["task_id"] = f"default_{campaign_id}"
        return params


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a simulation run. Defaults to a small debug experiment."
    )
    parser.add_argument(
        "campaign_id",
        default="debug_boundary_viz",
        nargs="?",
        help="Campaign to run (default: debug_boundary_viz)",
    )
    parser.add_argument(
        "task_id",
        default=None,
        nargs="?",
        help="Optional: specific task_id to visualize (can be memory-intensive).",
    )
    args = parser.parse_args()

    try:
        params = get_params(args.campaign_id, args.task_id)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Setup for Visualization ---
    task_id = params["task_id"]
    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    snapshot_dir = output_dir / task_id
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)

    params_viz = params.copy()
    params_viz["run_mode"] = "debug_viz"
    params_viz["output_dir_viz"] = str(output_dir)
    params_viz["snapshot_time_interval"] = (
        params_viz.get("max_steps", 50000) / 5000
    )  # Aim for ~10-20 frames

    print("\n--- Running visualization with parameters: ---")
    print(json.dumps(params, indent=2))

    sim = GillespieSimulation(**params_viz)
    trajectory = []
    max_steps = params.get("max_steps", 50000)

    with tqdm(total=max_steps, desc="Simulating") as pbar:
        while sim.step_count < max_steps:
            active, boundary_hit = sim.step()
            trajectory.append((sim.time, sim.mutant_sector_width))
            pbar.update(1)
            if not active or boundary_hit or sim.mutant_sector_width == 0:
                pbar.n = sim.step_count
                pbar.total = sim.step_count
                break

    # --- Plotting and GIF Creation ---
    trajectory = np.array(trajectory)
    plt.style.use("seaborn-v0_8-talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="navy", lw=2.5)
    ax.set_title(f'Sector Width Trajectory (s={params["s"]:.2f})')
    ax.set_xlabel("Time")
    ax.set_ylabel("Sector Width")
    ax.grid(True, linestyle=":")
    ax.set_ylim(bottom=0)
    sns.despine(fig=fig)
    plot_path = output_dir / f"{task_id}_trajectory.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Trajectory plot saved to: {plot_path}")

    snapshot_files = sorted(snapshot_dir.glob("*.png"))
    if snapshot_files:
        gif_path = output_dir / f"{task_id}_evolution.gif"
        with imageio.get_writer(gif_path, mode="I", duration=200, loop=0) as writer:
            for filename in tqdm(snapshot_files, desc="Assembling GIF"):
                writer.append_data(imageio.imread(filename))
        print(f"✅ Animation saved to: {gif_path}")


if __name__ == "__main__":
    main()
