# FILE: scripts/run_bet_hedging_viz.py (Corrected for New Config)
import os
import sys
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import argparse
import imageio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS, PARAM_GRID
from src.core.model import GillespieSimulation


def generate_debug_title(
    sim: GillespieSimulation, snap_num: int, total_snaps: int
) -> str:
    """Creates a rich, multi-line title for visualizing bet-hedging dynamics."""

    line1 = f"Snapshot {snap_num:02d} / {total_snaps}"
    line2 = f"Step: {sim.step_count}, Time: {sim.time:.1f}"
    line3 = f"Params: k_total={sim.global_k_total}, φ={sim.global_phi}"

    num_patches = len(sim.patch_params)
    line4 = "Environment: Homogeneous"

    if num_patches == 1:
        patch0_b_wt, patch0_b_m = sim.patch_params[0]
        line4 = f"Homogeneous Patch: b_m={patch0_b_m:.1f} (b_wt={patch0_b_wt:.1f})"
    elif num_patches >= 2:
        patch0_b_wt, patch0_b_m = sim.patch_params[0]
        patch1_b_wt, patch1_b_m = sim.patch_params[1]
        line4 = (
            f"Patch 0 (Light BG): b_m={patch0_b_m:.1f} | "
            f"Patch 1 (Dark BG): b_m={patch1_b_m:.1f}"
        )

    return f"{line1}\n{line2}\n{line3}\n{line4}"


def main():
    parser = argparse.ArgumentParser(
        description="Run an enhanced visualization for bet-hedging."
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Type of patch environment to test: 'symmetric' or 'asymmetric'.",
    )
    args = parser.parse_args()

    print(
        f"--- Running Enhanced Bet-Hedging Visualization for '{args.env}' environment ---"
    )

    # --- FIX: Update ENV_CHOICES to use the correct names from the new config.py ---
    ENV_CHOICES = {
        "symmetric": "debug_viz_refuge",
        "asymmetric": "asymmetric_refuge_90_30w",
    }
    selected_env_name = ENV_CHOICES[args.env]
    # --- END FIX ---

    try:
        exp_config = EXPERIMENTS["debug_bet_hedging"]
        params = exp_config["sim_sets"]["main"]["base_params"].copy()

        # Override the environment with our selected one
        params["env_definition"] = selected_env_name

        params["run_mode"] = exp_config["run_mode"]
        params["campaign_id"] = exp_config["campaign_id"]
    except KeyError as e:
        print(f"Error: Could not find config keys. Details: {e}", file=sys.stderr)
        sys.exit(1)

    task_id = f"bet_hedging_viz_{args.env}"

    print("\nUsing parameters:")
    print(json.dumps(params, indent=2))

    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    snapshot_dir = output_dir / task_id

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    sim = GillespieSimulation(**params)

    if hasattr(sim, "max_snapshots"):
        sim.max_snapshots = 0
    max_steps = 150000
    num_snapshots = 10
    snapshot_interval = max_steps // num_snapshots

    if not sim.plotter:
        print("Error: Plotter was not initialized. Check run_mode in params.")
        sys.exit(1)

    with tqdm(total=max_steps, desc=f"Simulating ({args.env})") as pbar:
        title = generate_debug_title(sim, 0, num_snapshots)
        snapshot_path = snapshot_dir / "snap_00.png"
        sim.plotter.plot_population(
            sim.population,
            sim.mean_front_position,
            sim.width,
            title=title,
            q_to_patch_index=sim.q_to_patch_index,
        )
        sim.plotter.save_figure(snapshot_path)

        while sim.step_count < max_steps:
            active, boundary_hit = sim.step()
            pbar.update(1)
            if sim.step_count > 0 and sim.step_count % snapshot_interval == 0:
                snap_num = sim.step_count // snapshot_interval
                title = generate_debug_title(sim, snap_num, num_snapshots)
                snapshot_path = snapshot_dir / f"snap_{snap_num:02d}.png"
                sim.plotter.plot_population(
                    sim.population,
                    sim.mean_front_position,
                    sim.width,
                    title=title,
                    q_to_patch_index=sim.q_to_patch_index,
                )
                sim.plotter.save_figure(snapshot_path)
            if not active or boundary_hit:
                print("\nSimulation ended (extinction or boundary hit).")
                break

    sim.plotter.close()

    snapshot_files = sorted(snapshot_dir.glob("*.png"))
    if snapshot_files:
        gif_path = output_dir / f"{task_id}_animation.gif"
        print(f"\nAssembling {len(snapshot_files)} frames into GIF...")
        with imageio.get_writer(gif_path, mode="I", duration=200, loop=0) as writer:
            for filename in tqdm(snapshot_files, desc="Creating GIF"):
                writer.append_data(imageio.imread(filename))
        print(f"✅ Debug visualization saved to: {gif_path}")
    else:
        print("No snapshot frames were generated.")


if __name__ == "__main__":
    main()
