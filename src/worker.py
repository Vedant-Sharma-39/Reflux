# FILE: src/worker.py
#
# The single, unified entry point for all simulation types. It parses
# parameters, instantiates the correct simulation model, runs it, and
# returns the results.

import sys
import os
import json
import argparse
import traceback
import pandas as pd
import numpy as np
from collections import deque

# Ensure the src directory is in the path
sys.path.insert(0, os.path.dirname(__file__))

from linear_model import GillespieSimulation
from fluctuating_model import FluctuatingGillespieSimulation
from metrics import calculate_interface_density
from config import EXPERIMENTS


def run_simulation(params: dict):
    """
    Main simulation runner. Instantiates the correct model based on run_mode,
    executes the simulation loop, and returns results.
    """
    run_mode = params.get("run_mode")

    try:
        # --- Model and Loop Selection ---
        if run_mode == "spatial_fluctuation_analysis":
            # This mode uses the more complex fluctuating environment model
            sim_params = params.copy()
            env_map_name = sim_params.pop("environment_map")

            # Find the correct experiment config to look up environment details
            # This is robust to running debug tasks from different experiment contexts
            campaign_id = next(
                (
                    exp["CAMPAIGN_ID"]
                    for exp in EXPERIMENTS.values()
                    if env_map_name in exp.get("PARAM_GRID", {})
                ),
                None,
            )
            if not campaign_id:
                raise ValueError(
                    f"Could not find experiment providing environment_map '{env_map_name}'"
                )

            exp_config = next(
                exp for exp in EXPERIMENTS.values() if exp["CAMPAIGN_ID"] == campaign_id
            )
            actual_env_map = exp_config["PARAM_GRID"][env_map_name]

            # Handle symmetric vs asymmetric patch definitions
            if "environment_patch_sequence" in sim_params:
                patch_seq_name = sim_params.pop("environment_patch_sequence")
                sim_params["environment_patch_sequence"] = exp_config["PARAM_GRID"][
                    patch_seq_name
                ]

            constructor_args = (
                FluctuatingGillespieSimulation.__init__.__code__.co_varnames
            )
            final_sim_params = {
                k: v for k, v in sim_params.items() if k in constructor_args
            }

            sim = FluctuatingGillespieSimulation(
                environment_map=actual_env_map, **final_sim_params
            )
            summary, timeseries = _run_fluctuating_loop(sim, params)

        elif run_mode in ["calibration", "diffusion"]:
            # These modes use the simpler linear model and run for a max number of steps
            sim = GillespieSimulation(
                width=params["width"],
                length=params["length"],
                b_m=params["b_m"],
                k_total=0.0,
                phi=0.0,
                initial_condition_type=params.get("initial_condition_type", "patch"),
                initial_mutant_patch_size=params.get("initial_mutant_patch_size", 0),
            )
            summary, timeseries = _run_max_steps_loop(sim, params)

        elif run_mode in [
            "structure_analysis",
            "timeseries_from_pure_state",
            "perturbation",
        ]:
            # These modes use the linear model but run for a max simulation time
            patch_size = "width" if run_mode == "timeseries_from_pure_state" else 0
            sim = GillespieSimulation(
                width=params["width"],
                length=params["length"],
                b_m=params["b_m"],
                k_total=params["k_total"],
                phi=params["phi"],
                initial_condition_type=params.get("initial_condition_type", "patch"),
                initial_mutant_patch_size=params.get(
                    "initial_mutant_patch_size", patch_size
                ),
            )
            summary, timeseries = _run_max_time_loop(sim, params)

        else:
            return {"error": f"Unknown run_mode: {run_mode}"}

        # --- Return combined results ---
        return {"summary": summary, "timeseries": timeseries}

    except Exception as e:
        return {
            "error": f"Exception in worker: {e}",
            "traceback": traceback.format_exc(),
            "params": params,
        }


# ==============================================================================
# SPECIALIZED SIMULATION LOOPS
# ==============================================================================


def _run_max_steps_loop(sim, params):
    """Loop for 'calibration' and 'diffusion' modes."""
    trajectory = []
    max_steps = params.get("max_steps", 100000)
    run_mode = params.get("run_mode")

    for _ in range(max_steps):
        did_step, boundary_hit = sim.step()
        if not did_step or boundary_hit:
            break

        if run_mode == "calibration":
            m_front_coords = [c.r for c in sim.m_front_cells]
            if not m_front_coords:
                break
            width = np.ptp(m_front_coords) + 1
            trajectory.append((sim.mean_front_position, width))
        elif run_mode == "diffusion":
            front_q = [c.q for c in {**sim.wt_front_cells, **sim.m_front_cells}]
            if len(front_q) < 2:
                break
            roughness_sq = np.var(front_q, ddof=1)
            trajectory.append((sim.mean_front_position, roughness_sq))

    summary_key = "trajectory" if run_mode == "calibration" else "roughness_trajectory"
    summary = {summary_key: trajectory}
    return summary, None


def _run_max_time_loop(sim, params):
    """Loop for modes running up to a total_run_time or max_sim_time."""
    run_mode = params.get("run_mode")
    total_time = params.get("total_run_time", params.get("max_sim_time", 2000.0))
    log_interval = params.get("sample_interval", params.get("log_interval", 10.0))

    timeseries = []
    samples = []
    last_log_time = -np.inf

    in_pulse = False
    original_k = sim.k_total_base

    while sim.time < total_time:
        if run_mode == "perturbation":
            pulse_start = params["pulse_start_time"]
            pulse_end = pulse_start + params["pulse_duration"]
            if not in_pulse and sim.time >= pulse_start:
                in_pulse = True
                sim.set_switching_rate(params["k_total_pulse"], sim.phi_base)
            elif in_pulse and sim.time >= pulse_end:
                in_pulse = False
                sim.set_switching_rate(original_k, sim.phi_base)

        did_step, boundary_hit = sim.step()
        if not did_step or boundary_hit:
            break

        if (sim.time - last_log_time) >= log_interval:
            last_log_time = sim.time
            num_m, num_wt = len(sim.m_front_cells), len(sim.wt_front_cells)
            total_front = num_m + num_wt
            rho_m = num_m / total_front if total_front > 0 else 0

            if run_mode in ["timeseries_from_pure_state", "perturbation"]:
                timeseries.append({"time": sim.time, "mutant_fraction": rho_m})

            elif run_mode == "structure_analysis" and sim.time >= params["warmup_time"]:
                if len(samples) < params["num_samples"]:
                    samples.append(
                        {
                            "rho_M": rho_m,
                            "interface_density": calculate_interface_density(sim),
                        }
                    )

    summary = {}
    if samples:
        df = pd.DataFrame(samples)
        summary = {
            "avg_rho_M": df["rho_M"].mean(),
            "avg_interface_density": df["interface_density"].mean(),
        }
    return summary, timeseries


def _run_fluctuating_loop(sim, params):
    """Loop for 'spatial_fluctuation_analysis' with cycle-aware convergence."""
    timeseries = []
    log_q_interval = params["log_q_interval"]
    last_log_q, last_q, last_time = -np.inf, 0.0, 0.0

    cycle_len = sum(w for _, w in sim.patch_sequence)
    min_q_for_check = params["convergence_min_cycles"] * cycle_len
    conv_window = params["convergence_window"]
    conv_threshold = params["convergence_threshold"]
    running_avg_history = deque(maxlen=conv_window)

    while sim.mean_front_position < sim.length:
        did_step, boundary_hit = sim.step()
        if not did_step or boundary_hit:
            break

        if (sim.mean_front_position - last_log_q) >= log_q_interval:
            last_log_q = sim.mean_front_position
            num_m, num_wt = len(sim.m_front_cells), len(sim.wt_front_cells)
            total_front = num_m + num_wt
            rho_m = num_m / total_front if total_front > 0 else 0
            speed = (
                (sim.mean_front_position - last_q) / (sim.time - last_time)
                if (sim.time - last_time) > 0
                else 0
            )
            timeseries.append(
                {
                    "mean_front_q": sim.mean_front_position,
                    "mutant_fraction": rho_m,
                    "front_speed": speed,
                }
            )
            last_q, last_time = sim.mean_front_position, sim.time

            if (
                sim.mean_front_position > min_q_for_check
                and len(timeseries) > conv_window
            ):
                current_running_avg = (
                    pd.Series([d["front_speed"] for d in timeseries])
                    .expanding()
                    .mean()
                    .iloc[-1]
                )
                running_avg_history.append(current_running_avg)
                if len(running_avg_history) == conv_window:
                    history_std = np.std(list(running_avg_history), ddof=1)
                    history_mean = np.mean(list(running_avg_history))
                    if (
                        history_mean > 1e-6
                        and (history_std / history_mean) < conv_threshold
                    ):
                        break

    df = pd.DataFrame(timeseries)
    warmup_q = params["warmup_cycles_for_stats"] * cycle_len
    stats_df = df[df["mean_front_q"] > warmup_q]

    summary = {
        "avg_front_speed": (
            stats_df["front_speed"].mean() if not stats_df.empty else np.nan
        ),
        "var_front_speed": (
            stats_df["front_speed"].var(ddof=1) if len(stats_df) > 1 else np.nan
        ),
        "avg_rho_M": (
            stats_df["mutant_fraction"].mean() if not stats_df.empty else np.nan
        ),
        "final_q": sim.mean_front_position,
    }
    return summary, timeseries


def main():
    """Parses command-line arguments and runs the simulation."""
    parser = argparse.ArgumentParser(description="Run a single simulation worker.")
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="JSON string of simulation parameters.",
    )
    args = parser.parse_args()

    params = json.loads(args.params)
    results = run_simulation(params)

    summary_output = results.get("summary", {})
    if "error" in results:
        summary_output["error"] = results["error"]
        summary_output["params"] = params

    # Enrich summary with key parameters for easier aggregation
    summary_output["task_id"] = params.get("task_id")
    for key in [
        "b_m",
        "k_total",
        "phi",
        "patch_width",
        "environment_patch_sequence",
        "replicate_id",
        "width",
    ]:
        if key in params:
            summary_output[key] = params[key]

    timeseries_output = {
        "task_id": params.get("task_id"),
        "timeseries": results.get("timeseries"),
    }

    print(json.dumps(summary_output, allow_nan=False))
    print("---WORKER_PAYLOAD_SEPARATOR---")
    print(json.dumps(timeseries_output, allow_nan=False))


if __name__ == "__main__":
    main()
