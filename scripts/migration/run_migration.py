# FILE: scripts/migration/run_migration.py (VERSION 9 - The Final, Polished Script)
#
# PURPOSE:
# This script intelligently migrates all old summary CSV files into a new, unified
# CSV format. It is aware of multiple legacy data formats, infers the correct
# new parameters from the data itself, and saves the output with the standard
# `_summary_aggregated.csv` filename for direct use in analysis.

import csv
import json
import os
import glob
from collections import defaultdict

# --- THE "INFERENCE ENGINE" ---
DECOMPOSITION_MAP = {
    "symmetric_30_strong": {
        "new_campaign_id": "fig3_controls",
        "env_definition": "symmetric_strong_scan_bm_30w",
    },
    "symmetric_60_strong": {
        "new_campaign_id": "fig3_bet_hedging_final",
        "env_definition": "symmetric_strong_scan_bm_60w",
    },
    "symmetric_120_strong": {
        "new_campaign_id": "fig3_controls",
        "env_definition": "symmetric_strong_scan_bm_120w",
    },
    "asymmetric_90_30_strong": {
        "new_campaign_id": "fig4_asymmetric_adaptation",
        "env_definition": "asymmetric_90_30_scan_bm",
    },
    "asymmetric_30_90_strong": {
        "new_campaign_id": "fig4_relaxation",
        "env_definition": "asymmetric_30_90_scan_bm",
    },
    "scrambled_60_60_strong": {
        "new_campaign_id": "fig4_relaxation",
        "env_definition": "scrambled_60_60_scan_bm",
    },
    "tracking_legacy": {
        "new_campaign_id": "fig5_timescales",
        "env_definition": "symmetric_strong_scan_bm_60w",
    },
}
JSON_NAME_TO_KEY_MAP = {
    "30_30_strong": "symmetric_30_strong",
    "30_30": "symmetric_30_strong",
    "60_60_strong": "symmetric_60_strong",
    "60_60": "symmetric_60_strong",
    "120_120_strong": "symmetric_120_strong",
    "120_120": "symmetric_120_strong",
    "90_30_strong": "asymmetric_90_30_strong",
    "90_30": "asymmetric_90_30_strong",
    "30_90_strong": "asymmetric_30_90_strong",
    "30_90": "asymmetric_30_90_strong",
    "scrambled_60_60_strong": "scrambled_60_60_strong",
    "scrambled_60_60": "scrambled_60_60_strong",
}
MASTER_HEADER = [
    "width",
    "length",
    "b_m",
    "phi",
    "k_total",
    "env_definition",
    "replicate",
    "task_id",
    "run_mode",
    "campaign_id",
    "avg_rho_M",
    "var_rho_M",
    "avg_front_speed",
    "var_front_speed",
    "avg_front_length",
    "avg_domain_boundary_length",
    "num_cycles_completed",
    "termination_reason",
    "migrated_from_campaign",
]


def convert_type(value_str: str):
    if not isinstance(value_str, str):
        return value_str
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return value_str


def get_decomposition_key(row: dict, old_campaign_id: str) -> str | None:
    if (
        "env_definition" in row
        and row["env_definition"]
        and row["env_definition"].strip().startswith("{")
    ):
        try:
            json_str = (
                row["env_definition"]
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
            )
            env_dict = json.loads(json_str)
            name = env_dict.get("name")
            return JSON_NAME_TO_KEY_MAP.get(name)
        except (json.JSONDecodeError, AttributeError):
            return None

    if "patch_width" in row and row["patch_width"]:
        try:
            pw = int(float(row["patch_width"]))
            if pw == 30:
                return "symmetric_30_strong"
            if pw == 60:
                return "symmetric_60_strong"
            if pw == 120:
                return "symmetric_120_strong"
        except (ValueError, TypeError):
            return None

    if old_campaign_id in ["fig3_bet_hedging_final", "bet_hedging_final"]:
        return "symmetric_60_strong"

    if (
        old_campaign_id in ["fig5_tracking", "recovery_timescale"]
        or "environment_map" in row
    ):
        return "tracking_legacy"
    return None


def handle_passthrough(row: dict, old_id: str, new_id: str):
    rec = {k.strip(): convert_type(v) for k, v in row.items()}
    rec["campaign_id"], rec["migrated_from_campaign"] = new_id, old_id
    return rec


def handle_decomposed_env(row: dict, old_id: str):
    key = get_decomposition_key(row, old_id)
    if not key:
        return None

    rule = DECOMPOSITION_MAP.get(key)
    if not rule:
        return None

    rec = {k.strip(): convert_type(v) for k, v in row.items()}
    rec.update(rule)
    rec["campaign_id"] = rule["new_campaign_id"]
    rec["migrated_from_campaign"] = old_id
    for old_key in ["env_definition", "environment_map", "patch_width"]:
        rec.pop(old_key, None)
    return rec


def handle_ignore(row: dict, old_id: str):
    return None


MIGRATION_MAP = {
    "fig1_boundary_analysis": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig1_boundary_analysis")
    },
    "fig1_kpz_scaling": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig1_kpz_scaling")
    },
    "fig2_phase_diagram": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig2_phase_diagram")
    },
    "fig5_relaxation": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig5_relaxation")
    },
    "boundary_analysis": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig1_boundary_analysis")
    },
    "kpz_scaling": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig1_kpz_scaling")
    },
    "phase_diagram": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig2_phase_diagram")
    },
    "relaxation_dynamics": {
        "handler": lambda r, c: handle_passthrough(r, c, "fig5_relaxation")
    },
    "fig3_bet_hedging_final": {"handler": handle_decomposed_env},
    "fig4_asymmetric_adaptation": {"handler": handle_decomposed_env},
    "fig5_tracking": {"handler": handle_decomposed_env},
    "bet_hedging_final": {"handler": handle_decomposed_env},
    "asymmetric_patches": {"handler": handle_decomposed_env},
    "recovery_timescale": {"handler": handle_decomposed_env},
    "fig3_controls": {"handler": handle_decomposed_env},
    "fig4_relaxation": {"handler": handle_decomposed_env},
    "debug_bet_hedging_viz": {"handler": handle_ignore},
    "debug_boundary_viz": {"handler": handle_ignore},
    "homogeneous_fitness_cost": {"handler": handle_ignore},
    "sup_homogeneous_cost": {"handler": handle_ignore},
}


def migrate_all_data(input_root: str, output_root: str):
    print(f"Starting migration from '{input_root}' to '{output_root}'...")
    os.makedirs(output_root, exist_ok=True)

    csv_writers = {}
    stats = defaultdict(lambda: defaultdict(int))
    search_pattern = os.path.join(
        input_root, "*", "analysis", "*_summary_aggregated.csv"
    )

    for csv_path in sorted(glob.glob(search_pattern)):
        old_campaign_id = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        print(f"\nProcessing file: {os.path.relpath(csv_path)}")

        mapping_rule = MIGRATION_MAP.get(old_campaign_id)
        if not mapping_rule:
            print(f"  -> SKIPPING: No migration rule for campaign '{old_campaign_id}'.")
            continue

        handler = mapping_rule["handler"]
        with open(csv_path, "r", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                stats[old_campaign_id]["rows_read"] += 1
                new_record = handler(row, old_campaign_id)

                if new_record:
                    new_campaign_id = new_record["campaign_id"]
                    if new_campaign_id not in csv_writers:
                        new_dir = os.path.join(output_root, new_campaign_id, "analysis")
                        os.makedirs(new_dir, exist_ok=True)
                        # --- FILENAME CHANGE IS HERE ---
                        output_path = os.path.join(
                            new_dir, f"{new_campaign_id}_summary_aggregated.csv"
                        )
                        outfile = open(output_path, "w", newline="")
                        writer = csv.DictWriter(
                            outfile, fieldnames=MASTER_HEADER, extrasaction="ignore"
                        )
                        writer.writeheader()
                        csv_writers[new_campaign_id] = writer
                        print(
                            f"  -> Creating new summary CSV: {os.path.relpath(output_path)}"
                        )

                    csv_writers[new_campaign_id].writerow(new_record)
                    stats[old_campaign_id][f"migrated_to_{new_campaign_id}"] += 1
                else:
                    stats[old_campaign_id]["orphaned_or_ignored_rows"] += 1

    print("\n" + "=" * 50 + "\n           MIGRATION COMPLETE: SUMMARY\n" + "=" * 50)
    for campaign, campaign_stats in sorted(stats.items()):
        print(f"\nProcessed Old Campaign '{campaign}':")
        for key, val in sorted(campaign_stats.items()):
            print(f"  - {key.replace('_', ' ').title()}: {val}")
    print("\nMigration finished. New CSV summary data is in './data/'.")


def main():
    print("--- One-Click Data Migration Script (v9 - Final Filename) ---\n")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_dir = os.path.join(project_root, "data_old")
    output_dir = os.path.join(project_root, "data")

    if not os.path.isdir(input_dir):
        print(f"ERROR: Source directory not found at '{input_dir}'")
        print(
            "Please move your old 'data' directory to the project root and rename it to 'data_old'."
        )
        return

    migrate_all_data(input_dir, output_dir)


if __name__ == "__main__":
    main()
