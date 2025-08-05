# FILE: scripts/aggregate_results.py

import sys
import os
import argparse

# --- Robust Path and Config Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EXPERIMENTS
from data_utils import aggregate_data_cached


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate raw JSON results for a campaign."
    )
    parser.add_argument(
        "experiment_name",
        choices=EXPERIMENTS.keys(),
        help="Name of the experiment to aggregate.",
    )
    parser.add_argument(
        "--force-reaggregate",
        action="store_true",
        help="Force re-aggregation even if a cache file exists.",
    )
    args = parser.parse_args()

    campaign_id = EXPERIMENTS[args.experiment_name]["CAMPAIGN_ID"]

    aggregate_data_cached(campaign_id, project_root, args.force_reaggregate)


if __name__ == "__main__":
    main()
