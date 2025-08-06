# FILE: src/config_loader.py
# A robust loader for the project's config.yml file.
# This version loads the config once, provides better error handling,
# and prevents state corruption via deep copying.

import yaml
import os
import sys
from typing import Dict, Any
from copy import deepcopy


def get_project_root() -> str:
    """Finds the project root directory reliably."""
    start_path = os.path.abspath(os.path.dirname(__file__))
    project_root = start_path
    while not os.path.isdir(os.path.join(project_root, "src")):
        parent = os.path.dirname(project_root)
        if parent == project_root:
            raise FileNotFoundError("Could not find project root containing 'src'.")
        project_root = parent
    return project_root


def _load_and_validate_config() -> Dict[str, Any]:
    """Internal function to load and validate config from disk once."""
    project_root = get_project_root()
    config_path = os.path.join(project_root, "config.yml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yml not found at {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # This is the critical new error handling.
        print("=" * 80, file=sys.stderr)
        print(
            "FATAL: There is a syntax error in your config.yml file.", file=sys.stderr
        )
        print(f"YAML parsing failed with error: {e}", file=sys.stderr)
        print(
            "HINT: Check for stray tab characters used for indentation instead of spaces.",
            file=sys.stderr,
        )
        print("=" * 80, file=sys.stderr)
        sys.exit(1)

    if not config:
        raise ValueError("config.yml is empty or could not be parsed.")

    if "param_grid" not in config or "experiments" not in config:
        raise ValueError(
            "config.yml must contain 'param_grid' and 'experiments' top-level keys."
        )

    return config


# --- SINGLETON CONFIGURATION ---
# Load the configuration once at module import time.
# This is efficient and ensures a single source of truth.
CONFIG = _load_and_validate_config()
EXPERIMENTS = CONFIG.get("experiments", {})


def get_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    Returns a deep copy of the configuration for a specific experiment.

    This function retrieves the configuration from the globally loaded CONFIG
    and merges the required parameter grids into it. It returns a deep copy
    to prevent any downstream modifications from affecting the global state.
    """
    if experiment_name not in EXPERIMENTS:
        available = list(EXPERIMENTS.keys())
        raise KeyError(
            f"Experiment '{experiment_name}' not found in config.yml. Available: {available}"
        )

    # Use deepcopy to prevent side-effects. This is a critical bug fix.
    exp_config = deepcopy(EXPERIMENTS[experiment_name])

    # Create a dedicated key for the resolved parameter grid values for clarity.
    exp_config["PARAM_GRID_VALUES"] = {}
    param_grid_keys = exp_config.get("param_grid_keys", [])
    global_param_grid = CONFIG.get("param_grid", {})

    for key in param_grid_keys:
        if key not in global_param_grid:
            raise KeyError(
                f"Parameter grid key '{key}' needed by '{experiment_name}' not found in global 'param_grid' section."
            )
        exp_config["PARAM_GRID_VALUES"][key] = global_param_grid[key]

    return exp_config
