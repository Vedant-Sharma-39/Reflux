# Reflux: A Computational Framework for Simulating Microbial Range Expansions

## Synopsis

This repository contains the source code for a numerical simulation framework designed to study the population dynamics of competing species on a two-dimensional hexagonal lattice. The model is implemented as an agent-based Gillespie simulation, suitable for investigating stochastic processes in biological systems. The primary application of this framework is to explore the phase space of competition outcomes in microbial range expansions, considering factors such as relative fitness and phenotypic switching.

The software is engineered for large-scale, high-throughput simulations on High-Performance Computing (HPC) clusters managed by the Slurm workload manager. It features a robust, data-driven workflow that ensures reproducibility and simplifies the management of complex parameter sweeps.

## Model Description

The simulation models asexually reproducing populations of two competing cell types—Wild-Type (WT) and Mutant (M)—expanding into empty territory on a hexagonal grid with periodic boundary conditions in the transverse direction. The dynamics are governed by a set of stochastic events whose rates are defined by the user:

*   **Growth:** Cells at the expanding front can reproduce into adjacent empty sites. The birth rates for Wild-Type (`b_wt`, typically normalized to 1.0) and Mutant (`b_m`) cells can be set independently to model fitness differences.
*   **Phenotypic Switching:** Cells at the front can switch their type. The total switching rate (`k_total`) and the bias of switching (`phi`) are tunable parameters, allowing for the exploration of symmetric and asymmetric switching dynamics.

The simulation is implemented using a direct-method stochastic simulation algorithm (SSA), commonly known as the Gillespie algorithm, which is an exact method for simulating the time evolution of a system of discrete events.

## Computational Workflow

The framework is designed around a modular and reproducible workflow, managed by a centralized configuration and a command-line interface.

1.  **Experiment Configuration:** All experimental campaigns are defined as entries in a single Python dictionary located in `src/config.py`. This file serves as the single source of truth for all parameters, including the parameter space to be explored (`PARAM_GRID`), the specific simulation sets (`SIM_SETS`), and the computational resources required for execution (`HPC_PARAMS`). This design ensures that an entire experiment is fully specified within a version-controlled file.

2.  **Task Generation & Execution:** The `scripts/launch.sh` interface automates the process of preparing and executing a simulation campaign. It reads the specified experiment configuration, generates a complete list of individual simulation tasks, checks for already completed tasks to allow for resumability, and submits a job array to the Slurm scheduler.

3.  **Data Aggregation and Analysis:** Upon completion, raw simulation outputs (in JSON format) are aggregated into a single, analysis-ready CSV file using the `scripts/aggregate_results.py` script. Subsequent analysis and visualization are performed by dedicated Python scripts, which process the aggregated data to produce the final figures and quantitative results.

## Software Dependencies

*   Python (3.8 or newer)
*   NumPy
*   Pandas
*   Matplotlib
*   SciPy
*   Seaborn
*   tqdm

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Vedant-Sharma-39/Reflux.git
    cd Reflux
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install numpy pandas matplotlib scipy seaborn tqdm
    ```
3.  **Set script permissions:**
    ```bash
    chmod +x scripts/launch.sh scripts/run_chunk.sh
    ```

## Usage Guide

The primary interface for managing simulation campaigns is `scripts/launch.sh`. This script can be run interactively or with command-line arguments.

**Interactive Mode:**
To select an action and experiment from a menu-driven prompt, run the script without arguments:
```bash
./scripts/launch.sh
```

**Command-Line Mode:**
To run non-interactively, provide the action and experiment name as arguments:
```bash
./scripts/launch.sh <action> <experiment_name>
```

### Available Actions

| Action       | Description                                                                                             |
|--------------|---------------------------------------------------------------------------------------------------------|
| `launch`     | Generates the list of missing tasks and submits a job array to the Slurm scheduler for execution.         |
| `status`     | Calculates and displays the completion percentage of the selected campaign without submitting any jobs.    |
| `clean`      | **Deletes all data, logs, and task lists** associated with the selected campaign after user confirmation. |
| `debug-task` | Runs a single, specified simulation task locally in the terminal for debugging purposes.                  |


## Reproducibility

This framework is designed to facilitate reproducible research. A specific set of published results can be reproduced by:
1.  Checking out the specific Git commit hash associated with the publication.
2.  Executing the `launch.sh` script with the relevant experiment name as defined in the version-controlled `src/config.py` file.

## Citation

If you use this software in your research, please cite the associated publication. A BibTeX entry is provided below for your convenience.

```bibtex
@article{sharma2024reflux,
  title={To be decided},
  author={Sharma, Vedant},
  journal={Journal Name},
  volume={XX},
  pages={YY--ZZ},
  year={2024},
  publisher={Publisher Name}
}
```
