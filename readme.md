# Reflux: Code for "[Publication Title TBD]"

This repository contains the complete source code and configuration files used to generate the results and figures for the scientific publication:

> **[Full Citation of the Paper - TBD]**

## Abstract

> _[Insert the full abstract of your publication here. A placeholder is provided below.]_
>
> The evolution of cooperative strategies in fluctuating environments presents a complex puzzle. Phenotypic switching, or bet-hedging, allows isogenic populations to adapt to changing conditions. Here, we investigate spatial bet-hedging in microbial range expansions using an agent-based simulation framework. We explore how environmental asymmetry and predictability shape the optimal switching strategy. Our key findings reveal that predictable environmental sequences favor finely-tuned, asymmetric switching strategies, whereas unpredictable, stochastic environments select for more conservative, symmetric switching. Furthermore, we analyze the role of "polluting" strategies (`phi < 0`), where one phenotype actively produces another, and find that this cooperative behavior is only viable under specific environmental structures. This work provides a quantitative framework for understanding the evolutionary dynamics of cooperation and adaptation in spatially structured, fluctuating environments.

## Model Description

The simulation is an agent-based model on a 2D hexagonal lattice, representing a microbial range expansion. The dynamics are governed by a direct-method stochastic simulation algorithm (Gillespie algorithm) with the following events for cells at the expanding front:

*   **Growth:** Cells reproduce into adjacent empty sites. The birth rates for Wild-Type (`b_wt`, normalized to 1.0) and Mutant (`b_m`) cells can be set independently. The fitness difference is described by the selective advantage `s = b_m - 1`.
*   **Phenotypic Switching:** Cells can switch their type (WT ↔ M). The dynamics are controlled by the total switching rate (`k_total`) and the bias (`phi`), which map to individual rates as:
    *   `k_wt_m` (WT → M) = `(k_total / 2) * (1 - phi)`
    *   `k_m_wt` (M → WT) = `(k_total / 2) * (1 + phi)`

## Guide to Reproducibility

This repository is structured to ensure the full reproducibility of our findings. The key experimental campaigns are defined in `src/config.py`.

To reproduce a figure from the paper, follow these steps:

#### Step 1: Generate the Master Task List

Identify the `experiment_name` in `src/config.py` corresponding to the figure you wish to reproduce (e.g., `asymmetric_patches` for Figure 5). Run the task generation script:

```bash
# Usage: python scripts/utils/generate_tasks.py <experiment_name>
python scripts/utils/generate_tasks.py asymmetric_patches
```
This creates a master task file in `data/fig5_asymmetric_patches/`.

#### Step 2: Execute the Simulations

Submit the tasks to a Slurm-based HPC cluster using the `hpc_manager.sh` script.

```bash
bash scripts/hpc_manager.sh submit data/fig5_asymmetric_patches/fig5_asymmetric_patches_master_tasks.jsonl
```

#### Step 3: Aggregate Raw Data

Once all jobs are complete, aggregate the distributed results into a single dataset.

```bash
# Usage: python scripts/utils/aggregate_data.py <campaign_id>
python scripts/utils/aggregate_data.py fig5_asymmetric_patches
```

#### Step 4: Generate the Figure

Finally, execute the corresponding figure script.

```bash
# Usage: python scripts/paper_figures/<figure_script>.py
python scripts/paper_figures/fig5_asymmetric_patches.py
```
The final plot will be saved to `data/fig5_asymmetric_patches/analysis/`.

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Seaborn
- Matplotlib

## How to Cite

If you use this code or our findings in your research, please cite both the original publication and this software repository.

**1. Publication:**
> [Full citation will be added here upon publication.]

**2. Software:**
> Vedant Sharma. (2025). Reflux: A Computational Framework for Simulating Microbial Range Expansions (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX _[Note: DOI to be generated via a Zenodo release.]_

## Synopsis

This repository contains the source code for a numerical simulation framework designed to study the population dynamics of competing species on a two-dimensional hexagonal lattice. The model is implemented as an agent-based Gillespie simulation, suitable for investigating stochastic processes in biological systems. The primary application of this framework is to explore the phase space of competition outcomes in microbial range expansions, considering factors such as relative fitness and phenotypic switching.

The software is engineered for large-scale, high-throughput simulations on High-Performance Computing (HPC) clusters managed by the Slurm workload manager. It features a robust, data-driven workflow that ensures reproducibility and simplifies the management of complex parameter sweeps.

## Model Description

The simulation models asexually reproducing populations of two competing cell types—Wild-Type (WT) and Mutant (M)—expanding into empty territory on a hexagonal grid. The expansion proceeds along one axis, with periodic boundary conditions in the transverse direction. The dynamics are governed by a set of stochastic events whose rates are defined by the user:

*   **Growth:** Cells at the expanding front can reproduce into adjacent empty sites. The birth rates for Wild-Type (`b_wt`, normalized to 1.0) and Mutant (`b_m`) cells can be set independently. The fitness difference is described by the selective advantage `s = b_m - 1`.
*   **Phenotypic Switching:** Cells at the front can switch their type (WT ↔ M). The dynamics are controlled by two parameters: the total switching rate (`k_total`) and the bias of switching (`phi`). These map to the individual rates as:
    *   `k_wt_m` (WT → M) = `(k_total / 2) * (1 - phi)`
    *   `k_m_wt` (M → WT) = `(k_total / 2) * (1 + phi)`
    A `phi` of -1.0 represents a "polluting" strategy where WT cells primarily produce M cells, while a `phi` of 1.0 represents a "purging" strategy where M cells primarily revert to WT.

The simulation is implemented using a direct-method stochastic simulation algorithm (SSA), commonly known as the Gillespie algorithm, which is an exact method for simulating the time evolution of a system of discrete events.

