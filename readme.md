# FILE: readme.md (Updated for New Workflow)

# Reflux: A Computational Framework for Simulating Microbial Range Expansions

## Synopsis

This repository contains the source code for a numerical simulation framework designed to study the population dynamics of competing species on a two-dimensional hexagonal lattice. The model is implemented as an agent-based Gillespie simulation, suitable for investigating stochastic processes in biological systems. The primary application of this framework is to explore the phase space of competition outcomes in microbial range expansions, considering factors such as relative fitness and phenotypic switching.

The software is engineered for large-scale, high-throughput simulations on High-Performance Computing (HPC) clusters and features a robust, data-driven workflow that ensures reproducibility and simplifies the management of complex parameter sweeps via a single command-line tool, `manage.py`.

## Model Description

The simulation models asexually reproducing populations of two competing cell types—Wild-Type (WT) and Mutant (M)—expanding into empty territory on a hexagonal grid. The expansion proceeds along one axis, with periodic boundary conditions in the transverse direction. The dynamics are governed by a set of stochastic events whose rates are defined by the user:

*   **Growth:** Cells at the expanding front can reproduce into adjacent empty sites. The birth rates for Wild-Type (`b_wt`, normalized to 1.0) and Mutant (`b_m`) cells can be set independently. The fitness difference is described by the selective advantage `s = b_m - 1`.
*   **Phenotypic Switching:** Cells at the front can switch their type (WT ↔ M). The dynamics are controlled by two parameters: the total switching rate (`k_total`) and the bias of switching (`phi`). These map to the individual rates as:
    *   `k_wt_m` (WT → M) = `(k_total / 2) * (1 - phi)`
    *   `k_m_wt` (M → WT) = `(k_total / 2) * (1 + phi)`

## Quickstart & Reproducibility Guide

The entire project workflow is managed by `manage.py` and the simplified `Makefile`.

### Step 1: Installation

First, install the required Python packages.

```bash
make install
# Or manually:
# python3 -m venv venv && source venv/bin/activate
# pip install -r requirements.txt