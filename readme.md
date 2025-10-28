````markdown
# The Advantage of Phenotypic Switching in Range Expansions Emerges from a Crossover, Not a Critical Transition

**Authors:** Vedant Sharma, et al.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Preprint](https://img.shields.io/badge/Preprint-bioRxiv-blue.svg)](https://www.biorxiv.org/) This repository contains the full source code for the large-scale agent-based model (ABM) and analysis scripts used in our manuscript. This work investigates the interplay between the physical nature of phase transitions and the biological fitness of spatial bet-hedging strategies.

---

### Abstract

Phenotypic switching is a common bet-hedging strategy for surviving in fluctuating environments. While well-understood in well-mixed systems, in spatially structured populations undergoing range expansions, the physics of domain formation and local extinction introduce new dynamics. Previous work has shown that *irreversible* switching ($\phi = -1.0$) leads to a critical phase transition belonging to the Directed Percolation (DP) universality class. But what happens if the switch is *reversible*?

Here, we use a large-scale, continuous-time agent-based model to investigate the interplay between switching reversibility, phase transition physics, and biological fitness. We demonstrate that introducing even slight reversibility ($\phi > -1.0$) fundamentally alters the system, changing the sharp, critical phase transition into a smooth crossover. We then show that this physical shift is the key to the biological advantage: the ability to recover from local extinctions in the crossover regime allows reversible strategies to achieve a significant fitness advantage in heterogeneous environments—an advantage that is completely absent in the critical, irreversible case. Finally, we show how optimal switching strategies (both rate, $k_{\text{total}}$, and bias, $\phi$) are fine-tuned by the statistical structure of the environment. Our work establishes that the biological benefit of spatial bet-hedging is an emergent property of the crossover physics, not a critical one.

---

### Key Findings

![Simulation Animation](https://github.com/your-username/your-repo/blob/main/figures/simulation_comparison.gif)
> *An animation from our agent-based model. (Left) An irreversible (`$\phi = -1.0$`) strategy suffers local extinctions (gray patches) that can never be recovered. (Right) A reversible (`$\phi = 0.0$`) strategy allows mutant (blue) sectors to be regenerated from the wild-type (gray) population, enabling long-term persistence and fitness.*

1.  **Critical vs. Crossover:** Irreversible switching (`$\phi = -1.0$`) creates an absorbing state, leading to a sharp, critical phase transition. Reversible switching (`$\phi > -1.0$`) removes this absorbing state, resulting in a smooth crossover.
2.  **Physics Enables Biology:** The biological advantage of bet-hedging (i.e., a fitness gain in fluctuating environments) is *only* present in the crossover regime. The critical system (`$\phi = -1.0$`) provides no fitness benefit, as locally extinct sectors are lost forever.
3.  **Optimal Strategy:** The optimal switching bias (`$\phi_{\text{opt}}$`) is tuned to the environmental statistics, demonstrating a new layer of evolutionary adaptation.

---

### Installation

This project is written in Python 3. The core simulation is a continuous-time, event-driven model based on a Gillespie algorithm, and the analysis scripts use the standard scientific Python stack.

```bash
# 1. Clone the repository
git clone [https://github.com/vsharma/spatial-bet-hedging.git](https://github.com/vsharma/spatial-bet-hedging.git)  # <-- TODO: Update with actual repo URL
cd spatial-bet-hedging

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt
````

The main dependencies are listed in `requirements.txt` and include:

  * `numpy`
  * `pandas`
  * `scipy`
  * `matplotlib`
  * `seaborn`
  * `tqdm`

-----

### How to Reproduce Our Results

You can reproduce all figures from the paper by following these steps.

**Note:** Running the full simulation sweeps is computationally intensive and will generate gigabytes of data. Pre-computed, aggregated data is provided in `data/processed/`.

#### 1\. (Optional) Run Full Simulation Sweeps

To run the simulations from scratch, use the `run_sweep.py` script. The campaign configurations are defined in `src/config.py`.

```bash
# Run the homogeneous environment sweep (for Fig 2a)
python src/run_sweep.py --campaign homogeneous_sweep

# Run the heterogeneous environment sweep (for Fig 2b & 3)
python src/run_sweep.py --campaign heterogeneous_sweep
```

  * Raw simulation output will be saved to `data/raw/`.

#### 2\. (Optional) Aggregate Raw Data

If you ran your own simulations, you must aggregate the raw output into tidy CSV files for plotting.

```bash
# Process all campaigns
python src/analysis/aggregate_data.py --campaign all
```

  * Processed data will be saved to `data/processed/`.

#### 3\. Generate All Paper Figures

This will use the processed data (either the ones you just generated or the ones included in the repo) to create all figures from the manuscript.

```bash
# This script will run all plotting scripts in the /scripts/ folder
python scripts/generate_all_figures.py
```

  * All figures will be saved to the `figures/` directory.
  * To generate a specific figure (e.g., Figure 2), you can run its script directly:
    ```bash
    python scripts/plot_fig2_crossover.py
    ```

-----

### Repository Structure

```
spatial-bet-hedging/
├── data/
│   ├── raw/                # Raw simulation output (gitignored)
│   └── processed/          # Aggregated .csv files for plotting
├── figures/                # Final figures for the paper (.pdf, .png)
├── scripts/                # Python scripts to generate paper figures
│   ├── generate_all_figures.py
│   ├── plot_fig1_model_schematic.py
│   ├── plot_fig2_crossover.py
│   ├── plot_fig3_asymmetry.py
│   └── ...
├── src/                    # Core simulation and analysis code
│   ├── analysis/           # Scripts for data aggregation (aggregate_data.py)
│   ├── core/               # ABM logic, Gillespie algorithm, lattice, etc.
│   ├── io/                 # Data loading/saving utilities
│   ├── config.py           # Defines simulation parameters for all campaigns
│   └── run_sweep.py        # Main script to launch simulation batches
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

-----

### Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{Sharma2025Crossover,
  title   = {The advantage of phenotypic switching in range expansions emerges from a crossover, not a critical transition},
  author  = {Sharma, Vedant and [Co-author 1] and [Co-author 2] and Wang, Hao},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/TODO},
  url     = {[https://www.biorxiv.org/content/TODO](https://www.biorxiv.org/content/TODO)}
}
```

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

```
```
