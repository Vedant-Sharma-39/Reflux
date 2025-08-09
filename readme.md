# Manuscript Outline: The Role of Reversibility in Spatial Bet-Hedging

**Title:** The advantage of phenotypic switching in range expansions emerges from a crossover, not a critical transition

**Authors:** Vedant Sharma, et al.

---

### Abstract
_(Your abstract here)_

---

### Introduction
-   Microbial range expansions are a key paradigm for studying evolution in spatially structured populations.
-   Phenotypic switching (bet-hedging) is a known strategy for adapting to fluctuating environments.
-   Prior work (e.g., Skanata & Kussell) has established a theoretical framework for temporal bet-hedging, but the spatial component adds crucial new physics (memory, domain formation).
-   Work by Kuhr et al. has connected irreversible switching (`phi = -1.0`) in spatial models to the Directed Percolation (DP) universality class, a true phase transition.
-   **Key Question:** What happens when this switching is reversible (`phi > -1.0`)? How does the physical nature of the system's phase boundary relate to the biological advantage conferred by the strategy?
-   **Our Approach:** We use a large-scale agent-based model to systematically map the phase space of mutant invasion and directly measure the fitness of bet-hedging strategies in heterogeneous environments.

---

### Results

#### 1. Switching Reversibility Transforms the Phase Boundary from Critical to Crossover
-   We first characterize the steady-state mutant density (`<ρ_M>`) in a homogeneous environment.
-   **[INSERT NEW FIGURE 2 HERE]**
-   **Figure 2a:** Shows that for irreversible switching (`phi = -1.0`), the system exhibits a sharp, step-like transition in `<ρ_M>` as a function of selection `s`, characteristic of a critical phase transition.
-   **Figure 2a (cont.):** As reversibility is introduced (`phi = -0.5, 0.0`), this transition becomes a smooth, gradual crossover.
-   **Conclusion:** The presence of a reverse switching pathway (M -> WT) removes the absorbing state (pure WT phase) and fundamentally changes the physics of the system.

#### 2. The Bet-Hedging Advantage is Enabled by the Crossover Regime
-   We next measure the fitness of switching strategies in a spatially heterogeneous environment (patches of favorable/unfavorable territory).
-   **Figure 2b:** Shows the relative fitness gain (speed of bet-hedger vs. best pure strategy).
-   **Figure 2b (cont.):** For irreversible switching (`phi = -1.0`), there is no fitness advantage. Once a mutant sector is lost, it can never be recovered.
-   **Figure 2b (cont.):** For reversible switching (`phi = 0.0`), a clear optimal switching rate `k_total` emerges that provides a significant fitness advantage (>1.0).
-   **Synthesis:** The biological advantage of bet-hedging is not a feature of the critical (`phi = -1.0`) system. It is an emergent property of the crossover regime, where the ability to recover from local extinction events is crucial.

#### 3. Optimal Switching Strategies Adapt to Environmental Asymmetry
-   We then explored how the optimal strategy adapts to predictable, asymmetric environments.
-   **[INSERT NEW FIGURE 3 HERE]**
-   **Figure 3a:** Shows that the optimal switching bias (`phi_opt`) is tuned to the environmental statistics. When the environment is mostly favorable to WT (90_30), a positive (purging) bias is optimal. When mostly favorable to M (30_90), a negative (polluting) bias is optimal.
-   **Figure 3b:** Shows that these adapted, asymmetric strategies achieve a higher maximal fitness than the symmetric (`phi=0`) strategy in a symmetric environment.
-   **Conclusion:** In predictable environments, evolution can fine-tune the switching dynamics to match the environmental structure, leading to significant fitness gains. The "scrambled" control shows that this advantage is lost when predictability is removed.

---

### Discussion
-   Recap the main finding: The crossover nature of reversible switching is the key physical principle enabling spatial bet-hedging.
-   Connect to theory: Our results provide a quantitative, spatial extension to the temporal models of bet-hedging. We show how spatial memory (the persistence of sectors) adds a new dimension to the problem.
-   The cost of memory and the timescale of recovery (`τ_recovery` from your new Figure 4) can be discussed here as the physical mechanism limiting how fast the optimal `k_total` can be.
-   Implications for cooperative strategies (`phi < 0`): Our work shows that "polluting" is not just altruism; it's a winning strategy in environments that are biased against the "polluter."
-   Future work...

---