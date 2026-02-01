## Complex System Simulation project 2026: Highly Polar and the Chamber of Echoes

- Group 1: Yara Berkelmans, Yaqi Duan, Philipp Kreiter

## Overview

This repository contains the Python implementation of an Agent-Based Model (ABM), modeling the the dynamics of co-evolving networks and emergence of echo chambers.

The model explores how the "Opinion influence strength" ($\alpha$) and "Social influence strength" ($\beta$) compete against interaction thresholds to drive the system towards either consensus or polarization.

<p align="center">
  <img src="img/alpha_80.gif" width="30%" alt="Structure Phase Transition">
</p>

## Key Features

* **Co-evolving Dynamics**: Agents update both their opinions (Continuous) and social positions (Spatial) simultaneously via SDEs (Euler-Maruyama integration).
* **Dual-Threshold**: 
    * $R_{sp}$: Spatial visibility radius.
    * $R_{op}$: Opinion tolerance radius.
* **Speed of Dynamics**:
    Explicit control over **Opinion Influence ($\alpha$)** and **Social Sorting ($\beta$)**. This allows the study of how different parameters accelerates polarization.
* **Stochastic Perturbation**:
    Integrates Gaussian noise ($\sigma$) into both opinion and spatial updates to simulate real-world entropy.

## 📂 Project Structure

```text
├── main.py             # Main file to run the experiments as shown in the presentation
├── echochambers.pdf     # Presentation slide
├── src/                # Core simulation logic (Agent, Model, Metrics)
├── visualization/      # Plotting and animate
├── img/                # Generated png and gif
└── archive/            # Early prototypes and experiments that are no longer used
