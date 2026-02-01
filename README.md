# Highly Polar and the Chamber of Echoes

### Complex Systems Simulation 2026

- Group 1: Yara Berkelmans, Yaqi Duan, Philipp Kreiter

## Overview

This repository contains the Python implementation of an Agent-Based Model (ABM), modeling the dynamics of co-evolving networks and emergence of echo chambers. The implementation follows [Conrad et Al., 2024](https://arxiv.org/pdf/2407.00145).

The model explores how the "Opinion influence strength" ($\alpha$) and "Social influence strength" ($\beta$) compete against interaction thresholds to drive the system towards either consensus or polarization. We further explore if noise can revert polarized or converged systems towards a mixed-opinion state again. Our experiments conclude that once a system is polarised or converged, noise cannot repopulate a diverse opinion-space while the social space shows tendencies of mixing.

<p align="center">
  <img src="img/alpha_80.gif" width="30%" alt="Structure Phase Transition">
</p>

## Key Features

- **Co-evolving Dynamics** Agents update both their opinions (Continuous) and social positions (Spatial) simultaneously via SDEs (Euler-Maruyama integration).
- **Dual-Threshold**
  - $R_{sp}$: Spatial visibility radius.
  - $R_{op}$: Opinion tolerance radius.
- **Speed of Dynamics**
  Explicit control over **Opinion Influence ($\alpha$)** and **Social Sorting ($\beta$)**. This allows the study of how different parameters accelerate polarization.
- **Stochastic Perturbation**
  Integrates Gaussian noise ($\sigma$) into both opinion and spatial updates to simulate real-world entropy.

## Experiments

We reproduce a majority of the figures in the original paper, explore further pareamter sweeps across the model scope and additionally investigate the use of noise and edge removal as treatment for reverting a convergend opinion-social space state to a mixed opinion state.

## Project Structure

```text
├── sim.py             # Simulation file to run the experiments as shown in the presentation
├── echochambers.pdf     # Presentation slide
├── src/                # Core simulation logic (Agent, Model, Metrics)
├── experiments/        # Specifies simulation experiments that can be run from sim.py
├── visualization/      # Plotting and animation
├── img/                # Generated png and gif
└── test_smoke.py       # Smoke tests to ensure the integrity of the Agent-based model
```

## Quickstart

Install the required packages

```bash
pip install -r requirements.txt
```

Run the experiments via

```bash
python sim.py
```

Run the smoke test via

```bash
python -m pytest test_smoke.py
```

## GenAI Usage

The core model structure was designed manually without GenAI. We used ChatGPT, Gemini 3, and Claude Code for debugging, code generation (via detailed prompts and line-by-line comments), and heavily during the creation of the smoke test. All outputs were reviewed and integrated by us into the project.
