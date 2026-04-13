# Quantum Heat Engine – Vibronic Transport

Julia implementation of a quantum heat engine model for light-harvesting systems, inspired by:

- Killoran, Huelga, Plenio, *Enhancing light-harvesting power with coherent vibrational interactions* (2015)
- Somoza et al., *Dissipation-Assisted Matrix Product Factorization (DAMPF)*, Phys. Rev. Lett. 123, 130502 (2019)

## Overview

This repository contains a modular implementation of:

- Electronic Hamiltonian (single-excitation subspace, 3–4 levels)
- Lindblad master equation for hot bath, cold bath, and load
- Steady-state and time-dependent dynamics
- Current, voltage, power, and I‑V / P‑V curves
- (Future) Vibronic coupling and DAMPF tensor network extension
