# Quantum Heat Engine – Vibronic Transport

Julia implementation of a quantum heat engine model for light-harvesting systems, inspired by:

- Killoran, Huelga, Plenio, *Enhancing light-harvesting power with coherent vibrational interactions* (2015)
- Somoza et al., *Dissipation-Assisted Matrix Product Factorization (DAMPF)*, Phys. Rev. Lett. 123, 130502 (2019)

## Overview

This repository implements a complete quantum heat engine model combining electronic and vibronic dynamics in light-harvesting systems. The model couples electronic transitions to vibrational modes, enabling study of coherent energy transfer mechanisms.

## Features Implemented

### Core Model
- **Electronic Hamiltonian**: 3-site exciton system with electronic-vibrational coupling
- **Three thermal baths**: 
  - Hot bath (fluorescence at temperature $T_H$)
  - Cold bath (radiative recombination at temperature $T_C$)
  - Load (dissipative channel with tunable coupling $\Gamma_L$)
- **Lindblad master equation**: Dynamics in the single-excitation subspace
- **Steady-state solver**: Efficiently computes equilibrium populations and observables

### Vibronic Coupling
- **Two local vibrational modes** with frequency $\omega_v$
- **Electronic-vibrational coupling parameter** $g$ (tunable: $g=0$ electronic-only, $g=55\,\text{cm}^{-1}$ strong coupling)
- **Full Hilbert space dynamics**: Electronic + vibrational degrees of freedom

### Observables & Analysis
- **I-V characteristic curves**: Current vs. load voltage across 14 orders of magnitude in $\Gamma_L$
- **Electronic population distributions**: Steady-state excitation probabilities
- **Power output**: $P(\Gamma_L)$ analysis and optimal load conditions
- **Vibrational populations**: Thermal occupations at different coupling strengths

## Results

### Electronic-Only System
- Computed I-V curves for purely electronic transport
- Identified optimal load voltage for maximum power extraction
- Characterized steady-state populations across the exciton network

### Vibronic Coupling Effects
- Analyzed impact of electron-vibration coupling ($g=0$ vs. $g=55\,\text{cm}^{-1}$)
- Generated population dynamics with 4-5 vibrational modes
- Compared coherent vs. incoherent transport regimes
- Generated comparative visualizations

## Project Structure

```
src/
  ├─ Hamiltonian.jl      # Electronic and vibronic Hamiltonian construction
  ├─ Lindblad.jl         # Jump operators and dissipation channels
  ├─ Liouvillian.jl      # Liouvillian superoperator and time evolution
  ├─ Solver.jl           # Steady-state solver
  └─ Observables.jl      # Physical observables (current, power, populations)

config/
  └─ parameters.jl       # Physical constants and default parameters

experiments/
  ├─ IV_curve_electronic_populations.jl      # Electronic I-V + populations
  ├─ IV_curve_vib.jl                         # Vibronic I-V (main analysis)
  ├─ IV_curve_vib_test.jl                    # Convergence & numerical tests
  └─ IV_curve2.jl                            # Electronic configuration

imgs/
  ├─ electronic/         # Electronic transport results
  └─ vibrational/        # Vibronic coupling results (g=0, g=55)
```


## Parameters

All physical parameters are defined in `config/parameters.jl`:
- Exciton transition energies and couplings (cm⁻¹)
- Thermal bath properties: temperatures ($T_H$, $T_C$), decay rates ($\gamma_H$, $\gamma_C$)
- Vibrational frequency ($\omega_v$) and max mode number ($N_v$)
- Load coupling sweep: $\Gamma_L \in [10^{-8}, 10^{8}]$ cm⁻¹ (70 points)
