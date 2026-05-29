# Quantum Heat Engine – Vibronic Transport

Julia implementation of a quantum heat engine model for light-harvesting systems, inspired by:

- Killoran, Huelga, Plenio, *Enhancing light-harvesting power with coherent vibrational interactions* (2015)
- Somoza et al., *Dissipation-Assisted Matrix Product Factorization (DAMPF)*, Phys. Rev. Lett. 123, 130502 (2019)

## Overview

This repository contains a Julia implementation of a quantum heat engine model for light-harvesting systems. The code covers both the electronic reference model and the vibronic extension used in the current simulations.

## Current Simulation Status

- The **electronic model** is implemented and produces steady-state populations, I-V curves, and power curves as a function of the load coupling $\Gamma_L$.
- The **vibronic model** is also implemented, with local vibrational modes coupled to the electronic subsystem.
- Current comparisons focus on a **weak/none coupling case** ($g=0$) and a **stronger vibronic coupling case** ($g=55\,\text{cm}^{-1}$).
- The existing scripts generate publication-style plots for electronic populations, vibronic transport, and numerical checks.

## Main Components

- **Electronic Hamiltonian**: 3-site excitonic system, written in a 4-level basis that includes the ground state
- **Thermal baths**: hot bath, cold bath, and load
- **Lindblad dynamics**: steady-state description in the single-excitation sector
- **Vibronic extension**: electronic plus local vibrational degrees of freedom
- **Observables**: current, voltage, power, and populations

## Results So Far

- Electronic transport curves and steady-state populations have been computed.
- Vibronic simulations have been run for the main parameter sets used in the project.
- The code is structured to compare how vibronic coupling modifies transport and power extraction.

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
