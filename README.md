# Quantum Heat Engine – Vibronic Transport

Julia implementation of a quantum heat engine model for light-harvesting systems, inspired by:

- Killoran, Huelga, Plenio, *Enhancing light-harvesting power with coherent vibrational interactions* (2015)
- Somoza et al., *Dissipation-Assisted Matrix Product Factorization (DAMPF)*, Phys. Rev. Lett. 123, 130502 (2019)

## Overview

The repository has two parts:

- **`src/`** – Lindblad master-equation model (electronic and vibronic), solved directly for the steady state.
- **`DAMPF/`** – matrix-product-state (DAMPF/ITensors) extension, used to go beyond the Lindblad local-mode treatment: multiple modes per site, and larger site topologies (cascade, cluster, three- and four-site networks).

Both give steady-state populations, I-V curves, and power curves as a function of the load coupling $\Gamma_L$.

## Main Components

**Lindblad model (`src/`)**
- Electronic Hamiltonian: 3-site excitonic system in a 4-level basis including the ground state
- Hot/cold/load thermal baths, steady-state Lindblad dynamics in the single-excitation sector
- Vibronic extension: electronic system plus local vibrational modes

**DAMPF model (`DAMPF/`)**
- `system/` – Hamiltonian, mode indexing, dissipators, trace basis for arbitrary site/mode topologies
- `dynamics/` – Trotterized MPS time evolution and steady-state extraction
- `parameters/` – physical and numerical (bond dimension, timestep, cutoff) parameters
- Topologies covered: Killoran 2-mode benchmark, three-mode (v1v2v3), two modes per site (s1x2/s2x2, incl. frequency sweep), cascade, cluster trio, and four-site networks
- `omega_gammaL_scan_killoran.jl` scans vibrational frequency $\omega_v$ against $\Gamma_L$
- `verify_multimode_equivalence.jl` checks the analytical equivalence of two identical modes per site to one effective mode

## Project Structure

```
src/                      # Lindblad model
  ├─ Hamiltonian.jl
  ├─ Lindblad.jl
  ├─ Liouvillian.jl
  ├─ Solver.jl
  └─ Observables.jl

config/
  └─ parameters.jl        # Physical constants and default parameters (Lindblad model)

experiments/              # Lindblad experiments
  ├─ IV_curve_electronic_populations.jl
  ├─ IV_curve_vib.jl
  ├─ IV_curve_vib_populations.jl
  └─ IV_curve2.jl

DAMPF/                    # Matrix-product-state extension
  ├─ system/              # Hamiltonian, mode indexing, dissipators
  ├─ dynamics/             # Trotter evolution, steady state
  ├─ parameters/           # Physical & numerical (MPS) parameters
  ├─ experiments/          # I-V curves per topology, omega-ΓL scan, checks
  ├─ plots/                # Plotting utilities
  └─ utils/                # CSV I/O

outputs/                  # Generated figures and numeric vectors (both models)
  ├─ imgs/
  └─ numeric_vectors/
```

## Parameters

All physical parameters for the Lindblad model are defined in `config/parameters.jl`:
- Exciton transition energies and couplings (cm⁻¹)
- Thermal bath properties: temperatures ($T_H$, $T_C$), decay rates ($\gamma_H$, $\gamma_C$)
- Vibrational frequency ($\omega_v$) and max mode number ($N_v$)
- Load coupling sweep: $\Gamma_L \in [10^{-8}, 10^{8}]$ cm⁻¹

DAMPF-specific physical and numerical parameters (site/mode topology, MPS bond dimension, timestep, cutoff) are in `DAMPF/parameters/`.
