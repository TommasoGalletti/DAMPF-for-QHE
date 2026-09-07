include("../system/system.jl")
using .DampfSystem

include("../../src/Solver.jl")
using .Solver

include("../../src/Observables.jl")
using .Observables

include("../../config/parameters.jl")

include("../utils/csv_io.jl")
using .DampfCsvIO

"""
Exact g=0 (no vibronic coupling) steady-state baseline for the cluster and
cascade topologies, requested to isolate the vibronic enhancement from the
topology change itself: enhancement should be computed against the SAME
electronic topology without vibrations, not against the 2-mode Killoran
benchmark (which also differs in dim_e/site count).

g=0 exactly decouples the vibrational modes from the electronic dynamics, so
ρ_e obeys the same Lindblad equation it would with no modes at all -- solved
here directly as the steady state of the electronic-only Liouvillian
(`Solver.steady_state`, dim_e x dim_e), with no DAMPF/MPS machinery and no
chi_max/Nv truncation error. Cheap enough to run the full standard ΓL grid.
"""

sim = default_simulation_params()

ΓL_grid = [0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 20.0, 35.0, 40.0, 50.0, 100.0, 150.0, 200.0,
           300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0,
           1500.0, 2000.0, 3000.0, 5000.0, 10000.0, 15000.0, 20000.0, 30000.0]

function run_electronic_baseline(model, csv_path)
    dim_e = DampfSystem.dim_e(model)
    idx_low, idx_g = dim_e - 1, dim_e
    H_e = DampfSystem.build_electronic_hamiltonian(model)
    energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)

    results = NamedTuple[]
    for ΓL in ΓL_grid
        diss = DampfSystem.build_electronic_dissipators(
            H_e; idx_low=idx_low, idx_g=idx_g,
            γH=sim.γH, nH=sim.nH, ΓL=ΓL,
            γC=sim.γC, TC=sim.TC, kB=sim.kB,
        ).ops
        L_e = DampfSystem.DampfElectronic.Liouvillian.build_liouvillian(Matrix{ComplexF64}(H_e), diss)
        ρ_e = Solver.steady_state(L_e)

        I = Observables.current(ρ_e, U_e, ΓL, idx_low)
        V = Observables.voltage(ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
        P = Observables.power(I, V)
        push!(results, (ΓL=ΓL, I_dampf=I, V_dampf=V, P_dampf=P, steps=0, converged=true))
    end

    save_dampf_sweep_csv(csv_path, results)
    println("Saved $(length(results)) points: $(csv_path)")
    return results
end

cluster_model = DampfSystem.killoran_four_site_model()
cascade_model = DampfSystem.killoran_cascade_model()

cluster_csv = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_cluster_electronic_baseline.csv")
cascade_csv = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_cascade_electronic_baseline.csv")

println("--- cluster/trio electronic baseline (g=0) ---")
cluster_results = run_electronic_baseline(cluster_model, cluster_csv)

println("--- cascade electronic baseline (g=0) ---")
cascade_results = run_electronic_baseline(cascade_model, cascade_csv)

println("\nDone.")
