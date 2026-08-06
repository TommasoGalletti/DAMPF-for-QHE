include("../system/system.jl")
using .DampfSystem
using ITensors
using ITensorMPS
using LinearAlgebra
using Plots

include("../plots/iv_curve_plots.jl")
using .DampfPlots

include("../utils/csv_io.jl")
using .DampfCsvIO

include("../../src/Hamiltonian.jl")
include("../../src/Lindblad.jl")
include("../../src/Liouvillian.jl")
include("../../src/Solver.jl")
include("../../src/Observables.jl")
include("../../config/parameters.jl")
using .Hamiltonian
using .Lindblad
using .Liouvillian
using .Solver
using .Observables

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Re-validation of the original N=3/2-mode Killoran benchmark under the new,
generic ITensorMPS architecture: for each ΓL, propagate the DAMPF Trotter
dynamics from the ground state to an approximate steady state, reduce the
block representation to a dense electronic density matrix (`Tr(Ô_{m,n})`
per block), and feed it directly into `Observables.jl`'s PLAIN (non-`_vib`)
functions -- no new observable code, since the block trace already IS the
reduced electronic density matrix. Compared against `Solver.steady_state`
("Lindblad diretto": a direct linear solve of the same Nv=6-truncated
Fock-space Lindbladian used by the original Killoran-style calculation --
not "exact" in an absolute sense, since it shares the same Fock truncation
as DAMPF, just without any Trotter splitting or bond-dimension truncation).

The reusable pieces of this pipeline live elsewhere, so this file is just
the experiment-specific wiring: DAMPF/dynamics/steady_state.jl (generic
"run Trotter until steady state" driver), DAMPF/utils/csv_io.jl (load/save/
merge the comparison CSV), DAMPF/plots/iv_curve_plots.jl (all plotting).
"""

params = default_params()
sim = default_simulation_params()
model = DampfSystem.killoran_benchmark_model(Nb=params.Nv)
dim_e = DampfSystem.dim_e(model)
idx_low = dim_e - 1
idx_g = dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
H_ref = Hamiltonian.build_total_hamiltonian(params)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)

siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

# ----------------------------------------------------------------------------
# Numerical knobs. Δt/chi_max re-tuned in check/04's high-ΓL scan (ΓL in
# [200,1000], t_final=1e-2): at chi_max=3, error vs an RK4 reference is
# U-shaped in Δt (more steps at fixed chi_max means more accumulated
# truncation events, which dominates once Δt is small enough that Trotter
# splitting error is no longer the bottleneck) with a minimum around
# Δt=2e-4 (~0.34% state-level error, ~20x fewer steps than the previous
# Δt=1e-5). coeff_tol loosened from the 1e-14 default (still ~3 orders of
# magnitude below that error floor) to prune negligible terms from the
# electronic-mixing sum.
# ----------------------------------------------------------------------------

Δt = 2.0e-4
chi_max = 3
coeff_tol = 1.0e-10
max_steps = 10_000
check_every = 500
conv_tol = 1.0e-6

# Δt-dependent, ΓL-independent: build once.
role_props = [DampfSystem.build_mode_role_propagators(mode, siteinds[q], Δt) for (q, mode) in enumerate(model.modes)]

"""ΓL-specific wiring (electronic dissipators + mixing propagator), then
hand off to the generic driver in DampfSystem.run_to_steady_state."""
function run_dampf_steady_state(ΓL::Float64)
    diss = DampfSystem.build_electronic_dissipators(
        H_e; idx_low=idx_low, idx_g=idx_g,
        γH=sim.γH, nH=sim.nH, ΓL=ΓL,
        γC=sim.γC, TC=sim.TC, kB=sim.kB,
    ).ops
    P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)
    return DampfSystem.run_to_steady_state(
        model, role_props, siteinds, τ, P_e;
        chi_max=chi_max, coeff_tol=coeff_tol, Δt=Δt, max_steps=max_steps, check_every=check_every, conv_tol=conv_tol,
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL, idx_low)), label="ΓL=$(ΓL)",
    )
end

"""The "Lindblad diretto" reference: direct steady-state solve of the same
Nv=6-truncated dense Liouvillian, no Trotter splitting or bond truncation."""
function run_exact_steady_state(ΓL::Float64)
    nC = cold_occupations(energies_e, sim.TC; kB=sim.kB)
    jump_ops = Lindblad.JumpOperator[]
    append!(jump_ops, Lindblad.build_hot_operators_vib(U_e; γH=sim.γH, nH=sim.nH, Nv=params.Nv))
    push!(jump_ops, Lindblad.build_load_operator_vib(U_e; ΓL=ΓL, Nv=params.Nv))
    append!(jump_ops, Lindblad.build_cold_operators_vib(U_e; γC=sim.γC, nC_matrix=nC, Nv=params.Nv))
    append!(jump_ops, Lindblad.build_vibrational_damping_operators(dim_e, params.Nv; γM=sim.γM, ωv=params.ωv, TM=sim.TM, kB=sim.kB))
    L = Liouvillian.build_liouvillian(H_ref, jump_ops)
    return Solver.steady_state(L)
end

# ----------------------------------------------------------------------------
# Sweep: append-and-sort workflow -- set ΓL_list to just the new points you
# want to add THIS run; they get merged into outputs/numeric_vectors/
# iv_curve_comparison.csv (overwriting a matching ΓL, not duplicating it),
# and plots are regenerated from the full accumulated file. Empty by default
# so re-running as-is is a safe no-op (just replots the existing data).
# ----------------------------------------------------------------------------

ΓL_list = [0.1, 450.0, 675.0, 850.0, 30000.0]

println("Δt=$(Δt), chi_max=$(chi_max), coeff_tol=$(coeff_tol), max_steps=$(max_steps), check_every=$(check_every), conv_tol=$(conv_tol)")
flush(stdout)

results = NamedTuple[]
for ΓL in ΓL_list
    println("\n--- ΓL = $(ΓL) ---")
    flush(stdout)

    t0 = time()
    dampf = run_dampf_steady_state(ΓL)
    t_dampf = time() - t0
    println("DAMPF: steps=$(dampf.steps) converged=$(dampf.converged) wall_time=$(round(t_dampf, digits=1))s")
    flush(stdout)

    t0 = time()
    ρ_exact = run_exact_steady_state(ΓL)
    t_exact = time() - t0
    println("Lindblad diretto: wall_time=$(round(t_exact, digits=3))s")
    flush(stdout)

    I_dampf = Observables.current(dampf.ρ_e, U_e, ΓL, idx_low)
    V_dampf = Observables.voltage(dampf.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    P_dampf = Observables.power(I_dampf, V_dampf)

    I_exact = Observables.current_vib(ρ_exact, U_e, ΓL, idx_low, params.Nv)
    V_exact = Observables.voltage_vib(ρ_exact, U_e, energies_e, idx_low, idx_g, params.Nv; kB=sim.kB, TC=sim.TC)
    P_exact = Observables.power(I_exact, V_exact)

    dense_dampf = DampfSystem.blocks_to_dense(dampf.blocks, siteinds, model)
    trace_dist = 0.5 * sum(abs.(eigvals(Hermitian((dense_dampf .- ρ_exact .+ (dense_dampf .- ρ_exact)') ./ 2))))

    println("I: dampf=$(I_dampf)  Lindblad diretto=$(I_exact)  rel_err=$(abs(I_dampf - I_exact) / max(abs(I_exact), 1e-30))")
    println("V: dampf=$(V_dampf)  Lindblad diretto=$(V_exact)  abs_err=$(abs(V_dampf - V_exact))")
    println("P: dampf=$(P_dampf)  Lindblad diretto=$(P_exact)  rel_err=$(abs(P_dampf - P_exact) / max(abs(P_exact), 1e-30))")
    println("trace distance(ρ_dampf, ρ_exact) = $(trace_dist)")
    flush(stdout)

    push!(results, (ΓL=ΓL, I_dampf=I_dampf, I_exact=I_exact, V_dampf=V_dampf, V_exact=V_exact,
                     P_dampf=P_dampf, P_exact=P_exact, trace_dist=trace_dist,
                     steps=dampf.steps, converged=dampf.converged))
end

println("\n=== Summary ===")
for r in results
    println("ΓL=$(r.ΓL)  I_dampf=$(r.I_dampf)  I_exact=$(r.I_exact)  P_dampf=$(r.P_dampf)  P_exact=$(r.P_exact)  trace_dist=$(r.trace_dist)  steps=$(r.steps) converged=$(r.converged)")
end

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_comparison.csv")
all_results = merge_sorted_results(load_comparison_csv(csv_path), results)
save_comparison_csv(csv_path, all_results)
println("\nSaved CSV ($(length(all_results)) total points, $(length(results)) from this run): $(csv_path)")

reference_path = joinpath(@__DIR__, "..", "..", "outputs", "numeric_vectors", "vibrational", "full",
                           "$(params.Nv)_modes", "g_$(Int(round(model.modes[1].g)))", "observables_vs_gammaL.csv")
reference = load_reference_sweep(reference_path)
println(reference === nothing ? "\nNo dense reference sweep found at $(reference_path)" :
        "\nLoaded dense reference sweep ($(length(reference.ΓL)) points): $(reference_path)")

imgs_outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "killoran_validation")
plot_comparison(all_results, imgs_outdir; reference=reference)
plot_iv_dampf(all_results, imgs_outdir)
plot_current_vs_voltage_comparison(all_results, imgs_outdir; reference=reference)
plot_power_vs_voltage_comparison(all_results, imgs_outdir; reference=reference)
println("Saved plots in: $(imgs_outdir)")
