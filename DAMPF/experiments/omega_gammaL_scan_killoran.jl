include("../system/system.jl")
using .DampfSystem
using ITensors
using ITensorMPS
using Plots

include("../utils/csv_io.jl")
using .DampfCsvIO

include("../../src/Observables.jl")
include("../../config/parameters.jl")
using .Observables

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
2D generalization of `omega_scan_killoran.jl`: same physical question (how
does the vibronic power enhancement respond to detuning ω away from the
ε1-ε2=200 cm^-1 resonance -- Killoran, Huelga & Plenio arXiv:1412.4136v2,
sec. IV.A), now scanned jointly against ΓL as well, around the region where
P(ΓL) peaks (see outputs/numeric_vectors/iv_curve_comparison.csv).

Same reasoning for reuse as the 1D scan: only `role_props` depends on ω;
H_e/siteinds/τ never depend on ω or ΓL; P_e and the g=0 baseline depend on
ΓL but not ω, so both are built ONCE PER ΓL (outer loop) and reused across
the whole inner ω loop -- avoids recomputing the (more expensive) ΓL-
dependent pieces 9x per ΓL for no reason.

Writes a separate CSV (`omega_gammaL_scan_killoran.csv`, columns
omega,GammaL,...) rather than extending `omega_scan_killoran.csv`'s
single-ΓL schema -- that file is left untouched. `ΓL_list` here includes
900.0 (the single ΓL_ref already covered by the 1D scan) so this new file
is self-contained and doesn't depend on stitching two different schemas
together.
"""

params = default_params()
sim = default_simulation_params()
model = DampfSystem.killoran_benchmark_model(Nb=params.Nv)  # ω/g on model.modes are placeholders, never read by the dynamics -- see role_props below
dim_e = DampfSystem.dim_e(model)
idx_low = dim_e - 1
idx_g = dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)

siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

# Same validated knobs as omega_scan_killoran.jl (M=2, tuned against the
# dense/RK4 reference in check/04).
Δt = 2.0e-4
chi_max = 3
coeff_tol = 1.0e-10
max_steps = 10_000
check_every = 500
conv_tol = 1.0e-6

"""Run DAMPF to steady state for a given (ω, g) pair on both modes (s1, s2)
and a given P_e (built by the caller for the ΓL currently being processed)."""
function run_dampf_for_ωg(ω::Float64, g::Float64, P_e::AbstractMatrix, ΓL::Float64; label::String)
    modes_ωg = [
        DampfSystem.DampfMode(2, ω, g, 5.3, 293.0, params.Nv),
        DampfSystem.DampfMode(3, ω, g, 5.3, 293.0, params.Nv),
    ]
    role_props = [DampfSystem.build_mode_role_propagators(mode, siteinds[q], Δt) for (q, mode) in enumerate(modes_ωg)]
    return DampfSystem.run_to_steady_state(
        model, role_props, siteinds, τ, P_e;
        chi_max=chi_max, coeff_tol=coeff_tol, Δt=Δt, max_steps=max_steps, check_every=check_every, conv_tol=conv_tol,
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL, idx_low)), label=label,
    )
end

# ----------------------------------------------------------------------------
# Grid: 9 ω values (same as omega_scan_killoran.jl) x 11 ΓL values around the
# P(ΓL) maximum (outputs/numeric_vectors/iv_curve_comparison.csv peaks
# ~850-900) -- 900.0 included so this file is self-contained.
# ----------------------------------------------------------------------------

ω_list = [100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0]
g_coupling = 55.0
ΓL_list = [
    5.0, 10.0, 20.0, 30.0, 40.0,
    50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 550.0, 600.0,
    650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0,
    1200.0, 1350.0, 1500.0,
    1700.0, 2000.0, 2500.0, 3000.0, 4000.0,
]

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "omega_gammaL_scan_killoran.csv")

# The grid is a full product ω_list x ΓL_list. Extending either list later
# means only the NEWLY exposed (ω,ΓL) combinations need computing -- so skip
# anything already present in the CSV instead of recomputing the whole grid.
existing = load_omega_gammaL_scan_csv(csv_path)
done_keys = Set((r.ω, r.ΓL) for r in existing)

for ΓL in ΓL_list
    ω_needed = [ω for ω in ω_list if (ω, ΓL) ∉ done_keys]
    if isempty(ω_needed)
        println("\nΓL = $(ΓL): all $(length(ω_list)) ω already in CSV, skipping.")
        flush(stdout)
        continue
    end
    println("\n============ ΓL = $(ΓL)  ($(length(ω_needed))/$(length(ω_list)) ω to compute) ============")
    flush(stdout)

    diss = DampfSystem.build_electronic_dissipators(
        H_e; idx_low=idx_low, idx_g=idx_g,
        γH=sim.γH, nH=sim.nH, ΓL=ΓL,
        γC=sim.γC, TC=sim.TC, kB=sim.kB,
    ).ops
    P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)

    println("--- g=0 baseline (ΓL=$(ΓL)) ---")
    flush(stdout)
    baseline = run_dampf_for_ωg(200.0, 0.0, P_e, ΓL; label="ΓL=$(ΓL) g=0")
    I_g0 = Observables.current(baseline.ρ_e, U_e, ΓL, idx_low)
    V_g0 = Observables.voltage(baseline.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    P_g0 = Observables.power(I_g0, V_g0)
    println("baseline: I=$(I_g0)  V=$(V_g0)  P=$(P_g0)")
    flush(stdout)

    for ω in ω_needed
        δ = ω - 200.0
        println("--- ω = $(ω)  (δ=$(δ)),  ΓL = $(ΓL) ---")
        flush(stdout)

        t0 = time()
        dampf = run_dampf_for_ωg(ω, g_coupling, P_e, ΓL; label="ΓL=$(ΓL) ω=$(ω)")
        t_dampf = time() - t0
        println("DAMPF: steps=$(dampf.steps) converged=$(dampf.converged) wall_time=$(round(t_dampf, digits=1))s")
        flush(stdout)

        I_dampf = Observables.current(dampf.ρ_e, U_e, ΓL, idx_low)
        V_dampf = Observables.voltage(dampf.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
        P_dampf = Observables.power(I_dampf, V_dampf)
        enhancement_pct = (P_dampf - P_g0) / P_g0 * 100

        println("I=$(I_dampf)  V=$(V_dampf)  P=$(P_dampf)  enhancement=$(round(enhancement_pct, digits=2))%")
        flush(stdout)

        r = (ω=ω, ΓL=ΓL, I_dampf=I_dampf, V_dampf=V_dampf, P_dampf=P_dampf,
             enhancement_pct=enhancement_pct, steps=dampf.steps, converged=dampf.converged)

        all_so_far = merge_sorted_results(load_omega_gammaL_scan_csv(csv_path), [r]; key=x -> (x.ω, x.ΓL))
        save_omega_gammaL_scan_csv(csv_path, all_so_far)
        println("Saved incrementally ($(length(all_so_far)) total points in CSV): $(csv_path)")
        flush(stdout)
    end
end

println("\nDone. Full grid in: $(csv_path)")
