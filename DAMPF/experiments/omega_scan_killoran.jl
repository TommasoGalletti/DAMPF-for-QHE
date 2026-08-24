include("../system/system.jl")
using .DampfSystem
using ITensors
using ITensorMPS
using Plots

include("../plots/iv_curve_plots.jl")
using .DampfPlots

include("../utils/csv_io.jl")
using .DampfCsvIO

include("../../src/Observables.jl")
include("../../config/parameters.jl")
using .Observables

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Reproduces/extends Killoran, Huelga & Plenio (arXiv:1412.4136v2), section
IV.A "Off-resonant vibrations": ωv=200 cm^-1 is resonant with the
delocalized-exciton splitting ε1-ε2=2*J12=200 cm^-1 in the benchmark model
(s1/s2 coupled, s3 isolated); the paper shows the Jaynes-Cummings-like
vibronic power enhancement survives detuning (~9.3%->~9% at δ=50 cm^-1
mismatch). Here we scan ω around resonance with DAMPF at a fixed reference
ΓL (near the power-maximizing point) and quantify the same enhancement,
relative to a g=0 (no vibronic coupling) baseline.

Uses `killoran_benchmark_model` -- the ONLY physically sensible choice here,
not v1v2v3/s1x2_s2x2: the Jaynes-Cummings channel requires a mode on a site
that participates in electronic delocalization (s1/s2, coupled via J12);
a mode on the electronically-isolated s3 (as in v1v2v3) is a spectator to
this mechanism, per the paper's own remark. It's also the only model with
a genuine "Lindblad diretto" validation (M=2), and the cheapest per point.

Only `role_props` depends on ω (via H_v=ω*a†a and the vibronic term) -- NOT
H_e, NOT the dissipators, NOT P_e (ΓL-dependent only) -- so `model`, `H_e`,
`siteinds`, `τ`, and P_e (built once for the fixed ΓL_ref) are all reused
unchanged across the whole scan; only role_props gets rebuilt per ω (cheap:
a handful of 36x36 matrix exponentials, not the ~10_000-step Trotter loop).
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

# ----------------------------------------------------------------------------
# Numerical knobs: the ORIGINAL killoran-validation values (Δt/chi_max tuned
# against the RK4/dense reference for this exact M=2 model in check/04 --
# see SPIEGAZIONE_iv_curve_dampf_killoran.md section 2). Unlike v1v2v3/
# s1x2_s2x2, no re-tuning needed here.
# ----------------------------------------------------------------------------

Δt = 2.0e-4
chi_max = 3
coeff_tol = 1.0e-10
max_steps = 10_000
check_every = 500
conv_tol = 1.0e-6

# Fixed reference ΓL: near the power-maximizing point found in the original
# killoran sweep (outputs/numeric_vectors/iv_curve_comparison.csv, P_dampf
# peaks around ΓL~800-900).
ΓL_ref = 900.0

diss = DampfSystem.build_electronic_dissipators(
    H_e; idx_low=idx_low, idx_g=idx_g,
    γH=sim.γH, nH=sim.nH, ΓL=ΓL_ref,
    γC=sim.γC, TC=sim.TC, kB=sim.kB,
).ops
P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)

"""Run DAMPF to steady state for a given (ω, g) pair on both modes (s1, s2),
at the fixed ΓL_ref. Only role_props depends on (ω,g); everything else
above is reused unchanged."""
function run_dampf_for_ωg(ω::Float64, g::Float64; label::String)
    modes_ωg = [
        DampfSystem.DampfMode(2, ω, g, 5.3, 293.0, params.Nv),
        DampfSystem.DampfMode(3, ω, g, 5.3, 293.0, params.Nv),
    ]
    role_props = [DampfSystem.build_mode_role_propagators(mode, siteinds[q], Δt) for (q, mode) in enumerate(modes_ωg)]
    return DampfSystem.run_to_steady_state(
        model, role_props, siteinds, τ, P_e;
        chi_max=chi_max, coeff_tol=coeff_tol, Δt=Δt, max_steps=max_steps, check_every=check_every, conv_tol=conv_tol,
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL_ref, idx_low)), label=label,
    )
end

# ----------------------------------------------------------------------------
# g=0 baseline: independent of ω (no vibronic coupling -> the mode cannot
# affect the electronic dynamics at all), so computed ONCE and reused as the
# denominator for every ω's enhancement percentage.
# ----------------------------------------------------------------------------

println("--- g=0 baseline (no vibronic coupling) ---")
flush(stdout)
baseline = run_dampf_for_ωg(200.0, 0.0; label="g=0")
I_g0 = Observables.current(baseline.ρ_e, U_e, ΓL_ref, idx_low)
V_g0 = Observables.voltage(baseline.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
P_g0 = Observables.power(I_g0, V_g0)
println("baseline: I=$(I_g0)  V=$(V_g0)  P=$(P_g0)")
flush(stdout)

# ----------------------------------------------------------------------------
# ω sweep: δ = ω - 200 in [-100,+100] cm^-1 (covers and exceeds the paper's
# own δ=50 example), g=55 (the benchmark's coupling) throughout.
# ----------------------------------------------------------------------------

ω_list = [100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0]
g_coupling = 55.0

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "omega_scan_killoran.csv")

results = NamedTuple[]
for ω in ω_list
    δ = ω - 200.0
    println("\n--- ω = $(ω)  (δ=$(δ)) ---")
    flush(stdout)

    t0 = time()
    dampf = run_dampf_for_ωg(ω, g_coupling; label="ω=$(ω)")
    t_dampf = time() - t0
    println("DAMPF: steps=$(dampf.steps) converged=$(dampf.converged) wall_time=$(round(t_dampf, digits=1))s")
    flush(stdout)

    I_dampf = Observables.current(dampf.ρ_e, U_e, ΓL_ref, idx_low)
    V_dampf = Observables.voltage(dampf.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    P_dampf = Observables.power(I_dampf, V_dampf)
    enhancement_pct = (P_dampf - P_g0) / P_g0 * 100

    println("I=$(I_dampf)  V=$(V_dampf)  P=$(P_dampf)  enhancement=$(round(enhancement_pct, digits=2))%")
    flush(stdout)

    r = (ω=ω, I_dampf=I_dampf, V_dampf=V_dampf, P_dampf=P_dampf,
         enhancement_pct=enhancement_pct, steps=dampf.steps, converged=dampf.converged)
    push!(results, r)

    all_so_far = merge_sorted_results(load_omega_scan_csv(csv_path), [r]; key=x -> x.ω)
    save_omega_scan_csv(csv_path, all_so_far)
    println("Saved incrementally ($(length(all_so_far)) total points in CSV): $(csv_path)")
    flush(stdout)
end

println("\n=== Summary (ΓL_ref=$(ΓL_ref), g=0 baseline: I=$(I_g0) V=$(V_g0) P=$(P_g0)) ===")
for r in results
    println("ω=$(r.ω)  I=$(r.I_dampf)  V=$(r.V_dampf)  P=$(r.P_dampf)  enhancement=$(round(r.enhancement_pct, digits=2))%  steps=$(r.steps) converged=$(r.converged)")
end

all_results = load_omega_scan_csv(csv_path)
imgs_outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "omega_scan")
plot_omega_scan(all_results, imgs_outdir; ΓL_ref=ΓL_ref)
println("Saved plots in: $(imgs_outdir)")
