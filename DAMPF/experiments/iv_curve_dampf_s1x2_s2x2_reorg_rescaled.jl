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
Reorganization-energy-preserving variant of iv_curve_dampf_s1x2_s2x2_freq_sweep.jl.

That script attaches a second mode (frequency ω2) to s1/s2 alongside the
normal 200 cm^-1 mode, keeping BOTH couplings at the benchmark value g=55
cm^-1 unchanged. Per-site reorganization energy λ=Σᵢgᵢ²/ωᵢ (Caldeira-Leggett)
then grows from the single-mode benchmark's λ₀=55²/200≈15.125 cm^-1 to
λ₀+55²/ω2 -- e.g. ≈3λ₀ at ω2=100 -- over-dressing the excitonic states
relative to the benchmark they were tuned against.

This script instead rescales BOTH couplings by a common factor
s=√(λ₀/λ_unscaled) so that λ stays pinned at λ₀ regardless of ω2, isolating
the effect of detuning the second mode from the effect of simply adding
more total coupling. Output goes to iv_curve_s1x2_s2x2_reorg_<ω2>.csv /
imgs/s1x2_s2x2_reorg_<ω2>/ -- entirely separate from the unscaled sweep's
files, so nothing there is touched.

Restricted by design to a handful of ΓL points bracketing the maximum-power
region already found by the unscaled sweep (500-1200 cm^-1) rather than the
full 20/31-point grid: the question here is only whether Pmax moves once λ
is controlled for, not a full re-characterization of the I-V curve.

Usage: julia -t auto --project=... iv_curve_dampf_s1x2_s2x2_reorg_rescaled.jl <ω2_in_cm-1> [ΓL_list_comma_separated]
"""

if isempty(ARGS)
    error("Usage: julia iv_curve_dampf_s1x2_s2x2_reorg_rescaled.jl <ω2_in_cm-1> [ΓL_list_comma_separated]  (e.g. 100  or  100 500,800,1200)")
end
ω2 = parse(Float64, ARGS[1])
ω2_tag = string(Int(round(ω2)))
ΓL_list_override = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : nothing

const G0 = 55.0
const ω_RESONANT = 200.0
λ0 = G0^2 / ω_RESONANT
λ_unscaled = λ0 + G0^2 / ω2
s = sqrt(λ0 / λ_unscaled)
g_rescaled = G0 * s

println("=== Reorganization-energy rescaling ===")
println("λ0 (single-mode benchmark, g=$(G0), ω=$(ω_RESONANT)) = $(λ0) cm^-1")
println("λ_unscaled (two modes at g=$(G0) each, ω2=$(ω2)) = $(λ_unscaled) cm^-1  ($(round(λ_unscaled/λ0, digits=2))x λ0)")
println("rescale factor s = √(λ0/λ_unscaled) = $(s)")
println("g_rescaled = $(G0) * s = $(g_rescaled) cm^-1 (applied to BOTH modes on each site)")
flush(stdout)

params = default_params()
sim = default_simulation_params()
Nb = params.Nv

model = DampfSystem.DampfModel(
    -10000.0,
    [300.0, 300.0, 0.0],
    [0.0 100.0 0.0; 100.0 0.0 0.0; 0.0 0.0 0.0],
    [
        DampfSystem.DampfMode(2, ω_RESONANT, g_rescaled, 5.3, 293.0, Nb),  # s1 normal
        DampfSystem.DampfMode(2, ω2, g_rescaled, 5.3, 293.0, Nb),          # s1 at ω2
        DampfSystem.DampfMode(3, ω_RESONANT, g_rescaled, 5.3, 293.0, Nb),  # s2 normal
        DampfSystem.DampfMode(3, ω2, g_rescaled, 5.3, 293.0, Nb),          # s2 at ω2
    ],
)
dim_e = DampfSystem.dim_e(model)
idx_low = dim_e - 1
idx_g = dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)

siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

Δt = 2.0e-4
chi_max = 4
coeff_tol = 1.0e-10
max_steps = 12_000
check_every = 500
conv_tol = 1.0e-6

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
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL, idx_low)), label="ω2=$(ω2) [reorg-rescaled] ΓL=$(ΓL)",
    )
end

# ----------------------------------------------------------------------------
# Restricted to the ΓL region where the unscaled sweep's maximum power fell
# (500-1200 cm^-1, see the top-3-points-per-ω2 analysis of the unscaled
# scan), since that's the only region this comparison needs.
# ----------------------------------------------------------------------------

ΓL_list = ΓL_list_override === nothing ?
    [500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0] :
    ΓL_list_override

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_s1x2_s2x2_reorg_$(ω2_tag).csv")

println("\nThreads.nthreads() = $(Threads.nthreads())")
println("ω2=$(ω2) cm^-1 (reorg-rescaled, g=$(g_rescaled) cm^-1)")
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

    I_dampf = Observables.current(dampf.ρ_e, U_e, ΓL, idx_low)
    V_dampf = Observables.voltage(dampf.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    P_dampf = Observables.power(I_dampf, V_dampf)

    println("I=$(I_dampf)  V=$(V_dampf)  P=$(P_dampf)")
    flush(stdout)

    r = (ΓL=ΓL, I_dampf=I_dampf, V_dampf=V_dampf, P_dampf=P_dampf,
         steps=dampf.steps, converged=dampf.converged)
    push!(results, r)

    all_so_far = merge_sorted_results(load_dampf_sweep_csv(csv_path), [r])
    save_dampf_sweep_csv(csv_path, all_so_far)
    println("Saved incrementally ($(length(all_so_far)) total points in CSV): $(csv_path)")
    flush(stdout)
end

println("\n=== Summary (ω2=$(ω2), reorg-rescaled g=$(g_rescaled)) ===")
for r in results
    println("ΓL=$(r.ΓL)  I_dampf=$(r.I_dampf)  V_dampf=$(r.V_dampf)  P_dampf=$(r.P_dampf)  steps=$(r.steps) converged=$(r.converged)")
end

all_results = load_dampf_sweep_csv(csv_path)
println("\nCSV has $(length(all_results)) total points ($(length(results)) from this run): $(csv_path)")

imgs_outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "s1x2_s2x2_reorg_$(ω2_tag)")
plot_dampf_vs_gammaL(all_results, imgs_outdir)
plot_iv_dampf(all_results, imgs_outdir)
println("Saved plots in: $(imgs_outdir)")
