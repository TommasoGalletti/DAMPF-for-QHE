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
Scale-up experiment: same ground+3-site Killoran electronic structure as
`iv_curve_dampf_killoran.jl`, but now s1 and s2 each carry TWO independent
vibrational modes (`killoran_s1x2_s2x2_model`, s3 stays mode-less) -- 4
modes total, distributed 2/2/0 across the sites instead of 1/1/1
(`killoran_three_mode_model`) or 1/1/0 (the original benchmark). Exercises
`modes_on_site` returning more than one mode for the same electronic site,
a code path the architecture was always generic over but never actually
exercised with >1 mode/site before this experiment.

Like `iv_curve_dampf_v1v2v3.jl`, this runs DAMPF only -- with M=4 the dense
joint vibrational space (Nb^4=1296 for Nb=6) makes the "Lindblad diretto"
dense reference even less attainable than for M=3, so there is no
comparison here, just DAMPF's own I/V/P sweep.

Reuses the same generic pieces as the other two experiments: DAMPF/dynamics/
steady_state.jl (Trotter-to-steady-state driver), DAMPF/utils/csv_io.jl
(save_dampf_sweep_csv/load_dampf_sweep_csv), DAMPF/plots/iv_curve_plots.jl
(plot_dampf_vs_gammaL, plot_iv_dampf).
"""

params = default_params()
sim = default_simulation_params()
model = DampfSystem.killoran_s1x2_s2x2_model(Nb=params.Nv)
dim_e = DampfSystem.dim_e(model)
idx_low = dim_e - 1
idx_g = dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)

siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

# ----------------------------------------------------------------------------
# Numerical knobs: starting from the M=3 (v1v2v3) values (chi_max=4,
# max_steps=12_000), which were themselves bumped from the killoran M=2
# defaults (3 / 10_000) after chi_max=3 turned out to genuinely undersample
# the steady state, not just add noise (see iv_curve_dampf_v1v2v3.jl). NOT
# independently re-validated for this 2-modes-on-one-site topology -- if
# convergence looks off (large chi_max-dependence, or non-shrinking
# checkpoint deltas like the M=3 chi_max=3 case), re-check chi_max the same
# way before trusting the numbers.
# ----------------------------------------------------------------------------

Δt = 2.0e-4
chi_max = 4
coeff_tol = 1.0e-10
max_steps = 12_000
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

# ----------------------------------------------------------------------------
# Sweep: append-and-sort workflow -- set ΓL_list to just the new points you
# want to add THIS run; they get merged into outputs/numeric_vectors/
# iv_curve_s1x2_s2x2_200.csv (overwriting a matching ΓL, not duplicating it),
# and plots are regenerated from the full accumulated file. Empty by default
# so re-running as-is is a safe no-op (just replots the existing data).
# ----------------------------------------------------------------------------

ΓL_list = [0.1, 0.2, 0.5, 5.0]

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_s1x2_s2x2_200.csv")

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

    # Incremental save: merge just this point into the on-disk CSV right
    # away (same append-and-sort semantics as the end-of-run save), so a
    # stopped/killed run never loses points that already finished.
    all_so_far = merge_sorted_results(load_dampf_sweep_csv(csv_path), [r])
    save_dampf_sweep_csv(csv_path, all_so_far)
    println("Saved incrementally ($(length(all_so_far)) total points in CSV): $(csv_path)")
    flush(stdout)
end

println("\n=== Summary ===")
for r in results
    println("ΓL=$(r.ΓL)  I_dampf=$(r.I_dampf)  V_dampf=$(r.V_dampf)  P_dampf=$(r.P_dampf)  steps=$(r.steps) converged=$(r.converged)")
end

all_results = load_dampf_sweep_csv(csv_path)
println("\nCSV has $(length(all_results)) total points ($(length(results)) from this run): $(csv_path)")

imgs_outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "s1x2_s2x2_200")
plot_dampf_vs_gammaL(all_results, imgs_outdir)
plot_iv_dampf(all_results, imgs_outdir)
println("Saved plots in: $(imgs_outdir)")
