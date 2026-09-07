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
Cluster experiment: `killoran_four_site_model` (ground + 4 electronic
sites, dim_e=5, bright TRIO s1-s2-s4 with s3 isolated/mode-less, M=3 modes
total -- see check/cluster_smoke_test.jl for the verified exciton spectrum
and idx_low/idx_g). DAMPF-only, no "Lindblad diretto" comparison: with
dim_e=5, Nb=6, M=3, the dense reference would need
dim_e*Nb^3=5*216=1080-dimensional H_ref and a 1080^2 x 1080^2 Liouvillian,
even less attainable than the dim_e=4 cases.

chi_max/max_steps set from the diagnostic in check/
cluster_chi_max_diagnostic.jl (+cluster_chi_max_diagnostic_6.jl) -- see the
values below and the note on which chi_max was ultimately selected and why.

Reuses the same generic pieces as the other DAMPF-only experiments:
dynamics/steady_state.jl (driver), utils/csv_io.jl
(save_dampf_sweep_csv/load_dampf_sweep_csv), plots/iv_curve_plots.jl
(plot_dampf_vs_gammaL, plot_iv_dampf).
"""

params = default_params()
sim = default_simulation_params()
model = DampfSystem.killoran_four_site_model(Nb=params.Nv)
dim_e = DampfSystem.dim_e(model)
idx_low = dim_e - 1
idx_g = dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)

siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

# ----------------------------------------------------------------------------
# Numerical knobs -- SET FROM THE chi_max DIAGNOSTIC (check/
# cluster_chi_max_diagnostic.jl + cluster_chi_max_diagnostic_6.jl), single ΓL=500 point:
#   chi_max=3: I=7.1218   chi_max=4: I=7.1594 (+0.53%)
#   chi_max=5: I=7.0797 (-1.11%)   chi_max=6: I=7.0990 (+0.27%)
# The shift neither shrinks nor keeps a consistent sign across chi_max=3..6
# -- unlike v1v2v3/s1x2_s2x2, this dim_e=5 topology does not show clean
# convergence in bond dimension within the range tested (chasing chi_max
# higher than 6 would cost more, ~chi_max^3, for an uncertain payoff).
# chi_max=5 is used here (not 6) since 6 did not resolve the oscillation and
# costs ~27% more per point; the ~1% band across chi_max=3..6 should be
# treated as the approximate uncertainty on every result below.
# ----------------------------------------------------------------------------

Δt = 2.0e-4
chi_max = 5
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
# Sweep: append-and-sort workflow, same as v1v2v3/s1x2_s2x2 -- ΓL_list holds
# only the NEW points for this run; merged into outputs/numeric_vectors/
# iv_curve_cluster.csv (overwriting a matching ΓL, not duplicating it).
# Points chosen from the same ΓL values already tested for s1x2_s2x2.
# ----------------------------------------------------------------------------

ΓL_list = [40.0, 150.0, 300.0, 10000.0, 15000.0]

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_cluster.csv")

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
    # away, so a stopped/killed run never loses points that already finished.
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

imgs_outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "cluster")
plot_dampf_vs_gammaL(all_results, imgs_outdir)
plot_iv_dampf(all_results, imgs_outdir)
println("Saved plots in: $(imgs_outdir)")
