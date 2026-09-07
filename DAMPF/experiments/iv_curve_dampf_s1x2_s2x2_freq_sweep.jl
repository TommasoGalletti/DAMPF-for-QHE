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
Parameterized generalization of iv_curve_dampf_s1x2_s2x2_overtone.jl (which
fixed the second-mode frequency at 400 cm^-1) and _300.jl (300 cm^-1): takes
the second mode's frequency ω2 as a command-line argument, so a new value
(250, 350, 450, 500, ...) is one invocation instead of one new file. Same
ground+3-site electronic structure, same 2-modes-per-site topology (s1, s2
each carry one normal mode at 200 cm^-1 and one at ω2; s3 stays mode-less),
same g/γ/T/Nb, same numerical knobs, same 20 ΓL points as its two existing
siblings, so every ω2 run this produces stays directly comparable to those
and to each other -- the point of this generalization is exactly to build
an "observable vs ω2" figure later across several such runs.

Usage: julia -t auto --project=... iv_curve_dampf_s1x2_s2x2_freq_sweep.jl <ω2_in_cm-1> [ΓL_list_comma_separated]
Example (default 20-point ΓL_list): julia -t auto --project=... iv_curve_dampf_s1x2_s2x2_freq_sweep.jl 500
Example (just add these 3 points, merged into the existing CSV): julia -t auto --project=... iv_curve_dampf_s1x2_s2x2_freq_sweep.jl 500 35,200,400

Model is built inline here (not added to parameters/parameters_physical.jl)
so this experiment can't affect any other script.

Output: outputs/numeric_vectors/iv_curve_s1x2_s2x2_<ω2>.csv and
outputs/imgs/s1x2_s2x2_<ω2>/ -- <ω2> is the integer-rounded frequency (e.g.
"500"), matching the naming already used by the existing _300.jl output.
Never touches any other experiment's files, including the existing
_overtone (400) and _300 runs (those keep living in their own dedicated
scripts/outputs, untouched by this one).
"""

if isempty(ARGS)
    error("Usage: julia iv_curve_dampf_s1x2_s2x2_freq_sweep.jl <ω2_in_cm-1> [ΓL_list_comma_separated]  (e.g. 500  or  500 35,200,400)")
end
ω2 = parse(Float64, ARGS[1])
ω2_tag = string(Int(round(ω2)))
ΓL_list_override = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : nothing

params = default_params()
sim = default_simulation_params()
Nb = params.Nv

model = DampfSystem.DampfModel(
    -10000.0,
    [300.0, 300.0, 0.0],
    [0.0 100.0 0.0; 100.0 0.0 0.0; 0.0 0.0 0.0],
    [
        DampfSystem.DampfMode(2, 200.0, 55.0, 5.3, 293.0, Nb),  # s1 normal
        DampfSystem.DampfMode(2, ω2, 55.0, 5.3, 293.0, Nb),     # s1 at ω2
        DampfSystem.DampfMode(3, 200.0, 55.0, 5.3, 293.0, Nb),  # s2 normal
        DampfSystem.DampfMode(3, ω2, 55.0, 5.3, 293.0, Nb),     # s2 at ω2
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
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL, idx_low)), label="ω2=$(ω2) ΓL=$(ΓL)",
    )
end

# ----------------------------------------------------------------------------
# Same 20 ΓL points as the _overtone (400) and _300 siblings, for direct
# point-by-point comparability across every ω2 this script is run with.
# Append-and-sort workflow, same as the other iv_curve_dampf_* scripts.
# ----------------------------------------------------------------------------

ΓL_list = ΓL_list_override === nothing ?
    [0.1, 0.2, 0.5, 1.0, 5.0, 10.0, 20.0, 40.0, 50.0, 100.0, 150.0, 300.0, 500.0, 700.0, 900.0, 1000.0, 1200.0, 2000.0, 5000.0, 15000.0] :
    ΓL_list_override

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_s1x2_s2x2_$(ω2_tag).csv")

println("Threads.nthreads() = $(Threads.nthreads())")
println("ω2=$(ω2) cm^-1")
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

println("\n=== Summary (ω2=$(ω2)) ===")
for r in results
    println("ΓL=$(r.ΓL)  I_dampf=$(r.I_dampf)  V_dampf=$(r.V_dampf)  P_dampf=$(r.P_dampf)  steps=$(r.steps) converged=$(r.converged)")
end

all_results = load_dampf_sweep_csv(csv_path)
println("\nCSV has $(length(all_results)) total points ($(length(results)) from this run): $(csv_path)")

imgs_outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "s1x2_s2x2_$(ω2_tag)")
plot_dampf_vs_gammaL(all_results, imgs_outdir)
plot_iv_dampf(all_results, imgs_outdir)
println("Saved plots in: $(imgs_outdir)")
