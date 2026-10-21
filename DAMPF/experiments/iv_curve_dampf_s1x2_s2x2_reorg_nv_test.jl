include("../system/system.jl")
using .DampfSystem
using ITensors
using ITensorMPS

include("../utils/csv_io.jl")
using .DampfCsvIO

include("../../src/Observables.jl")
include("../../config/parameters.jl")
using .Observables

"""
Fock-truncation diagnostic for the reorganization-energy-rescaled
s1x2_s2x2 scan (iv_curve_dampf_s1x2_s2x2_reorg_rescaled.jl): identical
physics and numerical settings (chi_max=4, Δt=2e-4, 12000 steps), but with
each mode truncated at Nv Fock states instead of the production Nv=6.

Purpose: check whether the point-to-point scatter of the noisy curves
(e.g. ω2=250) shrinks with a larger Fock space, using a clean curve
(e.g. ω2=450) as control, on the same ΓL points as the production grid.

One (ω2, ΓL list) per invocation; each invocation writes its own CSV
numeric_vectors/s1x2_s2x2_reorg_nv<Nv>/iv_curve_s1x2_s2x2_reorg_nv<Nv>_<ω2>_<ΓLtag>.csv
so that one-point-per-job SLURM runs never share a file. Log label
"[ω2=… [reorg-nv<Nv>] ΓL=…]" keeps the checkpoint lines parseable for the
trailing-checkpoint average.

Usage: julia --project=... iv_curve_dampf_s1x2_s2x2_reorg_nv_test.jl <ω2> <ΓL_list_comma_separated> <Nv>
"""

length(ARGS) >= 3 || error("Usage: julia iv_curve_dampf_s1x2_s2x2_reorg_nv_test.jl <ω2> <ΓL_list> <Nv>  (e.g. 250 700 8)")
ω2 = parse(Float64, ARGS[1])
ΓL_list = parse.(Float64, split(ARGS[2], ","))
Nb = parse(Int, ARGS[3])
ω2_tag = string(Int(round(ω2)))
ΓL_tag = join([replace(string(x), ".0" => "") for x in ΓL_list], "_")

const G0 = 55.0
const ω_RESONANT = 200.0
λ0 = G0^2 / ω_RESONANT
λ_unscaled = λ0 + G0^2 / ω2
g_rescaled = G0 * sqrt(λ0 / λ_unscaled)

sim = default_simulation_params()

model = DampfSystem.DampfModel(
    -10000.0,
    [300.0, 300.0, 0.0],
    [0.0 100.0 0.0; 100.0 0.0 0.0; 0.0 0.0 0.0],
    [
        DampfSystem.DampfMode(2, ω_RESONANT, g_rescaled, 5.3, 293.0, Nb),
        DampfSystem.DampfMode(2, ω2, g_rescaled, 5.3, 293.0, Nb),
        DampfSystem.DampfMode(3, ω_RESONANT, g_rescaled, 5.3, 293.0, Nb),
        DampfSystem.DampfMode(3, ω2, g_rescaled, 5.3, 293.0, Nb),
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

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "s1x2_s2x2_reorg_nv$(Nb)",
                    "iv_curve_s1x2_s2x2_reorg_nv$(Nb)_$(ω2_tag)_$(ΓL_tag).csv")

println("=== s1x2_s2x2 reorg-rescaled, Fock truncation Nv=$(Nb) ===")
println("ω2=$(ω2) cm^-1, g_rescaled=$(g_rescaled) cm^-1, ΓL=$(ΓL_list)")
println("Δt=$(Δt), chi_max=$(chi_max), max_steps=$(max_steps), check_every=$(check_every), conv_tol=$(conv_tol), threads=$(Threads.nthreads())")
flush(stdout)

for ΓL in ΓL_list
    println("\n--- ΓL = $(ΓL) ---")
    flush(stdout)
    diss = DampfSystem.build_electronic_dissipators(
        H_e; idx_low=idx_low, idx_g=idx_g,
        γH=sim.γH, nH=sim.nH, ΓL=ΓL,
        γC=sim.γC, TC=sim.TC, kB=sim.kB,
    ).ops
    P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)

    t0 = time()
    dampf = DampfSystem.run_to_steady_state(
        model, role_props, siteinds, τ, P_e;
        chi_max=chi_max, coeff_tol=coeff_tol, Δt=Δt, max_steps=max_steps, check_every=check_every, conv_tol=conv_tol,
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL, idx_low)), label="ω2=$(ω2) [reorg-nv$(Nb)] ΓL=$(ΓL)",
    )
    println("DAMPF: steps=$(dampf.steps) converged=$(dampf.converged) wall_time=$(round(time() - t0, digits=1))s")

    I = Observables.current(dampf.ρ_e, U_e, ΓL, idx_low)
    V = Observables.voltage(dampf.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    P = Observables.power(I, V)
    println("I=$(I)  V=$(V)  P=$(P)")

    r = (ΓL=ΓL, I_dampf=I, V_dampf=V, P_dampf=P, steps=dampf.steps, converged=dampf.converged)
    all_so_far = merge_sorted_results(load_dampf_sweep_csv(csv_path), [r])
    save_dampf_sweep_csv(csv_path, all_so_far)
    println("Saved: $(csv_path)")
    flush(stdout)
end
