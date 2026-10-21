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
Reorganization-energy-preserving variant of omega_gammaL_scan_killoran.jl
(single mode per site on s1, s2 -- the benchmark model -- scanned over the
mode frequency ω and ΓL).

The original scan keeps g=55 cm^-1 fixed, so the per-site reorganization
energy λ=g²/ω changes with ω (3.0x... 0.67x the resonant value λ0=55²/200
between ω=100 and 300) and the scan mixes detuning with a change of the
overall coupling strength. Here g is set, for every ω, to

    g(ω) = sqrt(λ0 ω) = 55 sqrt(ω/200)

so that λ=λ0 at every point (g=38.9 at ω=100 ... 67.4 at ω=300); ω=200 is
identical to the original scan and serves as a consistency check.

One ω per invocation (one SLURM job per ω), each writing its own CSV
numeric_vectors/omega_gammaL_scan_killoran_reorg/omega_gammaL_scan_killoran_reorg_<ω>.csv
so parallel jobs never touch the same file. Same schema as the original CSV.

The g=0 baseline does not depend on ω and is NOT recomputed: P_g0(ΓL) is
recovered from the original scan's CSV as P/(1+enhancement/100) (same
model, same numerical settings, deterministic propagation), so the
enhancement column here is directly comparable with the original one.

A second, optional argument selects what is held fixed instead of λ:
`lambda` (default, g=sqrt(λ0 ω)) or `S` -- fixed Huang-Rhys factor
S=(g/ω)²=(55/200)², i.e. g(ω)=55 ω/200 (27.5 at ω=100 ... 82.5 at ω=300),
written to numeric_vectors/omega_gammaL_scan_killoran_fixedS/; or `g` --
g=55 fixed as in the original scan, used to extend that scan to new ω
values one job per ω, written to numeric_vectors/omega_gammaL_scan_killoran_fixedg_ext/
(the original single-CSV scan is left untouched).

Usage: julia --project=... omega_gammaL_scan_killoran_reorg.jl <ω_in_cm-1> [lambda|S|g]
"""

isempty(ARGS) && error("Usage: julia omega_gammaL_scan_killoran_reorg.jl <ω_in_cm-1> [lambda|S|g]")
fixed = length(ARGS) >= 2 ? ARGS[2] : "lambda"
fixed in ("lambda", "S", "g") || error("Second argument must be `lambda`, `S` or `g`, got `$(fixed)`")
ω = parse(Float64, ARGS[1])
ω_tag = string(Int(round(ω)))

const G0 = 55.0
const ω_RES = 200.0
λ0 = G0^2 / ω_RES
S0 = (G0 / ω_RES)^2
g = fixed == "lambda" ? sqrt(λ0 * ω) : fixed == "S" ? sqrt(S0) * ω : G0

params = default_params()
sim = default_simulation_params()
model = DampfSystem.killoran_benchmark_model(Nb=params.Nv)
dim_e = DampfSystem.dim_e(model)
idx_low = dim_e - 1
idx_g = dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)
siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

# Same numerical settings as omega_gammaL_scan_killoran.jl.
Δt = 2.0e-4
chi_max = 3
coeff_tol = 1.0e-10
max_steps = 10_000
check_every = 500
conv_tol = 1.0e-6

modes = [
    DampfSystem.DampfMode(2, ω, g, 5.3, 293.0, params.Nv),
    DampfSystem.DampfMode(3, ω, g, 5.3, 293.0, params.Nv),
]
role_props = [DampfSystem.build_mode_role_propagators(mode, siteinds[q], Δt) for (q, mode) in enumerate(modes)]

ΓL_list = [
    5.0, 10.0, 20.0, 30.0, 40.0,
    50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 550.0, 600.0,
    650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0,
    1200.0, 1350.0, 1500.0,
    1700.0, 2000.0, 2500.0, 3000.0, 4000.0,
]

# g=0 baseline per ΓL, from the original (unscaled) scan.
nv = joinpath(@__DIR__, "..", "outputs", "numeric_vectors")
orig = load_omega_gammaL_scan_csv(joinpath(nv, "omega_gammaL_scan_killoran.csv"))
P_g0 = Dict{Float64,Float64}()
for r in orig
    r.ω == ω_RES || continue
    P_g0[r.ΓL] = r.P_dampf / (1 + r.enhancement_pct / 100)
end
missing_ΓL = [x for x in ΓL_list if !haskey(P_g0, x)]
isempty(missing_ΓL) || error("No g=0 baseline in the original scan for ΓL = $(missing_ΓL)")

out_name = Dict("lambda" => "omega_gammaL_scan_killoran_reorg", "S" => "omega_gammaL_scan_killoran_fixedS",
                 "g" => "omega_gammaL_scan_killoran_fixedg_ext")[fixed]
csv_path = joinpath(nv, out_name, "$(out_name)_$(ω_tag).csv")
done = Set(r.ΓL for r in load_omega_gammaL_scan_csv(csv_path))

println("=== ω-ΓL scan at fixed $(Dict("lambda" => "reorganization energy λ", "S" => "Huang-Rhys factor S", "g" => "coupling g")[fixed]) ===")
println("ω=$(ω) cm^-1, g=$(g) cm^-1 (λ = g²/ω = $(g^2/ω), λ0 = $(λ0); S = (g/ω)² = $((g/ω)^2), S0 = $(S0))")
println("Δt=$(Δt), chi_max=$(chi_max), max_steps=$(max_steps), check_every=$(check_every), conv_tol=$(conv_tol), threads=$(Threads.nthreads())")
flush(stdout)

for ΓL in ΓL_list
    if ΓL in done
        println("\nΓL = $(ΓL): already in CSV, skipping.")
        continue
    end
    println("\n--- ω = $(ω), g = $(round(g, digits=3)), ΓL = $(ΓL) ---")
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
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL, idx_low)), label="$(Dict("lambda" => "reorg", "S" => "fixedS", "g" => "fixedg")[fixed]) ω=$(ω) ΓL=$(ΓL)",
    )
    println("DAMPF: steps=$(dampf.steps) converged=$(dampf.converged) wall_time=$(round(time() - t0, digits=1))s")

    I = Observables.current(dampf.ρ_e, U_e, ΓL, idx_low)
    V = Observables.voltage(dampf.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    P = Observables.power(I, V)
    enhancement_pct = (P - P_g0[ΓL]) / P_g0[ΓL] * 100
    println("I=$(I)  V=$(V)  P=$(P)  enhancement=$(round(enhancement_pct, digits=3))%")

    r = (ω=ω, ΓL=ΓL, I_dampf=I, V_dampf=V, P_dampf=P,
         enhancement_pct=enhancement_pct, steps=dampf.steps, converged=dampf.converged)
    all_so_far = merge_sorted_results(load_omega_gammaL_scan_csv(csv_path), [r]; key=x -> (x.ω, x.ΓL))
    save_omega_gammaL_scan_csv(csv_path, all_so_far)
    println("Saved ($(length(all_so_far)) points): $(csv_path)")
    flush(stdout)
end

println("\nDone: $(csv_path)")
