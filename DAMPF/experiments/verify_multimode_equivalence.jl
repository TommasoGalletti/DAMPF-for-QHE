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
Verifies the analytical point raised on the s1x2_s2x2 model: two identical
(same ω,g,γ,T), independent, non-interacting modes on the same site are
EXACTLY equivalent, via the symmetric/antisymmetric normal-mode
transformation b1=(a1+a2)/√2, b2=(a1-a2)/√2 (well-defined because ω1=ω2),
to a single effective mode b1 with coupling g_eff=g√2 coupled to the site,
plus a fully decoupled "dark" spectator mode b2: the coupling term
(a1+a1†)+(a2+a2†) = √2(b1+b1†) never involves b2, and the SUM of two
identical local dissipators γ(D[a1]+D[a2]) is itself invariant under this
orthogonal rotation of the jump-operator pair, so b2 keeps its own bath
(same γ, n̄) but never touches the electronic system. Since s1x2_s2x2's
two modes per site are identical and couple only to their own site, this
reduction applies independently on s1 and s2 -- so the WHOLE model should
be exactly equivalent to `killoran_benchmark_model` with g=55*sqrt(2)
instead of 55 (plus two inert dark modes that affect nothing).

This does NOT touch `killoran_benchmark_model`'s hardcoded g=55 -- the
g_eff=55*sqrt(2) modes are built as standalone `DampfMode` objects, the
same pattern used in the ω-scan experiments, so nothing shared is left in
a modified state after this script runs.
"""

params = default_params()
sim = default_simulation_params()
model = DampfSystem.killoran_benchmark_model(Nb=params.Nv)  # g/ω on model.modes are placeholders, overridden below
dim_e = DampfSystem.dim_e(model)
idx_low = dim_e - 1
idx_g = dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)
siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

# Same validated knobs as killoran_benchmark_model (dim_e=4, M=2 -- this
# effective-mode picture is exactly the M=2 benchmark, just with g rescaled).
Δt = 2.0e-4
chi_max = 3
coeff_tol = 1.0e-10
max_steps = 10_000
check_every = 500
conv_tol = 1.0e-6

g_eff = 55.0 * sqrt(2)
modes_eff = [
    DampfSystem.DampfMode(2, 200.0, g_eff, 5.3, 293.0, params.Nv),
    DampfSystem.DampfMode(3, 200.0, g_eff, 5.3, 293.0, params.Nv),
]
role_props = [DampfSystem.build_mode_role_propagators(mode, siteinds[q], Δt) for (q, mode) in enumerate(modes_eff)]

s1x2s2x2 = load_dampf_sweep_csv(joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_s1x2_s2x2.csv"))

ΓL_list = [0.1, 10.0, 100.0, 500.0, 800.0, 2000.0, 10000.0]

println("g_eff = 55*sqrt(2) = $(g_eff)")
flush(stdout)

results = NamedTuple[]
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
        observable_fn=(ρ_e -> Observables.current(ρ_e, U_e, ΓL, idx_low)), label="ΓL=$(ΓL) g_eff",
    )
    t_dampf = time() - t0
    println("DAMPF: steps=$(dampf.steps) converged=$(dampf.converged) wall_time=$(round(t_dampf,digits=1))s")
    flush(stdout)

    I_dampf = Observables.current(dampf.ρ_e, U_e, ΓL, idx_low)
    V_dampf = Observables.voltage(dampf.ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    P_dampf = Observables.power(I_dampf, V_dampf)

    ref_i = findfirst(r -> r.ΓL == ΓL, s1x2s2x2)
    if ref_i === nothing
        println("g_eff model:  I=$(I_dampf)  V=$(V_dampf)  P=$(P_dampf)  (no s1x2_s2x2 reference at this ΓL)")
        push!(results, (ΓL=ΓL, I_dampf=I_dampf, V_dampf=V_dampf, P_dampf=P_dampf,
                         I_s1x2s2x2=NaN, P_s1x2s2x2=NaN, rel_err_I_pct=NaN, rel_err_P_pct=NaN))
    else
        r = s1x2s2x2[ref_i]
        rel_err_I = abs(I_dampf - r.I_dampf) / abs(r.I_dampf) * 100
        rel_err_P = abs(P_dampf - r.P_dampf) / abs(r.P_dampf) * 100
        println("g_eff model:  I=$(I_dampf)  V=$(V_dampf)  P=$(P_dampf)")
        println("s1x2_s2x2:    I=$(r.I_dampf)  V=$(r.V_dampf)  P=$(r.P_dampf)")
        println("rel diff: I=$(round(rel_err_I,digits=3))%  P=$(round(rel_err_P,digits=3))%")
        push!(results, (ΓL=ΓL, I_dampf=I_dampf, V_dampf=V_dampf, P_dampf=P_dampf,
                         I_s1x2s2x2=r.I_dampf, P_s1x2s2x2=r.P_dampf,
                         rel_err_I_pct=rel_err_I, rel_err_P_pct=rel_err_P))
    end
    flush(stdout)
end

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "verify_multimode_equivalence.csv")
mkpath(dirname(csv_path))
open(csv_path, "w") do io
    println(io, "GammaL,I_geff,V_geff,P_geff,I_s1x2s2x2,P_s1x2s2x2,rel_err_I_pct,rel_err_P_pct")
    for r in results
        println(io, string(r.ΓL, ",", r.I_dampf, ",", r.V_dampf, ",", r.P_dampf, ",",
                            r.I_s1x2s2x2, ",", r.P_s1x2s2x2, ",", r.rel_err_I_pct, ",", r.rel_err_P_pct))
    end
end
println("\nSaved: $(csv_path)")
println("VERIFICATION DONE")
