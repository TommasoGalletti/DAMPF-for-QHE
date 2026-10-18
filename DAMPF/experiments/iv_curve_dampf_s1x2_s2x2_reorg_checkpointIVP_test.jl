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
Checkpoint-level I/V/P diagnostic, single (ω2,ΓL) point.

`run_to_steady_state` (dynamics/steady_state.jl) only calls `observable_fn`
on the CURRENT I at each checkpoint (for the convergence test), and the
production scripts only compute V and P once, from the single ρ_e
returned at the end. So we have per-checkpoint history of I (in old runs'
stdout logs, still on disk this session) but never of V or P -- meaning we
don't actually know whether V fluctuates checkpoint-to-checkpoint as much
as I does, which matters for deciding whether "keep the final V, trail-
average only I" is a defensible cheap retrofit of old data, or whether V
needs averaging too (requiring a full rerun with this instrumentation).

This script inlines the same Trotter loop as run_to_steady_state (reusing
its exact building blocks, all exported from DampfSystem) but computes and
prints I, V, AND P at every checkpoint, then reports both the raw final-
step values and a trailing average over the last K checkpoints, so we can
directly compare their relative fluctuation and decide.

Single point, chi_max=4, max_steps=12000 -- same window as production,
no attempt to probe the late-time instability found separately.

Usage: julia -t auto --project=... iv_curve_dampf_s1x2_s2x2_reorg_checkpointIVP_test.jl <ω2_in_cm-1> <ΓL>
"""

if length(ARGS) < 2
    error("Usage: julia iv_curve_dampf_s1x2_s2x2_reorg_checkpointIVP_test.jl <ω2_in_cm-1> <ΓL>  (e.g. 250 800)")
end
ω2 = parse(Float64, ARGS[1])
ΓL = parse(Float64, ARGS[2])
ω2_tag = string(Int(round(ω2)))
ΓL_tag = string(Int(round(ΓL)))

const G0 = 55.0
const ω_RESONANT = 200.0
λ0 = G0^2 / ω_RESONANT
λ_unscaled = λ0 + G0^2 / ω2
s = sqrt(λ0 / λ_unscaled)
g_rescaled = G0 * s

println("=== Checkpoint I/V/P diagnostic (ω2=$(ω2), ΓL=$(ΓL)) ===")
println("g_rescaled = $(g_rescaled) cm^-1")
flush(stdout)

params = default_params()
sim = default_simulation_params()
Nb = params.Nv

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

diss = DampfSystem.build_electronic_dissipators(
    H_e; idx_low=idx_low, idx_g=idx_g,
    γH=sim.γH, nH=sim.nH, ΓL=ΓL,
    γC=sim.γC, TC=sim.TC, kB=sim.kB,
).ops
P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)

swap_gates = DampfSystem.build_swap_gates(model.modes, siteinds)
blocks = DampfSystem.init_ground_state_triangular(model, siteinds)
ρ_e = DampfSystem.reduced_electronic_density_triangular(blocks, τ)

checkpoints = NamedTuple[]
t_start = time()
for step in 1:max_steps
    DampfSystem.one_trotter_step_triangular!(blocks, model, role_props, P_e, siteinds, τ, swap_gates; chi_max=chi_max, svd_tol=0.0, coeff_tol=coeff_tol)
    if step % check_every == 0
        ρ_e = DampfSystem.reduced_electronic_density_triangular(blocks, τ)
        I_now = Observables.current(ρ_e, U_e, ΓL, idx_low)
        V_now = Observables.voltage(ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
        P_now = Observables.power(I_now, V_now)
        elapsed = round(time() - t_start, digits=1)
        println("  step $(step)/$(max_steps)  t_phys=$(round(step * Δt, sigdigits=4))  elapsed=$(elapsed)s  I=$(I_now)  V=$(V_now)  P=$(P_now)")
        flush(stdout)
        push!(checkpoints, (step=step, I=I_now, V=V_now, P=P_now))
    end
end

csv_path = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "s1x2_s2x2_reorg_checkpointIVP_test", "checkpoints_omega2_$(ω2_tag)_GammaL_$(ΓL_tag).csv")
mkpath(dirname(csv_path))
open(csv_path, "w") do io
    println(io, "step,I,V,P")
    for c in checkpoints
        println(io, "$(c.step),$(c.I),$(c.V),$(c.P)")
    end
end
println("\nSaved $(length(checkpoints)) checkpoints to: $(csv_path)")

for K in (3, 5, 8)
    last_K = checkpoints[max(1, end - K + 1):end]
    I_avg = sum(c.I for c in last_K) / length(last_K)
    V_avg = sum(c.V for c in last_K) / length(last_K)
    P_avg = sum(c.P for c in last_K) / length(last_K)
    P_of_avgIV = Observables.power(I_avg, V_avg)
    println("\n--- Trailing average over last $(K) checkpoints ($(K*check_every) steps) ---")
    println("I_avg=$(I_avg)  V_avg=$(V_avg)  P_avg(mean of P)=$(P_avg)  P(mean I, mean V)=$(P_of_avgIV)")
end

I_vals = [c.I for c in checkpoints]
V_vals = [c.V for c in checkpoints]
P_vals = [c.P for c in checkpoints]
relstd(x) = (n = length(x); m = sum(x) / n; sqrt(sum((xi - m)^2 for xi in x) / n) / abs(m))
println("\n=== Relative fluctuation (std/mean) over ALL $(length(checkpoints)) checkpoints ===")
println("I: $(100*relstd(I_vals))%   V: $(100*relstd(V_vals))%   P: $(100*relstd(P_vals))%")
println("\nFinal step values: I=$(I_vals[end])  V=$(V_vals[end])  P=$(P_vals[end])")
