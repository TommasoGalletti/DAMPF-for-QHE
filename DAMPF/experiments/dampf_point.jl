include("../system/system.jl")
using .DampfSystem
using ITensors
using ITensorMPS
using Printf

include("../../src/Observables.jl")
include("../../config/parameters.jl")
using .Observables

"""
One DAMPF steady-state point, for every configuration used in the thesis,
with a single protocol: fixed number of Trotter steps (no early stop),
checkpoint every 100 steps, time series written to file. The steady-state
value is then the plateau average built by utils/build_plateau_average.jl.

Configurations (<param> is a frequency in cm^-1):
  bench_g       one mode on s1 and s2 at ω = <param>, g = 55            (Ch. 6, Sec. 7.2)
  bench_lambda  same, g = 55 √(ω/200)  (fixed reorganization energy)  (Sec. 7.2.1)
  v1v2v3        one mode on s1, s2, s3 at 200, g = 55 (<param> ignored) (Sec. 7.1.1)
  s2m_g         modes at 200 and ω2 = <param> on s1 and s2, g = 55     (Secs. 7.1.2, 7.3)
  s2m_lambda    same, couplings rescaled to λ = λ0                     (Sec. 7.3)
  s2m_{g,lambda}_{cold,damp2}  same, with the detuned mode at zero temperature
                or with twice its damping rate                          (Sec. 8.3.2)

Output: numeric_vectors/<outname>/<config>/ts_<param>_<ΓL>_chi<χ>.csv (ΓL = 0.5 -> 0p5) with
columns step, t, I, V, P, trace_loss_mean, trace_loss_max, wall_s, then the
excitonic (ε1, ε2, ε3, g) and site (g, s1, s2, s3) populations.

Usage: julia dampf_point.jl <config> <param> <ΓL> <chi_max> [max_steps=12000] [Δt=2e-4] [outname=dampf_regen]
"""

length(ARGS) >= 4 || error("Usage: julia dampf_point.jl <config> <param> <ΓL> <chi_max> [max_steps] [Δt] [outname]")
config = ARGS[1]
param = parse(Float64, ARGS[2])
ΓL = parse(Float64, ARGS[3])
chi_max = parse(Int, ARGS[4])
max_steps = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 12_000
Δt = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 2.0e-4
outname = length(ARGS) >= 7 ? ARGS[7] : "dampf_regen"

const G0, ω_RES, γM, TM, Nb = 55.0, 200.0, 5.3, 293.0, 6
λ0 = G0^2 / ω_RES
mode(site, ω, g) = DampfSystem.DampfMode(site, ω, g, γM, TM, Nb)
modes = if config == "bench_g"
    [mode(2, param, G0), mode(3, param, G0)]
elseif config == "bench_lambda"
    gq = G0 * sqrt(param / ω_RES)
    [mode(2, param, gq), mode(3, param, gq)]
elseif config == "v1v2v3"
    [mode(2, ω_RES, G0), mode(3, ω_RES, G0), mode(4, ω_RES, G0)]
elseif startswith(config, "s2m_")
    base, variant = startswith(config, "s2m_lambda") ? ("s2m_lambda", config[11:end]) : ("s2m_g", config[6:end])
    gq = base == "s2m_g" ? G0 : G0 * sqrt(λ0 / (λ0 + G0^2 / param))
    detuned(site) = variant == "" ? mode(site, param, gq) :
                    variant == "_cold" ? DampfSystem.DampfMode(site, param, gq, γM, 1e-3, Nb) :
                    variant == "_damp2" ? DampfSystem.DampfMode(site, param, gq, 2γM, TM, Nb) :
                    error("Unknown variant $(variant)")
    [mode(2, ω_RES, gq), detuned(2), mode(3, ω_RES, gq), detuned(3)]
else
    error("Unknown configuration $(config)")
end
model = DampfSystem.DampfModel(-10000.0, [300.0, 300.0, 0.0], [0.0 100.0 0.0; 100.0 0.0 0.0; 0.0 0.0 0.0], modes)
# ΓL below 1 (benchmark grid: 0.1, 0.2, 0.5) keeps its decimals, "." -> "p"
ΓL_tag = isinteger(ΓL) ? string(Int(ΓL)) : replace(string(ΓL), "." => "p")
tag = "$(Int(round(param)))_$(ΓL_tag)_chi$(chi_max)"

sim = default_simulation_params()
dim_e = DampfSystem.dim_e(model)
idx_low, idx_g = dim_e - 1, dim_e
H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)
siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)
coeff_tol = 1.0e-10
check_every = 100

role_props = [DampfSystem.build_mode_role_propagators(m, siteinds[q], Δt) for (q, m) in enumerate(model.modes)]
swap_gates = DampfSystem.build_swap_gates(model.modes, siteinds)
diss = DampfSystem.build_electronic_dissipators(H_e; idx_low=idx_low, idx_g=idx_g, γH=sim.γH, nH=sim.nH, ΓL=ΓL,
                                                γC=sim.γC, TC=sim.TC, kB=sim.kB).ops
P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)

outdir = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", outname, config)
mkpath(outdir)
io = open(joinpath(outdir, "ts_$(tag).csv"), "w")
println(io, "step,t,I,V,P,trace_loss_mean,trace_loss_max,wall_s,p_e1,p_e2,p_e3,p_g,p_site_g,p_s1,p_s2,p_s3")

println("=== DAMPF point: $(config), param=$(param), ΓL=$(ΓL), chi_max=$(chi_max), Δt=$(Δt), max_steps=$(max_steps), " *
        "modes=$([(m.site, m.ω, round(m.g, digits=3)) for m in model.modes]), threads=$(Threads.nthreads())")
flush(stdout)

blocks = DampfSystem.init_ground_state_triangular(model, siteinds)
losses = Float64[]
t0 = time()
for step in 1:max_steps
    tr_val = DampfSystem.one_trotter_step_triangular!(blocks, model, role_props, P_e, siteinds, τ, swap_gates;
                                                      chi_max=chi_max, svd_tol=0.0, coeff_tol=coeff_tol)
    push!(losses, 1 - real(tr_val))
    if step % check_every == 0
        ρ_e = DampfSystem.reduced_electronic_density_triangular(blocks, τ)
        I = Observables.current(ρ_e, U_e, ΓL, idx_low)
        V = Observables.voltage(ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
        ρ_x = U_e' * ρ_e * U_e
        wall = time() - t0
        @printf(io, "%d,%.6e,%.12e,%.12e,%.12e,%.6e,%.6e,%.1f", step, step * Δt, I, V, I * V,
                sum(losses) / length(losses), maximum(losses), wall)
        for x in (real.([ρ_x[k, k] for k in 1:dim_e])..., real.([ρ_e[k, k] for k in 1:dim_e])...)
            @printf(io, ",%.12e", x)
        end
        println(io)
        flush(io)
        empty!(losses)
        step % 2000 == 0 && (println("  [$(config) $(tag)] step $(step)/$(max_steps)  elapsed=$(round(wall, digits=1))s  I≈$(I)"); flush(stdout))
    end
end
close(io)
println("Done: wall_time=$(round(time() - t0, digits=1))s")
