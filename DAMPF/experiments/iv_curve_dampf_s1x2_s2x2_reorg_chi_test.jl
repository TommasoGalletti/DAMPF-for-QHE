include("../system/system.jl")
using .DampfSystem
using ITensors
using ITensorMPS
using Printf

include("../../src/Observables.jl")
include("../../config/parameters.jl")
using .Observables

"""
Bond-dimension diagnostic for the reorganization-energy-rescaled s1x2_s2x2
scan (iv_curve_dampf_s1x2_s2x2_reorg_rescaled.jl): same physics and Δt, one
(ω2, ΓL, chi_max) point per invocation, always run for the full `max_steps`
(no early stop) so that the whole time series is available.

Hypothesis under test: at chi_max=4 the SVD truncation of the mixing step
injects so much error per step (trace loss ~3e-4) that the truncated
dynamics has no fixed point -- a limit cycle on the smooth curves
(ω2=450), an irregular attractor on the noisy ones (ω2=250) -- and the
point-to-point scatter of P(ΓL) is the phase at which a run is stopped.

Also used for production runs (e.g. the chi_max=8 rescaled scan): the
plateau average of the time series, built by utils/build_plateau_average.jl,
replaces the single end-of-run value.

Writes, under numeric_vectors/<outname>/ (default s1x2_s2x2_reorg_chitest):
- ts_<ω2>_<ΓL>_chi<χ>.csv: every `check_every` steps, I, V, P, the mean
  and max trace loss 1-Tr(ρ) of the steps since the previous checkpoint, and
  (trailing columns) the electronic populations in the excitonic (ε1, ε2, ε3,
  g) and site (g, s1, s2, s3) basis plus |ρ_ε1ε2|;
- spec_<ω2>_<ΓL>_chi<χ>.csv: every `spec_every` steps, for every stored
  block and bond, the spectrum of the mixing step truncated at 4*chi_max: discarded
  weight fraction beyond chi_max, σ_χ/σ_1, and σ_{χ+1}/σ_χ (close to 1 means
  the cut falls inside a near-degenerate multiplet). Not written if
  spec_every = 0 (it costs ~10% of the run).
Both are flushed at every write, so a killed job keeps its partial series.

Usage: julia iv_curve_dampf_s1x2_s2x2_reorg_chi_test.jl <ω2> <ΓL> <chi_max> [max_steps] [spec_every] [outname]
"""

length(ARGS) >= 3 || error("Usage: julia iv_curve_dampf_s1x2_s2x2_reorg_chi_test.jl <ω2> <ΓL> <chi_max> [max_steps] [spec_every] [outname]")
ω2 = parse(Float64, ARGS[1])
ΓL = parse(Float64, ARGS[2])
chi_max = parse(Int, ARGS[3])
max_steps = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 15_000
spec_every = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1000
outname = length(ARGS) >= 6 ? ARGS[6] : "s1x2_s2x2_reorg_chitest"
tag = "$(Int(round(ω2)))_$(Int(round(ΓL)))_chi$(chi_max)"

const G0 = 55.0
const ω_RESONANT = 200.0
λ0 = G0^2 / ω_RESONANT
g_rescaled = G0 * sqrt(λ0 / (λ0 + G0^2 / ω2))

sim = default_simulation_params()
Nb = 6
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
idx_low, idx_g = dim_e - 1, dim_e

H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)
siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)

Δt = 2.0e-4
coeff_tol = 1.0e-10
check_every = 100

role_props = [DampfSystem.build_mode_role_propagators(mode, siteinds[q], Δt) for (q, mode) in enumerate(model.modes)]
swap_gates = DampfSystem.build_swap_gates(model.modes, siteinds)
diss = DampfSystem.build_electronic_dissipators(H_e; idx_low=idx_low, idx_g=idx_g, γH=sim.γH, nH=sim.nH, ΓL=ΓL,
                                                γC=sim.γC, TC=sim.TC, kB=sim.kB).ops
P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)

outdir = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", outname)
mkpath(outdir)
ts_io = open(joinpath(outdir, "ts_$(tag).csv"), "w")
spec_io = spec_every > 0 ? open(joinpath(outdir, "spec_$(tag).csv"), "w") : devnull
println(ts_io, "step,t,I,V,P,trace_loss_mean,trace_loss_max,wall_s,p_e1,p_e2,p_e3,p_g,p_site_g,p_s1,p_s2,p_s3,abs_coh_e1e2")
println(spec_io, "step,m,n,bond,rank,discarded_frac,sigma_chi_over_sigma1,sigma_chi1_over_sigma_chi")

"""Singular values across every bond of ψ (not normalized)."""
function bond_spectra(ψ::MPS)
    ψ = orthogonalize(ψ, 1)
    spectra = Vector{Vector{Float64}}()
    for b in 1:(length(ψ) - 1)
        orthogonalize!(ψ, b)
        left = b == 1 ? (siteinds[b],) : (linkind(ψ, b - 1), siteinds[b])
        _, S, _ = svd(ψ[b], left...)
        push!(spectra, [abs(S[i, i]) for i in 1:dim(S, 1)])
    end
    spectra
end

"""Spectrum of the next mixing step, truncated only at 4*chi_max instead of chi_max (on a copy
of the state): enough to see σ_{χ+1} and the weight discarded at chi_max, without the cost
of the full-rank sum. `discarded_frac` is therefore a lower bound (tail beyond 4χ ignored)."""
function write_untruncated_spectra(step, blocks)
    probe = similar(blocks)
    for m in 1:dim_e, n in m:dim_e
        probe[m, n] = copy(blocks[m, n])
    end
    DampfSystem.DampfTrotter._apply_local_step_triangular!(probe, model, role_props)
    DampfSystem.DampfTrotter._apply_electronic_mixing_triangular!(probe, P_e, siteinds, swap_gates;
                                                                  chi_max=4 * chi_max, svd_tol=0.0, coeff_tol=coeff_tol)
    for m in 1:dim_e, n in m:dim_e
        norm(probe[m, n]) == 0 && continue
        for (b, σ) in enumerate(bond_spectra(probe[m, n]))
            w = σ .^ 2
            disc = length(σ) > chi_max ? sum(w[(chi_max + 1):end]) / sum(w) : 0.0
            r_chi = length(σ) >= chi_max ? σ[chi_max] / σ[1] : 0.0
            r_deg = length(σ) > chi_max ? σ[chi_max + 1] / σ[chi_max] : 0.0
            @printf(spec_io, "%d,%d,%d,%d,%d,%.6e,%.6e,%.6e\n", step, m, n, b, length(σ), disc, r_chi, r_deg)
        end
    end
    flush(spec_io)
end

observables(ρ_e) = let I = Observables.current(ρ_e, U_e, ΓL, idx_low),
                       V = Observables.voltage(ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
    ρ_x = U_e' * ρ_e * U_e   # excitonic basis, energies descending: ε1, ε2, ε3, g
    (I=I, V=V, P=Observables.power(I, V), exc=real.([ρ_x[k, k] for k in 1:dim_e]), site=real.([ρ_e[k, k] for k in 1:dim_e]),
     coh12=abs(ρ_x[1, 2]))
end

println("=== s1x2_s2x2 reorg-rescaled, chi_max test ===")
println("ω2=$(ω2) cm^-1, g_rescaled=$(g_rescaled) cm^-1, ΓL=$(ΓL), chi_max=$(chi_max), Δt=$(Δt), max_steps=$(max_steps), " *
        "check_every=$(check_every), spec_every=$(spec_every), threads=$(Threads.nthreads())")
flush(stdout)

blocks = DampfSystem.init_ground_state_triangular(model, siteinds)
losses = Float64[]
t0 = time()
for step in 1:max_steps
    spec_every > 0 && step % spec_every == 0 && write_untruncated_spectra(step, blocks)
    tr_val = DampfSystem.one_trotter_step_triangular!(blocks, model, role_props, P_e, siteinds, τ, swap_gates;
                                                      chi_max=chi_max, svd_tol=0.0, coeff_tol=coeff_tol)
    push!(losses, 1 - real(tr_val))
    if step % check_every == 0
        o = observables(DampfSystem.reduced_electronic_density_triangular(blocks, τ))
        wall = time() - t0
        @printf(ts_io, "%d,%.6e,%.12e,%.12e,%.12e,%.6e,%.6e,%.1f", step, step * Δt, o.I, o.V, o.P,
                sum(losses) / length(losses), maximum(losses), wall)
        for x in (o.exc..., o.site..., o.coh12)
            @printf(ts_io, ",%.12e", x)
        end
        println(ts_io)
        flush(ts_io)
        empty!(losses)
        step % 1000 == 0 && (println("  [$(tag)] step $(step)/$(max_steps)  elapsed=$(round(wall, digits=1))s  I≈$(o.I)"); flush(stdout))
    end
end
close(ts_io)
spec_every > 0 && close(spec_io)
println("Done: wall_time=$(round(time() - t0, digits=1))s")
