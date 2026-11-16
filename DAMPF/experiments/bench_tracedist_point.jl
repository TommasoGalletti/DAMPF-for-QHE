include("../system/system.jl")
using .DampfSystem
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

include("../../src/Hamiltonian.jl")
include("../../src/Lindblad.jl")
include("../../src/Liouvillian.jl")
include("../../src/Solver.jl")
include("../../src/Observables.jl")
include("../../config/parameters.jl")
using .Hamiltonian
using .Lindblad
using .Liouvillian
using .Solver
using .Observables

"""
Benchmark point (two modes, Fig. 6.1) with the full-state trace distance
D = ½ Tr|ρ_DAMPF − ρ_direct| against the direct Lindblad steady state of the
same N_v = 6 Liouvillian. Same protocol as dampf_point.jl (fixed number of
steps, checkpoint every 100 steps, no early stop); D is computed at every
checkpoint with step ≥ 7000, i.e. on the plateau window.

Output: numeric_vectors/<outname>/ts_tracedist_<ΓL>_chi<χ>.csv with columns
step, t, I, V, P, trace_dist (trace_dist = NaN before step 7000).

Usage: julia bench_tracedist_point.jl <ΓL> <chi_max> [max_steps=12000] [Δt=2e-4] [outname=phase1_tracedist]
"""

length(ARGS) >= 2 || error("Usage: julia bench_tracedist_point.jl <ΓL> <chi_max> [max_steps] [Δt] [outname]")
ΓL = parse(Float64, ARGS[1])
chi_max = parse(Int, ARGS[2])
max_steps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12_000
Δt = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.0e-4
outname = length(ARGS) >= 5 ? ARGS[5] : "phase1_tracedist"
plateau_start = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 7000

params = default_params()
sim = default_simulation_params()
model = DampfSystem.killoran_benchmark_model(Nb=params.Nv)
dim_e = DampfSystem.dim_e(model)
idx_low, idx_g = dim_e - 1, dim_e
H_e = DampfSystem.build_electronic_hamiltonian(model)
energies_e, U_e = DampfSystem.diagonalize_electronic(H_e)
siteinds = DampfSystem.build_mode_siteinds(model)
τ = DampfSystem.build_trace_mps(siteinds, model.modes)
coeff_tol = 1.0e-10
check_every = 100

# direct steady state of the same Nv-truncated Liouvillian (as in iv_curve_dampf_killoran.jl)
H_ref = Hamiltonian.build_total_hamiltonian(params)
nC = cold_occupations(energies_e, sim.TC; kB=sim.kB)
jump_ops = Lindblad.JumpOperator[]
append!(jump_ops, Lindblad.build_hot_operators_vib(U_e; γH=sim.γH, nH=sim.nH, Nv=params.Nv))
push!(jump_ops, Lindblad.build_load_operator_vib(U_e; ΓL=ΓL, Nv=params.Nv))
append!(jump_ops, Lindblad.build_cold_operators_vib(U_e; γC=sim.γC, nC_matrix=nC, Nv=params.Nv))
append!(jump_ops, Lindblad.build_vibrational_damping_operators(dim_e, params.Nv; γM=sim.γM, ωv=params.ωv, TM=sim.TM, kB=sim.kB))
ρ_direct = Solver.steady_state(Liouvillian.build_liouvillian(H_ref, jump_ops))
trace_distance(ρ) = (Δ = ρ .- ρ_direct; 0.5 * sum(abs.(eigvals(Hermitian((Δ .+ Δ') ./ 2)))))

role_props = [DampfSystem.build_mode_role_propagators(m, siteinds[q], Δt) for (q, m) in enumerate(model.modes)]
swap_gates = DampfSystem.build_swap_gates(model.modes, siteinds)
diss = DampfSystem.build_electronic_dissipators(H_e; idx_low=idx_low, idx_g=idx_g, γH=sim.γH, nH=sim.nH, ΓL=ΓL,
                                                γC=sim.γC, TC=sim.TC, kB=sim.kB).ops
P_e = DampfSystem.build_electronic_mixing_propagator(H_e, diss, Δt)

ΓL_tag = isinteger(ΓL) ? string(Int(ΓL)) : replace(string(ΓL), "." => "p")
outdir = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", outname)
mkpath(outdir)
io = open(joinpath(outdir, "ts_tracedist_$(ΓL_tag)_chi$(chi_max).csv"), "w")
println(io, "step,t,I,V,P,trace_dist")
println("=== benchmark trace distance: ΓL=$(ΓL), chi_max=$(chi_max), Δt=$(Δt), max_steps=$(max_steps)"); flush(stdout)

blocks = DampfSystem.init_ground_state_triangular(model, siteinds)
t0 = time()
for step in 1:max_steps
    DampfSystem.one_trotter_step_triangular!(blocks, model, role_props, P_e, siteinds, τ, swap_gates;
                                             chi_max=chi_max, svd_tol=0.0, coeff_tol=coeff_tol)
    if step % check_every == 0
        ρ_e = DampfSystem.reduced_electronic_density_triangular(blocks, τ)
        I = Observables.current(ρ_e, U_e, ΓL, idx_low)
        V = Observables.voltage(ρ_e, U_e, energies_e, idx_low, idx_g; kB=sim.kB, TC=sim.TC)
        D = step >= plateau_start ? trace_distance(DampfSystem.blocks_to_dense_triangular(blocks, siteinds, model)) : NaN
        @printf(io, "%d,%.6e,%.12e,%.12e,%.12e,%.6e\n", step, step * Δt, I, V, I * V, D)
        flush(io)
        step % 2000 == 0 && (println("  step $(step)/$(max_steps)  elapsed=$(round(time() - t0, digits=1))s  D=$(D)"); flush(stdout))
    end
end
close(io)
println("Done: wall_time=$(round(time() - t0, digits=1))s")
