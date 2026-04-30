include("../src/Hamiltonian.jl")
include("../src/Lindblad.jl")
include("../src/Liouvillian.jl")
include("../src/Solver.jl")
include("../src/Observables.jl")
include("../config/parameters.jl")

using .Hamiltonian
using .Lindblad
using .Liouvillian
using .Solver
using .Observables

using Plots
default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

# Model setup for the two-mode vibronic case.

function bose(ω, T; kB=0.69503476)
    x = ω / (kB * T)
    if x < 1e-6
        return (kB * T) / ω
    elseif x > 700
        return 0.0
    else
        return 1 / (exp(x) - 1)
    end
end

params = default_params()
sim = default_simulation_params()

# Electronic Hamiltonian only.
H_e = Hamiltonian.build_hamiltonian(params)
energies_e, U_e = Hamiltonian.diagonalize_hamiltonian(H_e)

# Full paper-like Hamiltonian: electronics + two local vibrational modes + local coupling.
# It remains in the site basis.
H_total = Hamiltonian.build_total_hamiltonian(params)

dim_e = size(H_e, 1)
dim_v = params.Nv
dim_total = dim_e * dim_v * dim_v

# Electronic level indices.

idx_high = 1
idx_low  = dim_e - 1
idx_g    = dim_e

# Dissipation parameters.

γH = sim.γH
γC = sim.γC
nH = sim.nH
TC = sim.TC
kB = sim.kB

γM = sim.γM
TM = sim.TM

# Thermal occupation matrix for the cold bath in the electronic basis.
nC = cold_occupations(energies_e, TC; kB=kB)

println("=== VIBRONIC SYSTEM SETUP ===")
println("Electronic dimension: $dim_e")
println("Vibrational dimension: $dim_v")
println("Total dimension: $dim_total")
println("Liouvillian dimension: $(dim_total^2)")
println()
println("nC max = ", maximum(nC))
println("nC min (nonzero) = ", minimum(nC[nC .> 0]))
println()

# Sweep over ΓL on a logarithmic scale.

ΓL_list = gammaL_list(sim)

I_list = Float64[]
V_list = Float64[]
P_list = Float64[]
CM1_TO_EV = sim.cm1_to_ev

for (idx_ΓL, ΓL) in enumerate(ΓL_list)

    # Build jump operators extended to the two vibrational modes.
    hot_ops_vib  = build_hot_operators_vib(U_e; γH=γH, nH=nH, Nv=dim_v)
    cold_ops_vib = build_cold_operators_vib(U_e; γC=γC, nC_matrix=nC, Nv=dim_v)
    load_op_vib  = build_load_operator_vib(U_e; ΓL=ΓL, Nv=dim_v)
    
    # Thermal damping of the vibrational modes.
    vib_damp_ops = build_vibrational_damping_operators(dim_e, dim_v; γM=γM, ωv=params.ωv, TM=TM, kB=kB)

    # Assemble the full jump-operator list.
    jump_ops = JumpOperator[]
    append!(jump_ops, hot_ops_vib)
    append!(jump_ops, cold_ops_vib)
    push!(jump_ops, load_op_vib)
    append!(jump_ops, vib_damp_ops)

    # Build the full Liouvillian.
    L = build_liouvillian(H_total, jump_ops)

    # Solve the steady state: L ρ = 0 with Tr(ρ) = 1.
    ρ_ss = steady_state(L)

    # Vibronic observables: trace out both modes, reduce to the electronic density matrix,
    # then transform to the exciton basis.
    I = Observables.current_vib(ρ_ss, U_e, ΓL, idx_low, dim_v)
    V = Observables.voltage_vib(ρ_ss, U_e, energies_e, idx_low, idx_g, dim_v; kB=kB, TC=TC)
    P = Observables.power(I, V)

    push!(I_list, I)
    push!(V_list, V)
    push!(P_list, P)

    if mod(idx_ΓL, 10) == 1
        println("ΓL = $(ΓL): I = $(I), V = $(V) cm^-1 ($(V * CM1_TO_EV) eV), P = $(P)")
    end
end

println("\n=== SIMULATION COMPLETED ===\n")

# Sample output.

println("\n=== SAMPLE OUTPUT ===")
for i in 1:5:length(ΓL_list)
    println("ΓL=$(ΓL_list[i]) | I=$(I_list[i]) | V=$(V_list[i]) cm^-1 ($(V_list[i] * CM1_TO_EV) eV) | P=$(P_list[i])")
end

# Summary plots.

logΓL = log10.(ΓL_list)
V_list_eV = CM1_TO_EV .* V_list

p1 = plot(logΓL, I_list,
    xlabel="log10(ΓL)",
    ylabel="Current I",
    title="Current vs ΓL (vibronic)",
    lw=2,
    legend=false
)

p2 = plot(logΓL, V_list_eV,
    xlabel="log10(ΓL)",
    ylabel="Voltage V (eV)",
    title="Voltage vs ΓL (vibronic, eV)",
    lw=2,
    legend=false
)
hline!(p2, [0.0], linestyle=:dash)

p3 = plot(logΓL, P_list,
    xlabel="log10(ΓL)",
    ylabel="Power P",
    title="Power vs ΓL (vibronic)",
    lw=2,
    legend=false
)
hline!(p3, [0.0], linestyle=:dash)

imax = argmax(P_list)
scatter!(p3, [logΓL[imax]], [P_list[imax]],
         markersize=8, color=:red, label="Max power")

outdir = joinpath(@__DIR__, "..", "imgs", "vibrational")
mkpath(outdir)

savefig(p1, joinpath(outdir, "i_gammaL.png"))
savefig(p2, joinpath(outdir, "v_gammaL.png"))
savefig(p3, joinpath(outdir, "p_gammaL.png"))

plot(p1, p2, p3, layout=(1,3), size=(1200,400))