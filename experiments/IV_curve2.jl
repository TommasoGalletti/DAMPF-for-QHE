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

# Model setup.

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
H = Hamiltonian.build_hamiltonian(params)

energies, U = Hamiltonian.diagonalize_hamiltonian(H)
dim = size(H,1)

# Energy ordering is descending: ε1 ≥ ε2 ≥ ε3 ≥ ε0.

idx_high = 1
idx_low  = dim - 1
idx_g    = dim

# Dissipation parameters.

γH = sim.γH
γC = sim.γC
nH = sim.nH
TC = sim.TC
kB = sim.kB

nC = cold_occupations(energies, TC; kB=kB)

# Debug output.
println("nC max = ", maximum(nC))
println("nC min (nonzero) = ", minimum(nC[nC .> 0]))
println("Minimum level spacing = ", minimum(abs.(energies[i] - energies[j] for i in 1:dim for j in 1:dim if i != j)))

# Sweep over ΓL on a logarithmic scale.

ΓL_list = gammaL_list(sim)

I_list = Float64[]
V_list = Float64[]
P_list = Float64[]
CM1_TO_EV = sim.cm1_to_ev

for ΓL in ΓL_list

    hot_ops  = build_hot_operators(U; γH=γH, nH=nH)
    cold_ops = build_cold_operators(U; γC=γC, nC_matrix=nC)
    load_op  = build_load_operator(U; ΓL=ΓL)

    jump_ops = JumpOperator[]
    append!(jump_ops, hot_ops)
    append!(jump_ops, cold_ops)
    push!(jump_ops, load_op)

    L = build_liouvillian(H, jump_ops)

    ρ_ss = steady_state(L)

    I = Observables.current(ρ_ss, U, ΓL, idx_low)
    V = Observables.voltage(ρ_ss, U, energies, idx_low, idx_g; kB=kB, TC=TC)
    P = Observables.power(I, V)

    push!(I_list, I)
    push!(V_list, V)
    push!(P_list, P)
end

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
    ylabel="Current I (arb. units)",
    title="Current vs ΓL",
    lw=2,
    legend=false
)

p2 = plot(logΓL, V_list_eV,
    xlabel="log10(ΓL)",
    ylabel="Voltage V (eV)",
    title="Voltage vs ΓL (eV)",
    lw=2,
    legend=false
)
hline!(p2, [0.0], linestyle=:dash)

p3 = plot(logΓL, P_list,
    xlabel="log10(ΓL)",
    ylabel="Power P (arb. units)",
    title="Power vs ΓL",
    lw=2,
    legend=false
)
hline!(p3, [0.0], linestyle=:dash)

imax = argmax(P_list)
scatter!(p3, [logΓL[imax]], [P_list[imax]],
         markersize=8, color=:red, label="Max power")


outdir = joinpath(@__DIR__, "..", "imgs", "electronic")
mkpath(outdir)

savefig(p1, joinpath(outdir, "i_gammaL.png"))
savefig(p2, joinpath(outdir, "v_gammaL.png"))
savefig(p3, joinpath(outdir, "p_gammaL.png"))

plot(p1, p2, p3, layout=(1,3), size=(1200,400))