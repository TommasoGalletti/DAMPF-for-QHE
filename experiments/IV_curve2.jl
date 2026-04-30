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

# Sample output in paper units.

V_list_V = CM1_TO_EV .* V_list
I_list_paper = I_list ./ γH
P_list_paper = (I_list .* V_list_V) ./ γH

# Find the ΓL at which P is maximum (before sorting)
idx_max_P = argmax(P_list_paper)
ΓL_max_P = ΓL_list[idx_max_P]
P_max = P_list_paper[idx_max_P]
V_at_max_P = V_list_V[idx_max_P]
I_at_max_P = I_list_paper[idx_max_P]

sort_idx = sortperm(V_list_V)
V_plot = V_list_V[sort_idx]
I_plot = I_list_paper[sort_idx]
P_plot = P_list_paper[sort_idx]

println("\n=== SAMPLE OUTPUT (paper units) ===")
for i in 1:5:length(V_plot)
    println("V=$(V_plot[i]) V | I=$(I_plot[i]) [I_L/(eγ_H)] | P=$(P_plot[i]) eV")
end

println("\n=== MAXIMUM POWER ===")
println("Max P = $P_max eV at V = $V_at_max_P V (ΓL = $ΓL_max_P)")

outdir = joinpath(@__DIR__, "..", "imgs", "electronic")
mkpath(outdir)

p_iv = plot(V_plot, I_plot,
    xlabel="Voltage (V)",
    ylabel="Current (1/γ_H)",
    title="I(V), P(V)",
    lw=2,
    color=:black,
    guidefontsize=16,
    tickfontsize=13,
    titlefontsize=18,
    legend=false
)

p_twin = plot!(twinx(), V_plot, P_plot,
    ylabel="Power (eV)",
    lw=2,
    color=:red,
    guidefontcolor=:red,
    tickfontcolor=:red,
    foreground_color_axis=:red,
    guidefontsize=16,
    tickfontsize=13,
    legend=false
)

# Mark the maximum power point on both axes
idx_max_in_sorted = argmax(P_plot)
V_max_sorted = V_plot[idx_max_in_sorted]
P_max_sorted = P_plot[idx_max_in_sorted]
I_max_sorted = I_plot[idx_max_in_sorted]

# Scatter on current axis (left y-axis)
scatter!(p_iv, [V_max_sorted], [I_max_sorted],
    markersize=8, color=:black, markerstrokewidth=2, markerstrokecolor=:darkgray
)

# Scatter on power axis (right y-axis)
scatter!(p_twin, [V_max_sorted], [P_max_sorted],
    markersize=8, color=:red, markerstrokewidth=2, markerstrokecolor=:darkred
)

# Add text annotation with ΓL value
annotate!(p_iv, V_max_sorted * 0.98, I_max_sorted * 1.15,
    text("Max P\nΓL=$(round(ΓL_max_P, sigdigits=3))", 9, :red, :center)
)

savefig(p_iv, joinpath(outdir, "power_and_current_vs_voltage.png"))

p_iv