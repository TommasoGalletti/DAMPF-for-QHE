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


#import Pkg; Pkg.add("Plots")
using Plots
default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

# =========================
# MODEL SETUP
# =========================

function bose(ω, T)
    x = ω / T

    if x < 1e-6
        return T / ω
    elseif x > 700
        return 0.0
    else
        return 1 / (exp(x) - 1)
    end
end

params = default_params()
H = build_hamiltonian(params)

energies, U = diagonalize_hamiltonian(H)
dim = size(H,1)

# =========================
# ε₁ ≥ ε₂ ≥ ε₃ ≥ ε₀
# =========================

idx_high = 1
idx_low  = dim - 1
idx_g    = dim

# =========================
# PARAMETRI
# =========================

γH = 0.01
γC = 8.0
nH = 60000.0
TC = 293.0

nC = zeros(dim, dim)

for i in 1:dim
    for j in 1:dim

        if i != j
            ΔE = energies[i] - energies[j]

            if ΔE > 1e-8   # SOLO transizioni fisiche ben definite
                nC[i,j] = 1 / (exp(ΔE / TC) - 1)

            end
        end
    end
end

# DEBUG
println("nC max = ", maximum(nC))
println("nC min (nonzero) = ", minimum(nC[nC .> 0]))
println("ΔE min = ", minimum(abs.(energies[i] - energies[j] for i in 1:dim for j in 1:dim if i != j)))

# =========================
# SWEEP SU ΓL (LOG SCALE)
# =========================

ΓL_list = 10 .^ range(-3, 1, length=60)

I_list = Float64[]
V_list = Float64[]
P_list = Float64[]

for ΓL in ΓL_list

    # Lindblad
    hot_ops  = build_hot_operators(U; γH=γH, nH=nH)
    cold_ops = build_cold_operators(U; γC=γC, nC_matrix=nC)
    load_op  = build_load_operator(U; ΓL=ΓL)

    jump_ops = JumpOperator[]
    append!(jump_ops, hot_ops)
    append!(jump_ops, cold_ops)
    push!(jump_ops, load_op)

    # Liouvilliana
    L = build_liouvillian(H, jump_ops)

    # steady state
    ρ_ss = steady_state(L)

    # osservabili
    I = Observables.current(ρ_ss, U, ΓL, idx_low)
    V = Observables.voltage(ρ_ss, U, energies, idx_low, idx_g; TC=TC)
    P = Observables.power(I, V)

    push!(I_list, I)
    push!(V_list, V)
    push!(P_list, P)

    #(Zeno check)
    # p_low = population_exciton(ρ_ss, U, idx_low)
    # println("ΓL=", ΓL, "  p_low=", p_low)

end

# =========================
# =========================

println("\n=== SAMPLE OUTPUT ===")
for i in 1:5:length(ΓL_list)
    println("ΓL=$(ΓL_list[i]) | I=$(I_list[i]) | V=$(V_list[i]) | P=$(P_list[i])")
end

# =========================
# PLOTS
# =========================

using Plots

# scala log per ΓL
logΓL = log10.(ΓL_list)

# filtro regione fisica (V > 0)
valid = V_list .> 0

ΓL_valid = ΓL_list[valid]
logΓL_valid = log10.(ΓL_valid)
I_valid  = I_list[valid]
V_valid  = V_list[valid]
P_valid  = P_list[valid]

# -------------------------
# 1. Corrente
# -------------------------
p1 = plot(logΓL, I_list,
    xlabel="log10(ΓL)",
    ylabel="Current I",
    title="I vs ΓL",
    lw=2,
    legend=false
)

# -------------------------
# 2. Tensione
# -------------------------
p2 = plot(logΓL, V_list,
    xlabel="log10(ΓL)",
    ylabel="Voltage V",
    title="V vs ΓL",
    lw=2,
    legend=false
)
hline!(p2, [0.0], linestyle=:dash)

# -------------------------
# 3. Potenza (max marker)
# -------------------------
p3 = plot(logΓL, P_list,
    xlabel="log10(ΓL)",
    ylabel="Power P",
    title="P vs ΓL",
    lw=2,
    legend=false
)
hline!(p3, [0.0], linestyle=:dash)

# marker max power
imax = argmax(P_list)
scatter!(p3, [logΓL[imax]], [P_list[imax]], 
         markersize=8, color=:red, label="Max power")

# -------------------------
# 4. Potenza (regime fisico)
# -------------------------
p4 = plot(logΓL_valid, P_valid,
    xlabel="log10(ΓL)",
    ylabel="Power P",
    title="P vs ΓL (V > 0)",
    lw=2,
    legend=false
)

savefig(p1, "./julia/imgs/current_vs_GammaL.png")
savefig(p2, "./julia/imgs/voltage_vs_GammaL.png")
savefig(p3, "./julia/imgs/power_vs_GammaL.png")
savefig(p4, "./julia/imgs/power_vs_GammaL_valid.png")

# -------------------------
# combined
# -------------------------
plot(p1, p2, p3, p4, layout=(2,2), size=(900,700))