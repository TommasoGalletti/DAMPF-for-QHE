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
H = Hamiltonian.build_hamiltonian(params)

energies, U = Hamiltonian.diagonalize_hamiltonian(H)
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
            if ΔE > 1e-8
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
    V = Observables.voltage(ρ_ss, U, energies, idx_low, idx_g; TC=TC)
    P = Observables.power(I, V)

    push!(I_list, I)
    push!(V_list, V)
    push!(P_list, P)
end

# =========================
# SAMPLE OUTPUT
# =========================

println("\n=== SAMPLE OUTPUT ===")
for i in 1:5:length(ΓL_list)
    println("ΓL=$(ΓL_list[i]) | I=$(I_list[i]) | V=$(V_list[i]) | P=$(P_list[i])")
end

# =========================
# PLOTS
# =========================

logΓL = log10.(ΓL_list)

p1 = plot(logΓL, I_list,
    xlabel="log10(ΓL)",
    ylabel="Current I",
    title="I vs ΓL",
    lw=2,
    legend=false
)

p2 = plot(logΓL, V_list,
    xlabel="log10(ΓL)",
    ylabel="Voltage V",
    title="V vs ΓL",
    lw=2,
    legend=false
)
hline!(p2, [0.0], linestyle=:dash)

p3 = plot(logΓL, P_list,
    xlabel="log10(ΓL)",
    ylabel="Power P",
    title="P vs ΓL",
    lw=2,
    legend=false
)
hline!(p3, [0.0], linestyle=:dash)

imax = argmax(P_list)
scatter!(p3, [logΓL[imax]], [P_list[imax]],
         markersize=8, color=:red, label="Max power")


outdir = joinpath(@__DIR__, "..", "imgs")
mkpath(outdir)

savefig(p1, joinpath(outdir, "current_vs_GammaL_electronic.png"))
savefig(p2, joinpath(outdir, "voltage_vs_GammaL_electronic.png"))
savefig(p3, joinpath(outdir, "power_vs_GammaL_electronic.png"))

plot(p1, p2, p3, layout=(1,3), size=(1200,400))