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

using LinearAlgebra
using Plots
default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

# =========================
# MODEL SETUP
# =========================

function build_jump_ops(U, nC, γH, nH, γC, ΓL)
    hot_ops  = build_hot_operators(U; γH=γH, nH=nH)
    cold_ops = build_cold_operators(U; γC=γC, nC_matrix=nC)
    load_op  = build_load_operator(U; ΓL=ΓL)

    jump_ops = JumpOperator[]
    append!(jump_ops, hot_ops)
    append!(jump_ops, cold_ops)
    push!(jump_ops, load_op)

    return jump_ops
end

function pick_significant_indices(n, idx_max_power)
    idx = Int[]

    for k in [1, Int(round(0.20 * (n - 1) + 1)), idx_max_power, n]
        k_clamped = clamp(k, 1, n)
        if !(k_clamped in idx)
            push!(idx, k_clamped)
        end
    end

    for k in [Int(round(0.40 * (n - 1) + 1)), Int(round(0.60 * (n - 1) + 1)), Int(round(0.80 * (n - 1) + 1))]
        k_clamped = clamp(k, 1, n)
        if !(k_clamped in idx)
            push!(idx, k_clamped)
        end
        length(idx) == 4 && break
    end

    return sort(idx)
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
            if ΔE > 1e-8
                nC[i,j] = 1 / (exp(ΔE / TC) - 1)
            end
        end
    end
end

println("nC max = ", maximum(nC))
println("nC min (nonzero) = ", minimum(nC[nC .> 0]))
println("ΔE min = ", minimum(abs.(energies[i] - energies[j] for i in 1:dim for j in 1:dim if i != j)))

# =========================
# SWEEP SU ΓL
# =========================

ΓL_list = 10 .^ range(-3, 1, length=60)

I_list = Float64[]
V_list = Float64[]
P_list = Float64[]
pop_ss = [Float64[] for _ in 1:dim]

for ΓL in ΓL_list
    jump_ops = build_jump_ops(U, nC, γH, nH, γC, ΓL)
    L = build_liouvillian(H, jump_ops)

    ρ_ss = steady_state(L)

    I = Observables.current(ρ_ss, U, ΓL, idx_low)
    V = Observables.voltage(ρ_ss, U, energies, idx_low, idx_g; TC=TC)
    P = Observables.power(I, V)

    push!(I_list, I)
    push!(V_list, V)
    push!(P_list, P)

    for i in 1:dim
        push!(pop_ss[i], population_exciton(ρ_ss, U, i))
    end
end

println("\n=== SAMPLE OUTPUT ===")
for i in 1:5:length(ΓL_list)
    println("ΓL=$(ΓL_list[i]) | I=$(I_list[i]) | V=$(V_list[i]) | P=$(P_list[i])")
end

# =========================
# POPOLAZIONI NEL TEMPO
# =========================

logΓL = log10.(ΓL_list)
valid = V_list .> 0

ΓL_valid = ΓL_list[valid]
logΓL_valid = log10.(ΓL_valid)
P_valid  = P_list[valid]

ρ0 = zeros(ComplexF64, dim, dim)
ρ0[1,1] = 1.0
ρ0_vec = vec(ρ0)

t_list = range(0.0, stop=400.0, length=260)
idx_max_power = argmax(P_list)
idx_time = pick_significant_indices(length(ΓL_list), idx_max_power)

time_plots = Any[]

for idx in idx_time
    ΓL = ΓL_list[idx]
    jump_ops = build_jump_ops(U, nC, γH, nH, γC, ΓL)
    L = build_liouvillian(H, jump_ops)

    pop_t = [Float64[] for _ in 1:dim]

    for t in t_list
        ρ_t_vec = exp(L * t) * ρ0_vec
        ρ_t = reshape(ρ_t_vec, dim, dim)
        ρ_t = (ρ_t + ρ_t') / 2
        ρ_t ./= real(tr(ρ_t))

        for i in 1:dim
            push!(pop_t[i], population_exciton(ρ_t, U, i))
        end
    end

    pt = plot(
        t_list,
        pop_t[1],
        xlabel="t",
        ylabel="Population",
        title="Popolazioni nel tempo (ΓL=$(round(ΓL, sigdigits=3)))",
        lw=2,
        label="p(ε1)"
    )
    plot!(pt, t_list, pop_t[2], lw=2, label="p(ε2)")
    plot!(pt, t_list, pop_t[3], lw=2, label="p(ε3)")
    plot!(pt, t_list, pop_t[4], lw=2, label="p(ε0)")

    push!(time_plots, pt)
end

# =========================
# PLOTS
# =========================

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
scatter!(p3, [logΓL[imax]], [P_list[imax]], markersize=8, color=:red, label="Max power")

p4 = plot(logΓL_valid, P_valid,
    xlabel="log10(ΓL)",
    ylabel="Power P",
    title="P vs ΓL (V > 0)",
    lw=2,
    legend=false
)

p5 = plot(
    logΓL,
    pop_ss[1],
    xlabel="log10(ΓL)",
    ylabel="Steady-state populations",
    title="Popolazioni all'equilibrio vs ΓL",
    lw=2,
    label="p(ε1)"
)
plot!(p5, logΓL, pop_ss[2], lw=2, label="p(ε2)")
plot!(p5, logΓL, pop_ss[3], lw=2, label="p(ε3)")
plot!(p5, logΓL, pop_ss[4], lw=2, label="p(ε0)")

p6 = plot(time_plots..., layout=(2,2), size=(1100,700))

outdir = joinpath(@__DIR__, "..", "imgs")
mkpath(outdir)

savefig(p1, joinpath(outdir, "current_vs_GammaL_electronic_populations.png"))
savefig(p2, joinpath(outdir, "voltage_vs_GammaL_electronic_populations.png"))
savefig(p3, joinpath(outdir, "power_vs_GammaL_electronic_populations.png"))
savefig(p4, joinpath(outdir, "power_vs_GammaL_valid_electronic_populations.png"))
savefig(p5, joinpath(outdir, "populations_ss_vs_GammaL_electronic_populations.png"))
savefig(p6, joinpath(outdir, "populations_time_4GammaL_electronic_populations.png"))

plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1200,1300))