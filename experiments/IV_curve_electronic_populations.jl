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

function pick_significant_indices(ΓL_list, P_list)
    # Select 4 physically meaningful indices:
    # 1. Very low ΓL (≈ 10^-8)
    # 2. Transition point (where P > 50% of P_max)
    # 3. Maximum power point
    # 4. Very high ΓL (≈ 10^8)
    
    n = length(ΓL_list)
    idx_low = 1  # ΓL ≈ 10^-8
    idx_high = n  # ΓL ≈ 10^8
    
    idx_max_P = argmax(P_list)  # ΓL that maximizes P
    
    # Find transition point: first index where P > 50% of max P
    P_max = maximum(P_list)
    idx_transition = findfirst(p -> p > 0.5 * P_max, P_list)
    if idx_transition === nothing
        # Fallback: use log-space midpoint between low and max_P
        idx_transition = Int(round(sqrt(idx_low * idx_max_P)))
    end
    
    idx = sort(unique([idx_low, idx_transition, idx_max_P, idx_high]))
    
    println("\n=== SELECTED ΓL INDICES ===")
    for i in idx
        println("idx=$i: ΓL=$(round(ΓL_list[i], sigdigits=3)), log₁₀(ΓL)=$(round(log10(ΓL_list[i]), sigdigits=3)), P=$(round(P_list[i], sigdigits=3))")
    end
    
    return idx
end
"""Estimate the slowest relaxation timescale from the Liouvillian spectrum."""

function relaxation_time(L)
    vals = eigvals(Matrix(L))
    revals = real.(vals)
    candidates = [x for x in revals if x < -1e-9]
    slowest = maximum(candidates)
    return 1 / abs(slowest)
end

"""Site-basis population (rho is already represented in site basis)."""
function population_site(ρ, i)
    return real(ρ[i, i])
end

params = default_params()
sim = default_simulation_params()
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

γH = sim.γH
γC = sim.γC
nH = sim.nH
TC = sim.TC
kB = sim.kB
CM1_TO_EV = sim.cm1_to_ev

nC = cold_occupations(energies, TC; kB=kB)

println("nC max = ", maximum(nC))
println("nC min (nonzero) = ", minimum(nC[nC .> 0]))
println("ΔE min = ", minimum(abs.(energies[i] - energies[j] for i in 1:dim for j in 1:dim if i != j)))

# =========================
# SWEEP SU ΓL
# =========================

ΓL_list = gammaL_list(sim)

I_list = Float64[]
V_list = Float64[]
P_list = Float64[]
pop_ss = [Float64[] for _ in 1:dim]
pop_site_ss = [Float64[] for _ in 1:dim]

for ΓL in ΓL_list
    jump_ops = build_jump_ops(U, nC, γH, nH, γC, ΓL)
    L = build_liouvillian(H, jump_ops)

    ρ_ss = steady_state(L)

    I = Observables.current(ρ_ss, U, ΓL, idx_low)
    V = Observables.voltage(ρ_ss, U, energies, idx_low, idx_g; kB=kB, TC=TC)
    P = Observables.power(I, V)

    push!(I_list, I)
    push!(V_list, V)
    push!(P_list, P)

    for i in 1:dim
        push!(pop_ss[i], population_exciton(ρ_ss, U, i))
        push!(pop_site_ss[i], population_site(ρ_ss, i))
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

# Il transiente è molto rapido per i tassi scelti (ordine 1e-2 - 1e-1 in unità inverse),
# quindi una griglia fino a t=400 con pochi punti appare piatta. Usiamo una griglia adattata.
idx_time = pick_significant_indices(ΓL_list, P_list)

time_plots = Any[]

for idx in idx_time
    ΓL = ΓL_list[idx]
    jump_ops = build_jump_ops(U, nC, γH, nH, γC, ΓL)
    L = build_liouvillian(H, jump_ops)
    L_dense = Matrix(L)

    τ_relax = relaxation_time(L_dense)
    t_max = max(0.2, 8.0 * τ_relax)
    t_list = range(0.0, stop=t_max, length=600)

    pop_t = [Float64[] for _ in 1:dim]
    pop_t_site = [Float64[] for _ in 1:dim]

    for t in t_list
        ρ_t_vec = exp(L_dense * t) * ρ0_vec
        ρ_t = reshape(ρ_t_vec, dim, dim)
        ρ_t = (ρ_t + ρ_t') / 2
        ρ_t ./= real(tr(ρ_t))

        for i in 1:dim
            push!(pop_t[i], population_exciton(ρ_t, U, i))
            push!(pop_t_site[i], population_site(ρ_t, i))
        end
    end

    pt = plot(
        t_list,
        pop_t[1],
        xlabel="t",
        ylabel="Population",
        title="Popolazioni nel tempo (ΓL=$(round(ΓL, sigdigits=3)), τ≈$(round(τ_relax, sigdigits=3)))",
        lw=2,
        label="p(ε1)"
    )
    plot!(pt, t_list, pop_t[2], lw=2, label="p(ε2)")
    plot!(pt, t_list, pop_t[3], lw=2, label="p(ε3)")
    plot!(pt, t_list, pop_t[4], lw=2, label="p(ε0)")

    push!(time_plots, pt)

    ps = plot(
        t_list,
        pop_t_site[1],
        xlabel="t (arb. units)",
        ylabel="Population",
        title="Popolazioni siti nel tempo (ΓL=$(round(ΓL, sigdigits=3)))",
        lw=2,
        label="p(g)"
    )
    plot!(ps, t_list, pop_t_site[2], lw=2, label="p(s1)")
    plot!(ps, t_list, pop_t_site[3], lw=2, label="p(s2)")
    plot!(ps, t_list, pop_t_site[4], lw=2, label="p(s3)")
    push!(time_plots, ps)
end

p7 = plot(time_plots..., layout=(4,2), size=(1200,1400))

# =========================
# PLOTS
# =========================

p1 = plot(logΓL, I_list,
    xlabel="log10(ΓL)",
    ylabel="Current I (arb. units)",
    title="I vs ΓL",
    lw=2,
    legend=false
)

p2 = plot(logΓL, CM1_TO_EV .* V_list,
    xlabel="log10(ΓL)",
    ylabel="Voltage V (eV)",
    title="V vs ΓL (eV)",
    lw=2,
    legend=false
)
hline!(p2, [0.0], linestyle=:dash)

p3 = plot(logΓL, P_list,
    xlabel="log10(ΓL)",
    ylabel="Power P (arb. units)",
    title="P vs ΓL",
    lw=2,
    legend=false
)
hline!(p3, [0.0], linestyle=:dash)

imax = argmax(P_list)
scatter!(p3, [logΓL[imax]], [P_list[imax]], markersize=8, color=:red, label="Max power")

p4 = plot(logΓL_valid, P_valid,
    xlabel="log10(ΓL)",
    ylabel="Power P (arb. units)",
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

p6 = plot(
    logΓL,
    pop_site_ss[1],
    xlabel="log10(ΓL)",
    ylabel="Steady-state populations",
    title="Popolazioni siti all'equilibrio vs ΓL",
    lw=2,
    label="p(g)"
)
plot!(p6, logΓL, pop_site_ss[2], lw=2, label="p(s1)")
plot!(p6, logΓL, pop_site_ss[3], lw=2, label="p(s2)")
plot!(p6, logΓL, pop_site_ss[4], lw=2, label="p(s3)")

# p7 = plot(time_plots..., layout=(4,2), size=(1200,1400))  # Commented: too many subplots

outdir = joinpath(@__DIR__, "..", "imgs", "electronic", "populations")
mkpath(outdir)

savefig(p5, joinpath(outdir, "pop_ss_gammaL.png"))
savefig(p6, joinpath(outdir, "pop_site_ss_gammaL.png"))
savefig(p7, joinpath(outdir, "pop_time_4gammaL.png"))