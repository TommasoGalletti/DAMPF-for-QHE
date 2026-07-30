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
# STEADY-STATE POPULATIONS FOR THE FULL VIBRONIC HAMILTONIAN
#
# IV_curve_vib.jl already produces I(ΓL), V(ΓL), P(ΓL) and the I-V/P-V
# curves for this model, so none of those are regenerated here. This
# script only adds the steady-state electronic populations (exciton
# and site basis) after tracing out the vibrations.
# =========================

"""Reduced electronic density matrix (site basis) tracing out both vibrational modes.
Basis ordering matches build_total_hamiltonian: |e> ⊗ |v1> ⊗ |v2>,
global index idx = (e-1)*Nv^2 + (v1-1)*Nv + v2."""
function reduced_electronic_density(ρ, dim_e, Nv)
    ρ_red = zeros(ComplexF64, dim_e, dim_e)
    for e1 in 1:dim_e, e2 in 1:dim_e
        for v1 in 1:Nv, v2 in 1:Nv
            idx1 = (e1 - 1) * Nv * Nv + (v1 - 1) * Nv + v2
            idx2 = (e2 - 1) * Nv * Nv + (v1 - 1) * Nv + v2
            ρ_red[e1, e2] += ρ[idx1, idx2]
        end
    end
    return ρ_red
end

# ------------------------------
# Model and simulation setup
# ------------------------------

params_base = default_params()
sim = default_simulation_params()
Nv_tag = "$(params_base.Nv)_modes"

H_e = Hamiltonian.build_hamiltonian(params_base)
energies_e, U_e = Hamiltonian.diagonalize_hamiltonian(H_e)

g_list = [0.0, 55.0]
CM1_TO_EV = sim.cm1_to_ev

for g_val in g_list
    params = Params(
        params_base.Eg,
        params_base.E,
        params_base.J,
        params_base.ωv,
        params_base.Nv,
        g_val,
    )

    H_total = Hamiltonian.build_total_hamiltonian(params)

    dim_e = size(H_e, 1)
    dim_v = params.Nv
    dim_total = dim_e * dim_v * dim_v

    γH = sim.γH
    γC = sim.γC
    nH = sim.nH
    TC = sim.TC
    kB = sim.kB
    γM = sim.γM
    TM = sim.TM

    nC = cold_occupations(energies_e, TC; kB=kB)

    println("=== VIBRONIC POPULATIONS (g=$(g_val)) ===")
    println("Electronic dimension: $dim_e")
    println("Vibrational dimension: $dim_v")
    println("Total dimension: $dim_total")
    println("Liouvillian dimension: $(dim_total^2)")
    println()

    ΓL_list = gammaL_list(sim)

    pop_ss = [Float64[] for _ in 1:dim_e]
    pop_site_ss = [Float64[] for _ in 1:dim_e]

    for ΓL in ΓL_list
        hot_ops_vib = build_hot_operators_vib(U_e; γH=γH, nH=nH, Nv=dim_v)
        cold_ops_vib = build_cold_operators_vib(U_e; γC=γC, nC_matrix=nC, Nv=dim_v)
        load_op_vib = build_load_operator_vib(U_e; ΓL=ΓL, Nv=dim_v)
        vib_damp_ops = build_vibrational_damping_operators(
            dim_e,
            dim_v;
            γM=γM,
            ωv=params.ωv,
            TM=TM,
            kB=kB,
        )

        jump_ops = JumpOperator[]
        append!(jump_ops, hot_ops_vib)
        append!(jump_ops, cold_ops_vib)
        push!(jump_ops, load_op_vib)
        append!(jump_ops, vib_damp_ops)

        L = build_liouvillian(H_total, jump_ops)
        ρ_ss = steady_state(L)

        ρ_red = reduced_electronic_density(ρ_ss, dim_e, dim_v)
        ρ_red_exc = U_e' * ρ_red * U_e

        for i in 1:dim_e
            push!(pop_ss[i], real(ρ_red_exc[i, i]))
            push!(pop_site_ss[i], real(ρ_red[i, i]))
        end
    end

    println("=== POPULATIONS COMPUTED (g=$(g_val)) ===\n")

    logΓL = log10.(ΓL_list)

    # =========================
    # PLOTS
    # =========================

    p5 = plot(
        logΓL,
        pop_ss[1],
        xlabel="log10(ΓL)",
        ylabel="Steady-state populations",
        title="Vibronic Simulation - steady-state exciton populations (g=$(g_val))",
        lw=2,
        label="p(ε1)",
        xlims=(-6, 6),
    )
    plot!(p5, logΓL, pop_ss[2], lw=2, label="p(ε2)")
    plot!(p5, logΓL, pop_ss[3], lw=2, label="p(ε3)")
    plot!(p5, logΓL, pop_ss[4], lw=2, label="p(ε0)")

    p6 = plot(
        logΓL,
        pop_site_ss[1],
        xlabel="log10(ΓL)",
        ylabel="Steady-state populations",
        title="Vibronic Simulation - steady-state site populations (g=$(g_val))",
        lw=2,
        label="p(g)",
        xlims=(-6, 6),
    )
    plot!(p6, logΓL, pop_site_ss[2], lw=2, label="p(s1)")
    plot!(p6, logΓL, pop_site_ss[3], lw=2, label="p(s2)")
    plot!(p6, logΓL, pop_site_ss[4], lw=2, label="p(s3)")

    outdir = joinpath(
        @__DIR__,
        "..",
        "outputs",
        "imgs",
        "vibrational",
        Nv_tag,
        "g_$(Int(round(g_val)))",
        "populations",
    )
    mkpath(outdir)

    savefig(p5, joinpath(outdir, "pop_ss_gammaL.png"))
    savefig(p6, joinpath(outdir, "pop_site_ss_gammaL.png"))

    println("Saved population plots to: $outdir\n")
end
