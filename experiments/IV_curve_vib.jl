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

"""Save raw and normalized observables to CSV for later comparisons."""
function save_observables_csv(filepath, ΓL_list, I_list, V_list, P_list, γH, cm1_to_ev)
    logΓL = log10.(ΓL_list)
    V_list_eV = cm1_to_ev .* V_list
    I_list_norm = I_list ./ γH
    P_list_norm = (I_list .* V_list_eV) ./ γH

    mkpath(dirname(filepath))
    open(filepath, "w") do io
        println(io, "GammaL,log10_GammaL,I_raw,V_cm1,P_raw,V_eV,I_norm,P_norm")
        for i in eachindex(ΓL_list)
            println(
                io,
                string(
                    ΓL_list[i], ",",
                    logΓL[i], ",",
                    I_list[i], ",",
                    V_list[i], ",",
                    P_list[i], ",",
                    V_list_eV[i], ",",
                    I_list_norm[i], ",",
                    P_list_norm[i],
                ),
            )
        end
    end
end

# ------------------------------
# Model and simulation setup
# ------------------------------

params_base = default_params()
sim = default_simulation_params()

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

    idx_low = dim_e - 1
    idx_g = dim_e

    γH = sim.γH
    γC = sim.γC
    nH = sim.nH
    TC = sim.TC
    kB = sim.kB
    γM = sim.γM
    TM = sim.TM

    nC = cold_occupations(energies_e, TC; kB=kB)

    println("=== VIBRONIC SYSTEM SETUP (g=$(g_val)) ===")
    println("Electronic dimension: $dim_e")
    println("Vibrational dimension: $dim_v")
    println("Total dimension: $dim_total")
    println("Liouvillian dimension: $(dim_total^2)")
    println()
    println("nC max = ", maximum(nC))
    println("nC min (nonzero) = ", minimum(nC[nC .> 0]))
    println()

    ΓL_list = gammaL_list(sim)

    I_list = Float64[]
    V_list = Float64[]
    P_list = Float64[]

    for (idx_ΓL, ΓL) in enumerate(ΓL_list)
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

        I = Observables.current_vib(ρ_ss, U_e, ΓL, idx_low, dim_v)
        V = Observables.voltage_vib(
            ρ_ss,
            U_e,
            energies_e,
            idx_low,
            idx_g,
            dim_v;
            kB=kB,
            TC=TC,
        )
        P = Observables.power(I, V)

        push!(I_list, I)
        push!(V_list, V)
        push!(P_list, P)

        if mod(idx_ΓL, 10) == 1
            println("ΓL = $(ΓL): I = $(I), V = $(V) cm^-1 ($(V * CM1_TO_EV) eV), P = $(P)")
        end
    end

    println("\n=== SIMULATION COMPLETED (g=$(g_val)) ===\n")

    println("\n=== SAMPLE OUTPUT ===")
    for i in 1:5:length(ΓL_list)
        println(
            "ΓL=$(ΓL_list[i]) | I=$(I_list[i]) | V=$(V_list[i]) cm^-1 ($(V_list[i] * CM1_TO_EV) eV) | P=$(P_list[i])",
        )
    end

    logΓL = log10.(ΓL_list)
    V_list_eV = CM1_TO_EV .* V_list
    I_list_paper = I_list ./ γH
    P_list_paper = (I_list .* V_list_eV) ./ γH

    idx_max_P = argmax(P_list_paper)
    ΓL_max_P = ΓL_list[idx_max_P]
    P_max = P_list_paper[idx_max_P]
    V_at_max_P = V_list_eV[idx_max_P]

    sort_idx = sortperm(V_list_eV)
    V_plot = V_list_eV[sort_idx]
    I_plot = I_list_paper[sort_idx]
    P_plot = P_list_paper[sort_idx]

    println("\n=== MAXIMUM POWER ===")
    println("Max P = $P_max eV at V = $V_at_max_P eV (ΓL = $ΓL_max_P)")

    p1 = plot(
        logΓL,
        I_list;
        xlabel="log10(ΓL)",
        ylabel="Current I",
        title="Current vs ΓL (vibronic, g=$(g_val))",
        lw=2,
        legend=false,
    )

    p2 = plot(
        logΓL,
        V_list_eV;
        xlabel="log10(ΓL)",
        ylabel="Voltage V (eV)",
        title="Voltage vs ΓL (vibronic, eV, g=$(g_val))",
        lw=2,
        legend=false,
    )
    hline!(p2, [0.0]; linestyle=:dash)

    p3 = plot(
        logΓL,
        P_list;
        xlabel="log10(ΓL)",
        ylabel="Power P",
        title="Power vs ΓL (vibronic, g=$(g_val))",
        lw=2,
        legend=false,
    )
    hline!(p3, [0.0]; linestyle=:dash)

    imax = argmax(P_list)
    scatter!(p3, [logΓL[imax]], [P_list[imax]]; markersize=8, color=:red, label="Max power")

    outdir = joinpath(@__DIR__, "..", "imgs", "vibrational", "g_$(Int(round(g_val)))")
    mkpath(outdir)

    savefig(p1, joinpath(outdir, "i_gammaL.png"))
    savefig(p2, joinpath(outdir, "v_gammaL.png"))
    savefig(p3, joinpath(outdir, "p_gammaL.png"))

    p_iv = plot(
        V_plot,
        I_plot;
        xlabel="Voltage (V)",
        ylabel="Current (1/γ_H)",
        title="I(V), P(V) (g=$(g_val))",
        lw=2,
        color=:black,
        guidefontsize=16,
        tickfontsize=13,
        titlefontsize=18,
        legend=false,
    )

    p_twin = plot!(
        twinx(),
        V_plot,
        P_plot;
        ylabel="Power (eV)",
        lw=2,
        color=:red,
        guidefontcolor=:red,
        tickfontcolor=:red,
        foreground_color_axis=:red,
        guidefontsize=16,
        tickfontsize=13,
        legend=false,
    )

    idx_max_in_sorted = argmax(P_plot)
    V_max_sorted = V_plot[idx_max_in_sorted]
    P_max_sorted = P_plot[idx_max_in_sorted]
    I_max_sorted = I_plot[idx_max_in_sorted]
    P_max_label = round(P_max_sorted, sigdigits=3)

    scatter!(
        p_iv,
        [V_max_sorted],
        [I_max_sorted];
        markersize=8,
        color=:black,
        markerstrokewidth=2,
        markerstrokecolor=:darkgray,
    )

    scatter!(
        p_twin,
        [V_max_sorted],
        [P_max_sorted];
        markersize=8,
        color=:red,
        markerstrokewidth=2,
        markerstrokecolor=:darkred,
    )

    annotate!(
        p_iv,
        V_max_sorted * 0.98,
        I_max_sorted * 1.15,
        text("Max P = $(P_max_label)\nΓL=$(round(ΓL_max_P, sigdigits=3))", 9, :red, :center),
    )

    savefig(p_iv, joinpath(outdir, "power_and_current_vs_voltage_vibronic.png"))

    data_outdir = joinpath(
        @__DIR__,
        "..",
        "outputs",
        "numeric_vectors",
        "vibrational",
        "full",
        "6_modes",
        "g_$(Int(round(g_val)))",
    )
    data_filepath = joinpath(data_outdir, "observables_vs_gammaL.csv")
    save_observables_csv(data_filepath, ΓL_list, I_list, V_list, P_list, γH, CM1_TO_EV)
    println("Saved numeric vectors: $data_filepath")

    plot(p_iv, p2, p3; layout=(1, 3), size=(1200, 400))
end
