module DampfPlots

using Plots

export load_reference_sweep, plot_comparison, plot_iv_dampf,
       plot_current_vs_voltage_comparison, plot_power_vs_voltage_comparison

"""
Load a full-sweep exact-Lindblad reference CSV (columns
`GammaL,log10_GammaL,I_raw,V_cm1,P_raw,V_eV,I_norm,P_norm`, as produced by
`experiments/IV_curve_vib.jl`). Returns `nothing` if the file is missing, so
callers can degrade gracefully rather than erroring.
"""
function load_reference_sweep(filepath)
    isfile(filepath) || return nothing
    ΓL = Float64[]
    I_raw = Float64[]
    V_cm1 = Float64[]
    P_raw = Float64[]
    for (i, line) in enumerate(eachline(filepath))
        i == 1 && continue
        cols = split(line, ",")
        push!(ΓL, parse(Float64, cols[1]))
        push!(I_raw, parse(Float64, cols[3]))
        push!(V_cm1, parse(Float64, cols[4]))
        push!(P_raw, parse(Float64, cols[5]))
    end
    return (ΓL=ΓL, I=I_raw, V=V_cm1, P=P_raw)
end

"""Overlay DAMPF vs Direct Lindblad I, V, P vs log10(ΓL), plus a trace-distance
panel. `reference`, if given (see `load_reference_sweep`), is drawn as a
dense background curve behind the sparse DAMPF/Direct-Lindblad comparison."""
function plot_comparison(results, outdir; reference=nothing)
    mkpath(outdir)
    logΓL = log10.([r.ΓL for r in results])
    ref_logΓL = reference === nothing ? nothing : log10.(reference.ΓL)

    p1 = plot(; xlabel="log10(ΓL / cm^-1)", ylabel="Current I (cm^-1)", title="Current vs load rate ΓL")
    reference !== nothing && plot!(p1, ref_logΓL, reference.I; label="Direct Lindblad (dense sweep)", lw=1, color=:gray, alpha=0.6)
    plot!(p1, logΓL, [r.I_exact for r in results]; label="Direct Lindblad", lw=2, marker=:circle)
    plot!(p1, logΓL, [r.I_dampf for r in results]; label="DAMPF", lw=2, marker=:diamond, linestyle=:dash)
    savefig(p1, joinpath(outdir, "current_vs_gammaL.png"))

    p2 = plot(; xlabel="log10(ΓL / cm^-1)", ylabel="Voltage V (cm^-1)", title="Voltage vs load rate ΓL", legend=:bottomleft)
    reference !== nothing && plot!(p2, ref_logΓL, reference.V; label="Direct Lindblad (dense sweep)", lw=1, color=:gray, alpha=0.6)
    plot!(p2, logΓL, [r.V_exact for r in results]; label="Direct Lindblad", lw=2, marker=:circle)
    plot!(p2, logΓL, [r.V_dampf for r in results]; label="DAMPF", lw=2, marker=:diamond, linestyle=:dash)
    savefig(p2, joinpath(outdir, "voltage_vs_gammaL.png"))

    p3 = plot(; xlabel="log10(ΓL / cm^-1)", ylabel="Power P (cm^-2)", title="Power vs load rate ΓL")
    reference !== nothing && plot!(p3, ref_logΓL, reference.P; label="Direct Lindblad (dense sweep)", lw=1, color=:gray, alpha=0.6)
    plot!(p3, logΓL, [r.P_exact for r in results]; label="Direct Lindblad", lw=2, marker=:circle)
    plot!(p3, logΓL, [r.P_dampf for r in results]; label="DAMPF", lw=2, marker=:diamond, linestyle=:dash)
    savefig(p3, joinpath(outdir, "power_vs_gammaL.png"))

    p4 = plot(logΓL, [r.trace_dist for r in results]; label=false, lw=2, marker=:square,
              xlabel="log10(ΓL / cm^-1)", ylabel="Trace distance (ρ_DAMPF, ρ_direct)", title="Full-state error vs load rate ΓL")
    savefig(p4, joinpath(outdir, "trace_distance_vs_gammaL.png"))

    panel = plot(p1, p2, p3, p4; layout=(2, 2), size=(1000, 800))
    savefig(panel, joinpath(outdir, "iv_curve_comparison_panel.png"))
end

"""Twin-axis Current(V) and Power(V) plot for DAMPF alone (points sorted by
voltage so the line does not zigzag -- V decreases monotonically with ΓL, so
this amounts to reversing the ΓL order)."""
function plot_iv_dampf(results, outdir)
    mkpath(outdir)
    V = [r.V_dampf for r in results]
    I = [r.I_dampf for r in results]
    P = [r.P_dampf for r in results]
    idx = sortperm(V)
    V, I, P = V[idx], I[idx], P[idx]

    p = plot(V, I; xlabel="Voltage V (cm^-1)", ylabel="Current I (cm^-1)",
              title="DAMPF: current and power vs voltage", lw=2, marker=:diamond,
              color=:black, guidefontcolor=:black, legend=false)
    twin = twinx(p)
    plot!(twin, V, P; ylabel="Power P (cm^-2)", lw=2, marker=:diamond, color=:red,
          guidefontcolor=:red, tickfontcolor=:red, foreground_color_axis=:red, legend=false)

    imax = argmax(P)
    scatter!(twin, [V[imax]], [P[imax]]; markersize=8, color=:red, markerstrokewidth=2, markerstrokecolor=:darkred, label=false)
    annotate!(p, minimum(V) + 0.05 * (maximum(V) - minimum(V)), minimum(I) + 0.10 * (maximum(I) - minimum(I)),
              text("Max P = $(round(P[imax], sigdigits=4))\nat V = $(round(V[imax], sigdigits=5))", 10, :red, :left);
              subplot=1)

    savefig(p, joinpath(outdir, "power_and_current_vs_voltage_dampf.png"))
end

"""Current(V) comparison, single axis: DAMPF vs Direct Lindblad (each
against its own voltage, since V_dampf and V_exact differ slightly at the
same ΓL). `reference`, if given, adds the dense-sweep curve as background."""
function plot_current_vs_voltage_comparison(results, outdir; reference=nothing)
    mkpath(outdir)
    Vd, Id = [r.V_dampf for r in results], [r.I_dampf for r in results]
    Ve, Ie = [r.V_exact for r in results], [r.I_exact for r in results]
    id_d, id_e = sortperm(Vd), sortperm(Ve)
    Vd, Id = Vd[id_d], Id[id_d]
    Ve, Ie = Ve[id_e], Ie[id_e]

    p = plot(; xlabel="Voltage V (cm^-1)", ylabel="Current I (cm^-1)",
              title="Current vs voltage: DAMPF vs Direct Lindblad", legend=:bottomleft)
    if reference !== nothing
        idr = sortperm(reference.V)
        plot!(p, reference.V[idr], reference.I[idr]; label="Direct Lindblad (dense sweep)", lw=1, color=:gray, alpha=0.6)
    end
    plot!(p, Ve, Ie; label="Direct Lindblad", lw=2, marker=:circle, color=:black)
    plot!(p, Vd, Id; label="DAMPF", lw=2, marker=:diamond, linestyle=:dash, color=:blue)

    savefig(p, joinpath(outdir, "current_vs_voltage_comparison.png"))
end

"""Power(V) comparison, single axis: DAMPF vs Direct Lindblad (each against
its own voltage). `reference`, if given, adds the dense-sweep curve as
background."""
function plot_power_vs_voltage_comparison(results, outdir; reference=nothing)
    mkpath(outdir)
    Vd, Pd = [r.V_dampf for r in results], [r.P_dampf for r in results]
    Ve, Pe = [r.V_exact for r in results], [r.P_exact for r in results]
    id_d, id_e = sortperm(Vd), sortperm(Ve)
    Vd, Pd = Vd[id_d], Pd[id_d]
    Ve, Pe = Ve[id_e], Pe[id_e]

    p = plot(; xlabel="Voltage V (cm^-1)", ylabel="Power P (cm^-2)",
              title="Power vs voltage: DAMPF vs Direct Lindblad", legend=:bottomleft)
    if reference !== nothing
        idr = sortperm(reference.V)
        plot!(p, reference.V[idr], reference.P[idr]; label="Direct Lindblad (dense sweep)", lw=1, color=:gray, alpha=0.6)
    end
    plot!(p, Ve, Pe; label="Direct Lindblad", lw=2, marker=:circle, color=:black)
    plot!(p, Vd, Pd; label="DAMPF", lw=2, marker=:diamond, linestyle=:dash, color=:red)

    savefig(p, joinpath(outdir, "power_vs_voltage_comparison.png"))
end

end
