include("../utils/csv_io.jl")
using .DampfCsvIO
using Plots

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Overlays DAMPF's own I/V/P results (never the "Lindblad diretto" reference
columns) from the three experiments on the same axes, so the effect of mode
topology alone -- 1 mode on s1/s2 only (killoran), 1 mode on every site
(v1v2v3), 2 modes each on s1 and s2 (s1x2_s2x2) -- is directly visible.
Each dataset keeps its own ΓL grid (no interpolation/matching needed, just
overlaid line+markers). Reads the three CSVs as they stand -- rerun the
relevant experiment first if you want a plot to reflect new points.
"""

killoran = load_comparison_csv(joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_comparison.csv"))
v1v2v3 = load_dampf_sweep_csv(joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_v1v2v3.csv"))
s1x2s2x2 = load_dampf_sweep_csv(joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "iv_curve_s1x2_s2x2_200.csv"))

datasets = [
    (results=killoran, label="Killoran (1 modo su s1,s2)", color=:black, marker=:circle),
    (results=v1v2v3, label="v1v2v3 (1 modo per sito, 3 modi)", color=:blue, marker=:diamond),
    (results=s1x2s2x2, label="s1x2_s2x2 (2 modi su s1, 2 su s2)", color=:red, marker=:utriangle),
]

outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "confronti")
mkpath(outdir)

sorted_by(results, xfield) = results[sortperm([getfield(r, xfield) for r in results])]

# --- I, V, P vs log10(ΓL) ---
p1 = plot(; xlabel="log10(ΓL / cm^-1)", ylabel="Current I (cm^-1)", title="Corrente vs ΓL", legend=:bottomright)
p2 = plot(; xlabel="log10(ΓL / cm^-1)", ylabel="Voltage V (cm^-1)", title="Voltaggio vs ΓL", legend=:bottomleft)
p3 = plot(; xlabel="log10(ΓL / cm^-1)", ylabel="Power P (cm^-2)", title="Potenza vs ΓL", legend=:bottomright)
for d in datasets
    r = sorted_by(d.results, :ΓL)
    logΓL = log10.([x.ΓL for x in r])
    plot!(p1, logΓL, [x.I_dampf for x in r]; label=d.label, lw=2, marker=d.marker, color=d.color)
    plot!(p2, logΓL, [x.V_dampf for x in r]; label=d.label, lw=2, marker=d.marker, color=d.color)
    plot!(p3, logΓL, [x.P_dampf for x in r]; label=d.label, lw=2, marker=d.marker, color=d.color)
end
savefig(p1, joinpath(outdir, "current_vs_gammaL_confronto.png"))
savefig(p2, joinpath(outdir, "voltage_vs_gammaL_confronto.png"))
savefig(p3, joinpath(outdir, "power_vs_gammaL_confronto.png"))

panel = plot(p1, p2, p3; layout=(1, 3), size=(1500, 450))
savefig(panel, joinpath(outdir, "gammaL_panel_confronto.png"))

# --- I(V), P(V): points sorted by voltage so lines don't zigzag ---
p4 = plot(; xlabel="Voltage V (cm^-1)", ylabel="Current I (cm^-1)", title="Corrente vs voltaggio", legend=:bottomleft)
p5 = plot(; xlabel="Voltage V (cm^-1)", ylabel="Power P (cm^-2)", title="Potenza vs voltaggio", legend=:bottomleft)
for d in datasets
    r = sorted_by(d.results, :V_dampf)
    V = [x.V_dampf for x in r]
    plot!(p4, V, [x.I_dampf for x in r]; label=d.label, lw=2, marker=d.marker, color=d.color)
    plot!(p5, V, [x.P_dampf for x in r]; label=d.label, lw=2, marker=d.marker, color=d.color)
end
savefig(p4, joinpath(outdir, "current_vs_voltage_confronto.png"))
savefig(p5, joinpath(outdir, "power_vs_voltage_confronto.png"))

println("Saved comparison plots in: $(outdir)")
