include("../utils/csv_io.jl")
using .DampfCsvIO
using Plots

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Visualizes Power against V from experiments/omega_gammaL_scan_killoran.jl's
output as a family of P(V) curves, one per ω (color-graded) -- the classic
power-vs-voltage view already used elsewhere (plot_iv_dampf,
plot_power_vs_voltage_comparison) but now overlaid across ω instead of
across models, showing how the whole P-V curve shifts/rescales as ω is
detuned from resonance.
"""

results = load_omega_gammaL_scan_csv(joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "omega_gammaL_scan_killoran.csv"))
isempty(results) && error("No data in omega_gammaL_scan_killoran.csv -- run experiments/omega_gammaL_scan_killoran.jl first.")

outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "omega_gammaL_scan")
mkpath(outdir)

# --- Family of P(V) curves, one line per ω ---
ωs = sort(unique([r.ω for r in results]))
palette_colors = cgrad(:viridis, length(ωs); categorical=true)

ΓL_min, ΓL_max = extrema([r.ΓL for r in results])
pfam = plot(; xlabel="Voltage V (cm^-1)", ylabel="Power P (cm^-2)",
            title="Power vs voltage", subtitle="ΓL ∈ [$(ΓL_min), $(ΓL_max)] cm^-1",
            legend=:outertopright, size=(900, 600))
for (k, ω) in enumerate(ωs)
    rω = [r for r in results if r.ω == ω]
    idx = sortperm([r.V_dampf for r in rω])
    V = [rω[i].V_dampf for i in idx]
    P = [rω[i].P_dampf for i in idx]
    plot!(pfam, V, P; label="ω=$(ω)", lw=2, marker=:circle, markersize=2, color=palette_colors[k])
end
savefig(pfam, joinpath(outdir, "power_vs_V_family_by_omega.png"))

println("Saved: power_vs_V_family_by_omega.png in $(outdir)")
