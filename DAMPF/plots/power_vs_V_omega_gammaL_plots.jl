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

# Optional argument: `reorg` plots the fixed-reorganization-energy scan, `S`
# the fixed-Huang-Rhys scan (experiments/omega_gammaL_scan_killoran_reorg.jl,
# one CSV per ω); no argument plots the original fixed-g scan.
const VARIANT = isempty(ARGS) ? "" : ARGS[1]
VARIANT in ("", "reorg", "S") || error("Argument must be `reorg` or `S`")
nv_dir = joinpath(@__DIR__, "..", "outputs", "numeric_vectors")
load_csv_dir(d) = isdir(d) ? reduce(vcat, [load_omega_gammaL_scan_csv(joinpath(d, f)) for f in readdir(d) if endswith(f, ".csv")]; init=NamedTuple[]) : NamedTuple[]
results = if VARIANT == ""
    # original single-CSV scan + any later ω extension (one CSV per ω)
    vcat(load_omega_gammaL_scan_csv(joinpath(nv_dir, "omega_gammaL_scan_killoran.csv")),
         load_csv_dir(joinpath(nv_dir, "omega_gammaL_scan_killoran_fixedg_ext")))
else
    vdir = joinpath(nv_dir, VARIANT == "reorg" ? "omega_gammaL_scan_killoran_reorg" : "omega_gammaL_scan_killoran_fixedS")
    reduce(vcat, [load_omega_gammaL_scan_csv(joinpath(vdir, f)) for f in readdir(vdir) if endswith(f, ".csv")])
end
isempty(results) && error("No omega-GammaL scan data found -- run the corresponding experiment first.")
title_suffix = Dict("" => "", "reorg" => " (fixed λ)", "S" => " (fixed S)")[VARIANT]
img_subdir = Dict("" => "omega_gammaL_scan", "reorg" => "omega_gammaL_scan_reorg", "S" => "omega_gammaL_scan_fixedS")[VARIANT]

outdir = joinpath(@__DIR__, "..", "outputs", "imgs", img_subdir)
mkpath(outdir)

# --- Family of P(V) curves, one line per ω ---
ωs = sort(unique([r.ω for r in results]))
palette_colors = cgrad(:viridis, length(ωs); categorical=true)

ΓL_min, ΓL_max = extrema([r.ΓL for r in results])
pfam = plot(; xlabel="Voltage V (cm^-1)", ylabel="Power P (cm^-2)",
            title="Power vs voltage$(title_suffix)", subtitle="ΓL ∈ [$(ΓL_min), $(ΓL_max)] cm^-1",
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
