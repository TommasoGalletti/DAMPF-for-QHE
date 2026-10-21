include("../utils/csv_io.jl")
using .DampfCsvIO
using Plots

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Side-by-side cut of the three single-mode ω scans (benchmark model) that
differ only in what is held fixed while ω changes: g (original scan), the
reorganization energy λ=g²/ω, or the Huang-Rhys factor S=(g/ω)². Plots the
vibronic enhancement vs ω at a few representative ΓL, one panel per ΓL.
"""

nv = joinpath(@__DIR__, "..", "outputs", "numeric_vectors")
load_dir(d) = isdir(joinpath(nv, d)) ? reduce(vcat, [load_omega_gammaL_scan_csv(joinpath(nv, d, f)) for f in readdir(joinpath(nv, d)) if endswith(f, ".csv")]; init=NamedTuple[]) : NamedTuple[]

datasets = [
    (label="g fixed (g=55)", results=vcat(load_omega_gammaL_scan_csv(joinpath(nv, "omega_gammaL_scan_killoran.csv")), load_dir("omega_gammaL_scan_killoran_fixedg_ext")), color=:steelblue, marker=:circle),
    (label="λ fixed (g ∝ √ω)", results=load_dir("omega_gammaL_scan_killoran_reorg"), color=:darkred, marker=:diamond),
    (label="S fixed (g ∝ ω)", results=load_dir("omega_gammaL_scan_killoran_fixedS"), color=:darkgreen, marker=:utriangle),
]

ΓL_cuts = [30.0, 850.0]
panels = Plots.Plot[]
for ΓL in ΓL_cuts
    p = plot(; xlabel="ω (cm^-1)", ylabel="Enhancement vs g=0 (%)", title="ΓL = $(Int(ΓL)) cm^-1",
             legend=:topleft)
    vline!(p, [200.0]; color=:gray, ls=:dash, label="resonance")
    for d in datasets
        r = sort([x for x in d.results if x.ΓL == ΓL]; by=x -> x.ω)
        plot!(p, [x.ω for x in r], [x.enhancement_pct for x in r]; label=d.label, lw=2,
              marker=d.marker, markersize=5, color=d.color)
    end
    push!(panels, p)
end

outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "omega_scan_conventions")
mkpath(outdir)
fig = plot(panels...; layout=(1, length(panels)), size=(1500, 550), left_margin=6Plots.mm, bottom_margin=6Plots.mm)
savefig(fig, joinpath(outdir, "enhancement_vs_omega_g_lambda_S.png"))
println("Saved: $(joinpath(outdir, "enhancement_vs_omega_g_lambda_S.png"))")
