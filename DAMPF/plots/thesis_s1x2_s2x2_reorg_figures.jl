include("../utils/csv_io.jl")
using .DampfCsvIO
using Plots

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Thesis figures for the reorganization-energy-rescaled ω2 scan (Chapter 7,
Section "Detuning one mode in the two-modes-per-site configuration", and the
reorganization-energy appendix). Written straight into
thesis_latex/figures/s1x2_s2x2_omega2_scan_rescaled/.

Rescaled data: the checkpoint-averaged dataset (numeric_vectors/s1x2_s2x2_reorg_avg5/).
Unscaled data: numeric_vectors/s1x2_s2x2/, restricted to the same ΓL window
(500-1200 cm^-1) so both P* estimates come from the same region.
Single-mode benchmark: DAMPF column of numeric_vectors/iv_curve_comparison.csv.
P* is the mean of the three highest-power points, as in the rest of Chapter 7.

Titles and legends in English, like the rest of the thesis figures.
"""

const NV = joinpath(@__DIR__, "..", "outputs", "numeric_vectors")
const OUTDIR = joinpath(@__DIR__, "..", "..", "thesis_latex", "figures", "s1x2_s2x2_omega2_scan_rescaled")
mkpath(OUTDIR)

const ω2_LIST = [100, 150, 200, 250, 300, 350, 400, 450, 500]
const ω_RES = 200.0
const ΓL_WINDOW = (500.0, 1200.0)

top3(results) = begin
    P = sort([r.P_dampf for r in results]; rev=true)[1:3]
    (mean=sum(P) / 3, lo=minimum(P), hi=maximum(P))
end
sorted_by_ΓL(results) = results[sortperm([r.ΓL for r in results])]
in_window(r) = ΓL_WINDOW[1] <= r.ΓL <= ΓL_WINDOW[2]

rescaled = Dict(w => load_dampf_sweep_csv(joinpath(NV, "s1x2_s2x2_reorg_avg5", "iv_curve_s1x2_s2x2_reorg_avg5_$(w).csv")) for w in ω2_LIST)
unscaled = Dict(w => filter(in_window, load_dampf_sweep_csv(joinpath(NV, "s1x2_s2x2", "iv_curve_s1x2_s2x2_$(w).csv"))) for w in ω2_LIST)
bench = load_comparison_csv(joinpath(NV, "iv_curve_comparison.csv"))
P_bench = maximum(r.P_dampf for r in bench)

stats_r = [top3(rescaled[w]) for w in ω2_LIST]
stats_u = [top3(unscaled[w]) for w in ω2_LIST]
λ_ratio = [1 + ω_RES / w for w in ω2_LIST]

println("Benchmark P* = $(round(P_bench, digits=1))")
println("ω2   λ/λ0   P*_unscaled (vs bench %)   P*_rescaled (vs bench %)")
for (i, w) in enumerate(ω2_LIST)
    pu, pr = stats_u[i].mean, stats_r[i].mean
    println("$(w)  $(round(λ_ratio[i], digits=2))   $(round(pu, digits=1)) ($(round(100 * (pu - P_bench) / P_bench, digits=2))%)   " *
            "$(round(pr, digits=1)) ($(round(100 * (pr - P_bench) / P_bench, digits=2))%)")
end

color_for(w) = cgrad(:viridis)[(w - first(ω2_LIST)) / (last(ω2_LIST) - first(ω2_LIST))]

# 1. P(ΓL) of the rescaled scan, one curve per ω2 (linear ΓL: the window is narrow).
p1 = plot(; xlabel="ΓL (cm^-1)", ylabel="Power P (cm^-2)", title="Power vs ΓL at fixed λ",
          legend=:outertopright, size=(1000, 600), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
for w in ω2_LIST
    r = sorted_by_ΓL(rescaled[w])
    plot!(p1, [x.ΓL for x in r], [x.P_dampf for x in r]; label="ω2=$(w)", lw=2, marker=:circle, markersize=2.5,
          color=color_for(w))
end
hline!(p1, [P_bench]; color=:black, ls=:dash, lw=1.5, label="benchmark")
savefig(p1, joinpath(OUTDIR, "power_vs_gammaL_rescaled_by_omega2.png"))

# 2. P* vs ω2: unscaled vs rescaled, single-mode benchmark as reference.
err(s) = ([x.mean - x.lo for x in s], [x.hi - x.mean for x in s])
p2 = plot(; xlabel="ω2 (cm^-1)", ylabel="Power (cm^-2)", title="Maximum power vs ω2",
          legend=:outertopright, size=(1000, 600), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
plot!(p2, ω2_LIST, [s.mean for s in stats_u]; yerror=err(stats_u), label="g=55 (unscaled)",
      marker=:circle, markersize=5, lw=2, color=:steelblue)
plot!(p2, ω2_LIST, [s.mean for s in stats_r]; yerror=err(stats_r), label="λ=λ0 (rescaled)",
      marker=:diamond, markersize=5, lw=2, color=:darkred)
hline!(p2, [P_bench]; color=:black, ls=:dash, lw=1.5, label="benchmark (single mode)")
savefig(p2, joinpath(OUTDIR, "Pmax_vs_omega2_rescaled_vs_unscaled.png"))

# 3. (appendix) relative P* vs total reorganization energy: unscaled points
# lie at λ/λ0 = 1 + ω_res/ω2, every rescaled point at λ/λ0 = 1.
rel(p) = 100 * (p - P_bench) / P_bench
p3 = plot(; xlabel="λ / λ0", ylabel="(P* - P*_bench) / P*_bench  (%)", title="Maximum power vs reorganization energy",
          legend=:topleft, size=(1000, 650), left_margin=5Plots.mm, bottom_margin=5Plots.mm,
          xlims=(0.85, 3.2), ylims=(-1.3, 4.9))
hline!(p3, [0.0]; color=:gray, ls=:dash, lw=1, label=false)
scatter!(p3, λ_ratio, [rel(s.mean) for s in stats_u]; label="g=55 (unscaled)", marker=:circle, markersize=5,
         color=:steelblue)
# Hand-placed label offsets: the ω2=350..500 points sit within ~0.2 of each other in λ/λ0.
label_offset = Dict(500 => (0.03, -0.15, :left), 450 => (-0.03, 0.0, :right), 400 => (0.03, -0.15, :left),
                    350 => (0.03, 0.15, :left), 300 => (0.03, 0.15, :left), 250 => (0.03, 0.15, :left),
                    200 => (0.03, 0.15, :left), 150 => (0.03, 0.15, :left), 100 => (-0.03, 0.0, :right))
for (i, w) in enumerate(ω2_LIST)
    dx, dy, halign = label_offset[w]
    annotate!(p3, λ_ratio[i] + dx, rel(stats_u[i].mean) + dy, text("ω2=$(w)", 9, halign, :steelblue, "Computer Modern"))
end
scatter!(p3, fill(1.0, length(ω2_LIST)), [rel(s.mean) for s in stats_r]; label="λ=λ0 (rescaled, all ω2)",
         marker=:diamond, markersize=5, color=:darkred)
scatter!(p3, [1.0], [0.0]; label="benchmark (single mode)", marker=:star5, markersize=9, color=:black)
savefig(p3, joinpath(OUTDIR, "Pmax_rel_vs_reorganization_energy.png"))

println("Saved thesis figures in: $(OUTDIR)")
