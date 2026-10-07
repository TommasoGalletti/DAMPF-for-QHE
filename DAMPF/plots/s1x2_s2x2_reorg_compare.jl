include("../utils/csv_io.jl")
using .DampfCsvIO
using Plots

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Two families of figures for the reorganization-energy-rescaled s1x2_s2x2
ω2 scan (see SPIEGAZIONI/SPIEGAZIONE_reorganization_energy_rescaling.md):

1. "Rescaled-only" figures, the exact analogue of the 4 figures already in
   the thesis for the unscaled scan (plots/s1x2_s2x2_freq_compare.jl) --
   same style, same layout, just fed with the rescaled datasets instead.
2. "vs unscaled" comparison figures, isolating what the rescaling changed:
   P vs ω2 for both (the headline result -- unscaled decreases, rescaled
   flattens/reverses), the per-ω2 % difference, and a small-multiples panel
   overlaying both P(ΓL) curves per ω2 on their common window.

Rescaled datasets only cover ΓL ∈ {500,600,700,800,900,1000,1200} (the
peak region, by design -- see the explanation doc); the unscaled side is
filtered to the SAME ΓL points for every comparison, so no curve shape
differences here are an artifact of different sampling grids.
"""

const NUMERIC_UNSCALED = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "s1x2_s2x2")
const NUMERIC_RESCALED = joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "s1x2_s2x2_reorg")
const OUTDIR_RESCALED = joinpath(@__DIR__, "..", "outputs", "imgs", "s1x2_s2x2_reorg", "s1x2_s2x2_reorg_compare")
const OUTDIR_VS = joinpath(@__DIR__, "..", "outputs", "imgs", "s1x2_s2x2_reorg", "s1x2_s2x2_reorg_vs_unscaled")
mkpath(OUTDIR_RESCALED)
mkpath(OUTDIR_VS)

"""Find every `iv_curve_s1x2_s2x2_reorg_<ω2>.csv` in NUMERIC_RESCALED, parse
`<ω2>`, pair it with the matching unscaled dataset (same ω2, filtered down
to only the ΓL points the rescaled run has), and return sorted-by-ω2
`(ω2=..., rescaled=..., unscaled=...)` tuples."""
function discover_paired_datasets()
    pattern = r"^iv_curve_s1x2_s2x2_reorg_([0-9]+(?:\.[0-9]+)?)\.csv$"
    found = NamedTuple[]
    for fname in readdir(NUMERIC_RESCALED)
        m = match(pattern, fname)
        m === nothing && continue
        ω2 = parse(Float64, m.captures[1])
        rescaled = load_dampf_sweep_csv(joinpath(NUMERIC_RESCALED, fname))
        isempty(rescaled) && continue
        ω2_tag = string(Int(round(ω2)))
        unscaled_path = joinpath(NUMERIC_UNSCALED, "iv_curve_s1x2_s2x2_$(ω2_tag).csv")
        unscaled_all = load_dampf_sweep_csv(unscaled_path)
        rescaled_ΓLs = Set(x.ΓL for x in rescaled)
        unscaled = filter(x -> x.ΓL in rescaled_ΓLs, unscaled_all)
        push!(found, (ω2=ω2, rescaled=rescaled, unscaled=unscaled))
    end
    return sort(found; by=d -> d.ω2)
end

datasets = discover_paired_datasets()
isempty(datasets) && error("No iv_curve_s1x2_s2x2_reorg_<ω2>.csv files found in $(NUMERIC_RESCALED)")
println("Found $(length(datasets)) rescaled ω2 datasets: $(join([d.ω2 for d in datasets], ", ")) cm^-1")

sorted_by(results, field) = results[sortperm([getfield(r, field) for r in results])]

ω2_values = [d.ω2 for d in datasets]
ω2_min, ω2_max = extrema(ω2_values)
color_for(ω2) = ω2_max > ω2_min ? cgrad(:viridis)[(ω2 - ω2_min) / (ω2_max - ω2_min)] : cgrad(:viridis)[0.5]

top3_mean(results) = begin
    P_sorted = sort([x.P_dampf for x in results]; rev=true)
    top3 = P_sorted[1:min(3, length(P_sorted))]
    (mean=sum(top3) / length(top3), lo=minimum(top3), hi=maximum(top3))
end

# ==============================================================================
# PART 1 -- rescaled-only figures (same 4 as the thesis's unscaled set)
# ==============================================================================

p1 = plot(; xlabel="ΓL (cm^-1)", ylabel="Current I (cm^-1)", title="Corrente vs ΓL (riscalato, al variare di ω2)",
          xaxis=:log10, legend=:outertopright)
p2 = plot(; xlabel="ΓL (cm^-1)", ylabel="Voltage V (cm^-1)", title="Voltaggio vs ΓL (riscalato, al variare di ω2)",
          xaxis=:log10, legend=:outertopright)
p3 = plot(; xlabel="ΓL (cm^-1)", ylabel="Power P (cm^-2)", title="Potenza vs ΓL (riscalato, al variare di ω2)",
          xaxis=:log10, legend=:outertopright)
for d in datasets
    r = sorted_by(d.rescaled, :ΓL)
    ΓL = [x.ΓL for x in r]
    label = "ω2=$(d.ω2) cm^-1"
    color = color_for(d.ω2)
    plot!(p1, ΓL, [x.I_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
    plot!(p2, ΓL, [x.V_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
    plot!(p3, ΓL, [x.P_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
end
panel = plot(p1, p2, p3; layout=(1, 3), size=(1800, 500))
savefig(panel, joinpath(OUTDIR_RESCALED, "gammaL_panel_rescaled_by_omega2.png"))

p4 = plot(; xlabel="Voltage V (cm^-1)", ylabel="Current I (cm^-1)", title="Corrente vs voltaggio (riscalato, al variare di ω2)", legend=:outertopright)
p5 = plot(; xlabel="Voltage V (cm^-1)", ylabel="Power P (cm^-2)", title="Potenza vs voltaggio (riscalato, al variare di ω2)", legend=:outertopright)
for d in datasets
    r = sorted_by(d.rescaled, :V_dampf)
    V = [x.V_dampf for x in r]
    label = "ω2=$(d.ω2) cm^-1"
    color = color_for(d.ω2)
    plot!(p4, V, [x.I_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
    plot!(p5, V, [x.P_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
end
savefig(p4, joinpath(OUTDIR_RESCALED, "current_vs_voltage_rescaled_by_omega2.png"))
savefig(p5, joinpath(OUTDIR_RESCALED, "power_vs_voltage_rescaled_by_omega2.png"))

rescaled_top3 = [top3_mean(d.rescaled) for d in datasets]
Pmax_rescaled = [s.mean for s in rescaled_top3]
yerr_rescaled = ([s.mean - s.lo for s in rescaled_top3], [s.hi - s.mean for s in rescaled_top3])

p6 = plot(; xlabel="ω2 (cm^-1)", ylabel="Power (cm^-2)",
          title="Potenza massima vs ω2 (riscalato, media dei 3 punti piu alti)", legend=false)
for (d, s) in zip(datasets, rescaled_top3)
    P_sorted = sort([x.P_dampf for x in d.rescaled]; rev=true)
    top3 = P_sorted[1:min(3, length(P_sorted))]
    scatter!(p6, fill(d.ω2, length(top3)), top3; color=:gray, markersize=3, alpha=0.6)
end
plot!(p6, ω2_values, Pmax_rescaled; yerror=yerr_rescaled, marker=:diamond, markersize=5, lw=2, color=:darkred)
savefig(p6, joinpath(OUTDIR_RESCALED, "Pmax_vs_omega2_rescaled.png"))

println("Saved $(length(datasets)) rescaled-only comparison plots in: $(OUTDIR_RESCALED)")

# ==============================================================================
# PART 2 -- rescaled vs unscaled: what the rescaling actually changed
# ==============================================================================

unscaled_top3 = [top3_mean(d.unscaled) for d in datasets]
Pmax_unscaled = [s.mean for s in unscaled_top3]
yerr_unscaled = ([s.mean - s.lo for s in unscaled_top3], [s.hi - s.mean for s in unscaled_top3])

# 1: the headline figure -- both P vs ω2 trends on the same axes.
p_head = plot(; xlabel="ω2 (cm^-1)", ylabel="Power (cm^-2)",
              title="Potenza massima vs ω2: non scalato vs riscalato", legend=:outertopright)
plot!(p_head, ω2_values, Pmax_unscaled; yerror=yerr_unscaled, label="non scalato (g=55 fisso)",
      marker=:circle, markersize=5, lw=2, color=:steelblue)
plot!(p_head, ω2_values, Pmax_rescaled; yerror=yerr_rescaled, label="riscalato (lambda fissa)",
      marker=:diamond, markersize=5, lw=2, color=:darkred)
savefig(p_head, joinpath(OUTDIR_VS, "Pmax_vs_omega2_rescaled_vs_unscaled.png"))

# 2: per-ω2 % difference (rescaled vs unscaled), point-by-point mean -- the
# number quoted in the explanation doc / summary message, as a figure.
ΔP_pct = [100 * (Pmax_rescaled[i] - Pmax_unscaled[i]) / Pmax_unscaled[i] for i in eachindex(datasets)]
p_delta = plot(ω2_values, ΔP_pct; xlabel="ω2 (cm^-1)", ylabel="ΔP (%)",
               title="Variazione di potenza dovuta al riscalamento", legend=false,
               marker=:diamond, markersize=5, lw=2, color=:darkorange)
hline!(p_delta, [0.0]; color=:gray, ls=:dash, label=false)
savefig(p_delta, joinpath(OUTDIR_VS, "deltaP_pct_vs_omega2.png"))

# 3: small multiples, one subplot per ω2, both P(ΓL) curves overlaid on
# their shared ΓL window -- shows directly that rescaling shifts the whole
# curve down by a roughly constant amount without changing its shape.
subplots = Plots.Plot[]
for d in datasets
    ru = sorted_by(d.unscaled, :ΓL)
    rr = sorted_by(d.rescaled, :ΓL)
    sp = plot(; xlabel="ΓL (cm^-1)", ylabel="Power P (cm^-2)", title="ω2=$(d.ω2)", legend=false)
    plot!(sp, [x.ΓL for x in ru], [x.P_dampf for x in ru]; lw=2, marker=:circle, markersize=3, color=:steelblue)
    plot!(sp, [x.ΓL for x in rr], [x.P_dampf for x in rr]; lw=2, marker=:diamond, markersize=3, color=:darkred)
    push!(subplots, sp)
end
p_panel = plot(subplots...; layout=(3, 3), size=(1500, 1300),
               plot_title="Potenza vs ΓL per ω2: non scalato (blu) vs riscalato (rosso)")
savefig(p_panel, joinpath(OUTDIR_VS, "power_vs_gammaL_rescaled_vs_unscaled_panel.png"))

println("Saved $(length(datasets))-dataset rescaled-vs-unscaled comparison plots in: $(OUTDIR_VS)")
