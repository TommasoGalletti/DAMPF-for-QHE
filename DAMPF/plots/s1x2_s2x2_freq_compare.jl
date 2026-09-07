include("../utils/csv_io.jl")
using .DampfCsvIO
using Plots

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
Compares every `outputs/numeric_vectors/iv_curve_s1x2_s2x2_<ω2>.csv` sweep
(2 modes per site, one fixed at 200 cm^-1 and one at ω2 -- see
experiments/iv_curve_dampf_s1x2_s2x2.jl for ω2=200 itself and
iv_curve_dampf_s1x2_s2x2_freq_sweep.jl for every other ω2), one line per
ω2, colored by a continuous colormap so the ω2 ordering reads at a glance.

Auto-discovers datasets by globbing the numeric_vectors directory instead
of hardcoding the ω2 list or point count: rerunning this script after
adding more ΓL points to an existing ω2, or a brand-new ω2 file of the same
name pattern, picks them up with no code change. Each dataset keeps its own
ΓL grid (no interpolation/matching needed, just overlaid line+markers) --
they need not even share the same points, only the family-curve plots
implicitly assume enough overlap to compare visually.

ΓL axis is always log-scaled (`xaxis=:log10`) per the request that motivated
this file, plotting raw ΓL directly rather than log10(ΓL) (unlike the older
plots/*.jl siblings) so tick labels show physical units.
"""

const NUMERIC_DIR = joinpath(@__DIR__, "..", "outputs", "numeric_vectors")
const OUTDIR = joinpath(@__DIR__, "..", "outputs", "imgs", "s1x2_s2x2_freq_compare")
mkpath(OUTDIR)

"""Find every `iv_curve_s1x2_s2x2_<ω2>.csv` in `NUMERIC_DIR`, parse `<ω2>`
as a number, and return `(ω2=..., results=...)` tuples sorted by ω2."""
function discover_datasets()
    pattern = r"^iv_curve_s1x2_s2x2_([0-9]+(?:\.[0-9]+)?)\.csv$"
    found = NamedTuple[]
    for fname in readdir(NUMERIC_DIR)
        m = match(pattern, fname)
        m === nothing && continue
        ω2 = parse(Float64, m.captures[1])
        results = load_dampf_sweep_csv(joinpath(NUMERIC_DIR, fname))
        isempty(results) && continue
        push!(found, (ω2=ω2, results=results))
    end
    return sort(found; by=d -> d.ω2)
end

datasets = discover_datasets()
isempty(datasets) && error("No iv_curve_s1x2_s2x2_<ω2>.csv files found in $(NUMERIC_DIR)")
println("Found $(length(datasets)) ω2 datasets: $(join([d.ω2 for d in datasets], ", ")) cm^-1")

sorted_by(results, field) = results[sortperm([getfield(r, field) for r in results])]

# Continuous colormap over ω2, so color order == frequency order at a glance.
ω2_values = [d.ω2 for d in datasets]
ω2_min, ω2_max = extrema(ω2_values)
color_for(ω2) = ω2_max > ω2_min ? cgrad(:viridis)[(ω2 - ω2_min) / (ω2_max - ω2_min)] : cgrad(:viridis)[0.5]

# ----------------------------------------------------------------------------
# 1-3: I, V, P vs ΓL (log-x), one line per ω2.
# ----------------------------------------------------------------------------
p1 = plot(; xlabel="ΓL (cm^-1)", ylabel="Current I (cm^-1)", title="Corrente vs ΓL (al variare di ω2)",
          xaxis=:log10, legend=:outertopright)
p2 = plot(; xlabel="ΓL (cm^-1)", ylabel="Voltage V (cm^-1)", title="Voltaggio vs ΓL (al variare di ω2)",
          xaxis=:log10, legend=:outertopright)
p3 = plot(; xlabel="ΓL (cm^-1)", ylabel="Power P (cm^-2)", title="Potenza vs ΓL (al variare di ω2)",
          xaxis=:log10, legend=:outertopright)
for d in datasets
    r = sorted_by(d.results, :ΓL)
    ΓL = [x.ΓL for x in r]
    label = "ω2=$(d.ω2) cm^-1"
    color = color_for(d.ω2)
    plot!(p1, ΓL, [x.I_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
    plot!(p2, ΓL, [x.V_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
    plot!(p3, ΓL, [x.P_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
end
savefig(p1, joinpath(OUTDIR, "current_vs_gammaL_by_omega2.png"))
savefig(p2, joinpath(OUTDIR, "voltage_vs_gammaL_by_omega2.png"))
savefig(p3, joinpath(OUTDIR, "power_vs_gammaL_by_omega2.png"))

panel = plot(p1, p2, p3; layout=(1, 3), size=(1800, 500))
savefig(panel, joinpath(OUTDIR, "gammaL_panel_by_omega2.png"))

# ----------------------------------------------------------------------------
# 4: I(V) and P(V), one line per ω2 (points sorted by voltage so lines don't
# zigzag -- V decreases roughly monotonically with ΓL).
# ----------------------------------------------------------------------------
p4 = plot(; xlabel="Voltage V (cm^-1)", ylabel="Current I (cm^-1)", title="Corrente vs voltaggio (al variare di ω2)", legend=:outertopright)
p5 = plot(; xlabel="Voltage V (cm^-1)", ylabel="Power P (cm^-2)", title="Potenza vs voltaggio (al variare di ω2)", legend=:outertopright)
for d in datasets
    r = sorted_by(d.results, :V_dampf)
    V = [x.V_dampf for x in r]
    label = "ω2=$(d.ω2) cm^-1"
    color = color_for(d.ω2)
    plot!(p4, V, [x.I_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
    plot!(p5, V, [x.P_dampf for x in r]; label=label, lw=2, marker=:circle, markersize=3, color=color)
end
savefig(p4, joinpath(OUTDIR, "current_vs_voltage_by_omega2.png"))
savefig(p5, joinpath(OUTDIR, "power_vs_voltage_by_omega2.png"))

# ----------------------------------------------------------------------------
# 5: Summary trend -- P_max (over each dataset's own ΓL sweep) and I at a
# few representative fixed ΓL, both vs ω2. One point per dataset instead of
# one curve per dataset -- the systematic generalization of the by-hand
# ΓL-by-ΓL comparison table done earlier in this ω2 exploration.
# ----------------------------------------------------------------------------
REPRESENTATIVE_ΓL = [0.1, 100.0, 5000.0]


# The raw per-dataset maximum is noisy: P(ΓL) is very flat near its peak and
# no run reaches full convergence (converged=false everywhere at max_steps),
# so which single ΓL sample happens to be largest jitters from dataset to
# dataset. Averaging the 3 highest points per ω2 (and showing their spread
# as an error bar, plus the raw points in gray) is far more robust to that
# jitter than a single argmax while still tracking the real peak region.
top3_stats = map(datasets) do d
    P_sorted = sort([x.P_dampf for x in d.results]; rev=true)
    top3 = P_sorted[1:min(3, length(P_sorted))]
    return (mean=sum(top3) / length(top3), lo=minimum(top3), hi=maximum(top3), vals=top3)
end
Pmax_by_ω2 = [s.mean for s in top3_stats]
yerr = ([s.mean - s.lo for s in top3_stats], [s.hi - s.mean for s in top3_stats])

p6 = plot(; xlabel="ω2 (cm^-1)", ylabel="Power (cm^-2)",
          title="Potenza massima vs ω2 (media dei 3 punti piu alti)", legend=false)
for (d, s) in zip(datasets, top3_stats)
    scatter!(p6, fill(d.ω2, length(s.vals)), s.vals; color=:gray, markersize=3, alpha=0.6)
end
plot!(p6, ω2_values, Pmax_by_ω2; yerror=yerr, marker=:diamond, markersize=5, lw=2, color=:darkred)
savefig(p6, joinpath(OUTDIR, "Pmax_vs_omega2.png"))

# One subplot per ΓL (not one shared overlay): I spans orders of magnitude
# across the representative ΓL (~0.03 at ΓL=0.1 vs ~6.7 at ΓL=5000), so a
# single shared y-axis would flatten the low-ΓL curve to an invisible line
# at the bottom -- exactly the point being compared there.
subplots = Plots.Plot[]
for ΓL_target in REPRESENTATIVE_ΓL
    I_at_ΓL = Float64[]
    ω2_available = Float64[]
    for d in datasets
        match_idx = findfirst(x -> x.ΓL == ΓL_target, d.results)
        match_idx === nothing && continue
        push!(I_at_ΓL, d.results[match_idx].I_dampf)
        push!(ω2_available, d.ω2)
    end
    isempty(ω2_available) && continue
    sp = plot(ω2_available, I_at_ΓL; xlabel="ω2 (cm^-1)", ylabel="Current I (cm^-1)",
              title="ΓL=$(ΓL_target)", legend=false, lw=2, marker=:circle, color=:darkblue)
    push!(subplots, sp)
end
p7 = plot(subplots...; layout=(1, length(subplots)), size=(500 * length(subplots), 400),
          plot_title="Corrente a ΓL fisso vs ω2")
savefig(p7, joinpath(OUTDIR, "I_vs_omega2_fixed_gammaL.png"))

# ----------------------------------------------------------------------------
# 6: 3D surfaces over (ω2, ΓL) -- same approach as
# plots/omega_gammaL_3d_plots.jl: union (not intersection) of every ΓL seen
# in ANY dataset, so a (ω2,ΓL) combination one dataset hasn't reached yet
# shows as a gap (NaN) in that dataset's row rather than shrinking the whole
# grid down to the least-complete dataset. Rerunning after more ΓL points
# land (any ω2, any batch) fills in more of the grid automatically.
# ----------------------------------------------------------------------------
ΓLs_union = sort(unique(vcat([[x.ΓL for x in d.results] for d in datasets]...)))
logΓLs_union = log10.(ΓLs_union)
surf_lookup = Dict{Tuple{Float64,Float64},NamedTuple}()
for d in datasets, x in d.results
    surf_lookup[(d.ω2, x.ΓL)] = x
end

"""Zᵢⱼ = getfield(result, field) at (ω2=ω2_values[j], ΓL=ΓLs_union[i]); NaN
where that (ω2,ΓL) hasn't been computed yet -- matches Plots.jl's
`surface`'s z[iy, ix] convention."""
function grid_for_surface(field::Symbol)
    Z = fill(NaN, length(ΓLs_union), length(ω2_values))
    for (i, ΓL) in enumerate(ΓLs_union), (j, ω2) in enumerate(ω2_values)
        r = get(surf_lookup, (ω2, ΓL), nothing)
        r === nothing && continue
        Z[i, j] = getfield(r, field)
    end
    return Z
end

surface_specs = [
    (field=:P_dampf, zlabel="Power P (cm^-2)", title="Potenza", fname="power"),
    (field=:I_dampf, zlabel="Current I (cm^-1)", title="Corrente", fname="current"),
    (field=:V_dampf, zlabel="Voltage V (cm^-1)", title="Voltaggio", fname="voltage"),
]
for s in surface_specs
    Z = grid_for_surface(s.field)
    psurf = surface(ω2_values, logΓLs_union, Z; xlabel="ω2 (cm^-1)", ylabel="log10(ΓL / cm^-1)", zlabel=s.zlabel,
                     title="$(s.title) vs (ω2, ΓL)", camera=(50, 30), color=:viridis, size=(800, 650))
    savefig(psurf, joinpath(OUTDIR, "$(s.fname)_surface_3d_by_omega2.png"))
end
println("Saved 3D surfaces (grid: $(length(ω2_values)) ω2 x $(length(ΓLs_union)) ΓL, gaps where a dataset hasn't reached a given ΓL yet)")

println("Saved $(length(datasets))-dataset comparison plots in: $(OUTDIR)")
