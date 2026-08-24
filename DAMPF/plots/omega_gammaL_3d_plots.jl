include("../utils/csv_io.jl")
using .DampfCsvIO
using Plots

default(fontfamily="Computer Modern", framestyle=:box, dpi=300)

"""
3D analysis of experiments/omega_gammaL_scan_killoran.jl's output: one
surface per observable (I, V, P, vibronic enhancement %), with ω and ΓL as
the two horizontal axes. Missing (ω,ΓL) combinations (e.g. if the scan grid
isn't a perfect rectangle -- more ω or ΓL points added later than on the
other axis) show as gaps (NaN) rather than erroring. Rerun this after
adding points to the CSV to refresh the plots.
"""

results = load_omega_gammaL_scan_csv(joinpath(@__DIR__, "..", "outputs", "numeric_vectors", "omega_gammaL_scan_killoran.csv"))
isempty(results) && error("No data in omega_gammaL_scan_killoran.csv -- run experiments/omega_gammaL_scan_killoran.jl first.")

ωs = sort(unique([r.ω for r in results]))
ΓLs = sort(unique([r.ΓL for r in results]))
lookup = Dict((r.ω, r.ΓL) => r for r in results)

"""Zᵢⱼ = getfield(result, field) at (ω=ωs[j], ΓL=ΓLs[i]); NaN where missing
-- matches Plots.jl's `surface`/`contourf` convention of z[iy, ix]."""
function grid_for(field::Symbol)
    Z = fill(NaN, length(ΓLs), length(ωs))
    for (i, ΓL) in enumerate(ΓLs), (j, ω) in enumerate(ωs)
        r = get(lookup, (ω, ΓL), nothing)
        r === nothing && continue
        Z[i, j] = getfield(r, field)
    end
    return Z
end

outdir = joinpath(@__DIR__, "..", "outputs", "imgs", "omega_gammaL_scan")
mkpath(outdir)

specs = [
    (field=:I_dampf, zlabel="Current I (cm^-1)", title="Corrente", fname="current"),
    (field=:V_dampf, zlabel="Voltage V (cm^-1)", title="Voltaggio", fname="voltage"),
    (field=:P_dampf, zlabel="Power P (cm^-2)", title="Potenza", fname="power"),
    (field=:enhancement_pct, zlabel="Enhancement vs g=0 (%)", title="Vibronic enhancement", fname="enhancement"),
]

for s in specs
    Z = grid_for(s.field)

    psurf = surface(ωs, ΓLs, Z; xlabel="ω (cm^-1)", ylabel="ΓL (cm^-1)", zlabel=s.zlabel,
                     title="$(s.title) vs (ω, ΓL)", camera=(50, 30), color=:viridis, size=(800, 650))
    savefig(psurf, joinpath(outdir, "$(s.fname)_surface_3d.png"))
end

println("Saved 3D surface plots (", length(specs), " observables) in: $(outdir)")
println("Grid: $(length(ωs)) ω values x $(length(ΓLs)) ΓL values = $(length(results)) points loaded")
