include("../utils/csv_io.jl")
include("thesis_style.jl")
using .DampfCsvIO
using .ThesisStyle
using Plots, LaTeXStrings

"""
Thesis figures of Chapters 6-7 and of the reorganization-energy appendix
(DAMPF), in the common style of thesis_style.jl. Written straight into
thesis_latex/figures/. Reads only saved CSVs from outputs/numeric_vectors/;
rerun the corresponding experiment first if a figure must include new points.
"""

theme!()
const FIG = joinpath(@__DIR__, "..", "..", "thesis_latex", "figures")
const NV = joinpath(@__DIR__, "..", "outputs", "numeric_vectors")
const P_SCALE = 1e4                  # powers plotted in units of 10^4 cm^-2
const ΓL_DECADES = (-1, 5)           # Γ_L range of the DAMPF sweeps (0.1 to 3x10^4 cm^-1)
const MARGINS = (left_margin=3mm, bottom_margin=4mm)

load_dir(d) = reduce(vcat, [load_omega_gammaL_scan_csv(joinpath(d, f)) for f in readdir(d) if endswith(f, ".csv")])

# ------------------------------------------------------------
# Phase 1 data (chi_max = 8, plateau average over steps 7000-12000; SPIEGAZIONI/00_PIANO_rigenerazione_chi.md)
# ------------------------------------------------------------
const P1 = joinpath(NV, "phase1")
"""Plateau-averaged sweep of one Phase 1 configuration, e.g. phase1_sweep("bench_g", 200)."""
phase1_sweep(cfg, param) = sorted_by(load_dampf_sweep_csv(joinpath(P1, cfg, "iv_$(cfg)_chi8_$(param).csv")), r -> r.ΓL)

"""Direct references on the Phase 1 points (experiments/phase1_exact.jl): (config, param, ΓL) => (I, V, P);
config "electronic" is the g = 0 baseline."""
const EXACT = Dict(map(readlines(joinpath(@__DIR__, "..", "..", "outputs", "numeric_vectors", "phase1_exact.csv"))[2:end]) do l
    c = split(l, ",")
    (String(c[1]), parse(Float64, c[2]), parse(Float64, c[3])) => (I=parse(Float64, c[4]), V=parse(Float64, c[5]), P=parse(Float64, c[6]))
end)

"""Trace distance to the direct steady state, averaged over the plateau (experiments/bench_tracedist_point.jl)."""
function plateau_trace_distance()
    d = joinpath(NV, "phase1_tracedist")
    Dict(map(filter(f -> startswith(f, "ts_tracedist_"), readdir(d))) do f
        ΓL = parse(Float64, replace(split(f, "_")[3], "p" => "."))
        D = [parse(Float64, split(l, ",")[6]) for l in readlines(joinpath(d, f))[2:end] if parse(Int, split(l, ",")[1]) >= 7000]
        ΓL => sum(D) / length(D)
    end)
end

"""Single-mode ω × Γ_L scan at chi_max = 8, with the enhancement over the exact g = 0 baseline
(scan_omega_gammaL rows: ω, ΓL, I_dampf, V_dampf, P_dampf, enhancement_pct). ω = 200 at fixed λ is the benchmark."""
function phase1_scan(cfg)
    files = filter(f -> startswith(f, "iv_$(cfg)_chi8_"), readdir(joinpath(P1, cfg)))
    ωs = [parse(Float64, split(f[1:end-4], "_")[end]) for f in files]
    cfg == "bench_lambda" && push!(ωs, 200.0)
    ΓLs = Set(r.ΓL for r in phase1_sweep(cfg, Int(first(filter(!=(200.0), ωs)))))   # scan grid (ω = 200 has more points)
    [(ω=ω, ΓL=r.ΓL, I_dampf=r.I_dampf, V_dampf=r.V_dampf, P_dampf=r.P_dampf,
      enhancement_pct=100 * (r.P_dampf / EXACT[("electronic", 0.0, r.ΓL)].P - 1))
     for ω in ωs for r in phase1_sweep(cfg == "bench_lambda" && ω == 200.0 ? "bench_g" : cfg, Int(ω)) if r.ΓL in ΓLs]
end

"""Dense direct-Lindblad sweep of the benchmark (experiments/IV_curve_vib.jl, N_v=6, g=55)."""
function load_dense_reference()
    path = joinpath(@__DIR__, "..", "..", "outputs", "numeric_vectors", "vibrational", "full", "6_modes", "g_55",
                    "observables_vs_gammaL.csv")
    rows = map(readlines(path)[2:end]) do line
        v = parse.(Float64, split(line, ","))
        (ΓL=v[1], I=v[3], V=v[4], P=v[5])
    end
    filter(r -> 10.0^ΓL_DECADES[1] <= r.ΓL <= 10.0^ΓL_DECADES[2], rows)
end

# ============================================================
# Chapter 6: DAMPF vs direct Lindblad (benchmark)
# ============================================================
bench = let D = plateau_trace_distance(), dampf = Dict(r.ΓL => r for r in phase1_sweep("bench_g", 200))
    sorted_by([(ΓL=G, I_dampf=dampf[G].I_dampf, V_dampf=dampf[G].V_dampf, P_dampf=dampf[G].P_dampf,
                I_exact=EXACT[("bench_g", 200.0, G)].I, V_exact=EXACT[("bench_g", 200.0, G)].V, P_exact=EXACT[("bench_g", 200.0, G)].P,
                trace_dist=D[G]) for G in keys(D)], r -> r.ΓL)
end
dense = load_dense_reference()
ΓLb = [r.ΓL for r in bench]

function compare_panel(label, f_exact, f_dampf, f_dense, ylab; legend=false)
    p = panel(label; xlabel=LBL.ΓL, ylabel=ylab, legend=legend)
    plot!(p, [r.ΓL for r in dense], f_dense.(dense); color=:grey60, lw=1, label="Direct Lindblad, dense sweep")
    for (f, fP, st) in ((f_exact, r -> r.P_exact, MODEL.direct), (f_dampf, r -> r.P_dampf, MODEL.dampf))
        plot!(p, ΓLb, f.(bench); color=st.color, ls=st.ls, marker=st.marker, label=st.label)
        k = argmax(fP.(bench))
        mark_max!(p, ΓLb[k], f(bench[k]); color=st.color)
    end
    logx!(p, ΓL_DECADES...; step=2)
end
fig = plot(
    compare_panel("(a)", r -> r.I_exact, r -> r.I_dampf, r -> r.I, LBL.I; legend=:bottomright),
    compare_panel("(b)", r -> r.V_exact, r -> r.V_dampf, r -> r.V, LBL.V),
    compare_panel("(c)", r -> r.P_exact / P_SCALE, r -> r.P_dampf / P_SCALE, r -> r.P / P_SCALE, LBL.P),
    let p = panel("(d)"; xlabel=LBL.ΓL, ylabel=LBL.trace, legend=false)
        plot!(p, ΓLb, [r.trace_dist for r in bench]; color=MODEL.dampf.color, marker=MODEL.dampf.marker)
        logx!(p, ΓL_DECADES...; step=2)
    end;
    layout=(2, 2), size=SIZE.grid22, MARGINS...)
save_figure(fig, joinpath(FIG, "dampf", "iv_curve_comparison_panel.png"))

"""(a) I(V) and (b) P(V) for several datasets, each a (rows, V, I, P, style) tuple."""
function iv_pv_panels(sets; legend=:bottomleft)
    pa = panel("(a)"; xlabel=LBL.V, ylabel=LBL.I, legend=legend)
    pb = panel("(b)"; xlabel=LBL.V, ylabel=LBL.P, legend=false)
    for s in sets
        rows = sorted_by(s.rows, s.V)
        V, I, P = s.V.(rows), s.I.(rows), s.P.(rows) ./ P_SCALE
        plot!(pa, V, I; color=s.style.color, ls=s.style.ls, marker=s.style.marker, label=s.style.label)
        plot!(pb, V, P; color=s.style.color, ls=s.style.ls, marker=s.style.marker, label=s.style.label)
        k = argmax(P)
        mark_max!(pa, V[k], I[k]; color=s.style.color)
        mark_max!(pb, V[k], P[k]; color=s.style.color)
    end
    plot(pa, pb; layout=(1, 2), size=SIZE.row2, MARGINS...)
end
save_figure(iv_pv_panels([(rows=bench, V=r -> r.V_exact, I=r -> r.I_exact, P=r -> r.P_exact, style=MODEL.direct),
                          (rows=bench, V=r -> r.V_dampf, I=r -> r.I_dampf, P=r -> r.P_dampf, style=MODEL.dampf)]),
            joinpath(FIG, "dampf", "iv_pv_comparison.png"))

# ============================================================
# Chapter 7: mode configurations
# ============================================================
configs = [
    (rows=sorted_by([(ΓL=r.ΓL, I=r.I_dampf, V=r.V_dampf, P=r.P_dampf) for r in bench], r -> r.ΓL), style=MODEL.benchmark),
    (rows=phase1_sweep("v1v2v3", 200), style=MODEL.one_per_site),
    (rows=phase1_sweep("s2m_g", 200), style=MODEL.two_per_site),
]
getI(r) = hasproperty(r, :I_dampf) ? r.I_dampf : r.I
getV(r) = hasproperty(r, :V_dampf) ? r.V_dampf : r.V
getP(r) = hasproperty(r, :P_dampf) ? r.P_dampf : r.P

"""xlims/ylims of a maximum-power zoom: points within `depth` (plot units) of the highest maximum."""
function zoom_limits(Vs, Ps, depth)
    top = maximum(maximum.(Ps))
    V = vcat([v[p .>= top - depth] for (v, p) in zip(Vs, Ps)]...)
    pad = 0.04 * (maximum(V) - minimum(V))
    (xlims=(minimum(V) - pad, maximum(V) + pad), ylims=(top - depth, top + 0.06 * depth))
end

"""(a) I, (b) V, (c) P vs Γ_L for one or more configurations; the legend goes in
panel (c), whose lower-right corner is always empty."""
function ivp_vs_gammaL(sets; legend=false)
    panels = map(zip(("(a)", "(b)", "(c)"), (getI, getV, r -> getP(r) / P_SCALE), (LBL.I, LBL.V, LBL.P))) do (lab, f, ylab)
        p = panel(lab; xlabel=LBL.ΓL, ylabel=ylab, legend=lab == "(c)" ? legend : false)
        for s in sets
            ΓL, y = [r.ΓL for r in s.rows], f.(s.rows)
            plot!(p, ΓL, y; color=s.style.color, ls=s.style.ls, marker=s.style.marker, label=s.style.label)
            k = argmax(getP.(s.rows))
            mark_max!(p, ΓL[k], y[k]; color=s.style.color)
        end
        logx!(p, ΓL_DECADES...; step=2)
    end
    plot(panels...; layout=(1, 3), size=SIZE.row3, MARGINS...)
end
save_figure(ivp_vs_gammaL([configs[2]]), joinpath(FIG, "v1v2v3", "iv_curve_panel.png"))
save_figure(ivp_vs_gammaL([configs[3]]), joinpath(FIG, "s1x2_s2x2", "iv_curve_panel.png"))
save_figure(ivp_vs_gammaL(configs; legend=:bottomright), joinpath(FIG, "confronti", "gammaL_panel_confronto.png"))

# P(V) of the three configurations (the maximum-power region is shown vs Γ_L in Ch. 8)
let p = panel(; xlabel=LBL.V, ylabel=LBL.P, legend=:bottomleft)
    for s in configs
        rows = sorted_by(s.rows, getV)
        V, P = getV.(rows), getP.(rows) ./ P_SCALE
        k = argmax(P)
        plot!(p, V, P; color=s.style.color, ls=s.style.ls, marker=s.style.marker, label=s.style.label)
        mark_max!(p, V[k], P[k]; color=s.style.color)
    end
    save_figure(plot(p; size=SIZE.single, MARGINS...), joinpath(FIG, "confronti", "power_vs_voltage_confronto.png"))
end

# Configurations in the maximum-power window (Ch. 8): P vs Γ_L
let p = panel(; xlabel=LBL.ΓL, ylabel=LBL.P, legend=:bottom, size=SIZE.single, MARGINS...)
    for s in configs
        r = filter(x -> 300 <= x.ΓL <= 3000, s.rows)
        ΓL, P = [x.ΓL for x in r], getP.(r) ./ P_SCALE
        plot!(p, ΓL, P; color=s.style.color, ls=s.style === MODEL.one_per_site ? :dash : s.style.ls, marker=s.style.marker, label=s.style.label)
        k = argmax(P)
        mark_max!(p, ΓL[k], P[k]; color=s.style.color)
    end
    plot!(p; xscale=:log10, xticks=([300, 1000, 3000], ["300", "1000", "3000"]))
    save_figure(p, joinpath(FIG, "confronti", "power_maxwindow_configs.png"))
end

# ============================================================
# Chapter 7: single-mode ω × Γ_L scan, fixed g (main) and fixed λ (comparison)
# ============================================================
fixed_g = phase1_scan("bench_g")
fixed_λ = phase1_scan("bench_lambda")

function surface_grid(results)
    ωs = sort(unique([r.ω for r in results]))
    ΓLs = sort(unique([r.ΓL for r in results]))
    lookup = Dict((r.ω, r.ΓL) => r.enhancement_pct for r in results)
    ωs, log10.(ΓLs), [get(lookup, (ω, ΓL), NaN) for ΓL in ΓLs, ω in ωs]
end
grids = Dict(:g => surface_grid(fixed_g), :λ => surface_grid(fixed_λ))
zlims = extrema(filter(!isnan, vcat(vec(grids[:g][3]), vec(grids[:λ][3]))))   # same color scale for both
for (key, fname) in ((:g, "enhancement_surface_fixedg.png"), (:λ, "enhancement_surface_fixedlambda.png"))
    ωs, logΓ, Z = grids[key]
    p = surface(ωs, logΓ, Z; xlabel=L"$\omega$ ($\mathrm{cm}^{-1}$)", ylabel=L"$\Gamma_L$ ($\mathrm{cm}^{-1}$)",
                zlabel=L"$\Delta P$ (%)", xticks=[200, 300, 400], yticks=(1:3, [L"10^{1}", L"10^{2}", L"10^{3}"]),
                camera=(50, 30), color=:viridis, clims=zlims, zlims=zlims, colorbar=false, size=SIZE.single)
    save_figure(p, joinpath(FIG, "omega_gammaL_scan", fname))
end

# P(V) family at fixed g (main result): every second ω, resonance in black dashed
const ω_FAMILY = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
const ω_RES = 200.0
function family!(p, results; legend_on=false)
    for ω in ω_FAMILY
        rω = sorted_by([r for r in results if r.ω == ω], r -> r.V_dampf)
        st = ω == ω_RES ? (color=:black, ls=:dash) : (color=family_color(ω, first(ω_FAMILY), last(ω_FAMILY)), ls=:solid)
        plot!(p, [r.V_dampf for r in rω], [r.P_dampf for r in rω] ./ P_SCALE; label=legend_on ? "$(Int(ω))" : "", st...)
    end
    p
end
let fig = plot(
        family!(panel("(a)"; xlabel=LBL.V, ylabel=LBL.P, legend=:bottomleft,
                      legendtitle=L"$\omega$ ($\mathrm{cm}^{-1}$)", legendtitlefontsize=9), fixed_g; legend_on=true),
        family!(panel("(b) Maximum-power region"; xlabel=LBL.V, ylabel=LBL.P, legend=false,
                      zoom_limits([[r.V_dampf for r in fixed_g if r.ω == ω] for ω in ω_FAMILY],
                                  [[r.P_dampf / P_SCALE for r in fixed_g if r.ω == ω] for ω in ω_FAMILY], 0.3)...), fixed_g);
        layout=(1, 2), size=SIZE.row2, MARGINS...)
    save_figure(fig, joinpath(FIG, "omega_gammaL_scan", "power_vs_V_fixedg.png"))
end

# Fixed g vs fixed λ: enhancement vs ω near the maximum-power Γ_L (there equal to the enhancement of P*)
let G = 850.0, p = panel(; xlabel=LBL.ω, ylabel=LBL.enh, legend=:bottomright, size=SIZE.single, MARGINS...)
    for (rows, st) in ((fixed_g, MODEL.fixed_g), (fixed_λ, MODEL.fixed_lambda))
        r = sorted_by([x for x in rows if x.ΓL == G], x -> x.ω)
        plot!(p, [x.ω for x in r], [x.enhancement_pct for x in r]; color=st.color, ls=st.ls, marker=st.marker, label=st.label)
    end
    vline!(p, [ω_RES]; color=:grey55, ls=:dash, lw=1.2, label="")
    save_figure(p, joinpath(FIG, "omega_gammaL_scan", "enhancement_vs_omega_fixedg_fixedlambda.png"))
end

# ============================================================
# Chapter 7 / appendix: detuned second mode, two modes per site
# chi_max = 8; P* = mean of the three largest powers in the Γ_L window, benchmark included
# ============================================================
ω2_LIST = [100, 150, 200, 250, 300, 350, 400, 450, 500]
ΓL_WINDOW = (500.0, 1200.0)
top3(rows) = let P = sort([r.P_dampf for r in rows]; rev=true)[1:3]
    (mean=sum(P) / 3, lo=minimum(P), hi=maximum(P))
end
in_window(rows) = filter(r -> ΓL_WINDOW[1] <= r.ΓL <= ΓL_WINDOW[2], rows)
rescaled = Dict(w => in_window(load_dampf_sweep_csv(joinpath(NV, "s1x2_s2x2_reorg_chi8", "iv_curve_s1x2_s2x2_reorg_chi8_$(w).csv")))
                for w in ω2_LIST)
unscaled = Dict(w => in_window(phase1_sweep("s2m_g", w)) for w in ω2_LIST)
P_bench = top3(in_window(phase1_sweep("bench_g", 200))).mean
stats_r = [top3(rescaled[w]) for w in ω2_LIST]
stats_u = [top3(unscaled[w]) for w in ω2_LIST]

let p = panel(; xlabel=LBL.ΓL, ylabel=LBL.P, legend=:outerright, legendtitle=L"$\omega_2$ ($\mathrm{cm}^{-1}$)",
              legendtitlefontsize=9, size=(760, 400), MARGINS...)
    for w in ω2_LIST
        r = sorted_by(rescaled[w], x -> x.ΓL)
        st = w == ω_RES ? (color=:black, ls=:dash) : (color=family_color(w, first(ω2_LIST), last(ω2_LIST)), ls=:solid)
        plot!(p, [x.ΓL for x in r], [x.P_dampf for x in r] ./ P_SCALE; marker=:circle, markersize=2, label="$(w)", st...)
    end
    ref_line!(p, P_bench / P_SCALE; label="Benchmark maximum")
    plot!(p; xlabel=L"Load rate $\Gamma_L$ ($\mathrm{cm}^{-1}$)")
    save_figure(p, joinpath(FIG, "s1x2_s2x2_omega2_scan_rescaled", "power_vs_gammaL_rescaled_by_omega2.png"))
end

err(s) = ([x.mean - x.lo for x in s] ./ P_SCALE, [x.hi - x.mean for x in s] ./ P_SCALE)
let p = panel(; xlabel=LBL.ω2, ylabel=L"Maximum power $P^\ast$ ($10^4\,\mathrm{cm}^{-2}$)", legend=:bottomright, size=SIZE.single, MARGINS...)
    for (stats, st) in ((stats_u, MODEL.fixed_g), (stats_r, MODEL.fixed_lambda))
        plot!(p, ω2_LIST, [s.mean for s in stats] ./ P_SCALE; yerror=err(stats), color=st.color, ls=st.ls, marker=st.marker,
              markersize=4.5, label=st.label)
    end
    ref_line!(p, P_bench / P_SCALE; label="Benchmark maximum")
    save_figure(p, joinpath(FIG, "s1x2_s2x2_omega2_scan_rescaled", "Pmax_vs_omega2_rescaled_vs_unscaled.png"))
end

# Appendix: relative P* vs total reorganization energy (fixed-g points at λ/λ0 = 1 + ω_res/ω2)
rel(P) = 100 * (P - P_bench) / P_bench
λ_ratio = [1 + ω_RES / w for w in ω2_LIST]
let p = panel(; xlabel=L"Reorganization energy $\lambda/\lambda_0$", ylabel=L"$(P^\ast-P^\ast_{\mathrm{bench}})/P^\ast_{\mathrm{bench}}$ (%)",
              legend=:topleft, size=SIZE.single, xlims=(0.85, 3.25), ylims=(-2.6, 4.6), MARGINS...)
    ref_line!(p, 0.0)
    scatter!(p, λ_ratio, [rel(s.mean) for s in stats_u]; color=MODEL.fixed_g.color, marker=MODEL.fixed_g.marker, markersize=4.5,
             label=MODEL.fixed_g.label)
    # ω2 labels next to the fixed-g points; offsets by hand, the ω2 = 350-500 points crowd within 0.2 in λ/λ0
    offset = Dict(500 => (0.03, -0.18, :left), 450 => (-0.03, 0.0, :right), 400 => (0.03, -0.18, :left),
                  350 => (0.03, 0.18, :left), 300 => (0.03, 0.18, :left), 250 => (0.03, 0.18, :left),
                  200 => (0.03, 0.18, :left), 150 => (0.03, 0.18, :left), 100 => (-0.03, 0.0, :right))
    for (i, w) in enumerate(ω2_LIST)
        dx, dy, ha = offset[w]
        annotate!(p, λ_ratio[i] + dx, rel(stats_u[i].mean) + dy, text(latexstring("\\omega_2=$(w)"), 8, ha, MODEL.fixed_g.color))
    end
    scatter!(p, fill(1.0, length(ω2_LIST)), [rel(s.mean) for s in stats_r]; color=MODEL.fixed_lambda.color,
             marker=MODEL.fixed_lambda.marker, markersize=4.5, label=latexstring("Fixed \$\\lambda=\\lambda_0\$ (all \$\\omega_2\$)"))
    scatter!(p, [1.0], [0.0]; color=MODEL.benchmark.color, marker=MODEL.benchmark.marker, markersize=5, label="Benchmark")
    save_figure(p, joinpath(FIG, "s1x2_s2x2_omega2_scan_rescaled", "Pmax_rel_vs_reorganization_energy.png"))
end

# ------------------------------------------------------------
# Chapter 8. Single-mode references from the direct calculation (experiments/second_mode_alone.jl,
# experiments/resonant_saturation.jl, Γ_L = 600-1100 step 25); two-modes P* from the DAMPF scans above,
# with the spread of the three highest points as error bar.
# ------------------------------------------------------------
const NV_DIRECT = joinpath(@__DIR__, "..", "..", "outputs", "numeric_vectors")
read_rows(f) = [split(l, ",") for l in readlines(joinpath(NV_DIRECT, f))[2:end]]
const ALONE = Dict((r[1], parse(Float64, r[2])) => (alone=parse(Float64, r[3]), res=parse(Float64, r[4]), el=parse(Float64, r[5]))
                   for r in read_rows("second_mode_alone.csv"))
const P_EL = first(values(ALONE)).el
enh(P) = 100 * (P / P_EL - 1)
enh_err(s) = (enh(s.mean) - enh(s.lo), enh(s.hi) - enh(s.mean))

# Redundancy, both conventions (top fixed g, bottom fixed λ). (a, c) Enhancement if the two modes added up
# (direct single-mode gains) and the real one (DAMPF); the shaded gap is the redundancy. (b, d) The redundancy
# as a fraction of the sum.
xl_ω2 = L"Frequency $\omega_2$ ($\mathrm{cm}^{-1}$)"
function redundancy_row(conv, stats, st, la, lb; ann)
    add = [enh(ALONE[(conv, Float64(w))].alone) + enh(ALONE[(conv, Float64(w))].res) for w in ω2_LIST]
    both = [enh(s.mean) for s in stats]
    pa = panel(la; xlabel=xl_ω2, ylabel=LBL.enh, legend=false, ylims=(6, 20), xlims=(75, 525))
    plot!(pa, ω2_LIST, add; fillrange=both, fillalpha=0.12, color=ThesisStyle.GREY, lw=0)
    plot!(pa, ω2_LIST, add; color=ThesisStyle.GREY, ls=:dash)
    plot!(pa, ω2_LIST, both; color=st.color, marker=st.marker, markersize=4)
    for (x, y, s, c) in ann
        annotate!(pa, x, y, text(s, 9, "Computer Modern", c == :grey ? ThesisStyle.GREY : st.color))
    end
    mark_max!(pa, ω2_LIST[argmax(add)], maximum(add); color=ThesisStyle.GREY)
    mark_max!(pa, ω2_LIST[argmax(both)], maximum(both); color=st.color)
    pb = panel(lb; xlabel=xl_ω2, ylabel="Redundancy (% of the sum)", legend=false, ylims=(0, 40), xlims=(75, 525))
    plot!(pb, ω2_LIST, 100 .* (1 .- both ./ add); color=st.color, marker=st.marker, markersize=4)
    println(conv, ": sum ", round.(add; digits=2), "\n   both ", round.(both; digits=2),
            "\n   red% ", round.(100 .* (1 .- both ./ add); digits=1))
    pa, pb
end
let (pa, pb) = redundancy_row("fixed_g", stats_u, MODEL.fixed_g, L"(a) Fixed $g$", L"(b) Fixed $g$";
        ann=[(300, 19.3, "Sum of the single-mode enhancements", :grey), (300, 15.6, "Redundancy", :grey),
             (330, 10.8, "Two modes per site (DAMPF)", :model)]),
    (pc, pd) = redundancy_row("fixed_lambda", stats_r, MODEL.fixed_lambda, L"(c) Fixed $\lambda$", L"(d) Fixed $\lambda$";
        ann=[(300, 14.6, "Sum of the single-mode enhancements", :grey), (300, 11.8, "Redundancy", :grey),
             (330, 7.0, "Two modes per site (DAMPF)", :model)])
    save_figure(plot(pa, pb, pc, pd; layout=(2, 2), size=SIZE.grid22, MARGINS...), joinpath(FIG, "discussion", "redundancy_vs_omega2.png"))
end

# Who gets the coupling at fixed λ. Both couplings are rescaled by the same factor, so the resonant mode keeps
# λ_res/λ0 = ω2/(ω2+ω_m) and the second mode carries λ_2/λ0 = ω_m/(ω2+ω_m). (a) The two shares. (b) Enhancement
# of each mode alone with its share (direct) and of the two together (DAMPF), against the benchmark.
let ωc = range(75, 525; length=200), sres = ωc ./ (ωc .+ ω_RES)
    RES_C, DET_C = RGB(0.35, 0.35, 0.35), ThesisStyle.GREEN
    pa = panel("(a)"; xlabel=xl_ω2, ylabel=L"Share of $\lambda_0$", legend=false, ylims=(0, 1), xlims=(75, 525),
               yticks=(0:0.25:1, ["0", "1/4", "1/2", "3/4", "1"]))
    plot!(pa, ωc, sres; fillrange=0, fillalpha=0.25, color=RES_C, lw=1.5)
    plot!(pa, ωc, ones(length(ωc)); fillrange=sres, fillalpha=0.25, color=DET_C, lw=0)
    vline!(pa, [ω_RES]; color=:grey55, ls=:dash, lw=1.0)
    annotate!(pa, 360, 0.30, text("Resonant mode", 10, "Computer Modern", RES_C))
    annotate!(pa, 360, 0.86, text(L"Detuned mode ($\omega_2$)", 10, "Computer Modern", DET_C))

    res = [enh(ALONE[("fixed_lambda", Float64(w))].res) for w in ω2_LIST]
    alone = [enh(ALONE[("fixed_lambda", Float64(w))].alone) for w in ω2_LIST]
    both = [enh(s.mean) for s in stats_r]
    pb = panel("(b)"; xlabel=xl_ω2, ylabel=LBL.enh, legend=false, ylims=(3, 12), xlims=(75, 525))
    ref_line!(pb, enh(P_bench))
    plot!(pb, ω2_LIST, res; color=RES_C, marker=:circle, markersize=3.5)
    plot!(pb, ω2_LIST, alone; color=DET_C, marker=:utriangle, markersize=4)
    plot!(pb, ω2_LIST, both; color=MODEL.fixed_lambda.color, marker=MODEL.fixed_lambda.marker, markersize=4)
    annotate!(pb, 300, 11.4, text("Both modes (DAMPF)", 10, "Computer Modern", MODEL.fixed_lambda.color))
    annotate!(pb, 450, 9.6, text("Benchmark", 9, "Computer Modern", ThesisStyle.GREY))
    annotate!(pb, 385, 8.3, text("Resonant mode alone", 10, "Computer Modern", RES_C))
    annotate!(pb, 395, 3.9, text("Detuned mode alone", 10, "Computer Modern", DET_C))
    println("fixed_lambda: res ", round.(res; digits=2), "\n   alone ", round.(alone; digits=2),
            "\n   both ", round.(both; digits=2), "\n   bench ", round(enh(P_bench); digits=2))
    save_figure(plot(pa, pb; layout=(1, 2), size=SIZE.row2, MARGINS...), joinpath(FIG, "discussion", "coupling_share_fixed_lambda.png"))
end
