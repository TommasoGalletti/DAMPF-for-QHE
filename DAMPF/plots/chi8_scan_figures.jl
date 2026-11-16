include("../utils/csv_io.jl")
include("thesis_style.jl")
using .DampfCsvIO
using .ThesisStyle
using Plots, LaTeXStrings

"""
Figures of the chi_max = 8 rerun of the reorganization-energy-rescaled
two-modes-per-site scan (plateau averages from utils/build_plateau_average.jl),
compared with the chi_max = 4 production data (avg5) and with the exact
direct solution through the relative-displacement reduction
(experiments/two_mode_rd_exact.jl). Working figures, written to
outputs/imgs/s1x2_s2x2_reorg_chi8/, not to the thesis.
"""

theme!()
const NV = joinpath(@__DIR__, "..", "outputs", "numeric_vectors")
const OUT = joinpath(@__DIR__, "..", "outputs", "imgs", "s1x2_s2x2_reorg_chi8")
const P_SCALE = 1e4
const ω2_LIST = 100:50:500
const ω_RES = 200

chi8 = Dict(w => sorted_by(load_dampf_sweep_csv(joinpath(NV, "s1x2_s2x2_reorg_chi8", "iv_curve_s1x2_s2x2_reorg_chi8_$(w).csv")), r -> r.ΓL)
            for w in ω2_LIST)
chi4 = Dict(w => sorted_by(load_dampf_sweep_csv(joinpath(NV, "s1x2_s2x2_reorg_avg5", "iv_curve_s1x2_s2x2_reorg_avg5_$(w).csv")), r -> r.ΓL)
            for w in ω2_LIST)
exact_rows = map(readlines(joinpath(@__DIR__, "..", "..", "outputs", "numeric_vectors", "two_mode_rd_exact.csv"))[2:end]) do l
    c = split(l, ",")
    (conv=c[1], ω2=parse(Float64, c[2]), ΓL=parse(Float64, c[4]), P=parse(Float64, c[7]))
end
exact(conv, w) = sorted_by([r for r in exact_rows if r.conv == conv && r.ω2 == w], r -> r.ΓL)

"""Maximum of a smooth P(ΓL): vertex of the parabola through the highest point and its neighbours."""
function pstar(ΓL, P)
    k = argmax(P)
    (k == 1 || k == length(P)) && return P[k]
    x, y = ΓL[k-1:k+1], P[k-1:k+1]
    A = [x .^ 2 x ones(3)]
    a, b, c = A \ y
    c - b^2 / (4a)
end
pstar(rows) = pstar([r.ΓL for r in rows], [hasproperty(r, :P_dampf) ? r.P_dampf : r.P for r in rows])

P_bench = pstar(exact("benchmark", 200.0))
rel(P) = 100 * (P / P_bench - 1)

# 1. P(ΓL) at fixed λ, chi_max = 8, one curve per ω2
let p = panel(; xlabel=L"Load rate $\Gamma_L$ ($\mathrm{cm}^{-1}$)", ylabel=LBL.P, legend=:outerright,
              legendtitle=L"$\omega_2$ ($\mathrm{cm}^{-1}$)", legendtitlefontsize=9, size=(760, 400),
              left_margin=3mm, bottom_margin=4mm)
    for w in ω2_LIST
        r = chi8[w]
        st = w == ω_RES ? (color=:black, ls=:dash) : (color=family_color(w, first(ω2_LIST), last(ω2_LIST)), ls=:solid)
        plot!(p, [x.ΓL for x in r], [x.P_dampf for x in r] ./ P_SCALE; marker=:circle, markersize=2, label="$(w)", st...)
    end
    ref_line!(p, P_bench / P_SCALE; label="Benchmark maximum (exact)")
    save_figure(p, joinpath(OUT, "power_vs_gammaL_chi8_by_omega2.png"))
end

# 2. P*(ω2) relative to the exact benchmark: chi_max = 4, chi_max = 8, exact
let p = panel(; xlabel=LBL.ω2, ylabel=L"$(P^\ast-P^\ast_{\mathrm{bench}})/P^\ast_{\mathrm{bench}}$ (%)", legend=:bottomright,
              size=SIZE.single, left_margin=3mm, bottom_margin=4mm)
    ref_line!(p, 0.0)
    plot!(p; xlims=(80, 520))
    plot!(p, collect(ω2_LIST), [rel(pstar(exact("fixed_lambda", Float64(w)))) for w in ω2_LIST];
          color=MODEL.direct.color, ls=:solid, marker=:circle, label="Exact")
    plot!(p, collect(ω2_LIST), [rel(pstar(chi8[w])) for w in ω2_LIST];
          color=MODEL.dampf.color, ls=:dash, marker=:diamond, label=L"DAMPF, $\chi_{\max}=8$")
    plot!(p, collect(ω2_LIST), [rel(pstar(chi4[w])) for w in ω2_LIST];
          color=:grey55, ls=:dot, marker=:utriangle, label=L"DAMPF, $\chi_{\max}=4$")
    save_figure(p, joinpath(OUT, "Pmax_vs_omega2_chi4_chi8_exact.png"))
end

# 3. P(ΓL), DAMPF chi_max = 8 vs exact, for a few ω2
let panels = map(zip(("(a)", "(b)", "(c)"), (200, 350, 500))) do (lab, w)
        p = panel(latexstring("$(lab) \$\\omega_2=$(w)\\,\\mathrm{cm}^{-1}\$"); xlabel=LBL.ΓL, ylabel=lab == "(a)" ? LBL.P : "",
                  legend=lab == "(a)" ? :bottomright : false)
        e = exact("fixed_lambda", Float64(w))
        plot!(p, [r.ΓL for r in e], [r.P for r in e] ./ P_SCALE; color=MODEL.direct.color, marker=:circle, label="Exact")
        d = chi8[w]
        plot!(p, [r.ΓL for r in d], [r.P_dampf for r in d] ./ P_SCALE; color=MODEL.dampf.color, ls=:dash, marker=:diamond,
              markersize=2.5, label=L"DAMPF, $\chi_{\max}=8$")
        plot!(p; xscale=:identity)
    end
    save_figure(plot(panels...; layout=(1, 3), size=SIZE.row3, left_margin=3mm, bottom_margin=4mm),
                joinpath(OUT, "power_vs_gammaL_chi8_vs_exact.png"))
end
