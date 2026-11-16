"""
Common graphic style for every figure of the thesis (thesis_latex/figures/).
Included by main/thesis_figures_direct.jl (Chapters 3-4) and by
DAMPF/plots/thesis_figures_dampf.jl (Chapters 6-7 and appendix).

Conventions:
- No figure titles: the LaTeX caption says what is shown. Multi-panel figures
  carry left-aligned panel labels "(a)", "(b)", ..., optionally followed by a
  short descriptor ("(a) Fixed g").
- Each model or convention always has the same color, line style and marker
  (MODEL below); states always have the same color (STATE below); scans over
  a parameter use the FAMILY gradient, with the resonant/benchmark curve in
  black dashed.
- Dense sweeps are drawn as lines only, sparse simulation points as line +
  markers.
- The maximum-power operating point is marked with a star (`mark_max!`) on
  every curve of a figure with at most three curves (also on I and V, at the
  Γ_L of maximum power); numerical values go in the caption or text, never as
  annotations inside the figure. Reference levels (benchmark maximum) are
  grey dashed horizontal lines (`ref_line!`).
- Γ_L is always on a logarithmic axis with ticks at the decades.
- Sizes are tied to the LaTeX width the figure is included at (WIDTHS), so
  that the printed font size is the same in every figure: FULL figures go in
  at width=\\textwidth, SINGLE at width=0.8\\textwidth.
"""
module ThesisStyle

using Plots, LaTeXStrings
using Plots.PlotMeasures

export MODEL, STATE, LBL, LBL_KILLORAN, SIZE, family_color, theme!, panel, logx!, mark_max!,
       ref_line!, sorted_by, save_figure, mm

const BLACK = RGB(0, 0, 0)
const BLUE = RGB(0x00 / 255, 0x72 / 255, 0xB2 / 255)
const VERMILLION = RGB(0xD5 / 255, 0x5E / 255, 0x00 / 255)
const GREEN = RGB(0x00 / 255, 0x9E / 255, 0x73 / 255)
const GREY = RGB(0.55, 0.55, 0.55)

# Okabe-Ito colors (color-blind safe); one fixed style per model/convention.
const MODEL = (
    direct=(color=BLACK, ls=:solid, marker=:circle, label="Direct Lindblad"),
    dampf=(color=VERMILLION, ls=:dash, marker=:diamond, label="DAMPF"),
    electronic=(color=BLACK, ls=:solid, marker=:none, label="Electronic model"),
    vibronic=(color=BLUE, ls=:solid, marker=:none, label="Vibronic model"),
    vibronic_g0=(color=VERMILLION, ls=:dash, marker=:none, label=L"Vibronic model, $g=0$"),
    benchmark=(color=BLACK, ls=:solid, marker=:circle, label="Benchmark"),
    one_per_site=(color=BLUE, ls=:solid, marker=:diamond, label="One mode per site"),
    two_per_site=(color=GREEN, ls=:solid, marker=:utriangle, label="Two modes per site"),
    fixed_g=(color=BLUE, ls=:solid, marker=:circle, label=L"Fixed $g$"),
    fixed_lambda=(color=VERMILLION, ls=:solid, marker=:diamond, label=L"Fixed $\lambda$"),
)

# Same color for the same index in both bases: ground, then 1, 2, 3.
const STATE = (g=BLACK, s1=BLUE, s2=VERMILLION, s3=GREEN)

# Viridis without its pale-yellow end, which is unreadable on white.
family_color(x, lo, hi) = cgrad(:viridis)[0.88 * (x - lo) / (hi - lo)]

const LBL = (
    ΓL=L"Load rate $\Gamma_L$ ($\mathrm{cm}^{-1}$)",
    V=L"Voltage $V$ ($\mathrm{cm}^{-1}$)",
    I=L"Current $I$ ($\mathrm{cm}^{-1}$)",
    P=L"Power $P$ ($10^4\,\mathrm{cm}^{-2}$)",
    ω=L"Mode frequency $\omega$ ($\mathrm{cm}^{-1}$)",
    ω2=L"Frequency $\omega_2$ of the detuned mode ($\mathrm{cm}^{-1}$)",
    enh=L"Enhancement $\Delta P$ (%)",
    pop="Population",
    trace=L"Trace distance $D(\rho_{\mathrm{DAMPF}},\rho_{\mathrm{direct}})$",
)
# Chapters 3-4 keep the units of Killoran et al.
const LBL_KILLORAN = (
    ΓL=LBL.ΓL,
    V=L"Voltage $V$ (V)",
    I=L"Current $I/(e\Gamma_H)$",
    P=L"Power $P/\Gamma_H$ (eV)",
    t=L"Time $t$ ($1/\mathrm{cm}^{-1}$)",
)
const P_SCALE = 1e4   # DAMPF powers are plotted in units of 10^4 cm^-2

# Plot size units are 1/100 inch: FULL is \textwidth, SINGLE is 0.8\textwidth.
const SIZE = (single=(600, 400), row2=(760, 320), row3=(760, 270), grid22=(760, 620), grid42=(760, 900))

function theme!()
    default(fontfamily="Computer Modern", framestyle=:box, dpi=300, linewidth=1.8, markersize=3.5,
            markerstrokewidth=0.6, guidefontsize=11, tickfontsize=9, legendfontsize=9,
            titlefontsize=11, titlelocation=:left, legend_background_color=RGBA(1, 1, 1, 0.85),
            gridalpha=0.15, foreground_color_legend=RGB(0.6, 0.6, 0.6))
end

"""Empty panel with a left-aligned label such as "(a)" or "(a) Fixed g"."""
panel(label=""; kw...) = plot(; title=label, kw...)

"""Logarithmic Γ_L axis with ticks at the decades between 10^lo and 10^hi."""
function logx!(p, lo::Int, hi::Int; step=1)
    ticks = lo:step:hi
    plot!(p; xscale=:log10, xlims=(10.0^lo, 10.0^hi), xticks=(10.0 .^ ticks, [latexstring("10^{$k}") for k in ticks]))
end

"""Star on the maximum-power operating point, in the color of its curve."""
mark_max!(p, x, y; color, kw...) =
    scatter!(p, [x], [y]; marker=:star5, markersize=9, color=color, markerstrokecolor=BLACK, markerstrokewidth=0.8,
             label="", kw...)

"""Horizontal reference level (benchmark maximum, zero, ...): always grey dashed, so
that it is never confused with the black-dashed resonant curve of a family."""
ref_line!(p, y; label="", color=GREY, ls=:dash) = hline!(p, [y]; color=color, ls=ls, lw=1.2, label=label)

sorted_by(rows, f) = rows[sortperm([f(r) for r in rows])]

function save_figure(fig, path)
    mkpath(dirname(path))
    savefig(fig, path)
    println("Saved: ", path)
end

end
