# Figures for slides2.tex, slide 12 (single mode, fixed g only).
# Same data and style as DAMPF/plots/thesis_figures_dampf.jl (Chapter 7 omega x Gamma_L scan);
# writes only into presentazione/figures/, the thesis figures are not touched.
const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "DAMPF", "utils", "csv_io.jl"))
include(joinpath(ROOT, "DAMPF", "plots", "thesis_style.jl"))
using .DampfCsvIO
using .ThesisStyle
using Plots, LaTeXStrings

theme!()
const OUT = @__DIR__
const P1 = joinpath(ROOT, "DAMPF", "outputs", "numeric_vectors", "phase1")
const MARGINS = (left_margin=3mm, bottom_margin=4mm)
const G_SECTION = 850.0
const ω_RES = 200.0

phase1_sweep(cfg, param) = sorted_by(load_dampf_sweep_csv(joinpath(P1, cfg, "iv_$(cfg)_chi8_$(param).csv")), r -> r.ΓL)
const EXACT = Dict(map(readlines(joinpath(ROOT, "outputs", "numeric_vectors", "phase1_exact.csv"))[2:end]) do l
    c = split(l, ",")
    (String(c[1]), parse(Float64, c[2]), parse(Float64, c[3])) => (I=parse(Float64, c[4]), V=parse(Float64, c[5]), P=parse(Float64, c[6]))
end)
function scan_fixed_g()
    cfg = "bench_g"
    files = filter(f -> startswith(f, "iv_$(cfg)_chi8_"), readdir(joinpath(P1, cfg)))
    ωs = [parse(Float64, split(f[1:end-4], "_")[end]) for f in files]
    ΓLs = Set(r.ΓL for r in phase1_sweep(cfg, Int(first(filter(!=(200.0), ωs)))))
    [(ω=ω, ΓL=r.ΓL, enh=100 * (r.P_dampf / EXACT[("electronic", 0.0, r.ΓL)].P - 1))
     for ω in ωs for r in phase1_sweep(cfg, Int(ω)) if r.ΓL in ΓLs]
end
rows = scan_fixed_g()

# 3D surface, as in the thesis, with the section at Γ_L = 850 cm^-1 drawn on it
ωs = sort(unique([r.ω for r in rows]))
ΓLs = sort(unique([r.ΓL for r in rows]))
lookup = Dict((r.ω, r.ΓL) => r.enh for r in rows)
Z = [get(lookup, (ω, Γ), NaN) for Γ in ΓLs, ω in ωs]
p3 = surface(ωs, log10.(ΓLs), Z; xlabel=L"$\omega$ ($\mathrm{cm}^{-1}$)", ylabel=L"$\Gamma_L$ ($\mathrm{cm}^{-1}$)",
             zlabel=L"$\Delta P$ (%)", xticks=[200, 300, 400], yticks=(1:3, [L"10^{1}", L"10^{2}", L"10^{3}"]),
             camera=(50, 30), color=:viridis, colorbar=false, size=SIZE.single)
sec = sorted_by([r for r in rows if r.ΓL == G_SECTION], r -> r.ω)
plot!(p3, [r.ω for r in sec], fill(log10(G_SECTION), length(sec)), [r.enh for r in sec] .+ 0.05;
      st=:path3d, color=:black, lw=2.5, label="")
save_figure(p3, joinpath(OUT, "enhancement_surface_fixedg_section.png"))

# Section at Γ_L = 850 cm^-1, fixed g only
p2 = panel(; xlabel=LBL.ω, ylabel=LBL.enh, legend=false, size=SIZE.single, MARGINS...)
st = MODEL.fixed_g
plot!(p2, [r.ω for r in sec], [r.enh for r in sec]; color=st.color, ls=st.ls, marker=st.marker, label="")
vline!(p2, [ω_RES]; color=:grey55, ls=:dash, lw=1.2, label="")
save_figure(p2, joinpath(OUT, "enhancement_vs_omega_fixedg.png"))
