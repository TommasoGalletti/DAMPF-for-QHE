"""
Review figures (not thesis figures) for the Phase 1 chi_max = 8 datasets:
each DAMPF configuration at chi = 8, the previous low-chi data in the thesis,
and the exact references (outputs/numeric_vectors/phase1_exact.csv,
two_mode_rd_exact.csv), with the relative error panels.
Output: DAMPF/outputs/imgs/chi8_review/*.png

Usage: python3 DAMPF/plots/chi8_review_figures.py
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
NV = os.path.join(HERE, "..", "outputs", "numeric_vectors")
NV_EXACT = os.path.join(HERE, "..", "..", "outputs", "numeric_vectors")
OUT = os.path.join(HERE, "..", "outputs", "imgs", "chi8_review")
os.makedirs(OUT, exist_ok=True)

C_EXACT, C_NEW, C_OLD = "#333333", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 2})


def rows(path):
    return list(csv.DictReader(open(path)))


def sweep(path, col="P_dampf"):
    r = sorted(rows(path), key=lambda x: float(x["GammaL"]))
    G = np.array([float(x["GammaL"]) for x in r])
    P = np.array([float(x[col]) for x in r])
    sd = np.array([float(x.get("P_std_pct", 0) or 0) for x in r])
    return G, P, sd


EX = {(r["config"], float(r["param"]), float(r["GammaL"])): float(r["P"]) for r in rows(os.path.join(NV_EXACT, "phase1_exact.csv"))}
ex = lambda c, p, G: np.array([EX[(c, float(p), float(g))] for g in G])
base = lambda G: ex("electronic", 0.0, G)


def compare_sweep(name, title, new, old, exact_cfg, exact_p, old_label):
    Gn, Pn, sdn = new
    Go, Po, _ = old
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})
    Ge = np.array(sorted(set(Gn) | set(Go)))
    a.plot(Ge, ex(exact_cfg, exact_p, Ge) / 1e4, color=C_EXACT, label="exact (direct, reduced)")
    a.plot(Go, Po / 1e4, "s", ms=5, mfc="none", color=C_OLD, label=old_label)
    a.plot(Gn, Pn / 1e4, "o", ms=5, color=C_NEW, label="DAMPF χ = 8")
    a.set_xscale("log"); a.set_ylabel("P  [10⁴ cm⁻²]"); a.set_title(title, loc="left"); a.legend(frameon=False)
    b.axhline(0, color=C_EXACT, lw=1)
    b.plot(Go, 100 * (Po / ex(exact_cfg, exact_p, Go) - 1), "s-", ms=4, mfc="none", lw=1, color=C_OLD, label=old_label)
    b.errorbar(Gn, 100 * (Pn / ex(exact_cfg, exact_p, Gn) - 1), yerr=sdn, fmt="o-", ms=4, lw=1, capsize=2, color=C_NEW, label="DAMPF χ = 8 (bar: plateau std)")
    b.set_xscale("log"); b.set_xlabel("Γ_L  [cm⁻¹]"); b.set_ylabel("P / P_exact − 1  [%]"); b.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=160); plt.close(fig)


# 1-3: benchmark, one mode per site, two modes per site (ω2 = 200, fixed g)
bench_old = rows(os.path.join(NV, "iv_curve_comparison.csv"))
bench_old = (np.array([float(r["GammaL"]) for r in bench_old]), np.array([float(r["P_dampf"]) for r in bench_old]), None)
compare_sweep("1_benchmark.png", "Benchmark (2 modes): fig. 6.1–6.2",
              sweep(os.path.join(NV, "phase1", "bench_g", "iv_bench_g_chi8_200.csv")), bench_old, "bench_g", 200, "thesis now (DAMPF χ = 3)")
compare_sweep("2_one_mode_per_site.png", "One mode per site (3 modes): fig. 7.1 — exact = benchmark",
              sweep(os.path.join(NV, "phase1", "v1v2v3", "iv_v1v2v3_chi8_200.csv")), sweep(os.path.join(NV, "iv_curve_v1v2v3.csv")),
              "bench_g", 200, "thesis now (DAMPF χ = 4)")
compare_sweep("3_two_modes_per_site.png", "Two modes per site, ω2 = 200, fixed g (4 modes): fig. 7.2",
              sweep(os.path.join(NV, "phase1", "s2m_g", "iv_s2m_g_chi8_200.csv")), sweep(os.path.join(NV, "s1x2_s2x2", "iv_curve_s1x2_s2x2_200.csv")),
              "s2m_g", 200, "thesis now (DAMPF χ = 4)")

# 4: configuration comparison, P* window (fig. 7.3–7.4)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, src in ((a, "DAMPF χ = 8"), (b, "exact")):
    for lab, cfg_new, cfg_ex, col in (("benchmark", ("bench_g", "iv_bench_g_chi8_200.csv"), ("bench_g", 200), "#2a78d6"),
                                      ("one mode per site", ("v1v2v3", "iv_v1v2v3_chi8_200.csv"), ("bench_g", 200), "#1baf7a"),
                                      ("two modes per site", ("s2m_g", "iv_s2m_g_chi8_200.csv"), ("s2m_g", 200), "#eb6834")):
        G, P, _ = sweep(os.path.join(NV, "phase1", *cfg_new))
        m = (G >= 300) & (G <= 3000)
        y = P[m] if src.startswith("DAMPF") else ex(*cfg_ex, G[m])
        ls = "--" if lab == "one mode per site" else "-"
        ax.plot(G[m], y / 1e4, ls, marker="o" if src.startswith("DAMPF") else None, ms=4, color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("Γ_L  [cm⁻¹]"); ax.set_ylabel("P  [10⁴ cm⁻²]"); ax.set_title(f"{src}: configurations, max-power window", loc="left")
    ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "4_configurations_maxpower.png"), dpi=160); plt.close(fig)

# 5-6: single-mode ω×ΓL scans (fig. 7.5–7.8)
def scan_new(cfg):
    out = {}
    for f in glob.glob(os.path.join(NV, "phase1", cfg, f"iv_{cfg}_chi8_*.csv")):
        w = float(f.rsplit("_", 1)[1][:-4])
        for G, P, _ in zip(*sweep(f)):
            out[(w, G)] = P
    if cfg == "bench_lambda":  # ω = 200 is the benchmark itself
        out.update({k: v for k, v in scan_new("bench_g").items() if k[0] == 200.0})
    return out


def scan_old(files):
    out = {}
    for f in files:
        for r in rows(f):
            out[(float(r["omega"]), float(r["GammaL"]))] = float(r["enhancement_pct"])
    return out


old_g = scan_old([os.path.join(NV, "omega_gammaL_scan_killoran.csv")] + glob.glob(os.path.join(NV, "omega_gammaL_scan_killoran_fixedg_ext", "*.csv")))
old_l = scan_old(glob.glob(os.path.join(NV, "omega_gammaL_scan_killoran_reorg", "*.csv")))
for tag, cfg, old, label in (("fixedg", "bench_g", old_g, "fixed g"), ("fixedlambda", "bench_lambda", old_l, "fixed λ")):
    new = scan_new(cfg)
    W = sorted({k[0] for k in new}); G = sorted(set.intersection(*[{k[1] for k in new if k[0] == w} for w in W]))
    enh = lambda P, g: 100 * (P / EX[("electronic", 0.0, g)] - 1)
    exk = lambda w, g: EX[(cfg if (cfg, w, g) in EX else "bench_g", w, g)]
    E_new = np.array([[enh(new[(w, g)], g) for w in W] for g in G])
    E_ex = np.array([[enh(exk(w, g), g) for w in W] for g in G])
    E_old = np.array([[old.get((w, g), np.nan) for w in W] for g in G])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, M, t in ((axs[0], E_new - E_ex, "DAMPF χ = 8 − exact"), (axs[1], E_old - E_ex, "thesis now (χ = 3) − exact")):
        im = ax.pcolormesh(np.arange(len(W) + 1), np.arange(len(G) + 1), M, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-3, vmax=3))
        ax.set_xticks(np.arange(len(W))[::2] + 0.5, [f"{w:g}" for w in W][::2]); ax.set_yticks(np.arange(len(G))[::4] + 0.5, [f"{g:g}" for g in G][::4])
        ax.set_xlabel("ω  [cm⁻¹]"); ax.set_ylabel("Γ_L  [cm⁻¹]"); ax.set_title(f"{label}: ΔP error, {t}", loc="left", fontsize=9)
        fig.colorbar(im, ax=ax, label="percentage points")
    ax = axs[2]
    for gl, mk in ((30.0, "o"), (850.0, "D")):
        i = G.index(gl)
        ax.plot(W, E_ex[i], color=C_EXACT, lw=1.5)
        ax.plot(W, E_old[i], mk, mfc="none", ms=5, color=C_OLD)
        ax.plot(W, E_new[i], mk, ms=5, color=C_NEW, label=f"Γ_L = {gl:g}")
    ax.plot([], [], color=C_EXACT, label="exact"); ax.plot([], [], "s", mfc="none", color=C_OLD, label="thesis now (χ = 3)"); ax.plot([], [], "s", color=C_NEW, label="DAMPF χ = 8")
    ax.set_xlabel("ω  [cm⁻¹]"); ax.set_ylabel("ΔP  [%]"); ax.set_title(f"{label}: enhancement vs ω (fig. 7.8)", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"5_omega_scan_{tag}.png"), dpi=160); plt.close(fig)

# 7: ω2 scans (fig. 7.9–7.10)
E6 = {}
for r in rows(os.path.join(NV_EXACT, "two_mode_rd_exact.csv")):
    E6[(r["convention"], float(r["omega2"]), float(r["GammaL"]))] = float(r["P"])
W2 = list(range(100, 501, 50))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
pstar = {}
for conv, new_f, old_f in (("fixed_g", lambda w: os.path.join(NV, "phase1", "s2m_g", f"iv_s2m_g_chi8_{w}.csv"),
                            lambda w: os.path.join(NV, "s1x2_s2x2", f"iv_curve_s1x2_s2x2_{w}.csv")),
                           ("fixed_lambda", lambda w: os.path.join(NV, "s1x2_s2x2_reorg_chi8", f"iv_curve_s1x2_s2x2_reorg_chi8_{w}.csv"),
                            lambda w: os.path.join(NV, "s1x2_s2x2_reorg_avg5", f"iv_curve_s1x2_s2x2_reorg_avg5_{w}.csv"))):
    d = {}
    for w in W2:
        Gn, Pn, _ = sweep(new_f(w)); Go, Po, _ = sweep(old_f(w))
        mn, mo = (Gn >= 600) & (Gn <= 1100), (Go >= 600) & (Go <= 1100)
        pe = max(P for (c, ww, g), P in E6.items() if c == conv and ww == w) if conv == "fixed_lambda" else max(EX[("s2m_g", float(w), float(g))] for g in range(600, 1101, 25))
        d[w] = (Pn[mn].max(), Po[mo].max(), pe)
    pstar[conv] = d
for ax, conv, t in ((axs[0], "fixed_g", "fixed g"), (axs[1], "fixed_lambda", "fixed λ")):
    d = pstar[conv]
    rel = lambda k: [100 * (d[w][k] / d[200][k] - 1) for w in W2]
    ax.axhline(0, color="#999999", lw=1)
    ax.plot(W2, rel(2), color=C_EXACT, label="exact")
    ax.plot(W2, rel(1), "s--", mfc="none", ms=5, lw=1, color=C_OLD, label="thesis now (χ = 4)")
    ax.plot(W2, rel(0), "o-", ms=5, lw=1.5, color=C_NEW, label="DAMPF χ = 8")
    ax.set_xlabel("ω2  [cm⁻¹]"); ax.set_ylabel("P*(ω2) / P*(200) − 1  [%]"); ax.set_title(f"{t}: P* vs ω2 (fig. 7.10)", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=8)
ax = axs[2]
for conv, mk, t in (("fixed_g", "o", "fixed g"), ("fixed_lambda", "D", "fixed λ")):
    d = pstar[conv]
    ax.plot(W2, [100 * (d[w][0] / d[w][2] - 1) for w in W2], mk + "-", ms=5, lw=1.5, color=C_NEW, mfc=C_NEW if conv == "fixed_g" else "white", label=f"χ = 8, {t}")
    ax.plot(W2, [100 * (d[w][1] / d[w][2] - 1) for w in W2], mk + "--", ms=5, lw=1, color=C_OLD, mfc=C_OLD if conv == "fixed_g" else "white", label=f"χ = 4, {t}")
ax.axhline(0, color=C_EXACT, lw=1)
ax.set_xlabel("ω2  [cm⁻¹]"); ax.set_ylabel("P*_DAMPF / P*_exact − 1  [%]"); ax.set_title("error on P*", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "6_omega2_scan_pstar.png"), dpi=160); plt.close(fig)

# 7: P vs ΓL in the max-power window for every ω2 (fig. 7.9), both conventions
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, conv, t, new_f in ((axs[0], "fixed_g", "fixed g", lambda w: os.path.join(NV, "phase1", "s2m_g", f"iv_s2m_g_chi8_{w}.csv")),
                           (axs[1], "fixed_lambda", "fixed λ", lambda w: os.path.join(NV, "s1x2_s2x2_reorg_chi8", f"iv_curve_s1x2_s2x2_reorg_chi8_{w}.csv"))):
    for i, w in enumerate(W2):
        col = cmap(i / (len(W2) - 1))
        G, P, sd = sweep(new_f(w)); m = (G >= 600) & (G <= 1100)
        ax.errorbar(G[m], P[m] / 1e4, yerr=sd[m] * P[m] / 1e6, fmt="o", ms=3, lw=0.8, color=col)
        if conv == "fixed_g":
            ax.plot(G[m], ex("s2m_g", w, G[m]) / 1e4, color=col, lw=1.2, label=f"{w}")
        else:
            ge = sorted(g for (c, ww, g) in E6 if c == conv and ww == w)
            ax.plot(ge, [E6[(conv, w, g)] / 1e4 for g in ge], color=col, lw=1.2, label=f"{w}")
    ax.set_xlabel("Γ_L  [cm⁻¹]"); ax.set_title(f"{t}: lines exact, dots DAMPF χ = 8 (fig. 7.9)", loc="left", fontsize=9)
axs[0].set_ylabel("P  [10⁴ cm⁻²]"); axs[1].legend(title="ω2", frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "7_omega2_scan_curves.png"), dpi=160); plt.close(fig)
print("figures ->", os.path.abspath(OUT))
