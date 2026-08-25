module DampfPhysical

export DampfMode, DampfModel, dim_e, killoran_benchmark_model, killoran_three_mode_model,
       killoran_s1x2_s2x2_model, killoran_four_site_model, killoran_cascade_model,
       KB_CM1_PER_K, thermal_occupation

const KB_CM1_PER_K = 0.69503476

"""
A single vibrational mode attached to one electronic site.

`site` is the 1-based electronic label it's attached to (2..dim_e; site 1 is
always the ground state and never carries a mode). `T` is this mode's own
bath temperature (Kelvin) -- kept per-mode (not global) so different sites
can later have different thermal environments without changing the model.
"""
struct DampfMode
    site::Int
    ω::Float64
    g::Float64
    γ::Float64
    T::Float64
    Nb::Int
end

"""
Thermal (Bose-Einstein) occupation number for a mode of frequency `ω` at
temperature `T`, in the same cm^-1/K convention as `config/parameters.jl`.
"""
thermal_occupation(ω::Float64, T::Float64) = 1 / (exp(ω / (KB_CM1_PER_K * T)) - 1)

"""
Generic vibronic model: `N` electronic sites (dim_e = N+1 including the
ground state), site energies `E` (length N), coupling matrix `J` (N x N),
and an arbitrary list of vibrational modes (any per-site count, incl. zero).
Nothing here is hardcoded to a specific N or mode count.
"""
struct DampfModel
    Eg::Float64
    E::Vector{Float64}
    J::Matrix{Float64}
    modes::Vector{DampfMode}
end

dim_e(model::DampfModel) = length(model.E) + 1

"""
Original Killoran benchmark: ground + 3 sites (s1,s2,s3), one mode each on
s1/s2 (s3 has none), matching `config/parameters.jl`'s
`default_params()`/`default_simulation_params()`. `site` is a SITE-BASIS
index (1=ground,2=s1,3=s2,4=s3) -- not to be confused with `idx_low`/`idx_g`
elsewhere, which index the sorted EXCITON basis instead (idx_low=3 lands on
s3, the isolated lowest exciton; idx_g=4 is the true ground state, since Eg
is far below everything else).
"""
function killoran_benchmark_model(; Nb::Int=6)
    return DampfModel(
        -10000.0,
        [300.0, 300.0, 0.0],
        [0.0 100.0 0.0; 100.0 0.0 0.0; 0.0 0.0 0.0],
        [
            DampfMode(2, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(3, 200.0, 55.0, 5.3, 293.0, Nb),
        ],
    )
end

"""
Scale-up variant: same electronic structure as `killoran_benchmark_model`,
but every site carries a mode -- s1, s2, AND s3 (same ω/g/γ/T). Electronic
structure is untouched, so `idx_low`/`idx_g` are unchanged: s3 stays an
exact eigenstate of `H_e` regardless of attached modes.

With M=3, the dense joint vibrational space is Nb^3 (=216 for Nb=6), so the
exact Liouvillian reference ((dim_e*Nb^3)^2 x same = 746496x746496) is no
longer attainable -- DAMPF-only from here on.
"""
function killoran_three_mode_model(; Nb::Int=6)
    return DampfModel(
        -10000.0,
        [300.0, 300.0, 0.0],
        [0.0 100.0 0.0; 100.0 0.0 0.0; 0.0 0.0 0.0],
        [
            DampfMode(2, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(3, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(4, 200.0, 55.0, 5.3, 293.0, Nb),
        ],
    )
end

"""
Another scale-up variant: s1 and s2 each carry TWO independent modes (s3
stays mode-less), 4 modes total. Exercises `modes_on_site` returning more
than one mode for the same site (already generic in
local_generators.jl/trotter.jl, never run before). All four modes share the
same ω/g/γ/T -- literal duplicates, not distinct vibrations; change `ω`/`g`
here if that's wanted instead.
"""
function killoran_s1x2_s2x2_model(; Nb::Int=6)
    return DampfModel(
        -10000.0,
        [300.0, 300.0, 0.0],
        [0.0 100.0 0.0; 100.0 0.0 0.0; 0.0 0.0 0.0],
        [
            DampfMode(2, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(2, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(3, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(3, 200.0, 55.0, 5.3, 293.0, Nb),
        ],
    )
end

"""
Ground + FOUR sites, dim_e=5. Extends the benchmark's bright pair (s1,s2)
into a symmetric bright TRIO (s1,s2,s4, all pairwise J=100); s3 stays
isolated/mode-less as before. One mode each on s1,s2,s4 (M=3).

By hand: the symmetric 3x3 block gives a non-degenerate state at E=500 and
a degenerate pair at E=200; s3 stays an exact eigenstate at E=0, the
lowest excited state -- so idx_low/idx_g still land on dim_e-1/dim_e,
confirmed by diagonalizing `H_e` in check/cluster_smoke_test.jl. The
E=200 degeneracy doesn't matter: neither the hot bath (unique highest
state) nor the load (unique lowest, s3) touches it.
"""
function killoran_four_site_model(; Nb::Int=6)
    return DampfModel(
        -10000.0,
        [300.0, 300.0, 0.0, 300.0],
        [0.0 100.0 0.0 100.0; 100.0 0.0 0.0 100.0; 0.0 0.0 0.0 0.0; 100.0 100.0 0.0 0.0],
        [
            DampfMode(2, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(3, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(5, 200.0, 55.0, 5.3, 293.0, Nb),
        ],
    )
end

"""
Second scale-up variant, same dim_e=5, different topology: an energetic
CASCADE instead of a wider bright cluster. Bright pair s1-s2 unchanged;
the isolated funnel s3 becomes a two-stage chain instead -- relay s3
(E=150, weakly coupled to s2, J23=30) then funnel s4 (E=0, J34=30).
J13=J14=J24=0: a genuine chain, no shortcuts. J23/J34 are weak vs J12 and
the 150 cm^-1 gap (~4% admixture expected), but unlike the trio, no site
here is exactly isolated -- idx_low/idx_g are confirmed numerically in
check/cascade_smoke_test.jl, not read off by hand.

Modes stay on s1/s2 only (same as the benchmark) so this isolates the
cascade topology's effect from any change in vibronic structure.
"""
function killoran_cascade_model(; Nb::Int=6)
    return DampfModel(
        -10000.0,
        [300.0, 300.0, 150.0, 0.0],
        [0.0 100.0 0.0 0.0; 100.0 0.0 30.0 0.0; 0.0 30.0 0.0 30.0; 0.0 0.0 30.0 0.0],
        [
            DampfMode(2, 200.0, 55.0, 5.3, 293.0, Nb),
            DampfMode(3, 200.0, 55.0, 5.3, 293.0, Nb),
        ],
    )
end

end
