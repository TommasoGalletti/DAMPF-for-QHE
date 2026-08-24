module DampfPhysical

export DampfMode, DampfModel, dim_e, killoran_benchmark_model, killoran_three_mode_model,
       killoran_s1x2_s2x2_model, KB_CM1_PER_K, thermal_occupation

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
The original Killoran benchmark: ground + 3 sites (s1,s2,s3), one mode each
on s1 and s2 (s3 has none), matching `config/parameters.jl`'s
`default_params()`/`default_simulation_params()` exactly. `site` here is a
SITE-BASIS index (1=ground,2=s1,3=s2,4=s3, matching `Hamiltonian.jl`'s
`build_hamiltonian`/`build_total_hamiltonian` P_s1/P_s2 convention) -- not
to be confused with `idx_low`/`idx_g` elsewhere, which index the sorted
EXCITON (eigen-)basis instead (there idx_low=3 lands on s3, the isolated
lowest exciton, and idx_g=4 lands on the true ground state, since Eg is so
far below everything else that diagonalization does not mix it).
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
Scale-up variant of the Killoran benchmark: same ground+3-site electronic
structure (Eg, E, J unchanged), but now EVERY excitonic site carries its own
vibrational mode -- s1, s2, AND s3 (the original benchmark left s3
mode-less). The third mode reuses the same per-mode physical parameters as
the other two (ω=200, g=55, γ=5.3, T=293), since nothing in the model
distinguishes s3 physically from s1/s2 other than being electronically
isolated (J has no s3 row/column). Electronic structure is untouched, so
`idx_low`/`idx_g` keep the same values as for `killoran_benchmark_model`
(dim_e-1, dim_e) -- s3 remains an exact eigenstate of `H_e` regardless of
which modes are attached to it, since `diagonalize_electronic` only ever
sees the 4x4 electronic Hamiltonian.

With M=3 modes, the dense joint vibrational space is Nb^3 (=216 for Nb=6),
so the "Lindblad diretto" dense reference used to validate
`killoran_benchmark_model` is no longer computationally attainable (the
vectorized Liouvillian would be (dim_e*Nb^3)^2 x (dim_e*Nb^3)^2 =
746496x746496) -- this model is meant to be run through DAMPF only.
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
Another scale-up variant: same ground+3-site electronic structure as
`killoran_benchmark_model` (Eg, E, J unchanged), but now s1 and s2 each
carry TWO independent vibrational modes (s3 stays mode-less, as in the
original benchmark) -- 4 modes total, vs. 1-per-site (2 modes) in the
original benchmark and 1-per-site-on-all-3 (3 modes) in
`killoran_three_mode_model`. This exercises `modes_on_site` returning more
than one mode for the same site, a code path that was already generic
(local_generators.jl/trotter.jl loop mode-by-mode and never assume
uniqueness per site) but never actually run with >1 mode per site before.

All four modes reuse the same per-mode physical parameters (ω=200, g=55,
γ=5.3, T=293) as every other model here -- so the two modes on a given site
are literal duplicates of each other (independent, uncoupled oscillators
with identical ω/g/γ/T). If instead you want each site's two modes to
represent physically distinct vibrations (e.g. a low- and a high-frequency
mode), change their `ω`/`g` here.
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

end
