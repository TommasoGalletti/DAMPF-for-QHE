module DampfPhysical

export DampfMode, DampfModel, dim_e, killoran_benchmark_model, KB_CM1_PER_K, thermal_occupation

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

end
