module DAMPFPhysical

export DampfPhysicalParams, default_physical_params

"""
Physical parameters (symbols mapped to paper notation):

- `N`: number of electronic sites
- `Q`: number of local modes per site
- `Nb`: truncation for each oscillator
- `Ω_n`: site energies (vector length N)
- `J_{mn}`: electronic couplings (N x N)
- `ω_q`: vibrational frequencies (length Q)
- `g_{nq}`: vibronic couplings (N x Q)
- `γ_q`: damping rates for pseudomodes (length Q)
- `T`: bath temperature
"""
struct DampfPhysicalParams
    n_sites::Int          # N
    n_modes_per_site::Int # Q
    local_truncation::Int # Nb
    site_energies::Vector{Float64} # Ω_n
    electronic_couplings::Matrix{Float64} # J_{mn}
    vib_frequencies::Vector{Float64} # ω_q
    vibronic_couplings::Matrix{Float64} # g_{nq}
    damping_rates::Vector{Float64} # γ_q
    bath_temperature::Float64 # T
end

function default_physical_params()
    N = 3
    Q = 2
    Nb = 6
    site_energies = [300.0, 300.0, 0.0]
    J = [0.0 100.0 0.0;
         100.0 0.0 0.0;
         0.0 0.0 0.0]
    ω = fill(200.0, Q)
    g = fill(55.0, N, Q)
    γ = fill(1.0/500.0, Q)
    T = 300.0

    return DampfPhysicalParams(N, Q, Nb, site_energies, J, ω, g, γ, T)
end

end
