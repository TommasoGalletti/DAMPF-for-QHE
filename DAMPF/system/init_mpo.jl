module DampfInitMPO

using ITensors
using LinearAlgebra
using Main.DampfSystem.DAMPFPhysical: DampfPhysicalParams

export build_rho0_product_mpo

function _thermal_diag(Nb::Int, ω::Float64, T::Float64)
    if T <= 0.0
        ρ = zeros(Float64, Nb, Nb)
        ρ[1,1] = 1.0
        return ρ
    end
    β = 1.0 / T
    vals = [exp(-β * ω * (n-1)) for n in 1:Nb]
    Z = sum(vals)
    vals .= vals ./ Z
    ρ = zeros(Float64, Nb, Nb)
    for n in 1:Nb
        ρ[n,n] = vals[n]
    end
    return ρ
end

"""
Build a product initial density operator rho0 as an ITensors MPO (product state, bond dim 1).

Returns a Dict with fields:
- `:mpo` => vector of `ITensor` objects (one per site)
- `:site_inds` => vector of site `Index`
- `:site_dims` => vector of local Hilbert dims

Behavior:
- electronic site is first and has dimension `N+1` (ground + N excited sites)
- then follow N*Q vibrational modes (one tensor per mode) with truncation `Nb`

Options:
- `vib_state` = :vacuum or :thermal
- `T` temperature for thermal vibrational states
"""
function build_rho0_product_mpo(phys::DampfPhysicalParams; vib_state::Symbol = :vacuum, T::Float64 = 0.0)
    N = phys.n_sites
    Q = phys.n_modes_per_site
    Nb = phys.local_truncation

    # site dims: first electronic (N+1), then N*Q vibrational sites (each Nb)
    dims = Int[]
    push!(dims, N + 1)
    for i in 1:(N*Q)
        push!(dims, Nb)
    end

    # local density matrices
    local_rhos = Vector{Matrix{Float64}}(undef, length(dims))
    # electronic ground projector |g><g|
    ρe = zeros(Float64, N+1, N+1)
    ρe[1,1] = 1.0
    local_rhos[1] = ρe

    # vibrational local states
    k = 2
    for site in 1:N
        for q in 1:Q
            if vib_state == :vacuum
                local_rhos[k] = _thermal_diag(Nb, phys.vib_frequencies[q], 0.0)
            else
                local_rhos[k] = _thermal_diag(Nb, phys.vib_frequencies[q], T)
            end
            k += 1
        end
    end

    # Build MPO as array of ITensors with bond dim 1 (product operator)
    siteinds = [Index(d, "s$i") for (i,d) in enumerate(dims)]
    mpo_tensors = Vector{ITensor}(undef, length(dims))
    # link indices of dimension 1
    links = [Index(1, "link$i") for i in 1:(length(dims)-1)]

    for i in 1:length(dims)
        s = siteinds[i]
        sp = prime(s)
        linkL = i == 1 ? Index(1, "l0") : links[i-1]
        linkR = i == length(dims) ? Index(1, "l$(length(dims))") : links[i]
        T = ITensor(linkL, linkR, s, sp)
        ρloc = local_rhos[i]
        for a in 1:size(ρloc,1), b in 1:size(ρloc,2)
            T[linkL=>1, linkR=>1, s=>a, sp=>b] = ρloc[a,b]
        end
        mpo_tensors[i] = T
    end

    return Dict(:mpo=>mpo_tensors, :site_inds=>siteinds, :site_dims=>dims)
end

end
