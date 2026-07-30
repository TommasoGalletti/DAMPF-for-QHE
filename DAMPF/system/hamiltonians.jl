module DampfHamiltonians

using Main.DampfSystem.DAMPFPhysical: DampfPhysicalParams
using LinearAlgebra

export build_He, build_Hv_local, build_Hev_local

"""
Build electronic Hamiltonian `H_e` in basis |g>, |s1>, ..., |sN|.
Returns dense matrix of size (N+1)x(N+1).
"""
function build_He(phys::DampfPhysicalParams; ground_energy::Float64 = 0.0)
    N = phys.n_sites
    dim = N + 1
    H = zeros(ComplexF64, dim, dim)
    H[1,1] = ground_energy
    for i in 1:N
        H[i+1, i+1] = phys.site_energies[i]
    end
    for i in 1:N, j in 1:N
        if i != j
            H[i+1, j+1] = phys.electronic_couplings[i,j]
        end
    end
    return H
end

"""
Return local vibrational Hamiltonians for each mode as `ω * n` on the single-mode Nb space.
Returns a vector of NamedTuples with fields (site, mode, idx, Hv_local, Nb).
"""
function build_Hv_local(phys::DampfPhysicalParams)
    N = phys.n_sites
    Q = phys.n_modes_per_site
    Nb = phys.local_truncation
    modes = Vector{NamedTuple}(undef, N * Q)
    k = 1
    for i in 1:N
        for q in 1:Q
            a = zeros(ComplexF64, Nb, Nb)
            for n in 2:Nb
                a[n-1,n] = sqrt(n-1)
            end
            adag = a'
            n_op = adag * a
            ω = phys.vib_frequencies[q]
            Hv = ω * n_op
            modes[k] = (site=i, mode=q, idx=k, Hv_local=Hv, Nb=Nb)
            k += 1
        end
    end
    return modes
end

"""
Return vibronic coupling descriptors for each (site,mode): g_{nq} |s_n><s_n| ⊗ (a + a†)
The function returns a vector of NamedTuples with (site, mode, idx, g, X_local)
where X_local is the single-mode operator (a + a†) on the local Nb space.
"""
function build_Hev_local(phys::DampfPhysicalParams)
    N = phys.n_sites
    Q = phys.n_modes_per_site
    Nb = phys.local_truncation
    couplings = Vector{NamedTuple}(undef, N * Q)
    k = 1
    for i in 1:N
        for q in 1:Q
            a = zeros(ComplexF64, Nb, Nb)
            for n in 2:Nb
                a[n-1,n] = sqrt(n-1)
            end
            X = a + a'
            g = phys.vibronic_couplings[i,q]
            couplings[k] = (site=i, mode=q, idx=k, g=g, X_local=X)
            k += 1
        end
    end
    return couplings
end

end
