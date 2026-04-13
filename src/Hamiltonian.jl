module Hamiltonian

using LinearAlgebra

export build_hamiltonian, diagonalize_hamiltonian

"""
Hamiltoniana 4x4 - base:
|g>, |s1>, |s2>, |s3>
"""
function build_hamiltonian(params)

    Eg = params.Eg
    E  = params.E
    J  = params.J

    H = zeros(ComplexF64, 4, 4)

    # Ground
    H[1,1] = Eg

    # Excited states
    for i in 1:3
        H[i+1, i+1] = E[i]
    end

    # Couplings
    for i in 1:3
        for j in 1:3
            if i != j
                H[i+1, j+1] = J[i,j]
            end
        end
    end

    return H
end

function build_total_hamiltonian(params)

    H_e = build_hamiltonian(params)
    dim_e = size(H_e,1)

    Nv = params.Nv
    ωv = params.ωv
    g  = params.g

    a    = vib_annihilation(Nv)
    adag = vib_creation(Nv)

    I_e = Matrix{ComplexF64}(I, dim_e, dim_e)
    I_v = Matrix{ComplexF64}(I, Nv, Nv)

    H_v = ωv * (adag * a)

    H_e_ext = kron(H_e, I_v)
    H_v_ext = kron(I_e, H_v)

    # coupling (SITE BASIS)
    A_site = zeros(ComplexF64, dim_e, dim_e)
    A_site[2,3] = 1.0
    A_site[3,2] = 1.0

    H_ev = g * kron(A_site, (a + adag))

    return H_e_ext + H_v_ext + H_ev
end

"""
Diagonalizzo (Lindblad is in base degli eccitoni) e ordino eigenstuff:
ε₁ ≥ ε₂ ≥ ε₃ ≥ ε₀
"""
function diagonalize_hamiltonian(H)

    eig = eigen(H)

    energies = eig.values
    states   = eig.vectors

    # ordinamento decrescente
    idx = sortperm(energies, rev=true)

    energies_sorted = energies[idx]
    states_sorted   = states[:, idx]

    return energies_sorted, states_sorted
end

function vib_annihilation(N)
    a = zeros(ComplexF64, N, N)
    for n in 2:N
        a[n-1, n] = sqrt(n-1)
    end
    return a
end

function vib_creation(N)
    return vib_annihilation(N)'
end

end