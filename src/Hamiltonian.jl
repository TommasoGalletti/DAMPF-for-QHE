module Hamiltonian

using LinearAlgebra

export build_hamiltonian, diagonalize_hamiltonian, build_total_hamiltonian, vib_annihilation, vib_creation

"""Build the 4x4 electronic Hamiltonian in the basis |g>, |s1>, |s2>, |s3>."""
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
    dim_e = size(H_e, 1)

    Nv = params.Nv
    ωv = params.ωv
    g  = params.g

    a = vib_annihilation(Nv)
    adag = vib_creation(Nv)

    I_e = Matrix{ComplexF64}(I, dim_e, dim_e)
    I_v = Matrix{ComplexF64}(I, Nv, Nv)

    n_op = adag * a
    H_v = ωv * n_op

    # Two local vibrational modes (paper prototype):
    # Hm = ωv (a1†a1 + a2†a2)
    H_e_ext = kron(H_e, kron(I_v, I_v))
    H_v_ext = kron(I_e, kron(H_v, I_v) + kron(I_v, H_v))

    # Paper-like local vibronic coupling:
    # HI = g |s1><s1| ⊗ (a1 + a1†) + g |s2><s2| ⊗ (a2 + a2†)
    X = a + adag
    P_s1 = zeros(ComplexF64, dim_e, dim_e)
    P_s2 = zeros(ComplexF64, dim_e, dim_e)
    P_s1[2, 2] = 1.0
    P_s2[3, 3] = 1.0

    H_ev = g * kron(P_s1, kron(X, I_v)) +
           g * kron(P_s2, kron(I_v, X))

    return H_e_ext + H_v_ext + H_ev
end

"""Diagonalize H and return eigenpairs sorted by descending energy."""
function diagonalize_hamiltonian(H)

    eig = eigen(H)

    energies = eig.values
    states   = eig.vectors

    # descending order
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