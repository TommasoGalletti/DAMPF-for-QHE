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

end