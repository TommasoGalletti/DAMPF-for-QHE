module Solver

using LinearAlgebra

export steady_state

"""
Costruisce vettore che implementa Tr(ρ)=1 nello spazio vettorializzato
"""
function trace_vector(dim)

    v = zeros(ComplexF64, dim^2)

    for i in 1:dim
        idx = (i-1)*dim + i  # posizione diagonale in vec
        v[idx] = 1.0
    end

    return v
end

"""
Risolve Lρ = 0 con vincolo Tr(ρ)=1
"""
function steady_state(L)

    dim2 = size(L,1)
    dim  = Int(sqrt(dim2))

    # Copia
    L_mod = copy(L)

    b = zeros(ComplexF64, dim2)

    # Sostituisci ultima riga con condizione di traccia
    L_mod[end, :] .= trace_vector(dim)
    b[end] = 1.0

    # Risolvi sistema lineare
    ρ_vec = L_mod \ b

    # reshape → matrice densità
    ρ = reshape(ρ_vec, dim, dim)
    #ρ = (ρ + ρ') / 2  # Assicura che ρ sia hermitiana

    return ρ
end

end