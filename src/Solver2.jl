function steady_state(L)

    dim2 = size(L,1)
    dim  = Int(sqrt(dim2))

    L_mod = copy(L)
    b = zeros(ComplexF64, dim2)

    # scelgo riga da sostituire
    row = dim2

    L_mod[row, :] .= trace_vector(dim)
    b[row] = 1.0

    ρ_vec = L_mod \ b
    ρ = reshape(ρ_vec, dim, dim)

    # hermiticity fix
    ρ = (ρ + ρ') / 2

    # normalizzazione
    ρ /= tr(ρ)

    return ρ
end