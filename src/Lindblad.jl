module Lindblad

using LinearAlgebra

export JumpOperator,
       build_jump_operator,
       transform_to_site_basis,
       build_hot_operators,
       build_load_operator,
       build_cold_operators

# =========================
# STRUCT
# =========================

struct JumpOperator
    A::Matrix{ComplexF64}
    rate::Float64
end

# =========================
# BASIC OPERATORS
# =========================

"""
Operatore |i><j| in exciton basis
"""
function build_jump_operator(i, j, dim)
    A = zeros(ComplexF64, dim, dim)
    A[i,j] = 1.0
    return A
end

"""
Trasformazione exciton → site basis
"""
function transform_to_site_basis(A_exc, U)
    return U * A_exc * U'
end

# =========================
# HOT BATH
# =========================

# ε₁ ↔ g  → (1 ↔ dim)
function build_hot_operators(U; γH, nH)

    dim = size(U,1)

    A_up_exc   = build_jump_operator(1, dim, dim)   # |ε1><g|
    A_down_exc = build_jump_operator(dim, 1, dim)   # |g><ε1|

    A_up   = transform_to_site_basis(A_up_exc, U)
    A_down = transform_to_site_basis(A_down_exc, U)

    return [
        JumpOperator(A_up,   γH * nH),
        JumpOperator(A_down, γH * (nH + 1))
    ]
end

# =========================
# LOAD
# =========================

# ε_low → g  → (3 → dim)
function build_load_operator(U; ΓL)

    dim = size(U,1)

    A_exc = build_jump_operator(dim, 3, dim)  # |g><ε_low|

    A = transform_to_site_basis(A_exc, U)

    return JumpOperator(A, ΓL)
end

# =========================
# COLD BATH
# =========================

"""
Cold bath: condizione ε_i > ε_j (no doppio counting)
"""
function build_cold_operators(U; γC, nC_matrix)

    dim = size(U,1)
    ops = JumpOperator[]

    for i in 1:dim
        for j in 1:dim

            if i != j && i < j

                n = nC_matrix[i,j]

                # -------------------------
                # emissione i → j
                # |ε_j><ε_i|
                # -------------------------
                A_exc = build_jump_operator(j, i, dim)
                A = transform_to_site_basis(A_exc, U)

                push!(ops, JumpOperator(A, γC * (n + 1)))

                # -------------------------
                # assorbimento j → i
                # |ε_i><ε_j|
                # -------------------------
                A_exc_rev = build_jump_operator(i, j, dim)
                A_rev = transform_to_site_basis(A_exc_rev, U)

                push!(ops, JumpOperator(A_rev, γC * n))
            end
        end
    end

    return ops
end

end