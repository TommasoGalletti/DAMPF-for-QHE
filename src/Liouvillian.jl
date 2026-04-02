module Liouvillian

using LinearAlgebra
using ..Lindblad

export build_liouvillian

# =========================
# COMMUTATOR
# =========================

function commutator_superoperator(H)

    dim = size(H,1)
    I_d = Matrix{ComplexF64}(I, dim, dim)

    return -1im * (kron(I_d, H) - kron(transpose(H), I_d))
end

# =========================
# SINGLE DISSIPATOR
# =========================

function dissipator_superoperator(A)

    dim = size(A,1)
    I_d = Matrix{ComplexF64}(I, dim, dim)

    AdagA = A' * A

    term1 = kron(conj(A), A)
    term2 = kron(I_d, AdagA)
    term3 = kron(transpose(AdagA), I_d)

    return term1 - 0.5 * term2 - 0.5 * term3
end

# =========================
# LIOUVILLIAN
# =========================

"""
Costruisce Liouvilliana completa (16x16)
"""
function build_liouvillian(H, jump_operators::Vector{JumpOperator})

    dim = size(H,1)
    L = commutator_superoperator(H)

    for op in jump_operators
        L += op.rate * dissipator_superoperator(op.A)
    end

    return L
end

end