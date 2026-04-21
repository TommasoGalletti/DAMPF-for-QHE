module Liouvillian

using LinearAlgebra
using SparseArrays
using ..Lindblad

export build_liouvillian

# =========================
# COMMUTATOR
# =========================

function commutator_superoperator(H)

    dim = size(H,1)
    I_d = sparse(I, dim, dim)
    Hs = sparse(H)

    return -1im * (kron(I_d, Hs) - kron(transpose(Hs), I_d))
end

# =========================
# SINGLE DISSIPATOR
# =========================

function dissipator_superoperator(A)

    dim = size(A,1)
    I_d = sparse(I, dim, dim)
    As = sparse(A)

    AdagA = As' * As

    term1 = kron(conj(As), As)
    term2 = kron(I_d, AdagA)
    term3 = kron(transpose(AdagA), I_d)

    return term1 - 0.5 * term2 - 0.5 * term3
end

# =========================
# LIOUVILLIAN
# =========================

"""
Build the full Liouvillian superoperator for the given Hamiltonian and jump operators.
"""
function build_liouvillian(H, jump_operators::Vector{JumpOperator})

    dim = size(H, 1)
    L = commutator_superoperator(H)

    for op in jump_operators
        L += op.rate * dissipator_superoperator(op.A)
    end

    return L
end

end