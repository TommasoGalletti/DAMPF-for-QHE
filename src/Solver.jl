module Solver

using LinearAlgebra
using SparseArrays

export steady_state

"""
Build the vectorized trace constraint Tr(ρ)=1.
"""
function trace_vector(dim)

    v = zeros(ComplexF64, dim^2)

    for i in 1:dim
        idx = (i - 1) * dim + i  # diagonal position in vec(ρ)
        v[idx] = 1.0
    end

    return v
end

"""
Solve Lρ = 0 with the trace constraint Tr(ρ)=1.
"""
function steady_state(L)

    dim2 = size(L, 1)
    dim  = Int(sqrt(dim2))

    # Copy before enforcing the trace row constraint
    L_mod = copy(L)

    b = zeros(ComplexF64, dim2)

    # Replace last row with trace condition
    if issparse(L_mod)
        L_mod[end, :] = sparse(trace_vector(dim)')
    else
        L_mod[end, :] .= trace_vector(dim)
    end
    b[end] = 1.0

    # Solve linear system
    ρ_vec = L_mod \ b

    # reshape -> density matrix
    ρ = reshape(ρ_vec, dim, dim)

    return ρ
end

end