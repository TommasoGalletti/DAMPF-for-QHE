module Lindblad

using LinearAlgebra

export JumpOperator,
       build_jump_operator,
       transform_to_exciton_basis,
       transform_to_site_basis,
       build_hot_operators,
       build_load_operator,
       build_cold_operators,
       extend_operator_to_vib,
       build_hot_operators_vib,
       build_load_operator_vib,
       build_cold_operators_vib,
             build_vibrational_damping_operators

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

function build_jump_operator(i, j, dim)
    A = zeros(ComplexF64, dim, dim)
    A[i, j] = 1.0
    return A
end

"""
Transform an operator from site basis to exciton basis.
"""
function transform_to_exciton_basis(A_site, U)
    return U' * A_site * U
end

"""
Transform an operator from exciton basis to site basis.
"""
function transform_to_site_basis(A_exc, U)
    return U * A_exc * U'
end

# =========================
# VIBRONIC EXTENSION
# =========================

function extend_operator_to_vib(A_e, Nv)
    I_v = Matrix{ComplexF64}(I, Nv, Nv)
    return kron(A_e, kron(I_v, I_v))
end

# =========================
# HOT BATH
# =========================

function build_hot_operators(U; γH, nH)

    dim = size(U, 1)

    A_up_exc   = build_jump_operator(1, dim, dim)
    A_down_exc = build_jump_operator(dim, 1, dim)

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

function build_load_operator(U; ΓL)

    dim = size(U, 1)

    A_exc = build_jump_operator(dim, 3, dim)
    A = transform_to_site_basis(A_exc, U)

    return JumpOperator(A, ΓL)
end

# =========================
# COLD BATH
# =========================

function build_cold_operators(U; γC, nC_matrix)

    dim = size(U, 1)
    ops = JumpOperator[]

    for i in 1:dim
        for j in 1:dim

            if i != j && i < j

                n = nC_matrix[i, j]

                # emission
                A_exc = build_jump_operator(j, i, dim)
                A = transform_to_site_basis(A_exc, U)
                push!(ops, JumpOperator(A, γC * (n + 1)))

                # absorption
                A_exc_rev = build_jump_operator(i, j, dim)
                A_rev = transform_to_site_basis(A_exc_rev, U)
                push!(ops, JumpOperator(A_rev, γC * n))
            end
        end
    end

    return ops
end

# =========================
# VIBRONIC VERSIONS
# =========================

function build_hot_operators_vib(U; γH, nH, Nv)

    ops = build_hot_operators(U; γH=γH, nH=nH)

    return [JumpOperator(extend_operator_to_vib(op.A, Nv), op.rate) for op in ops]
end

function build_load_operator_vib(U; ΓL, Nv)

    op = build_load_operator(U; ΓL=ΓL)

    return JumpOperator(extend_operator_to_vib(op.A, Nv), op.rate)
end

function build_cold_operators_vib(U; γC, nC_matrix, Nv)

    ops = build_cold_operators(U; γC=γC, nC_matrix=nC_matrix)

    return [JumpOperator(extend_operator_to_vib(op.A, Nv), op.rate) for op in ops]
end

# =========================
# VIBRATIONAL DAMPING
# =========================

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

"""
Vibrational thermal damping for two local modes.

Returns 4 jump operators:
- mode 1 damping/excitation
- mode 2 damping/excitation
"""
function build_vibrational_damping_operators(dim_e, Nv; γM, nM=nothing, ωv=nothing, TM=nothing, kB=0.69503476)

    nM_val = nM
    if nM_val === nothing
        if ωv === nothing || TM === nothing
            error("Pass nM directly, or provide both ωv and TM.")
        end
        nM_val = 1 / (exp(ωv / (kB * TM)) - 1)
    end

    I_e = Matrix{ComplexF64}(I, dim_e, dim_e)
    I_v = Matrix{ComplexF64}(I, Nv, Nv)
    a = vib_annihilation(Nv)
    adag = vib_creation(Nv)

    # mode 1 acts on first vibrational subspace
    A1_down = kron(I_e, kron(a, I_v))
    A1_up = kron(I_e, kron(adag, I_v))

    # mode 2 acts on second vibrational subspace
    A2_down = kron(I_e, kron(I_v, a))
    A2_up = kron(I_e, kron(I_v, adag))

    return [
        JumpOperator(A1_down, γM * (nM_val + 1)),
        JumpOperator(A1_up, γM * nM_val),
        JumpOperator(A2_down, γM * (nM_val + 1)),
        JumpOperator(A2_up, γM * nM_val)
    ]
end

end