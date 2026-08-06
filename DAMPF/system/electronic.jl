module DampfElectronic

using LinearAlgebra
include("../../src/Lindblad.jl")
include("../../src/Liouvillian.jl")
using .Lindblad
using .Liouvillian
using Main.DampfSystem.DampfPhysical: DampfModel

export build_electronic_hamiltonian, diagonalize_electronic, build_electronic_dissipators,
       build_electronic_mixing_propagator

"""
Build the dense (dim_e x dim_e) electronic Hamiltonian in site basis
|g>,|s1>,...,|sN>, independently from `model` (mirrors, but does not call,
`Hamiltonian.build_hamiltonian` -- cross-checked in check/01).
"""
function build_electronic_hamiltonian(model::DampfModel)
    N = length(model.E)
    dim = N + 1
    H = zeros(ComplexF64, dim, dim)
    H[1, 1] = model.Eg
    for i in 1:N
        H[i+1, i+1] = model.E[i]
    end
    for i in 1:N, j in 1:N
        if i != j
            H[i+1, j+1] = model.J[i, j]
        end
    end
    return H
end

"""
Diagonalize a Hermitian electronic Hamiltonian, sorted by descending energy
-- matches `Hamiltonian.diagonalize_hamiltonian`'s convention exactly.
"""
function diagonalize_electronic(H_e::AbstractMatrix)
    eig = eigen(H_e)
    idx = sortperm(real.(eig.values), rev=true)
    return eig.values[idx], eig.vectors[:, idx]
end

function _cold_occupations(energies, TC, kB)
    dim = length(energies)
    nC = zeros(dim, dim)
    for i in 1:dim, j in 1:dim
        if i != j
            ΔE = energies[i] - energies[j]
            if ΔE > 1e-8
                nC[i, j] = 1 / (exp(ΔE / (kB * TC)) - 1)
            end
        end
    end
    return nC
end

"""
Build every electronic Lindblad channel (hot bath, load, cold bath) in site
basis. `idx_low`/`idx_g` mirror `Observables.jl`'s convention (load feeds
`idx_low`, ground exciton is `idx_g`) -- for the Killoran benchmark these are
`dim_e-1` and `dim_e`, but this function makes no assumption about their
values, unlike `Lindblad.build_load_operator` (which hardcodes exciton index
3 and is therefore only valid for that one N=3 benchmark). Reuses
`Lindblad.build_hot_operators`/`build_cold_operators` unchanged.

Returns `(ops, energies, U)`: the jump operators, and the electronic
eigendecomposition used to build them (exposed so `check/` can cross-verify
against `Hamiltonian.diagonalize_hamiltonian`).
"""
function build_electronic_dissipators(
    H_e::AbstractMatrix;
    idx_low::Int, idx_g::Int,
    γH::Float64, nH::Float64,
    ΓL::Float64,
    γC::Float64, TC::Float64,
    kB::Float64,
)
    energies, U = diagonalize_electronic(H_e)
    dim = size(H_e, 1)
    nC = _cold_occupations(energies, TC, kB)

    ops = Lindblad.JumpOperator[]
    append!(ops, Lindblad.build_hot_operators(U; γH=γH, nH=nH))
    A_load = Lindblad.transform_to_site_basis(Lindblad.build_jump_operator(idx_g, idx_low, dim), U)
    push!(ops, Lindblad.JumpOperator(A_load, ΓL))
    append!(ops, Lindblad.build_cold_operators(U; γC=γC, nC_matrix=nC))

    return (ops=ops, energies=energies, U=U)
end

"""
The electronic-label mixing propagator: `exp(Δt * L)` where `L` is the
standard vectorized Lindblad generator restricted to the `dim_e`-dimensional
electronic Hilbert space, built via `Liouvillian.build_liouvillian`
completely unmodified. This is the ONLY generator that mixes different
`(m,n)` blocks together -- every other physical term (mode Hamiltonians,
vibrational damping, the vibronic coupling) is single-mode local, handled in
`local_generators.jl`.
"""
function build_electronic_mixing_propagator(H_e::AbstractMatrix, dissipators, Δt::Float64)
    L = Liouvillian.build_liouvillian(Matrix{ComplexF64}(H_e), dissipators)
    return exp(Matrix(Δt .* L))
end

end
