module DampfTrotter

using ITensors
using ITensorMPS
using LinearAlgebra
using Main.DampfSystem.DampfPhysical: DampfModel
using Main.DampfSystem.DampfLocalGenerators: ModeRolePropagators
using Main.DampfSystem.DampfTraceBasis: zero_block, trace_of_block
using Main.DampfSystem.DampfDagger: dagger_block

export one_trotter_step!, one_trotter_step_triangular!

_flatidx(m::Int, n::Int, dim::Int) = (n - 1) * dim + m

"""
Local (single-mode) step: for every electronic block (m,n) and every mode
q, select the role gate (P0/Pdiag/Pket/Pbra) depending on whether m and/or n
equal that mode's owning electronic site, and apply it in place. No bond
dimension change here; every physical term (H_v, damping, vibronic coupling)
is single-mode local, so this step is order-independent across (m,n,q).
"""
function _apply_local_step!(blocks::Matrix{MPS}, model::DampfModel, role_props::Vector{ModeRolePropagators})
    dim = size(blocks, 1)
    for (q, mode) in enumerate(model.modes)
        owner = mode.site
        props = role_props[q]
        for m in 1:dim, n in 1:dim
            gate = if m == owner && n == owner
                props.Pdiag
            elseif m == owner
                props.Pket
            elseif n == owner
                props.Pbra
            else
                props.P0
            end
            blocks[m, n][q] = noprime(gate * blocks[m, n][q])
        end
    end
    return nothing
end

"""
Electronic-mixing step: `Ô'_{m,n} = Σ_{k,l} P_e[flat(m,n),flat(k,l)] · Ô_{k,l}`,
via ITensorMPS `+` with truncation -- the only step that grows/truncates
bond dimension (matches the paper's own Fig. 2c compression step). `old` is
snapshotted first since every new block potentially reads every old one.
ρ(0) is a pure product state (only block (1,1) nonzero), so "most source
terms are exactly zero" is the very first step, not a rare edge case --
skip identically-zero sources explicitly rather than paying for a zero-MPS
sum.
"""
function _apply_electronic_mixing!(
    blocks::Matrix{MPS}, P_e::AbstractMatrix, siteinds::Vector{<:Index};
    chi_max::Int, svd_tol::Float64, coeff_tol::Float64=1e-14,
)
    dim = size(blocks, 1)
    old = copy(blocks)

    for m in 1:dim, n in 1:dim
        terms = MPS[]
        coeffs = ComplexF64[]
        for k in 1:dim, l in 1:dim
            c = P_e[_flatidx(m, n, dim), _flatidx(k, l, dim)]
            abs(c) < coeff_tol && continue
            norm(old[k, l]) == 0 && continue
            push!(terms, old[k, l])
            push!(coeffs, c)
        end

        if isempty(terms)
            blocks[m, n] = zero_block(siteinds)
        elseif length(terms) == 1
            blocks[m, n] = coeffs[1] * terms[1]
        else
            blocks[m, n] = +((c * t for (c, t) in zip(coeffs, terms))...; cutoff=svd_tol, maxdim=chi_max)
        end
    end
    return nothing
end

"""
One (first-order, Lie-Trotter) DAMPF step: the local step, then the
electronic-mixing step, then trace renormalization. SVD truncation in the
mixing step does not preserve Tr(ρ) (it minimizes Hilbert-Schmidt/Frobenius
error, not trace error -- the same root cause discovered in the old
hand-rolled architecture), so every block is rescaled by `1/Tr(ρ)` at the
end. Returns the pre-renormalization trace (diagnostic: how much the
truncation drifted this step).
"""
function one_trotter_step!(
    blocks::Matrix{MPS}, model::DampfModel, role_props::Vector{ModeRolePropagators},
    P_e::AbstractMatrix, siteinds::Vector{<:Index}, τ::MPS;
    chi_max::Int, svd_tol::Float64=1e-12, coeff_tol::Float64=1e-10,
)
    _apply_local_step!(blocks, model, role_props)
    _apply_electronic_mixing!(blocks, P_e, siteinds; chi_max=chi_max, svd_tol=svd_tol, coeff_tol=coeff_tol)

    dim = size(blocks, 1)
    tr_val = sum(trace_of_block(τ, blocks[m, m]) for m in 1:dim)
    for m in 1:dim, n in 1:dim
        blocks[m, n] = (1 / tr_val) * blocks[m, n]
    end

    return tr_val
end

"""
Triangular-storage variant of `_apply_local_step!`: only touches blocks with
m<=n (the m>n half is never allocated in the triangular-storage `blocks`
matrix -- see `one_trotter_step_triangular!`). Same gate-selection logic,
restricted to the stored half.
"""
function _apply_local_step_triangular!(blocks::Matrix{MPS}, model::DampfModel, role_props::Vector{ModeRolePropagators})
    dim = size(blocks, 1)
    for (q, mode) in enumerate(model.modes)
        owner = mode.site
        props = role_props[q]
        for m in 1:dim, n in m:dim
            gate = if m == owner && n == owner
                props.Pdiag
            elseif m == owner
                props.Pket
            elseif n == owner
                props.Pbra
            else
                props.P0
            end
            blocks[m, n][q] = noprime(gate * blocks[m, n][q])
        end
    end
    return nothing
end

"""
Triangular-storage variant of `_apply_electronic_mixing!`: only computes
output blocks (m,n) with m<=n (`dim*(dim+1)/2` instead of `dim^2` -- 10 vs
16 for dim_e=4). Exploits Ô_{n,m}=Ô_{m,n}^dagger, which always holds for a
physical ρ and is preserved by every piece of this dynamics (Hamiltonian
commutators and Lindblad dissipators are Hermiticity-preserving by
construction; see the DAMPF SPIEGAZIONE docs for the full argument).
Source blocks (k,l) with k>l are reconstructed once per step via
`dagger_block` (cheap: per-site conjugate + fixed permutation gate, no SVD)
instead of being stored/truncated independently -- this doesn't just save
work, it makes the m>n half EXACTLY the dagger of the stored half. Today's
unoptimized `_apply_electronic_mixing!` truncates both halves independently
via separate SVDs, so hermiticity there only holds approximately (drifts
slowly, like `Tr(ρ)` does before renormalization).
"""
function _apply_electronic_mixing_triangular!(
    blocks::Matrix{MPS}, P_e::AbstractMatrix, siteinds::Vector{<:Index}, swap_gates::Vector{ITensor};
    chi_max::Int, svd_tol::Float64, coeff_tol::Float64=1e-14,
)
    dim = size(blocks, 1)
    old = copy(blocks)  # only entries with k<=l are ever populated/read

    # Precompute each not-directly-stored mirror source ONCE (cheap, no SVD)
    # rather than re-deriving it on every (m,n) output that happens to need it.
    mirrors = Dict{Tuple{Int,Int},MPS}()
    for l in 1:dim, k in (l + 1):dim
        mirrors[(k, l)] = dagger_block(old[l, k], swap_gates)
    end
    get_src(k, l) = k <= l ? old[k, l] : mirrors[(k, l)]

    for m in 1:dim, n in m:dim
        terms = MPS[]
        coeffs = ComplexF64[]
        for k in 1:dim, l in 1:dim
            c = P_e[_flatidx(m, n, dim), _flatidx(k, l, dim)]
            abs(c) < coeff_tol && continue
            src = get_src(k, l)
            norm(src) == 0 && continue
            push!(terms, src)
            push!(coeffs, c)
        end

        if isempty(terms)
            blocks[m, n] = zero_block(siteinds)
        elseif length(terms) == 1
            blocks[m, n] = coeffs[1] * terms[1]
        else
            blocks[m, n] = +((c * t for (c, t) in zip(coeffs, terms))...; cutoff=svd_tol, maxdim=chi_max)
        end
    end
    return nothing
end

"""
Triangular-storage variant of `one_trotter_step!`: stores/evolves only the
m<=n half of `blocks` (see `_apply_electronic_mixing_triangular!` for why
this is exact, not an approximation). `swap_gates` (from
`DampfDagger.build_swap_gates`, one per mode) is Δt-independent, build it
once alongside `role_props`. Trace renormalization is unaffected in spirit
(still uses only the diagonal, always stored) but now loops m<=n only.
"""
function one_trotter_step_triangular!(
    blocks::Matrix{MPS}, model::DampfModel, role_props::Vector{ModeRolePropagators},
    P_e::AbstractMatrix, siteinds::Vector{<:Index}, τ::MPS, swap_gates::Vector{ITensor};
    chi_max::Int, svd_tol::Float64=1e-12, coeff_tol::Float64=1e-10,
)
    _apply_local_step_triangular!(blocks, model, role_props)
    _apply_electronic_mixing_triangular!(blocks, P_e, siteinds, swap_gates; chi_max=chi_max, svd_tol=svd_tol, coeff_tol=coeff_tol)

    dim = size(blocks, 1)
    tr_val = sum(trace_of_block(τ, blocks[m, m]) for m in 1:dim)
    for m in 1:dim, n in m:dim
        blocks[m, n] = (1 / tr_val) * blocks[m, n]
    end

    return tr_val
end

end
