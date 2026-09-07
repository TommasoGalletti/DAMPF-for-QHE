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
Local step: for every block (m,n) and mode q, pick the role gate
(P0/Pdiag/Pket/Pbra) from whether m/n match that mode's owning site, and
apply it in place. No bond-dimension change -- every term here (H_v,
damping, vibronic coupling) is single-mode local.
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
via ITensorMPS `+` with truncation -- the only step where bond dimension
grows (matches the paper's Fig. 2c compression step). `old` is snapshotted
first since every new block can read every old one. Zero sources (e.g. most
blocks at t=0) are skipped explicitly rather than summed.
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
One first-order Lie-Trotter DAMPF step: local step, electronic mixing, trace
renormalization. SVD truncation in the mixing step minimizes Frobenius
error, not trace error, so `Tr(ρ)` drifts each step; every block is
rescaled by `1/Tr(ρ)` at the end. Returns the pre-renormalization trace
(diagnostic: how far this step drifted).
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

"""Triangular-storage variant of `_apply_local_step!`: same gate-selection
logic, restricted to the stored half (m<=n).

Parallelized over (m,n) pairs (`Threads.@threads`, no-op if Julia was
started with a single thread): each pair only ever reads/writes its own
`blocks[m,n]` MPS, over all modes in turn, so different threads never touch
the same object -- no locking needed."""
function _apply_local_step_triangular!(blocks::Matrix{MPS}, model::DampfModel, role_props::Vector{ModeRolePropagators})
    dim = size(blocks, 1)
    pairs = [(m, n) for m in 1:dim for n in m:dim]
    Threads.@threads for idx in eachindex(pairs)
        m, n = pairs[idx]
        for (q, mode) in enumerate(model.modes)
            owner = mode.site
            props = role_props[q]
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
outputs (m,n) with m<=n (`dim*(dim+1)/2` instead of `dim^2`), exploiting
Ô_{n,m}=Ô_{m,n}^dagger (always true for physical ρ, preserved by every term
of this dynamics -- see the SPIEGAZIONE docs). Sources with k>l are
reconstructed via `dagger_block` (cheap, no SVD) instead of being stored and
truncated independently, so the m>n half is exactly the dagger of the
stored half rather than only approximately so.

Parallelized over output (m,n) pairs (`Threads.@threads`, no-op with a
single Julia thread): each pair only reads the immutable `old` snapshot and
`mirrors`, and writes only its own `new_blocks[idx]` slot, so no two threads
ever touch the same memory.

`mirrors` used to be a lazily-populated `Dict` (dagger a source only the
first time some target needed it) -- safe when built one entry at a time by
a single serial loop, but a plain `Dict` is NOT safe under concurrent
`get!` writes from multiple threads. Fixed by splitting into two passes:
first a cheap serial scan that just enumerates which (k,l) mirrors are
needed by ANY (m,n) (no MPS work, only reading `P_e`), then computing
exactly those daggers serially -- so by the time the parallel loop below
starts, `mirrors` is fully built and read-only.
"""
function _apply_electronic_mixing_triangular!(
    blocks::Matrix{MPS}, P_e::AbstractMatrix, siteinds::Vector{<:Index}, swap_gates::Vector{ITensor};
    chi_max::Int, svd_tol::Float64, coeff_tol::Float64=1e-14,
)
    dim = size(blocks, 1)
    old = copy(blocks)  # only entries with k<=l are ever populated/read
    pairs = [(m, n) for m in 1:dim for n in m:dim]

    needed_mirrors = Set{Tuple{Int,Int}}()
    for (m, n) in pairs, k in 1:dim, l in 1:dim
        k > l || continue
        abs(P_e[_flatidx(m, n, dim), _flatidx(k, l, dim)]) < coeff_tol && continue
        push!(needed_mirrors, (k, l))
    end
    mirrors = Dict{Tuple{Int,Int},MPS}(kl => dagger_block(old[kl[2], kl[1]], swap_gates) for kl in needed_mirrors)
    get_src(k, l) = k <= l ? old[k, l] : mirrors[(k, l)]

    new_blocks = Vector{MPS}(undef, length(pairs))
    Threads.@threads for idx in eachindex(pairs)
        m, n = pairs[idx]
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

        new_blocks[idx] = if isempty(terms)
            zero_block(siteinds)
        elseif length(terms) == 1
            coeffs[1] * terms[1]
        else
            +((c * t for (c, t) in zip(coeffs, terms))...; cutoff=svd_tol, maxdim=chi_max)
        end
    end
    for (idx, (m, n)) in enumerate(pairs)
        blocks[m, n] = new_blocks[idx]
    end
    return nothing
end

"""Triangular-storage variant of `one_trotter_step!`: evolves only the m<=n
half of `blocks`. `swap_gates` (`DampfDagger.build_swap_gates`) is
Δt-independent, build it once alongside `role_props`.

The two steps below are internally multithreaded over (m,n) blocks
(`Threads.@threads` in `_apply_local_step_triangular!` /
`_apply_electronic_mixing_triangular!`; a no-op if Julia has one thread).
LAPACK's own internal threading is turned off for their duration (restored
after) -- without this, LAPACK's per-call threads and Julia's per-block
threads would compete for the same cores, at best wasting the parallelism
and at worst running slower than the single-threaded version."""
function one_trotter_step_triangular!(
    blocks::Matrix{MPS}, model::DampfModel, role_props::Vector{ModeRolePropagators},
    P_e::AbstractMatrix, siteinds::Vector{<:Index}, τ::MPS, swap_gates::Vector{ITensor};
    chi_max::Int, svd_tol::Float64=1e-12, coeff_tol::Float64=1e-10,
)
    previous_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        _apply_local_step_triangular!(blocks, model, role_props)
        _apply_electronic_mixing_triangular!(blocks, P_e, siteinds, swap_gates; chi_max=chi_max, svd_tol=svd_tol, coeff_tol=coeff_tol)
    finally
        LinearAlgebra.BLAS.set_num_threads(previous_blas_threads)
    end

    dim = size(blocks, 1)
    tr_val = sum(trace_of_block(τ, blocks[m, m]) for m in 1:dim)
    for m in 1:dim, n in m:dim
        blocks[m, n] = (1 / tr_val) * blocks[m, n]
    end

    return tr_val
end

end
