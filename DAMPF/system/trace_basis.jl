module DampfTraceBasis

using ITensors
using ITensorMPS
using Main.DampfSystem.DampfPhysical: DampfMode

export product_mps_from_vectors, build_trace_mps, trace_of_block, zero_block

"""
Build a bond-dimension-1 MPS from explicit per-mode local vectors (each
`local_vectors[q]` indexed `1:dim(siteinds[q])`).
"""
function product_mps_from_vectors(siteinds::Vector{<:Index}, local_vectors::Vector{Vector{ComplexF64}})
    M = length(siteinds)
    if M == 1
        s = siteinds[1]
        v = local_vectors[1]
        T = ITensor(s)
        for k in 1:dim(s)
            T[s=>k] = v[k]
        end
        return MPS([T])
    end

    # Only M-1 INTERIOR links -- no dangling boundary links at the two ends,
    # matching standard MPS convention (a dangling dim-1 link at the boundary
    # would otherwise be mistaken for a second site index by ITensorMPS's
    # dot/inner, since it is not shared with any neighboring tensor).
    links = [Index(1, "Link,l$i") for i in 1:(M - 1)]
    tensors = ITensor[]
    for q in 1:M
        s = siteinds[q]
        v = local_vectors[q]
        if q == 1
            T = ITensor(s, links[1])
            for k in 1:dim(s)
                T[s=>k, links[1]=>1] = v[k]
            end
        elseif q == M
            T = ITensor(links[M-1], s)
            for k in 1:dim(s)
                T[links[M-1]=>1, s=>k] = v[k]
            end
        else
            T = ITensor(links[q-1], s, links[q])
            for k in 1:dim(s)
                T[links[q-1]=>1, s=>k, links[q]=>1] = v[k]
            end
        end
        push!(tensors, T)
    end
    return MPS(tensors)
end

"""
The "trace functional" MPS τ: `Tr(Ô) = inner(τ, Ô)` for any block Ô expressed
in the matrix-unit basis (`k=(b-1)*Nb+a` per mode). Per mode, 1 at each
diagonal matrix-unit position (`k=(a-1)*Nb+a`), 0 elsewhere -- since
Tr(x̂_j1⊗...⊗x̂_jM) = Π_q Tr(x̂_jq) for matrix-unit basis operators, and
Tr(E_ab) = δ_ab, this product structure is exactly a bond-dim-1 MPS.
"""
function build_trace_mps(siteinds::Vector{<:Index}, modes::Vector{DampfMode})
    vectors = Vector{Vector{ComplexF64}}(undef, length(modes))
    for (q, mode) in enumerate(modes)
        Nb = mode.Nb
        v = zeros(ComplexF64, Nb^2)
        for a in 1:Nb
            v[(a - 1) * Nb + a] = 1.0
        end
        vectors[q] = v
    end
    return product_mps_from_vectors(siteinds, vectors)
end

trace_of_block(τ::MPS, ψ::MPS) = inner(τ, ψ)

"""
An explicit all-zero block (bond dim 1), used when a Trotter step's
electronic-mixing sum has no surviving source terms (e.g. every block but
(1,1) at t=0).
"""
function zero_block(siteinds::Vector{<:Index})
    vectors = [zeros(ComplexF64, dim(s)) for s in siteinds]
    return product_mps_from_vectors(siteinds, vectors)
end

end
