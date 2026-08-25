module DampfDagger

using ITensors
using ITensorMPS
using Main.DampfSystem.DampfPhysical: DampfMode

export build_swap_gates, dagger_block

"""
Permutation gate mapping k=(b-1)*Nb+a to (a-1)*Nb+b, the ket/bra swap for one
mode's vectorized-operator index. Combined with conjugating the MPS data
(`dagger_block`), gives vec(Ô^dagger)[k]=conj(Ô[b,a]). Fixed, not
Δt-exponentiated.
"""
function _swap_gate(Nb::Int, s::Index)
    G = zeros(Float64, Nb^2, Nb^2)
    for k in 1:Nb^2
        b = ((k - 1) ÷ Nb) + 1
        a = ((k - 1) % Nb) + 1
        kswap = (a - 1) * Nb + b
        G[k, kswap] = 1.0
    end
    return ITensor(G, s', s)
end

"""One swap gate per mode, built once (depends only on Nb, not Δt)."""
function build_swap_gates(modes::Vector{DampfMode}, siteinds::Vector{<:Index})
    return [_swap_gate(mode.Nb, siteinds[q]) for (q, mode) in enumerate(modes)]
end

"""
Hermitian conjugate of a block MPS: conjugate each site tensor and apply the
swap gate. Reconstructs Ô_{n,m}=Ô_{m,n}^dagger on the fly in the
triangular-storage Trotter step (`dynamics/trotter.jl`) instead of storing
it independently.
"""
function dagger_block(ψ::MPS, swap_gates::Vector{ITensor})
    M = length(ψ)
    return MPS([noprime(swap_gates[q] * conj(ψ[q])) for q in 1:M])
end

end
