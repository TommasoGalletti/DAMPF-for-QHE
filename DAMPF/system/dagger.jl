module DampfDagger

using ITensors
using ITensorMPS
using Main.DampfSystem.DampfPhysical: DampfMode

export build_swap_gates, dagger_block

"""
Fixed (Δt-independent) permutation gate for one mode's vectorized-operator
site index: maps k=(b-1)*Nb+a to the "ket/bra swapped" index (a-1)*Nb+b.
Combined with complex conjugation of the MPS data (see `dagger_block`), this
implements the dagger of a single-mode block operator: vec(Ô)[k]=Ô[a,b] ->
vec(Ô^dagger)[k]=conj(Ô[b,a]). Real, symmetric, self-inverse (S^2=Id) --
a pure basis permutation, never exponentiated (unlike the role
propagators in `local_generators.jl`).
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

"""One swap gate per mode, built once -- depends only on Nb/siteinds, not Δt."""
function build_swap_gates(modes::Vector{DampfMode}, siteinds::Vector{<:Index})
    return [_swap_gate(mode.Nb, siteinds[q]) for (q, mode) in enumerate(modes)]
end

"""
Hermitian conjugate of a block MPS Ô (dagger acts independently, per mode,
on the vectorized-operator site index -- conjugate + ket/bra swap -- so the
MPS bond structure needs no special handling: conjugating each site tensor
in place already conjugates every bond-index entry correctly). Used to
reconstruct a not-directly-stored block Ô_{n,m}=Ô_{m,n}^dagger on the fly
in the triangular-storage Trotter step (`dynamics/trotter.jl`), instead of
storing/truncating it independently.
"""
function dagger_block(ψ::MPS, swap_gates::Vector{ITensor})
    M = length(ψ)
    return MPS([noprime(swap_gates[q] * conj(ψ[q])) for q in 1:M])
end

end
