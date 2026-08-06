module DampfLocalGenerators

using ITensors
using LinearAlgebra
using Main.DampfSystem.DampfPhysical: DampfMode, thermal_occupation

export ModeRolePropagators, build_mode_role_propagators

function _commutator_superop(H::AbstractMatrix)
    D = size(H, 1)
    Id = Matrix{ComplexF64}(I, D, D)
    Hd = Matrix{ComplexF64}(H)
    return -im .* (kron(Id, Hd) .- kron(transpose(Hd), Id))
end

function _dissipator_superop(A::AbstractMatrix)
    D = size(A, 1)
    Id = Matrix{ComplexF64}(I, D, D)
    Ad = Matrix{ComplexF64}(A)
    AdagA = Ad' * Ad
    return kron(conj(Ad), Ad) .- 0.5 .* kron(Id, AdagA) .- 0.5 .* kron(transpose(AdagA), Id)
end

_left_mult_superop(X::AbstractMatrix) = kron(Matrix{ComplexF64}(I, size(X, 1), size(X, 1)), Matrix{ComplexF64}(X))
_right_mult_superop(X::AbstractMatrix) = kron(transpose(Matrix{ComplexF64}(X)), Matrix{ComplexF64}(I, size(X, 1), size(X, 1)))

"""
The four possible local roles a mode can play for a given (m,n) block,
depending on whether m and/or n equal the mode's owning electronic site.
Every physical term in this model (H_v, vibrational damping, and the
vibronic coupling H_ev) acts on at most ONE mode at a time -- this is what
makes the whole architecture avoid 2-site gates entirely. Each role is a
dense Nb^2 x Nb^2 superoperator, already Δt-exponentiated and wrapped as an
ITensor gate with legs (s', s), ready for `noprime(gate * ψ[q])`.

- P0: neither m nor n is this mode's owning site -> only H_v + damping.
- Pdiag: m == n == owner -> full local commutator with H_v + g*X, + damping.
- Pket: m == owner, n != owner -> one-sided LEFT multiply by -i*g*X only.
- Pbra: n == owner, m != owner -> one-sided RIGHT multiply by +i*g*X only.
"""
struct ModeRolePropagators
    P0::ITensor
    Pdiag::ITensor
    Pket::ITensor
    Pbra::ITensor
end

"""
Build the Δt-exponentiated local generator for all 4 roles of a single mode.
`s` is that mode's shared site Index (dimension Nb^2, vectorized ket⊗bra
operator space via matrix units, k=(b-1)*Nb+a).
"""
function build_mode_role_propagators(mode::DampfMode, s::Index, Δt::Float64)
    Nb = mode.Nb
    a = zeros(ComplexF64, Nb, Nb)
    for n in 2:Nb
        a[n-1, n] = sqrt(n - 1)
    end
    adag = Matrix{ComplexF64}(a')
    X = a + adag
    H_v = mode.ω .* (adag * a)

    nth = thermal_occupation(mode.ω, mode.T)
    base = _commutator_superop(H_v) .+
           (mode.γ * (nth + 1)) .* _dissipator_superop(a) .+
           (mode.γ * nth) .* _dissipator_superop(adag)

    gX = mode.g .* X
    left = _left_mult_superop(gX)
    right = _right_mult_superop(gX)

    G0 = base
    Gdiag = base .- im .* left .+ im .* right
    Gket = base .- im .* left
    Gbra = base .+ im .* right

    _gate(G) = ITensor(exp(Matrix(Δt .* G)), s', s)

    return ModeRolePropagators(_gate(G0), _gate(Gdiag), _gate(Gket), _gate(Gbra))
end

end
