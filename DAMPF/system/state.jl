module DampfState

using ITensors
using ITensorMPS
using Main.DampfSystem.DampfPhysical: DampfModel, DampfMode, dim_e
using Main.DampfSystem.DampfTraceBasis: product_mps_from_vectors, zero_block, trace_of_block

export init_ground_state, reduced_electronic_density, block_to_dense, blocks_to_dense,
       init_ground_state_triangular, reduced_electronic_density_triangular, blocks_to_dense_triangular

"""Initial state ρ0 = |g><g| ⊗ (vacuum on every mode): only block (1,1) is
nonzero (matrix unit E_{1,1}=|0><0| on each mode), every other block zero."""
function init_ground_state(model::DampfModel, siteinds::Vector{<:Index})
    dim = dim_e(model)
    blocks = Matrix{MPS}(undef, dim, dim)
    vacuum_vectors = Vector{Vector{ComplexF64}}(undef, length(model.modes))
    for (q, mode) in enumerate(model.modes)
        v = zeros(ComplexF64, mode.Nb^2)
        v[1] = 1.0
        vacuum_vectors[q] = v
    end
    ground_block = product_mps_from_vectors(siteinds, vacuum_vectors)
    for m in 1:dim, n in 1:dim
        blocks[m, n] = (m == 1 && n == 1) ? ground_block : zero_block(siteinds)
    end
    return blocks
end

"""Reduced electronic density matrix ρ_e[m,n] = Tr(Ô_{m,n}), dense dim_e x
dim_e, site basis -- feeds directly into `Observables.jl`'s plain (non-`_vib`)
functions."""
function reduced_electronic_density(blocks::Matrix{MPS}, τ::MPS)
    dim = size(blocks, 1)
    ρ = zeros(ComplexF64, dim, dim)
    for m in 1:dim, n in 1:dim
        ρ[m, n] = trace_of_block(τ, blocks[m, n])
    end
    return ρ
end

"""Triangular-storage variant of `init_ground_state`: only populates m<=n
entries (see `DampfTrotter.one_trotter_step_triangular!`)."""
function init_ground_state_triangular(model::DampfModel, siteinds::Vector{<:Index})
    dim = dim_e(model)
    blocks = Matrix{MPS}(undef, dim, dim)
    vacuum_vectors = Vector{Vector{ComplexF64}}(undef, length(model.modes))
    for (q, mode) in enumerate(model.modes)
        v = zeros(ComplexF64, mode.Nb^2)
        v[1] = 1.0
        vacuum_vectors[q] = v
    end
    ground_block = product_mps_from_vectors(siteinds, vacuum_vectors)
    for m in 1:dim, n in m:dim
        blocks[m, n] = (m == 1 && n == 1) ? ground_block : zero_block(siteinds)
    end
    return blocks
end

"""Triangular-storage variant of `reduced_electronic_density`: computes m<=n
directly, fills m>n by scalar conjugation (Tr(Ô_{n,m})=conj(Tr(Ô_{m,n})) --
no MPS dagger needed here, unlike inside the mixing step)."""
function reduced_electronic_density_triangular(blocks::Matrix{MPS}, τ::MPS)
    dim = size(blocks, 1)
    ρ = zeros(ComplexF64, dim, dim)
    for m in 1:dim, n in m:dim
        val = trace_of_block(τ, blocks[m, n])
        ρ[m, n] = val
        m != n && (ρ[n, m] = conj(val))
    end
    return ρ
end

function _mixed_radix_index(digits::Vector{Int}, radices::Vector{Int})
    idx = 0
    for (d, r) in zip(digits, radices)
        idx = idx * r + (d - 1)
    end
    return idx + 1
end

"""Reconstruct one block Ô as a dense (ΠNb_q x ΠNb_q) matrix (mode 1 slowest,
matching `Hamiltonian.build_total_hamiltonian`'s `kron` convention).
CHECK-ONLY: contracts the full MPS, doesn't scale -- never call from
`dynamics/`."""
function block_to_dense(ψ::MPS, siteinds::Vector{<:Index}, modes::Vector{DampfMode})
    M = length(modes)
    Nbs = [mode.Nb for mode in modes]
    Dv = prod(Nbs)
    dense = zeros(ComplexF64, Dv, Dv)

    full = ψ[1]
    for q in 2:M
        full = full * ψ[q]
    end

    ranges = [1:(Nb^2) for Nb in Nbs]
    for jcombo in Iterators.product(ranges...)
        aqs = Int[]
        bqs = Int[]
        for (q, jq) in enumerate(jcombo)
            Nb = Nbs[q]
            bq = ((jq - 1) ÷ Nb) + 1
            aq = ((jq - 1) % Nb) + 1
            push!(aqs, aq)
            push!(bqs, bq)
        end
        ket = _mixed_radix_index(aqs, Nbs)
        bra = _mixed_radix_index(bqs, Nbs)
        val = full[[siteinds[q] => jcombo[q] for q in 1:M]...]
        dense[ket, bra] = val
    end

    return dense
end

"""Reconstruct the full (dim_e*ΠNb_q)-dim dense density operator from every
block, electronic label slowest (matching `Hamiltonian.build_total_hamiltonian`'s
`kron` convention). CHECK-ONLY -- see `block_to_dense`."""
function blocks_to_dense(blocks::Matrix{MPS}, siteinds::Vector{<:Index}, model::DampfModel)
    dim = dim_e(model)
    Dv = prod(mode.Nb for mode in model.modes)
    D = dim * Dv
    dense = zeros(ComplexF64, D, D)
    for m in 1:dim, n in 1:dim
        vib = block_to_dense(blocks[m, n], siteinds, model.modes)
        dense[((m - 1) * Dv + 1):(m * Dv), ((n - 1) * Dv + 1):(n * Dv)] = vib
    end
    return dense
end

"""Triangular-storage variant of `blocks_to_dense`: for m>n, the dense block
is the conjugate-transpose of the already-computed (n,m) block -- cheap,
avoids an MPS dagger. CHECK-ONLY, see `block_to_dense`."""
function blocks_to_dense_triangular(blocks::Matrix{MPS}, siteinds::Vector{<:Index}, model::DampfModel)
    dim = dim_e(model)
    Dv = prod(mode.Nb for mode in model.modes)
    D = dim * Dv
    dense = zeros(ComplexF64, D, D)
    for m in 1:dim, n in m:dim
        vib = block_to_dense(blocks[m, n], siteinds, model.modes)
        dense[((m - 1) * Dv + 1):(m * Dv), ((n - 1) * Dv + 1):(n * Dv)] = vib
        if m != n
            dense[((n - 1) * Dv + 1):(n * Dv), ((m - 1) * Dv + 1):(m * Dv)] = vib'
        end
    end
    return dense
end

end
