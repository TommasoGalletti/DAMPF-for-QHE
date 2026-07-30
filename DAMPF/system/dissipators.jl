module DampfDissipators

using LinearAlgebra
using Main.DampfSystem.DAMPFPhysical: DampfPhysicalParams

export build_vibrational_dissipators

"""
Build Lindblad jump operators for each local vibrational mode.

Returns a vector of NamedTuples with fields:
- `site`, `mode`, `idx` : identifiers
- `L` : jump operator matrix (scaled by sqrt(rate))
- `rate` : physical rate
- `type` : "loss" or "gain"

Arguments:
- `phys`: `DampfPhysicalParams` instance
- `n_th`: thermal occupancy for modes (default 0.0). If unknown, keep 0.0.
"""
function build_vibrational_dissipators(phys::DampfPhysicalParams; n_th::Float64 = 0.0)
    N = phys.n_sites
    Q = phys.n_modes_per_site
    Nb = phys.local_truncation
    jumps = Vector{NamedTuple}()
    k = 1
    for i in 1:N
        for q in 1:Q
            # single-mode lowering operator
            a = zeros(ComplexF64, Nb, Nb)
            for n in 2:Nb
                a[n-1,n] = sqrt(n-1)
            end
            adag = a'
            γ = phys.damping_rates[q]

            # loss (relaxation) with rate γ*(n_th + 1)
            rate_loss = γ * (n_th + 1.0)
            L_loss = sqrt(rate_loss) * a
            push!(jumps, (site=i, mode=q, idx=k, L=L_loss, rate=rate_loss, type="loss"))

            # gain (thermal excitation) with rate γ * n_th (may be zero)
            rate_gain = γ * n_th
            if rate_gain > 0.0
                L_gain = sqrt(rate_gain) * adag
                push!(jumps, (site=i, mode=q, idx=k, L=L_gain, rate=rate_gain, type="gain"))
            end

            k += 1
        end
    end
    return jumps
end

end
