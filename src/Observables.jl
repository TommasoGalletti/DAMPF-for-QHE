module Observables

using LinearAlgebra

export population_exciton,
       population_exciton_vib,
       current,
       current_vib,
       voltage,
       voltage_vib,
       power

# =========================
# STANDARD (Electronic, NO VIB)
# =========================

function population_exciton(ρ, U, i)
    ρ_exc = U' * ρ * U
    return real(ρ_exc[i, i])
end

function current(ρ, U, ΓL, idx_low)
    P_low = population_exciton(ρ, U, idx_low)
    return ΓL * P_low
end

function voltage(ρ, U, energies, idx_low, idx_g; kB=0.69503476, TC)
    p_low = population_exciton(ρ, U, idx_low)
    p_g   = population_exciton(ρ, U, idx_g)

    ΔE = energies[idx_low] - energies[idx_g]

    return ΔE + kB * TC * log(p_low / p_g)
end

# =========================
# VIBRONIC EXTENSION
# =========================

"""
Exciton population for the two-mode vibronic model,
tracing out both vibrational subspaces.
"""
function population_exciton_vib(ρ, U, i, Nv)
    dim_e = size(U, 1)
    
    # Trace out both vibrational modes.
    # Basis ordering matches build_total_hamiltonian: |e> ⊗ |v1> ⊗ |v2>.
    # Global index: idx = (e-1)*Nv^2 + (v1-1)*Nv + v2.
    ρ_red = zeros(ComplexF64, dim_e, dim_e)
    for e1 in 1:dim_e
        for e2 in 1:dim_e
            for v1 in 1:Nv
                for v2 in 1:Nv
                    idx1 = (e1 - 1) * Nv * Nv + (v1 - 1) * Nv + v2
                    idx2 = (e2 - 1) * Nv * Nv + (v1 - 1) * Nv + v2
                    ρ_red[e1, e2] += ρ[idx1, idx2]
                end
            end
        end
    end
    
    # DEBUG: check ρ_red 
    # println("[DEBUG] population_exciton_vib: sum(abs(ρ_red)) = ", sum(abs.(ρ_red)))
    # println("[DEBUG] ρ_red[1,1] = ", ρ_red[1,1])
    
    ρ_exc = U' * ρ_red * U
    return real(ρ_exc[i, i])
end

"""Vibronic current."""
function current_vib(ρ, U, ΓL, idx_low, Nv)
    P_low = population_exciton_vib(ρ, U, idx_low, Nv)
    return ΓL * P_low
end

"""Vibronic voltage."""
function voltage_vib(ρ, U, energies, idx_low, idx_g, Nv; kB=0.69503476, TC)

    p_low = population_exciton_vib(ρ, U, idx_low, Nv)
    p_g   = population_exciton_vib(ρ, U, idx_g, Nv)

    ΔE = energies[idx_low] - energies[idx_g]

    return ΔE + kB * TC * log(p_low / p_g)
end

# =========================
# POWER = V * I
# =========================

function power(I, V)
    return I * V
end

end