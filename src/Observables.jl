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
    return real(ρ_exc[i,i])
end

function current(ρ, U, ΓL, idx_low)
    P_low = population_exciton(ρ, U, idx_low)
    return ΓL * P_low
end

function voltage(ρ, U, energies, idx_low, idx_g; kB=1.0, TC)
    p_low = population_exciton(ρ, U, idx_low)
    p_g   = population_exciton(ρ, U, idx_g)

    ΔE = energies[idx_low] - energies[idx_g]

    return ΔE + kB * TC * log(p_low / p_g)
end

# =========================
# VIBRONIC EXTENSION
# =========================

"""
Popolazione eccitonica tracciando fuori il vibrone
"""
function population_exciton_vib(ρ, U, i, Nv)

    dim_e = size(U,1)

    # reshape → ρ[a,α; b,β]
    ρ_tensor = reshape(ρ, dim_e, Nv, dim_e, Nv)

    # traccia su vibrazioni
    ρ_red = zeros(ComplexF64, dim_e, dim_e)

    for α in 1:Nv
        ρ_red += ρ_tensor[:,α,:,α]
    end

    # torna in exciton basis
    ρ_exc = U' * ρ_red * U

    return real(ρ_exc[i,i])
end

"""
Corrente vibronica
"""
function current_vib(ρ, U, ΓL, idx_low, Nv)
    P_low = population_exciton_vib(ρ, U, idx_low, Nv)
    return ΓL * P_low
end

"""
Voltaggio vibronico
"""
function voltage_vib(ρ, U, energies, idx_low, idx_g, Nv; kB=1.0, TC)

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