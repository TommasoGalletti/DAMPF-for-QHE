module Observables

using LinearAlgebra

export population_exciton, current, voltage, power

"""
Popolazione stato eccitonico i
"""
function population_exciton(ρ, U, i)

    ρ_exc = U' * ρ * U

    return real(ρ_exc[i,i])
end

"""
Corrente load
"""
function current(ρ, U, ΓL, idx_low)

    P_low = population_exciton(ρ, U, idx_low)

    return ΓL * P_low
end

"""
Voltaggio
"""
function voltage(ρ, U, energies, idx_low, idx_g; kB=1.0, TC)

    p_low = population_exciton(ρ, U, idx_low)
    p_g   = population_exciton(ρ, U, idx_g)

    Eg = energies[idx_low] - energies[idx_g]

    return Eg + kB * TC * log(p_low / p_g)
end

"""
Potenza
"""
function power(I, V)
    return I * V
end

end