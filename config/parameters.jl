struct Params
    Eg::Float64
    E::Vector{Float64}
    J::Matrix{Float64}

    # vibrazioni
    ωv::Float64
    Nv::Int
    g::Float64
end

struct SimulationParams
    γH::Float64
    γC::Float64
    γM::Float64
    nH::Float64
    TC::Float64
    TM::Float64
    kB::Float64
    cm1_to_ev::Float64
    logΓL_min::Float64
    logΓL_max::Float64
    nΓL::Int
end

"""
Tabella II (prototype) + impostazioni numeriche usate negli esperimenti.
Energie/rate in cm^-1, temperature in K.
"""
function default_simulation_params()
    return SimulationParams(
        0.01,          # γH
        8.07,          # γC
        5.3,           # γM
        60000.0,       # nH
        293.0,         # TC
        293.0,         # TM
        0.69503476,    # kB in cm^-1/K
        1.239841984e-4,# cm^-1 -> eV
        -3.0,          # log10(ΓL)_min
        3.0,           # log10(ΓL)_max
        70             # punti sweep
    )
end

function gammaL_list(sim::SimulationParams)
    return 10 .^ range(sim.logΓL_min, sim.logΓL_max, length=sim.nΓL)
end

"""
Thermal occupations of the cold bath for transitions with ΔE > 0.
Uses Bose-Einstein occupation n = 1 / (exp(ΔE / (kB*TC)) - 1).
"""
function cold_occupations(energies, TC; kB=0.69503476)
    dim = length(energies)
    nC = zeros(dim, dim)

    for i in 1:dim
        for j in 1:dim
            if i != j
                ΔE = energies[i] - energies[j]
                if ΔE > 1e-8
                    nC[i, j] = 1 / (exp(ΔE / (kB * TC)) - 1)
                end
            end
        end
    end

    return nC
end

function default_params()

    Eg = -10000.0 

    # eccitoni
    E = [300.0, 300.0, 0.0]

    J = [0.0 100.0 0.0; 
        100.0 0.0 0.0; 
        0.0 0.0 0.0]

    ωv = 200.0   # Risonante (ε1 - ε2)
    g = 55.0     # Coupling
    Nv = 6 

    return Params(Eg, E, J, ωv, Nv, g)
end