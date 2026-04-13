struct Params
    Eg::Float64
    E::Vector{Float64}
    J::Matrix{Float64}

    # vibrazioni
    ωv::Float64
    Nv::Int
    g::Float64
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