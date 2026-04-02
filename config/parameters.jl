struct Params
    Eg::Float64
    E::Vector{Float64}
    J::Matrix{Float64}
end

function default_params()

    Eg = 0.0

    E  = [300.0, 290.0, 0.0]    #no degenerazione, ε₁ > ε₂ > ε3 > ε₀

    J = [
        0.0 100.0 0.0;
        100.0 0.0 0.0;
        0.0 0.0 0.0
    ]

    return Params(Eg, E, J)
end