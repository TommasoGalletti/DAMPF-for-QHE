module DampfNumerical

export DampfNumericalParams, default_numerical_params

struct DampfNumericalParams
    Δt::Float64
    chi_max::Int
    svd_tol::Float64
end

default_numerical_params() = DampfNumericalParams(1.0e-5, 8, 1.0e-10)

end
