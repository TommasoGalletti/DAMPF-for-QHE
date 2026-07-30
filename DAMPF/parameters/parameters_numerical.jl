module DAMPFNumerics

export DampfNumerics, default_numerics

"""
Numerical parameters for the tensor-network solver.

- `dt`: time step
- `t_final`: final simulation time
- `chi_max`: maximum bond dimension χ
- `svd_tol`: truncation tolerance for SVD
- `save_every`: save every N steps
"""
struct DampfNumerics
    dt::Float64
    t_final::Float64
    chi_max::Int
    svd_tol::Float64
    save_every::Int
end

function default_numerics()
    return DampfNumerics(0.01, 1.0, 64, 1e-10, 1)
end

end
