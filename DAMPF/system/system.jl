module DampfSystem

include("../parameters/parameters_physical.jl")
include("../parameters/parameters_numerical.jl")

using .DAMPFPhysical: DampfPhysicalParams, default_physical_params
using .DAMPFNumerics: DampfNumerics, default_numerics

export DampfSystemParams, build_system, default_system

"""
Container that aggregates physical and numerical parameters into a single object
used by higher-level builders.
"""
struct DampfSystemParams
    phys::DampfPhysicalParams
    numerics::DampfNumerics
end

function build_system(phys::DampfPhysicalParams, numerics::DampfNumerics)
    return DampfSystemParams(phys, numerics)
end

function default_system()
    phys = default_physical_params()
    numerics = default_numerics()
    return DampfSystemParams(phys, numerics)
end

end
