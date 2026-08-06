module DampfModeIndex

using ITensors
using Main.DampfSystem.DampfPhysical: DampfModel

export build_mode_siteinds, modes_on_site

"""
Build the shared per-mode site indices for the vibrational MPS chain, one
Index per mode with dimension `Nb^2` (vectorized ket⊗bra operator space,
matrix-unit basis). Built ONCE per model and threaded everywhere --
ITensorMPS's `+`/`truncate!` require identical Index objects (not just
matching dimensions) across every block MPS that will ever be summed.
"""
function build_mode_siteinds(model::DampfModel)
    return [Index(mode.Nb^2, "Site,mode$(q)") for (q, mode) in enumerate(model.modes)]
end

"""
Flat indices (into `model.modes`) of the modes attached to electronic site
`site` (1-based, matching `DampfModel`/`Hamiltonian.jl`'s convention: site 1
is the ground state and is never returned here).
"""
function modes_on_site(model::DampfModel, site::Int)
    return [q for (q, mode) in enumerate(model.modes) if mode.site == site]
end

end
