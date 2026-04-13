include("../src/Hamiltonian.jl")
include("../src/Lindblad.jl")
include("../src/Liouvillian.jl")
include("../src/Solver.jl")
include("../src/Observables.jl")
include("../config/parameters.jl")

using LinearAlgebra

println("\n=== TEST VIBRONIC HAMILTONIAN ===\n")

# =========================
# PARAMETRI
# =========================

params = default_params()

println("Nv = ", params.Nv)
println("ωv = ", params.ωv)
println("g  = ", params.g)

# =========================
# HAMILTONIANA ELETTRONICA
# =========================

H_e  = Hamiltonian.build_hamiltonian(params)

energies_e, U = Hamiltonian.diagonalize_hamiltonian(H_e)
dim = size(H_e,1)

println("\nElectronic Hamiltonian size:")
println(size(H_e))

# =========================
# HAMILTONIANA VIBRONICA
# =========================

H_tot = Hamiltonian.build_total_hamiltonian(params)

println("\nTotal Hamiltonian size:")
println(size(H_tot))

# =========================
# CHECK DIMENSIONI
# =========================

dim_expected = size(H_e,1) * params.Nv

println("\nExpected dimension:")
println(dim_expected)

@assert size(H_tot,1) == dim_expected
@assert size(H_tot,2) == dim_expected

println("✔ Dimension check passed")

# =========================
# HERMITICITY CHECK
# =========================

println("\nHermiticity check:")
println(H_tot ≈ H_tot')

# =========================
# DIAGONALIZZAZIONE
# =========================

println("\nDiagonalizing...")

eig = eigen(H_tot)
energies = eig.values

println("✔ Diagonalization OK")

# =========================
# CHECK ENERGIE
# =========================

println("\nEnergy spectrum (first 10):")
for i in 1:min(10, length(energies))
    println(energies[i])
end

# =========================
# GAP MINIMO
# =========================

ΔE_min = minimum(abs.(energies[i] - energies[j] 
    for i in 1:length(energies), j in 1:length(energies) if i != j))

println("\nMinimum energy gap:")
println(ΔE_min)

#%% CHECK JUMP OPERATORS

Nv = 4

hot_ops_vib  = Lindblad.build_hot_operators_vib(U; γH=0.01, nH=0.5, Nv=Nv)
load_op_vib  = Lindblad.build_load_operator_vib(U; ΓL=1.0, Nv=Nv)
cold_ops_vib = Lindblad.build_cold_operators_vib(U; γC=1.0, nC_matrix=ones(dim,dim), Nv=Nv)

println(size(hot_ops_vib[1].A))

println("\n=== TEST COMPLETED ===\n")

