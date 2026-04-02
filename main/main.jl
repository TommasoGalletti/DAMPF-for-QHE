include("../src/Hamiltonian.jl")
include("../src/Lindblad.jl")
include("../src/Liouvillian.jl")
include("../src/Solver.jl")
include("../src/Observables.jl")
include("../config/parameters.jl")

using .Hamiltonian
using .Lindblad
using .Liouvillian
using .Solver
using .Observables

# %% Definizione del modello
# 1. Hamiltoniana
params = default_params()
H = build_hamiltonian(params)

# 2. Diagonalizzazione
energies, U = diagonalize_hamiltonian(H)
dim = size(H,1)

# 3. Lindblad operators
hot_ops  = build_hot_operators(U; γH=0.01, nH=0.5)
load_op  = build_load_operator(U; ΓL=1.0)
cold_ops = build_cold_operators(U; γC=1.0, nC_matrix=ones(3,3))

# merge tutto
jump_ops = JumpOperator[]

append!(jump_ops, hot_ops)
push!(jump_ops, load_op)
append!(jump_ops, cold_ops)

# 4. Liouvilliana
L = build_liouvillian(H, jump_ops)

# %% Check Liouvillian
@assert size(L) == (16,16)

# %% Steady state
ρ_ss = steady_state(L)

#%% Check state
using LinearAlgebra

println("Trace = ", tr(ρ_ss))
println("Hermitian = ", ρ_ss ≈ ρ_ss')
println("Min eigenvalue = ", minimum(real(eigvals(ρ_ss))))

#%% Observables
idx_low = 3
idx_g   = dim

I = current(ρ_ss, U, 1.0, idx_low)
println("Current = ", I)

#%% Check populations and indices
idx_low = 3
idx_g   = dim

p_low = population_exciton(ρ_ss, U, idx_low)
p_g   = population_exciton(ρ_ss, U, idx_g)

println("I = ", I)

println("All exciton populations:")
for i in 1:dim
    println("i=$i → ", population_exciton(ρ_ss, U, i))
end