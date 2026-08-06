module DampfSteadyState

using ITensors
using ITensorMPS
using Main.DampfSystem.DampfPhysical: DampfModel
using Main.DampfSystem.DampfLocalGenerators: ModeRolePropagators
using Main.DampfSystem.DampfState: init_ground_state, reduced_electronic_density
using Main.DampfSystem.DampfTrotter: one_trotter_step!

export run_to_steady_state

"""
Propagate the DAMPF Trotter dynamics from the ground state until a scalar
`observable_fn(ρ_e)` stops changing by more than `conv_tol` between
checkpoints (every `check_every` steps), or `max_steps` is reached.

This driver is generic on purpose: `P_e` (the ΓL/model-specific electronic-
mixing propagator) is built by the caller, and `observable_fn` is whatever
scalar the caller wants to track for convergence (e.g. current, a
population, ...) -- this file knows nothing about ΓL, `Observables.jl`, or
any one experiment's model, only the Trotter-step machinery in
`dynamics/trotter.jl`. `label` is just a prefix for the progress log line.
"""
function run_to_steady_state(
    model::DampfModel, role_props::Vector{ModeRolePropagators}, siteinds::Vector{<:Index}, τ::MPS, P_e::AbstractMatrix;
    chi_max::Int, coeff_tol::Float64, Δt::Float64, max_steps::Int, check_every::Int, conv_tol::Float64,
    observable_fn, label::String="",
)
    blocks = init_ground_state(model, siteinds)
    ρ_e = reduced_electronic_density(blocks, τ)
    obs_prev = 0.0
    steps_taken = 0
    converged = false
    t_start = time()

    for step in 1:max_steps
        trace_before = one_trotter_step!(blocks, model, role_props, P_e, siteinds, τ; chi_max=chi_max, svd_tol=0.0, coeff_tol=coeff_tol)
        steps_taken = step
        if step % check_every == 0
            ρ_e = reduced_electronic_density(blocks, τ)
            obs_now = observable_fn(ρ_e)
            elapsed = round(time() - t_start, digits=1)
            println("  [$(label)] step $(step)/$(max_steps)  t_phys=$(round(step * Δt, sigdigits=4))  " *
                    "elapsed=$(elapsed)s  trace_before_renorm=$(trace_before)  obs≈$(obs_now)")
            flush(stdout)
            if abs(obs_now - obs_prev) < conv_tol * max(abs(obs_now), 1e-12)
                converged = true
                break
            end
            obs_prev = obs_now
        end
    end

    return (blocks=blocks, ρ_e=ρ_e, steps=steps_taken, converged=converged)
end

end
