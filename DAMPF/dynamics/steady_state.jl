module DampfSteadyState

using ITensors
using ITensorMPS
using Main.DampfSystem.DampfPhysical: DampfModel
using Main.DampfSystem.DampfLocalGenerators: ModeRolePropagators
using Main.DampfSystem.DampfDagger: build_swap_gates, dagger_block
using Main.DampfSystem.DampfState: init_ground_state_triangular, reduced_electronic_density_triangular
using Main.DampfSystem.DampfTrotter: one_trotter_step_triangular!

export run_to_steady_state

"""
Propagate the DAMPF Trotter dynamics from the ground state until
`observable_fn(ρ_e)` stops changing by more than `conv_tol` between
checkpoints (every `check_every` steps), or `max_steps` is reached.

Generic on purpose: `P_e` and `observable_fn` are supplied by the caller,
so this file knows nothing about ΓL, `Observables.jl`, or any one
experiment's model -- only the Trotter-step machinery of `dynamics/trotter.jl`.

Uses the triangular-storage Trotter step internally (only m<=n blocks are
evolved, exploiting Ô_{n,m}=Ô_{m,n}^dagger -- see `dynamics/trotter.jl`,
`system/dagger.jl`, and the SPIEGAZIONE docs). The returned `blocks` is
filled back to full dim_e x dim_e storage (cheap, one-time) so this stays a
drop-in replacement for callers that expect full storage, e.g.
`iv_curve_dampf_killoran.jl`'s `blocks_to_dense` call.
"""
function run_to_steady_state(
    model::DampfModel, role_props::Vector{ModeRolePropagators}, siteinds::Vector{<:Index}, τ::MPS, P_e::AbstractMatrix;
    chi_max::Int, coeff_tol::Float64, Δt::Float64, max_steps::Int, check_every::Int, conv_tol::Float64,
    observable_fn, label::String="",
)
    swap_gates = build_swap_gates(model.modes, siteinds)
    blocks = init_ground_state_triangular(model, siteinds)
    ρ_e = reduced_electronic_density_triangular(blocks, τ)
    obs_prev = 0.0
    steps_taken = 0
    converged = false
    t_start = time()

    for step in 1:max_steps
        trace_before = one_trotter_step_triangular!(blocks, model, role_props, P_e, siteinds, τ, swap_gates; chi_max=chi_max, svd_tol=0.0, coeff_tol=coeff_tol)
        steps_taken = step
        if step % check_every == 0
            ρ_e = reduced_electronic_density_triangular(blocks, τ)
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

    dim = size(blocks, 1)
    for l in 1:dim, k in (l + 1):dim
        blocks[k, l] = dagger_block(blocks[l, k], swap_gates)
    end

    return (blocks=blocks, ρ_e=ρ_e, steps=steps_taken, converged=converged)
end

end
