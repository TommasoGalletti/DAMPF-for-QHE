module DampfCsvIO

export save_comparison_csv, load_comparison_csv, save_dampf_sweep_csv, load_dampf_sweep_csv,
       save_omega_gammaL_scan_csv, load_omega_gammaL_scan_csv,
       merge_sorted_results

"""Save one row per result (ΓL, DAMPF vs Direct-Lindblad observables, trace
distance, convergence info) to CSV."""
function save_comparison_csv(filepath, results)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        println(io, "GammaL,I_dampf,I_exact,V_dampf,V_exact,P_dampf,P_exact,trace_dist,steps,converged")
        for r in results
            println(io, string(
                r.ΓL, ",", r.I_dampf, ",", r.I_exact, ",", r.V_dampf, ",", r.V_exact, ",",
                r.P_dampf, ",", r.P_exact, ",", r.trace_dist, ",", r.steps, ",", r.converged,
            ))
        end
    end
end

"""
Load a previously-saved comparison CSV (same schema as `save_comparison_csv`)
so a run can add new ΓL points to it instead of overwriting the whole sweep.
Returns an empty vector if the file doesn't exist yet (first run).
"""
function load_comparison_csv(filepath)
    isfile(filepath) || return NamedTuple[]
    rows = NamedTuple[]
    for (i, line) in enumerate(eachline(filepath))
        i == 1 && continue
        isempty(strip(line)) && continue
        cols = split(line, ",")
        push!(rows, (
            ΓL=parse(Float64, cols[1]), I_dampf=parse(Float64, cols[2]), I_exact=parse(Float64, cols[3]),
            V_dampf=parse(Float64, cols[4]), V_exact=parse(Float64, cols[5]),
            P_dampf=parse(Float64, cols[6]), P_exact=parse(Float64, cols[7]),
            trace_dist=parse(Float64, cols[8]), steps=parse(Int, cols[9]), converged=(strip(cols[10]) == "true"),
        ))
    end
    return rows
end

"""Save one row per result (ΓL, DAMPF-only observables, convergence info) --
for DAMPF-only sweeps that have no "Lindblad diretto" reference to compare
against (e.g. once M is large enough that the dense reference is no longer
attainable, see `killoran_three_mode_model`)."""
function save_dampf_sweep_csv(filepath, results)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        println(io, "GammaL,I_dampf,V_dampf,P_dampf,steps,converged")
        for r in results
            println(io, string(
                r.ΓL, ",", r.I_dampf, ",", r.V_dampf, ",", r.P_dampf, ",", r.steps, ",", r.converged,
            ))
        end
    end
end

"""
Load a previously-saved DAMPF-only sweep CSV (same schema as
`save_dampf_sweep_csv`) so a run can add new ΓL points to it instead of
overwriting the whole sweep. Returns an empty vector if the file doesn't
exist yet (first run).
"""
function load_dampf_sweep_csv(filepath)
    isfile(filepath) || return NamedTuple[]
    rows = NamedTuple[]
    for (i, line) in enumerate(eachline(filepath))
        i == 1 && continue
        isempty(strip(line)) && continue
        cols = split(line, ",")
        push!(rows, (
            ΓL=parse(Float64, cols[1]), I_dampf=parse(Float64, cols[2]),
            V_dampf=parse(Float64, cols[3]), P_dampf=parse(Float64, cols[4]),
            steps=parse(Int, cols[5]), converged=(strip(cols[6]) == "true"),
        ))
    end
    return rows
end

"""Save one row per result (mode frequency ω, ΓL, DAMPF I/V/P, power
enhancement relative to that ΓL's own g=0 baseline, convergence info) --
for scanning ω AND ΓL together (see experiments/omega_gammaL_scan_killoran.jl)."""
function save_omega_gammaL_scan_csv(filepath, results)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        println(io, "omega,GammaL,I_dampf,V_dampf,P_dampf,enhancement_pct,steps,converged")
        for r in results
            println(io, string(
                r.ω, ",", r.ΓL, ",", r.I_dampf, ",", r.V_dampf, ",", r.P_dampf, ",",
                r.enhancement_pct, ",", r.steps, ",", r.converged,
            ))
        end
    end
end

"""
Load a previously-saved (ω,ΓL)-scan CSV (same schema as
`save_omega_gammaL_scan_csv`) so a run can add new (ω,ΓL) points to it
instead of overwriting the whole scan. Returns an empty vector if the file
doesn't exist yet (first run).
"""
function load_omega_gammaL_scan_csv(filepath)
    isfile(filepath) || return NamedTuple[]
    rows = NamedTuple[]
    for (i, line) in enumerate(eachline(filepath))
        i == 1 && continue
        isempty(strip(line)) && continue
        cols = split(line, ",")
        push!(rows, (
            ω=parse(Float64, cols[1]), ΓL=parse(Float64, cols[2]), I_dampf=parse(Float64, cols[3]),
            V_dampf=parse(Float64, cols[4]), P_dampf=parse(Float64, cols[5]), enhancement_pct=parse(Float64, cols[6]),
            steps=parse(Int, cols[7]), converged=(strip(cols[8]) == "true"),
        ))
    end
    return rows
end

"""
Merge `results` into `existing` (a matching `key(r)` is OVERWRITTEN, not
duplicated), then sort by `key` ascending -- "append, then sort": repeated
small runs accumulate points into one growing, ordered CSV instead of each
run wiping the file. Defaults to keying on `.ΓL`, but any field works.
"""
function merge_sorted_results(existing, results; key=r -> r.ΓL)
    by_key = Dict{Any,Any}()
    for r in existing
        by_key[key(r)] = r
    end
    for r in results
        by_key[key(r)] = r
    end
    return sort(collect(values(by_key)); by=key)
end

end
