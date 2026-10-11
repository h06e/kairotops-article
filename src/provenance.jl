# optitopo5 — results-store / provenance infrastructure (campaign spec Section 1/0.3)
#
# Every study script calls `new_run(study_id, config)` to get a fresh,
# hash-keyed run directory under `results/`, then `finish_run!(run, extra)`
# once the computation is done, which writes `meta.json` (provenance) and
# appends a row to `results/index.csv`. Scripts never overwrite an existing
# run directory — a changed config always hashes to a different directory.

using TOML
using JSON3
using Dates
using Pkg
using CUDA

const CAMPAIGN_ROOT = normpath(joinpath(@__DIR__, ".."))
# Kairotop.jl as installed in this project's environment (a git checkout next to this repository if
# developed with Pkg.develop, else the package directory; its git SHA is then "unknown" and the version is
# in the stored Pkg.status()).
const LIBRARY_ROOT = let p = Base.find_package("Kairotop")
    p === nothing ? normpath(joinpath(CAMPAIGN_ROOT, "..", "Kairotop")) : normpath(joinpath(dirname(p), ".."))
end

#***************************************************************
#* GIT INFO (two repos: this campaign workspace, and the Kairotop library —
#* see docs/OPEN_QUESTIONS.md OQ-0 for why provenance needs both)

"""
    git_info(repo_path) -> (sha, dirty)

`sha` is `"unknown"` if `repo_path` isn't a git repository (or `git` isn't on
PATH); `dirty` is `true` whenever that can't be positively ruled out (i.e.
fails safe towards "not reproducible" rather than silently claiming a clean
tree).
"""
function git_info(repo_path::AbstractString)
    sha = try
        strip(read(`git -C $repo_path rev-parse HEAD`, String))
    catch
        "unknown"
    end
    dirty = try
        !isempty(strip(read(`git -C $repo_path status --porcelain`, String)))
    catch
        true
    end
    return (sha=sha, dirty=dirty)
end

#***************************************************************
#* CONFIG HASHING

"""
    canonicalize(x)

Recursively sort `Dict` keys (as `String`s) into a nested `Vector{Pair}`
structure with deterministic iteration order — the JSON serialization of two
`Dict`s with the same content but different insertion order must hash
identically, which requires sorting before serializing.
"""
canonicalize(x::AbstractDict) = [string(k) => canonicalize(x[k]) for k in sort(collect(keys(x)); by=string)]
canonicalize(x::AbstractVector) = canonicalize.(x)
canonicalize(x) = x

"""
    config_hash(config) -> String

Deterministic, order-independent 12-hex-character hash of a resolved config
(`canonicalize` then `Base.hash` over its JSON encoding) — this is
`run_hash`, the key under which `results/<study_id>/<run_hash>/` is created.
Not cryptographic (no `SHA` dependency needed for this purpose); a config
that resolves to the same value twice must produce the same hash, which is
all that's required here.
"""
function config_hash(config::AbstractDict)
    str = JSON3.write(canonicalize(config))
    h = hash(str)
    return string(h; base=16, pad=12)[1:12]
end

#***************************************************************
#* RUN LIFECYCLE

mutable struct Run
    study_id::String
    run_hash::String
    dir::String
    config::Dict{String,Any}
    resolved_config::Dict{String,Any}
    t_start::Float64
end

"""
    new_run(study_id, config; resolved_config=config) -> Run

Create (or reuse, if an identical config already produced one) the run
directory `results/<study_id>/<run_hash>/` and return a `Run` handle. Never
overwrites an existing run directory's contents — if `run_hash` already
exists, that directory is returned as-is (rerunning an identical config is a
no-op at the storage level, not a silent overwrite of different content,
since the hash guarantees the content would be identical anyway).
`resolved_config` (the config *after* defaults are applied) is recorded
separately from the given `config` if different; pass it once known, or
patch it in via `finish_run!`.
"""
function new_run(study_id::AbstractString, config::AbstractDict; resolved_config::AbstractDict=config)
    h = config_hash(config)
    dir = joinpath(CAMPAIGN_ROOT, "results", study_id, h)
    # Never overwrite a stored run: a rerun of an identical config (e.g. after
    # a library change) must carry a distinguishing config field.
    isfile(joinpath(dir, "meta.json")) && error("run directory $dir already holds a finished run; refusing to overwrite (add a distinguishing config field)")
    mkpath(dir)
    return Run(String(study_id), h, dir, Dict{String,Any}(config), Dict{String,Any}(resolved_config), time())
end

# Recursively replace non-finite floats (Inf, -Inf, NaN) with their string
# representation — the JSON spec has no literal for them, and JSON3 throws
# mid-write on encountering one (which previously left a truncated, 0-byte
# meta.json behind, poisoning rebuild_index! for every later run).
_json_safe(x::AbstractFloat) = isfinite(x) ? x : string(x)
_json_safe(x::AbstractDict) = Dict(k => _json_safe(v) for (k, v) in x)
_json_safe(x::AbstractVector) = _json_safe.(x)
_json_safe(x) = x

"""
    finish_run!(run::Run, extra::AbstractDict=Dict(); resolution_profile=:dev)

Write `meta.json` (full provenance block) into `run.dir` and append/update
`run`'s row in `results/index.csv`. `extra` holds the study's own scalar
outputs (whatever should be filterable from `index.csv` without opening
`data.jld2` — e.g. achieved solid fraction, final cost, iteration count).
`resolution_profile` must be `:dev` or `:prod`; every result produced at the
current default (`N=64`, see `src/common_campaign.jl`) is `:dev` and must be
labeled as such throughout (notebooks, figure titles) per the campaign's
ground rules (Section 0.1).
"""
function finish_run!(run::Run, extra::AbstractDict=Dict{String,Any}(); resolution_profile::Symbol=:dev)
    resolution_profile in (:dev, :prod) || throw(ArgumentError("resolution_profile must be :dev or :prod"))

    campaign_git = git_info(CAMPAIGN_ROOT)
    library_git = git_info(LIBRARY_ROOT)

    gpu_name = try
        CUDA.functional() ? CUDA.name(CUDA.device()) : "none (CPU fallback)"
    catch
        "unavailable"
    end

    manifest_path = joinpath(CAMPAIGN_ROOT, "Manifest.toml")
    manifest_hash = isfile(manifest_path) ? string(hash(read(manifest_path, String)); base=16) : "missing"

    meta = Dict(
        "study_id" => run.study_id,
        "run_hash" => run.run_hash,
        "resolution_profile" => string(resolution_profile),
        "config" => run.config,
        "resolved_config" => run.resolved_config,
        "extra" => extra,
        "provenance" => Dict(
            "campaign_repo_sha" => campaign_git.sha,
            "campaign_repo_dirty" => campaign_git.dirty,
            "library_repo_sha" => library_git.sha,
            "library_repo_dirty" => library_git.dirty,
            "manifest_hash" => manifest_hash,
            "julia_version" => string(VERSION),
            "gpu_device" => gpu_name,
            "utc_timestamp" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
            "wall_clock_seconds" => round(time() - run.t_start; digits=3),
        ),
    )

    meta_path = joinpath(run.dir, "meta.json")
    tmp_path = meta_path * ".tmp"
    open(tmp_path, "w") do io
        JSON3.pretty(io, _json_safe(meta))
    end
    mv(tmp_path, meta_path; force=true)

    _append_index_row!(run, meta)
    return meta
end

"""
    _append_index_row!(run, meta)

Regenerate `results/index.csv` from every `meta.json` currently on disk
(simpler and more robust than incremental appends against a possibly-stale
file — cheap since `meta.json`s are small and the run count stays in the
hundreds, not millions). One row per run: `study_id`, `run_hash`,
`resolution_profile`, `utc_timestamp`, `wall_clock_seconds`, plus every key
of `extra` (unioned across all rows, missing values blank).
"""
function _append_index_row!(::Run, ::Dict)
    rebuild_index!()
    return nothing
end

"""
    rebuild_index!()

Walk `results/<study_id>/<run_hash>/meta.json` for every study/run and
(re)write `results/index.csv` — see `_append_index_row!`'s docstring for why
this is a full rebuild rather than an incremental append.
"""
function rebuild_index!()
    results_root = joinpath(CAMPAIGN_ROOT, "results")
    isdir(results_root) || return nothing

    rows = Vector{Dict{String,Any}}()
    extra_keys = Set{String}()

    for study_id in sort(readdir(results_root))
        study_dir = joinpath(results_root, study_id)
        isdir(study_dir) || continue
        for run_hash in sort(readdir(study_dir))
            meta_path = joinpath(study_dir, run_hash, "meta.json")
            isfile(meta_path) || continue
            meta = JSON3.read(read(meta_path, String), Dict{String,Any})
            row = Dict{String,Any}(
                "study_id" => meta["study_id"],
                "run_hash" => meta["run_hash"],
                "resolution_profile" => meta["resolution_profile"],
                "utc_timestamp" => meta["provenance"]["utc_timestamp"],
                "wall_clock_seconds" => meta["provenance"]["wall_clock_seconds"],
            )
            for (k, v) in get(meta, "extra", Dict())
                row[k] = v
                push!(extra_keys, k)
            end
            push!(rows, row)
        end
    end

    columns = vcat(["study_id", "run_hash", "resolution_profile", "utc_timestamp", "wall_clock_seconds"], sort(collect(extra_keys)))

    open(joinpath(results_root, "index.csv"), "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((_csv_field(get(row, c, "")) for c in columns), ","))
        end
    end
    return nothing
end

function _csv_field(x)
    s = string(x)
    return occursin(",", s) || occursin("\"", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

#***************************************************************
#* CONFIG LOADING

"""
    load_config(path) -> Dict

Load a study TOML config from `configs/<study_id>.toml`. Every config file
has (at minimum) a `[dev]` and `[prod]` section — see `params/README.md` for
the switching convention; `resolve_config` picks one.
"""
load_config(path::AbstractString) = TOML.parsefile(path)

"""
    resolve_config(config, profile::Symbol=:dev) -> Dict

Merge `config`'s top-level (profile-independent) keys with its
`config[string(profile)]` section, profile-specific keys taking precedence.
"""
function resolve_config(config::AbstractDict, profile::Symbol=:dev)
    profile_key = string(profile)
    haskey(config, profile_key) || throw(ArgumentError("config has no [$profile_key] section."))
    resolved = Dict{String,Any}(k => v for (k, v) in config if k != "dev" && k != "prod")
    merge!(resolved, config[profile_key])
    return resolved
end
