# S2 -- morphological control via the initialization length l0 (Fig. 3 / Sec. 2.2).
# Design and rationale: configs/S2.toml. One C-Kai generation per call so the
# runs can be queued one at a time on the GPU (scripts/run_queue.sh):
#   julia --project=. scripts/S2_morphological_control.jl 0.03 2      # l0 = 0.03, seed 2, r_f from the config (fixed series)
#   julia --project=. scripts/S2_morphological_control.jl 0.03 2 0.0075   # same with r_f = 0.0075 (proportional series r_f = l0/4)
#   julia --project=. scripts/S2_morphological_control.jl refcov      # covariograms of the S1 references at f = 0.3
# Shared kairotop settings: configs/S1.toml [prod.kai_base] (same method as S1).
# Stores the covariogram of the thresholded physical field with each run.

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
include(joinpath(@__DIR__, "..", "src", "kai_runner.jl"))

using JSON3

# Covariograms of the S1 Gyroid and canonical RSA runs at f = 0.3 (same method,
# contrast 1000), stored as S2 reference entries that point back to their S1 run.
function refcov(c)
    s1 = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S1.toml")), :prod)
    rmax = s1["rsa_max_radius"][@sprintf("f%02d", round(Int, 100c["fraction"]))]
    dir = joinpath(CAMPAIGN_ROOT, "results", "S1")
    for h in readdir(dir)
        m = JSON3.read(read(joinpath(dir, h, "meta.json"), String), Dict{String,Any})
        e, cf = m["extra"], m["config"]
        isapprox(e["fraction"], c["fraction"]) || continue
        fam = e["family"]
        (fam == "Gyroid" || (fam == "RSA" && cf["rsa_max_radius"] == rmax)) || continue
        χ = Float64.(load(joinpath(dir, h, "data.jld2"), "chi"))
        r, γ, _ = directional_covariogram(χ)
        cfg = Dict{String,Any}("family" => fam, "fraction" => c["fraction"], "source" => "S1/$h", "phase_contrast" => c["phase_contrast"], "N" => c["N"])
        haskey(e, "seed") && (cfg["seed"] = e["seed"])
        res = merge(Dict{String,Any}(k => e[k] for k in ("kappa_eff", "mu_eff", "delta_iso", "D_eff", "r_cov", "void_percolates_xyz")), Dict("family" => fam, "fraction" => c["fraction"]))
        haskey(e, "seed") && (res["seed"] = e["seed"])
        run = new_run("S2", cfg)
        jldsave(joinpath(run.dir, "data.jld2"); covariogram_r=r, covariogram_gamma=γ)
        finish_run!(run, res; resolution_profile=:prod)
        println("[S2 refcov] $fam $(get(e, "seed", "")) from S1/$h r_cov=$(e["r_cov"])")
    end
    rebuild_index!()
end

function main(args)
    c = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S2.toml")), :prod)
    args[1] == "refcov" && return refcov(c)
    l0 = parse(Float64, args[1]); seed = parse(Int, args[2])
    rf = length(args) >= 3 ? parse(Float64, args[3]) : c["density_filter_radius"]
    base = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S1.toml")), :prod)["kai_base"]
    p = merge(kai_defaults(), base, Dict("family" => "C-Kai", "fraction" => c["fraction"], "seed" => seed, "N" => c["N"],
        "phase_contrast" => c["phase_contrast"], "init_smoothing_radius" => l0, "density_filter_radius" => rf))
    res, data = run_kai(p)
    r, γ, _ = directional_covariogram(data["rho_thresholded"])
    data["covariogram_r"] = r; data["covariogram_gamma"] = γ
    run = store_kai("S2", p, merge(res, Dict("l0" => l0, "density_filter_radius" => rf)), data)
    println(kai_summary_line("S2 C-Kai l0=$l0", p, res))
    rebuild_index!()
end

main(ARGS)
