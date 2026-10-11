# S1 -- stiffness benchmark across solid fractions (contrast 1000, revised
# algorithm). See configs/S1.toml for the protocol and parameter provenance.
#
#   julia --project=. scripts/S1_benchmark.jl refs [--fractions 0.1,0.2] [--rsa-only]   # Gyroid + RSA seeds
#   julia --project=. scripts/S1_benchmark.jl kai C 0.3 2                  # one C-Kai run (f=0.3, seed 2)
#   julia --project=. scripts/S1_benchmark.jl kai O 0.3 2                  # one O-Kai run
#
# `kai` runs one generation so that many can be queued on the GPU in parallel
# (scripts/run_queue.sh). Properties are reported on the thresholded physical
# field; renders are made from results afterwards.

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
include(joinpath(@__DIR__, "..", "src", "kai_runner.jl"))

fkey(f) = @sprintf("f%02d", round(Int, 100f))

function report_reference(config, χ)
    f = config["fraction"]
    Ceff, Λeff = homogenize_only(χ, Objectives(elastic=isotropic_elastic_target(f), solid_fraction=f, diffusion=open_diffusion_target()))
    kappa, mu = bulk_shear_moduli(Ceff)
    void = χ .< 0.5
    res = Dict{String,Any}("family" => config["family"], "fraction" => f, "kappa_eff" => kappa, "mu_eff" => mu,
        "delta_iso" => delta_iso(Ceff), "D_eff" => sum(Λeff[i, i] for i in 1:3) / 3, "r_cov" => kai_r_cov(χ),
        "achieved_solid_fraction" => mean(χ), "void_percolates_xyz" => count(percolates_periodic(void)),
        "void_largest_cluster" => largest_cluster_fraction(void))
    haskey(config, "seed") && (res["seed"] = config["seed"])
    return res, Dict{String,Any}("chi" => χ, "Ceff" => Ceff, "Λeff" => Λeff)
end

function store(config, res, data)
    run = new_run("S1", config)
    jldsave(joinpath(run.dir, "data.jld2"); NamedTuple(Symbol(k) => v for (k, v) in data)...)
    finish_run!(run, res; resolution_profile=:prod)
end

function run_refs(c, fractions; rsa_only=false)
    base = Dict{String,Any}("N" => c["N"], "phase_contrast" => c["phase_contrast"])
    for f in fractions
        if !rsa_only
            cfg = merge(base, Dict("family" => "Gyroid", "fraction" => f))
            res, data = report_reference(cfg, Float64.(gyroid_initialisation(c["N"], f)))
            store(cfg, res, data)
            @printf("[S1 Gyroid f=%.1f] kappa=%.3f mu=%.3f delta=%.4f D=%.2f\n", f, res["kappa_eff"], res["mu_eff"], res["delta_iso"], res["D_eff"]); flush(stdout)
        end
        rsa = Dict{String,Any}(k => c[k] for k in ("rsa_min_radius", "rsa_min_distance", "rsa_shrink_factor", "rsa_max_attempts"))
        rsa["rsa_max_radius"] = c["rsa_max_radius"][fkey(f)]
        for seed in c["rsa_seeds"]
            χ_void = rsa_initialisation(c["N"], 1 - f, rsa["rsa_max_radius"], rsa["rsa_min_radius"], rsa["rsa_min_distance"];
                seed=seed, shrink_factor=rsa["rsa_shrink_factor"], max_attempts=rsa["rsa_max_attempts"])
            χ = 1.0 .- Float64.(χ_void)
            cfg = merge(base, rsa, Dict("family" => "RSA", "fraction" => f, "seed" => seed))
            res, data = report_reference(cfg, χ)
            dev = abs(res["achieved_solid_fraction"] - f)
            res["failed"] = dev > c["rsa_deviation_tolerance"]
            res["failure_reason"] = res["failed"] ? "achieved solid fraction $(round(res["achieved_solid_fraction"]; digits=4)) misses target $f by $(round(dev; digits=4)) > $(c["rsa_deviation_tolerance"])" : ""
            store(cfg, res, data)
            @printf("[S1 RSA f=%.1f seed=%d] achieved f=%.4f%s kappa=%.3f mu=%.3f delta=%.4f D=%.2f\n", f, seed, res["achieved_solid_fraction"],
                res["failed"] ? " FAILED" : "", res["kappa_eff"], res["mu_eff"], res["delta_iso"], res["D_eff"]); flush(stdout)
        end
    end
    rebuild_index!()
end

function run_one_kai(c, fam, f, seed)
    table = fam == "C" ? c["ckai"] : c["okai"]
    haskey(table, fkey(f)) || error("no [prod.$(fam == "C" ? "ckai" : "okai").$(fkey(f))] in configs/S1.toml -- run the S1p tuning first")
    params = Dict(k => v for (k, v) in table[fkey(f)] if k != "tuning_run")   # provenance only
    p = merge(kai_defaults(), c["kai_base"], params,
        Dict("family" => fam == "C" ? "C-Kai" : "O-Kai", "fraction" => f, "seed" => seed, "N" => c["N"], "phase_contrast" => c["phase_contrast"]))
    (fam == "O") == (p["weight_D"] > 0) || error("O-Kai needs weight_D > 0, C-Kai weight_D = 0")
    res, data = run_kai(p)
    store(p, merge(res, Dict("family" => p["family"], "fraction" => f, "seed" => seed)), data)
    println(kai_summary_line("S1 $(p["family"])", p, res)); flush(stdout)
    rebuild_index!()
end

function main(args)
    c = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S1.toml")), :prod)
    if args[1] == "refs"
        i = findfirst(==("--fractions"), args)
        fr = i === nothing ? Float64.(c["fractions"]) : parse.(Float64, split(args[i + 1], ","))
        run_refs(c, fr; rsa_only="--rsa-only" in args)
    elseif args[1] == "kai"
        run_one_kai(c, args[2], parse(Float64, args[3]), parse(Int, args[4]))
    else
        error("usage: S1_benchmark.jl refs | kai C|O f seed")
    end
end

main(ARGS)
