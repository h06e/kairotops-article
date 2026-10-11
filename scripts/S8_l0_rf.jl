# S8 -- O-Kai at f = 0.2: initialization length l0 x density-filter radius r_f (OQ-53). Design: configs/S8.toml.
# One generation per call, so the 25 runs can be queued one at a time on the GPU (scripts/S8_pipeline.sh):
#   julia --project=. scripts/S8_l0_rf.jl 0.01 0.0075            # l0 = 0.01, r_f = 0.0075, seed from the config
#   julia --project=. scripts/S8_l0_rf.jl 0.02 0.01 --smoke      # dev profile (N = 64, few iterations), study S8_smoke
#   julia --project=. scripts/S8_l0_rf.jl 0.02 0.01 --family C-Kai --fraction 0.1   # C-Kai extension (OQ-55)
# Base parameters: the S1 O-Kai recipe at f = 0.2 (configs/S1.toml [prod.kai_base] + [prod.okai.f20]); with --family C-Kai,
# the S1 C-Kai recipe at that fraction ([prod.ckai.fXX], one of configs/S8.toml [prod.ckai] fractions).
# Stores the covariogram of the thresholded physical field with each run.

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
include(joinpath(@__DIR__, "..", "src", "kai_runner.jl"))

function main(args)
    smoke = "--smoke" in args; a = filter(!=("--smoke"), args)
    opt(name) = (i = findfirst(==(name), a); i === nothing ? nothing : a[i + 1])
    fam = opt("--family"); fstr = opt("--fraction")
    a = [x for (i, x) in enumerate(a) if !(x in ("--family", "--fraction")) && !(i > 1 && a[i - 1] in ("--family", "--fraction"))]
    l0 = parse(Float64, a[1]); rf = parse(Float64, a[2])
    c = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S8.toml")), smoke ? :dev : :prod)
    c["recipe"] == "S1.okai.f20" || error("unknown recipe $(c["recipe"])")
    s1 = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S1.toml")), :prod)
    if fam === nothing                                   # the O-Kai grid at f = 0.2 (unchanged, OQ-53)
        fam, f, tab = c["family"], c["fraction"], s1["okai"]["f20"]
    else
        fam == "C-Kai" || error("--family: only C-Kai (the O-Kai grid is the default)")
        f = parse(Float64, fstr); any(isapprox(f, x) for x in c["ckai"]["fractions"]) || error("fraction $f not in configs/S8.toml ckai.fractions")
        tab = s1["ckai"][@sprintf("f%02d", round(Int, 100f))]
    end
    recipe = Dict{String,Any}(k => v for (k, v) in tab if k != "tuning_run")
    p = merge(kai_defaults(), s1["kai_base"], recipe, Dict("family" => fam, "fraction" => f, "seed" => c["seed"],
        "N" => c["N"], "phase_contrast" => c["phase_contrast"], "init_smoothing_radius" => l0, "density_filter_radius" => rf))
    haskey(c, "maxiter") && (p["maxiter"] = c["maxiter"])
    (p["weight_D"] > 0) == (fam == "O-Kai") || error("$fam with weight_D = $(p["weight_D"])")
    smoke && (p["smoke_test"] = true)
    res, data = run_kai(p)
    r, γ, _ = directional_covariogram(data["rho_thresholded"])
    data["covariogram_r"] = r; data["covariogram_gamma"] = γ
    store_kai(smoke ? "S8_smoke" : "S8", p, merge(res, Dict("l0" => l0, "density_filter_radius" => rf)), data;
        resolution_profile=smoke ? :dev : :prod)
    println(kai_summary_line("S8 $fam l0=$l0 rf=$rf", p, res))
    rebuild_index!()
end

main(ARGS)
