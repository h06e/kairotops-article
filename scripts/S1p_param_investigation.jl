# S1p -- parameter investigations feeding the S1 benchmark (contrast 1000,
# corrected gradient): O-Kai at f=0.3, C-Kai/O-Kai at f=0.1. One run per
# invocation line; every parameter of src/kai_runner.jl's kai_defaults() can
# be overridden as key=value, plus family=C|O, fraction, seeds.
#   julia --project=. scripts/S1p_param_investigation.jl family=O fraction=0.3 weight_D=0.5
#   julia --project=. scripts/S1p_param_investigation.jl family=C fraction=0.1 density_filter_radius=0.003 seeds=1,2

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
include(joinpath(@__DIR__, "..", "src", "kai_runner.jl"))

function main(args)
    p = kai_defaults(); p["family"] = "C"; p["fraction"] = 0.3; seeds = [1]
    for a in args
        k, v = split(a, "="; limit=2)
        if k == "seeds"; seeds = parse.(Int, split(v, ",")); continue; end
        k == "family" && (p[k] = v; continue)
        haskey(p, k) || k == "fraction" || error("unknown parameter $k")
        p[k] = occursin(r"^-?\d+$", v) && k in ("maxiter", "projection_beta_every", "N") ? parse(Int, v) : parse(Float64, v)
    end
    p["family"] in ("C", "O") || error("family must be C or O")
    (p["family"] == "O") == (p["weight_D"] > 0) || error("family=O needs weight_D>0, family=C needs weight_D=0")
    for seed in seeds
        q = merge(p, Dict("seed" => seed, "family" => p["family"] == "C" ? "C-Kai" : "O-Kai"))
        res, data = run_kai(q)
        store_kai("S1p", q, res, data)
        extra = join(["$k=$(q[k])" for k in sort(collect(keys(q))) if haskey(kai_defaults(), k) && q[k] != kai_defaults()[k]], " ")
        println(kai_summary_line("S1p $(q["family"])", q, res), "  | overrides: ", extra)
        println("  r_cov(rho) history: ", join([@sprintf("it%d:%.3f", i, v) for (i, v) in zip(data["rcov_iters"], data["rcov_hist"])], " "))
        flush(stdout)
    end
    rebuild_index!()
end

main(ARGS)
