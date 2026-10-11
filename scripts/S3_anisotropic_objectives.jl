# S3 -- anisotropic objectives with the method in force. Design: configs/S3.toml.
# One generation per call (queued one at a time by scripts/S23_pipeline.sh):
#   julia --project=. scripts/S3_anisotropic_objectives.jl scan  E 7.0        # S3p: seed 1, weight scan
#   julia --project=. scripts/S3_anisotropic_objectives.jl prod  E 2          # S3: case E, seed 2, selected weight
#   julia --project=. scripts/S3_anisotropic_objectives.jl sweep 0.2 0.05 3   # S3_sweep: case E, f, l0 (r_f = l0/4), seed
#   julia --project=. scripts/S3_anisotropic_objectives.jl tune  F 3.0 0.03 pd # S3t: tuning, seed 1, target variant (OQ-39)
# Append --smoke to test the chain: N = 64, 5 iterations, stored as dev runs under
# <study>_smoke (never mixed with the prod studies read by the notebook).

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
include(joinpath(@__DIR__, "..", "src", "kai_runner.jl"))
using LinearAlgebra

include(joinpath(@__DIR__, "S3_case_targets.jl"))

function generate(study, case, f, seed, wK, l0, rf, c; smoke=false, variant="", save_every=10)
    base = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S1.toml")), :prod)["kai_base"]
    p = merge(kai_defaults(), base, Dict("family" => "Kai-$case", "case" => case, "fraction" => f, "seed" => seed, "N" => c["N"],
        "phase_contrast" => c["phase_contrast"], "init_smoothing_radius" => l0, "density_filter_radius" => rf, "weight_D" => wK))
    variant == "" || (p["target_variant"] = variant)
    save_every == 10 || (p["history_every"] = save_every)
    smoke && (p["N"] = 64; p["maxiter"] = 5; p["smoke_test"] = true; study *= "_smoke")
    C, K = case_targets(case, f; variant)
    res, data = run_kai(p; elastic=C, diffusion=K, save_every, save_microstructure=save_every == 10)   # no per-iteration 3D snapshots (memory, OQ-39)
    data["C_target"] = C; data["K_target"] = K
    res["case"] = case; res["l0"] = l0; res["weight_D"] = wK
    res["L1111_over_L2222"] = data["Ceff"][1, 1] / data["Ceff"][2, 2]
    res["K33_over_K11"] = data["Λeff"][3, 3] / data["Λeff"][1, 1]
    res["K13"] = data["Λeff"][1, 3]
    merge!(res, Dict("shape_" * k => v for (k, v) in shape_metrics(data["Ceff"], data["Λeff"])))
    variant == "" || (res["target_variant"] = variant)
    if save_every == 1   # exact stop reason (OQ-36): tolerance at beta_max, maxiter, or line-search failure
        res["stop_reason"] = res["converged"] ? "tolerance" : res["iterations"] >= p["maxiter"] ? "maxiter" : "line_search_failure"
    end
    store_kai(study, p, res, data; resolution_profile=smoke ? :dev : :prod)
    @printf("[%s case %s f=%.1f l0=%.3f rf=%.4f wK=%g seed=%d] J_L=%.4f J_K=%.4f L11/L22=%.2f K33/K11=%.2f K13=%.2f kappa=%.3f r_cov=%.3f iters=%d t=%.0fs\n",
        study, case, f, l0, rf, wK, seed, res["J_L"], res["J_K"], res["L1111_over_L2222"], res["K33_over_K11"], res["K13"],
        res["kappa_eff"], res["r_cov"], res["iterations"], res["elapsed_seconds"])
    rebuild_index!()
end

function main(args)
    smoke = "--smoke" in args; args = filter(!=("--smoke"), args)
    c = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S3.toml")), :prod)
    mode = args[1]
    if mode == "scan"
        generate("S3p", args[2], c["fraction"], 1, parse(Float64, args[3]), c["init_smoothing_radius"], c["density_filter_radius"], c; smoke)
    elseif mode == "prod"
        case = args[2]; haskey(c["weight_D"], case) || error("no selected weight for case $case (run scripts/S3_select_weights.jl --write)")
        generate("S3", case, c["fraction"], parse(Int, args[3]), c["weight_D"][case], c["init_smoothing_radius"], c["density_filter_radius"], c; smoke)
    elseif mode == "sweep"
        haskey(c["weight_D"], "E") || error("no selected weight for case E")
        f = parse(Float64, args[2]); l0 = parse(Float64, args[3])
        generate("S3_sweep", "E", f, parse(Int, args[4]), c["weight_D"]["E"], l0, l0 / 4, c; smoke)
    elseif mode == "tune"
        # S3t (OQ-39): seed 1, history at every iteration, optional target variant
        #   tune CASE wK l0 [variant]     (r_f = l0 / 4)
        case = args[2]; wK = parse(Float64, args[3]); l0 = parse(Float64, args[4])
        generate("S3t", case, c["fraction"], 1, wK, l0, l0 / 4, c; smoke, variant=get(args, 5, ""), save_every=1)
    else
        error("mode must be scan | prod | sweep | tune")
    end
end

main(ARGS)
