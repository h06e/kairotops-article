# S6 -- detailed optimization example for the supplementary material (v2 SI
# sec. 3.1, replaces Fig. S2): the S1 O-Kai at f = 0.3, seed 1, method in force,
# with the history recorded at every iteration. Config: configs/S6.toml.
#
# Stored per iteration k (data.jld2): mid-plane slices (z = N/2) of the design
# chi^k and of the physical field rho^k = H_beta_k(G * chi^k); cost terms
# J, J_L (= Kairotop's J_C), J_K (= J_D), J_Bin (= J_BW); w_Bin, beta_k, step
# length, number of line-search trials; greyness <4 x (1 - x)> of chi and rho;
# and, for rho^k thresholded at its (1 - f) quantile (exact fraction f):
# kappa, mu, D, delta_iso, r_cov. Full 3D fields kept: chi^0, rho^0 and its
# thresholded version, final chi, rho, rho thresholded at 0.5 (as in S1).
#
#   julia --project=. scripts/S6_detailed_example.jl --config configs/S6.toml [--profile dev|prod]

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
include(joinpath(@__DIR__, "..", "src", "kai_runner.jl"))

argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i+1])
const CONFIG_PATH = argval("--config", joinpath(CAMPAIGN_ROOT, "configs", "S6.toml"))
const PROFILE = Symbol(argval("--profile", "prod"))

# physical field of a design snapshot (same map as Kairotop.update_physical!)
function physical_field(χs, p, β)
    g = Float64.(χs)
    p["density_filter_radius"] > 0 && Kairotop.gaussian_smoothing!(g, p["density_filter_radius"])
    η = p["projection_eta"]
    return @.((tanh(β * η) + tanh(β * (g - η))) / (tanh(β * η) + tanh(β * (1 - η))))
end
beta_at(p, it) = min(p["projection_beta_init"] * 2.0^floor(it / p["projection_beta_every"]), p["projection_beta_max"])
threshold_fraction(ρ, f) = Float64.(ρ .>= quantile(vec(ρ), 1 - f))

function properties(ρb, f)
    Ceff, Λeff = homogenize_only(ρb, Objectives(elastic=isotropic_elastic_target(f), solid_fraction=f, diffusion=open_diffusion_target()))
    κ, μ = bulk_shear_moduli(Ceff)
    return κ, μ, sum(Λeff[i, i] for i in 1:3) / 3, delta_iso(Ceff), Ceff, Λeff
end

# the prod parameters must be the S1 ones (kai_base + okai.f30)
function check_against_S1(p)
    s1 = resolve_config(load_config(joinpath(CAMPAIGN_ROOT, "configs", "S1.toml")), :prod)
    ref = merge(Dict{String,Any}(), s1["kai_base"], s1["okai"]["f30"])
    for (k, v) in ref
        k == "tuning_run" && continue
        haskey(p, k) && p[k] != v && error("S6 $k = $(p[k]) differs from S1 ($v)")
    end
end

function main()
    cfg = resolve_config(load_config(CONFIG_PATH), PROFILE)
    delete!(cfg, "study_id")
    p = merge(kai_defaults(), cfg)
    PROFILE == :prod && check_against_S1(p)
    n, f = p["N"], p["fraction"]
    problem, options, _, _ = kai_setup(p)

    workspace = CUDAWorkspace(problem, options.threads_per_block)
    save = SaveOn(every=p["history_every"], save_microstructure=true, save_line_search=true)
    elapsed = @elapsed hist = Kairotop.solve!(options, workspace, problem; save_mode=save)
    Kairotop.update_physical!(workspace, options)
    χ_final = Array(workspace.χ); ρ_final = Array(workspace.ρ); ρ_bin = threshold_chi(ρ_final)
    final_β = workspace.projection_β[]
    workspace = nothing; GC.gc()
    @printf("[S6] solve done: %d iterations, %.0f s\n", hist.iter[end], elapsed)

    # consistency: the CPU physical map reproduces the solver's final rho
    ρ_check = physical_field(hist.microstructure[end], p, beta_at(p, hist.iter[end]))
    phys_err = maximum(abs.(ρ_check .- ρ_final))
    @printf("[S6] max |rho_cpu - rho_solver| at the last iteration: %.2e (beta=%.0f)\n", phys_err, final_β)

    k = n ÷ 2
    its = collect(hist.iter); m = length(its)
    chi_slices = Array{Float32}(undef, n, n, m); rho_slices = similar(chi_slices)
    betas = zeros(m); grey_chi = zeros(m); grey_rho = zeros(m); mean_rho = zeros(m)
    kap = zeros(m); mu = zeros(m); D = zeros(m); diso = zeros(m); rcov = zeros(m)
    ρ0 = ρ0_thr = nothing
    for (j, (it, χs)) in enumerate(zip(its, hist.microstructure))
        β = beta_at(p, it); ρs = physical_field(χs, p, β)
        chi_slices[:, :, j] = χs[:, :, k]; rho_slices[:, :, j] = ρs[:, :, k]
        betas[j] = β; grey_chi[j] = mean(4 .* χs .* (1 .- χs)); grey_rho[j] = mean(4 .* ρs .* (1 .- ρs)); mean_rho[j] = mean(ρs)
        ρt = threshold_fraction(ρs, f)
        kap[j], mu[j], D[j], diso[j], _, _ = properties(ρt, f)
        rcov[j] = kai_r_cov(ρt)
        j == 1 && (ρ0 = Float32.(ρs); ρ0_thr = ρt)
        @printf("[S6 post] it=%3d beta=%4.0f grey(rho)=%.4f kappa_thr=%.3f mu_thr=%.3f D_thr=%.2f r_cov=%.4f\n", it, β, grey_rho[j], kap[j], mu[j], D[j], rcov[j])
    end
    χ0 = hist.microstructure[1]
    ls_trials = [length(t.trials) for t in hist.line_search]
    ls_outcome = [String(t.outcome) for t in hist.line_search]
    hist.microstructure = Array{Float32,3}[]; GC.gc()

    # final structure, reported as in S1 (rho thresholded at 0.5)
    κ, μ, Dfin, δ, Ceff, Λeff = properties(ρ_bin, f)
    a = (Ceff[1,1]+Ceff[2,2]+Ceff[3,3])/3; b = (Ceff[1,2]+Ceff[1,3]+Ceff[2,3])/3; c = (Ceff[4,4]+Ceff[5,5]+Ceff[6,6])/3
    void = ρ_bin .< 0.5
    kHS, mHS = hashin_shtrikman_bounds(phase_soft, phase_stiff, f)
    tol_ok = hist.J_BW[end] <= p["tolerance"]
    res = Dict{String,Any}(
        "family" => p["family"], "fraction" => f, "seed" => p["seed"],
        "kappa_eff" => κ, "mu_eff" => μ, "kappa_over_HS" => κ / kHS, "mu_over_HS" => μ / mHS, "D_eff" => Dfin,
        "delta_iso" => δ, "zener_A" => c / (a - b), "r_cov" => kai_r_cov(ρ_bin), "achieved_solid_fraction" => mean(ρ_bin),
        "void_percolates_xyz" => count(percolates_periodic(void)), "void_largest_cluster" => largest_cluster_fraction(void),
        "iterations" => its[end], "elapsed_seconds" => round(elapsed; digits=2), "final_projection_beta" => final_β,
        "final_J_BW" => hist.J_BW[end], "final_J" => hist.J[end], "final_weight_black_white" => hist.weight_black_white[end],
        "converged" => tol_ok && final_β >= p["projection_beta_max"],
        "stop_reason" => (tol_ok && final_β >= p["projection_beta_max"]) ? "tolerance" : its[end] >= p["maxiter"] ? "maxiter" : "line_search_failure",
        "line_search_trials_total" => sum(ls_trials; init=0), "physical_map_check_maxabs" => phys_err,
    )
    # determinism check against the S1 production run of the same case in force (prod only; OQ-54: before the
    # revision of the O-Kai recipe this was results/S1/feba488635e9)
    s1cfg = resolve_config(load_config(joinpath(CAMPAIGN_ROOT, "configs", "S1.toml")), :prod); s1root = joinpath(CAMPAIGN_ROOT, "results", "S1")
    s1h = [h for h in readdir(s1root) if isfile(joinpath(s1root, h, "meta.json")) &&
           (m = JSON3.read(read(joinpath(s1root, h, "meta.json"), String), Dict{String,Any}); e = m["extra"];
            e["family"] == p["family"] && e["fraction"] ≈ p["fraction"] && get(e, "seed", -1) == p["seed"] && s1_in_force(m, s1cfg))]
    s1dir = joinpath(s1root, isempty(s1h) ? "none" : only(s1h))
    res["S1_reference_run"] = isempty(s1h) ? "none" : only(s1h)
    if PROFILE == :prod && isfile(joinpath(s1dir, "data.jld2"))
        ρs1 = JLD2.load(joinpath(s1dir, "data.jld2"), "rho_thresholded")
        res["S1_seed1_differing_voxels"] = count(ρs1 .!= ρ_bin)
    end

    run = new_run("S6", p)
    jldsave(joinpath(run.dir, "data.jld2"); iters=its, chi_slices, rho_slices, betas, grey_chi, grey_rho, mean_rho,
        kappa_thr=kap, mu_thr=mu, D_thr=D, delta_iso_thr=diso, rcov_thr=rcov,
        J=collect(hist.J), J_L=collect(hist.J_C), J_K=collect(hist.J_D), J_Bin=collect(hist.J_BW),
        w_Bin=collect(hist.weight_black_white), w_K=collect(hist.weight_diffusion), step=collect(hist.step_length),
        ls_trials, ls_outcome, chi0=χ0, rho0=ρ0, rho0_thresholded=ρ0_thr,
        chi_final=χ_final, rho_final=ρ_final, rho_thresholded=ρ_bin, Ceff, Λeff)
    finish_run!(run, res; resolution_profile=PROFILE)
    rebuild_index!()
    @printf("[S6 %s] run %s: kappa=%.3f (%.1f%% HS+) mu=%.3f (%.1f%% HS+) D=%.2f delta=%.4f r_cov=%.4f perc=%d/3 iters=%d J_BW=%.2e stop=%s t=%.0fs S1diff=%s\n",
        PROFILE, run.run_hash, κ, 100κ / kHS, μ, 100μ / mHS, Dfin, δ, res["r_cov"], res["void_percolates_xyz"], its[end],
        hist.J_BW[end], res["stop_reason"], elapsed, get(res, "S1_seed1_differing_voxels", "n/a"))
end

main()
