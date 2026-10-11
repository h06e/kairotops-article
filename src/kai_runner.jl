# Shared C-Kai / O-Kai generation + reporting, used by the S1 parameter
# investigation (scripts/S1p_param_investigation.jl) and the S1 production
# sweep (scripts/S1_benchmark.jl). Algorithm: docs/algorithm/ckai_algorithm.pdf
# (Algorithm 2: density filter + Heaviside projection; rf = beta0 = 0 gives
# the plain DirectClamp scheme with optional gradient smoothing lg).
# Requires src/common.jl and src/provenance.jl to be included first.

using JLD2, Printf, Statistics

"Default generation parameters (the S0r C-Kai recipe, OQ-27). Every key is part of the stored config."
kai_defaults() = Dict{String,Any}(
    "N" => 256, "phase_contrast" => 1000, "gradient" => "mandel_fixed",
    "init_smoothing_radius" => 0.02, "init_contrast" => 0.5,
    "density_filter_radius" => 0.005, "projection_beta_init" => 16.0, "projection_beta_max" => 64.0,
    "projection_beta_every" => 20, "projection_eta" => 0.5, "gradient_smoothing_radius" => 0.0,
    "weight_BW_init" => 0.5, "weight_update_alpha" => 0.99, "weight_update_beta" => 1.3,
    "maxiter" => 300, "tolerance" => 1e-3, "weight_D" => 0.0,
)

function kai_r_cov(χ)
    r, γ, _ = directional_covariogram(χ)
    return covariance_range(r, γ)
end

# r_cov of the physical field implied by a design snapshot: H_beta is monotone,
# so thresholding G*chi at its (1-f) quantile equals thresholding rho.
function kai_physical_rcov(χs, rf, f)
    g = Float64.(χs)
    rf > 0 && Kairotop.gaussian_smoothing!(g, rf)
    return kai_r_cov(Float64.(g .>= quantile(vec(g), 1 - f)))
end

"""
    kai_setup(p; elastic=nothing, diffusion=nothing) -> (problem, options, C_obj, K_obj)

Initial field, objectives and solver options of one kairotop generation (shared
by `run_kai` and the S6 detailed example, which records a denser history).
"""
function kai_setup(p; elastic=nothing, diffusion=nothing)
    isapprox(p["phase_contrast"], phase_stiff.E / phase_soft.E) || error("config phase_contrast $(p["phase_contrast"]) != materials in src/common.jl")
    n, f = p["N"], p["fraction"]
    χ0 = random_smoothed_initialisation(n, p["init_contrast"], p["init_smoothing_radius"], f; seed=p["seed"])
    C_obj = elastic === nothing ? isotropic_elastic_target(f) : elastic
    K_obj = diffusion === nothing ? open_diffusion_target() : diffusion
    open = p["weight_D"] > 0
    objectives = open ? Objectives(elastic=C_obj, solid_fraction=f, diffusion=K_obj) :
                        Objectives(elastic=C_obj, solid_fraction=f)
    problem = Problem(materials, objectives, χ0)
    options = SolverOptions(
        density_filter_radius=Float32(p["density_filter_radius"]), projection_β_init=Float32(p["projection_beta_init"]),
        projection_β_max=Float32(p["projection_beta_init"] > 0 ? p["projection_beta_max"] : 0), projection_β_every=p["projection_beta_every"],
        projection_η=Float32(p["projection_eta"]), gradient_smoothing_radius=Float32(p["gradient_smoothing_radius"]),
        weight_BW_init=Float32(p["weight_BW_init"]), weight_update_α=Float64(p["weight_update_alpha"]),
        weight_update_β=Float64(p["weight_update_beta"]), maxiter=p["maxiter"], tolerance=Float32(p["tolerance"]),
        weight_D=Float32(p["weight_D"]),
    )
    return problem, options, C_obj, K_obj
end

"""
    run_kai(p) -> (results::Dict, data::Dict)

Generate one kairotop with parameters `p` (kai_defaults() merged with
"fraction", "seed", "weight_D", ...), threshold the physical field at 0.5 and
report stiffness, anisotropy, feature size, diffusion and void connectivity.
Optional `elastic`/`diffusion` replace the default targets (isotropic Voigt
stiffness; the open-diffusion target when `weight_D > 0`) -- used by S3 for
anisotropic objectives, which then also reports the gaps J_L / J_K. `save_every` (default 10)
sets the history recording interval (`save_microstructure = false` keeps no 3D snapshots:
with `save_every = 1` they would cost 64 MB per iteration at N = 256 -- OQ-39); `iterations` / `final_J_BW` / `converged` are those of the
last *recorded* iteration (OQ-36), exact only with `save_every = 1`.
"""
function run_kai(p; elastic=nothing, diffusion=nothing, save_every=10, save_microstructure=true)
    n, f = p["N"], p["fraction"]
    problem, options, C_obj, K_obj = kai_setup(p; elastic, diffusion)
    workspace = CUDAWorkspace(problem, options.threads_per_block)
    elapsed = @elapsed history = Kairotop.solve!(options, workspace, problem; save_mode=SaveOn(every=save_every, save_microstructure=save_microstructure))
    Kairotop.update_physical!(workspace, options)
    χ = Array(workspace.χ); ρ = Array(workspace.ρ); ρ_bin = threshold_chi(ρ)
    final_β = workspace.projection_β[]
    workspace = nothing; GC.gc()
    Ceff, Λeff = homogenize_only(ρ_bin, Objectives(elastic=isotropic_elastic_target(f), solid_fraction=f, diffusion=open_diffusion_target()))
    kappa, mu = bulk_shear_moduli(Ceff)
    a = (Ceff[1,1]+Ceff[2,2]+Ceff[3,3])/3; b = (Ceff[1,2]+Ceff[1,3]+Ceff[2,3])/3; c = (Ceff[4,4]+Ceff[5,5]+Ceff[6,6])/3
    void = ρ_bin .< 0.5
    perc = percolates_periodic(void)
    keep = save_microstructure ? findall(it -> it % 10 == 0, history.iter) : Int[]   # r_cov history every 10 iterations, whatever save_every
    its = history.iter[keep]
    rcov_hist = [kai_physical_rcov(history.microstructure[k], p["density_filter_radius"], f) for k in keep]
    res = Dict{String,Any}(
        "kappa_eff" => kappa, "mu_eff" => mu, "delta_iso" => delta_iso(Ceff), "zener_A" => c / (a - b),
        "D_eff" => sum(Λeff[i, i] for i in 1:3) / 3, "void_percolates_xyz" => count(perc), "void_largest_cluster" => largest_cluster_fraction(void),
        "r_cov" => kai_r_cov(ρ_bin), "achieved_solid_fraction" => mean(ρ_bin), "iterations" => history.iter[end],
        "elapsed_seconds" => round(elapsed; digits=2), "final_projection_beta" => final_β, "final_J_BW" => history.J_BW[end],
        "converged" => history.J_BW[end] <= p["tolerance"] && (p["projection_beta_init"] == 0 || final_β >= p["projection_beta_max"]),
    )
    if elastic !== nothing || diffusion !== nothing
        res["J_L"] = gap_metric(Ceff, C_obj); res["J_K"] = gap_metric(Λeff, K_obj)
    end
    data = Dict{String,Any}("chi" => χ, "rho" => ρ, "rho_thresholded" => ρ_bin, "Ceff" => Ceff, "Λeff" => Λeff,
        "rcov_iters" => its, "rcov_hist" => rcov_hist)
    return res, data
end

"Store a run_kai result under `study` with provenance; returns the Run. `resolution_profile` defaults to :prod (N = 256 runs); smoke tests pass :dev."
function store_kai(study, config, res, data; resolution_profile=:prod)
    run = new_run(study, config)
    jldsave(joinpath(run.dir, "data.jld2"); NamedTuple(Symbol(k) => v for (k, v) in data)...)
    finish_run!(run, merge(Dict{String,Any}(k => config[k] for k in ("family", "fraction", "seed") if haskey(config, k)), res); resolution_profile)
    return run
end

function kai_summary_line(tag, p, r)
    kHS, mHS = hashin_shtrikman_bounds(phase_soft, phase_stiff, p["fraction"])
    return @sprintf("[%s f=%.2f seed=%d] kappa=%.3f (%.1f%% HS+) mu=%.3f (%.1f%% HS+) delta=%.4f A=%.3f r_cov=%.4f D=%.3f void_perc=%d/3 void_lcf=%.3f f=%.4f iters=%d beta=%.0f J_BW=%.1e conv=%s t=%.0fs",
        tag, p["fraction"], p["seed"], r["kappa_eff"], 100r["kappa_eff"]/kHS, r["mu_eff"], 100r["mu_eff"]/mHS, r["delta_iso"], r["zener_A"],
        r["r_cov"], r["D_eff"], r["void_percolates_xyz"], r["void_largest_cluster"], r["achieved_solid_fraction"], r["iterations"],
        r["final_projection_beta"], r["final_J_BW"], r["converged"], r["elapsed_seconds"])
end
