# S0s -- C-Kai optimization walkthrough (contrast 1000, corrected gradient):
# how the design chi and the physical field rho evolve under different
# binarization strategies, f=0.3, seed 1, N=256. Records every 2 iterations:
# mid-plane slices (z = N/2) of chi and rho, grey levels, r_cov(rho), and the
# cost terms.
#
# Variants (all share l0=0.02, rf=0.005 unless stated):
#   A_sharp_start      recipe (docs/algorithm/ckai_algorithm.pdf, Alg. 2): beta0=16, c0=0.5, w0=0.5
#   B_weak_start       weak binarization start: beta0=1, c0=0.05, w0=0.001
#   C_beta_only        beta continuation only, no binarization penalty: beta0=1, c0=0.05, w0=0
#   D_penalty_only     reference-style: no filter/projection, J_Bin only, gradient smoothing lg=0.005, c0=0.05, w0=0.001
#
#   julia --project=. scripts/S0s_optimization_walkthrough.jl A_sharp_start

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
include(joinpath(@__DIR__, "..", "src", "kai_runner.jl"))

const VARIANTS = Dict(
    "A_sharp_start" => Dict("projection_beta_init" => 16.0, "init_contrast" => 0.5, "weight_BW_init" => 0.5),
    "B_weak_start" => Dict("projection_beta_init" => 1.0, "init_contrast" => 0.05, "weight_BW_init" => 0.001),
    "C_beta_only" => Dict("projection_beta_init" => 1.0, "init_contrast" => 0.05, "weight_BW_init" => 0.0, "maxiter" => 200),
    "D_penalty_only" => Dict("projection_beta_init" => 0.0, "density_filter_radius" => 0.0, "gradient_smoothing_radius" => 0.005,
                             "init_contrast" => 0.05, "weight_BW_init" => 0.001),
)

# physical field for a design snapshot at iteration `it` (beta from the continuation schedule)
function physical_field(χs, p, it)
    g = Float64.(χs)
    p["density_filter_radius"] > 0 && Kairotop.gaussian_smoothing!(g, p["density_filter_radius"])
    β0 = p["projection_beta_init"]
    β0 > 0 || return g, 0.0
    β = min(β0 * 2.0^floor(it / p["projection_beta_every"]), p["projection_beta_max"])
    η = p["projection_eta"]
    return @.((tanh(β * η) + tanh(β * (g - η))) / (tanh(β * η) + tanh(β * (1 - η)))), β
end

function main(args)
    name = args[1]
    p = merge(kai_defaults(), VARIANTS[name], Dict("fraction" => 0.3, "seed" => 1, "family" => "C-Kai", "variant" => name))
    n, f = p["N"], p["fraction"]
    χ0 = random_smoothed_initialisation(n, p["init_contrast"], p["init_smoothing_radius"], f; seed=p["seed"])
    problem = Problem(materials, Objectives(elastic=isotropic_elastic_target(f), solid_fraction=f), χ0)
    options = SolverOptions(
        density_filter_radius=Float32(p["density_filter_radius"]), projection_β_init=Float32(p["projection_beta_init"]),
        projection_β_max=Float32(p["projection_beta_init"] > 0 ? p["projection_beta_max"] : 0), projection_β_every=p["projection_beta_every"],
        projection_η=Float32(p["projection_eta"]), gradient_smoothing_radius=Float32(p["gradient_smoothing_radius"]),
        weight_BW_init=Float32(p["weight_BW_init"]), weight_update_α=Float64(p["weight_update_alpha"]),
        weight_update_β=Float64(p["weight_update_beta"]), maxiter=p["maxiter"], tolerance=Float32(p["tolerance"]), weight_D=0f0,
    )
    workspace = CUDAWorkspace(problem, options.threads_per_block)
    elapsed = @elapsed hist = Kairotop.solve!(options, workspace, problem; save_mode=SaveOn(every=2, save_microstructure=true))
    Kairotop.update_physical!(workspace, options)
    ρ_final = Array(workspace.ρ); ρ_bin = threshold_chi(ρ_final)
    workspace = nothing; GC.gc()

    k = n ÷ 2
    its = hist.iter
    chi_slices = Array{Float32}(undef, n, n, length(its)); rho_slices = similar(chi_slices)
    grey_chi = Float64[]; grey_rho = Float64[]; betas = Float64[]; rcov = Float64[]; frac_rho = Float64[]
    for (j, (it, χs)) in enumerate(zip(its, hist.microstructure))
        ρs, β = physical_field(χs, p, it)
        chi_slices[:, :, j] = χs[:, :, k]; rho_slices[:, :, j] = ρs[:, :, k]
        push!(grey_chi, mean(4 .* χs .* (1 .- χs))); push!(grey_rho, mean(4 .* ρs .* (1 .- ρs)))
        push!(betas, β); push!(frac_rho, mean(ρs))
        push!(rcov, kai_r_cov(Float64.(ρs .>= quantile(vec(ρs), 1 - f))))
    end
    Ceff, _ = homogenize_only(ρ_bin, Objectives(elastic=isotropic_elastic_target(f), solid_fraction=f, diffusion=open_diffusion_target()))
    kappa, mu = bulk_shear_moduli(Ceff)
    kHS, mHS = hashin_shtrikman_bounds(phase_soft, phase_stiff, f)
    res = Dict{String,Any}("kappa_eff" => kappa, "mu_eff" => mu, "kappa_over_HS" => kappa / kHS, "delta_iso" => delta_iso(Ceff),
        "r_cov" => kai_r_cov(ρ_bin), "iterations" => its[end], "elapsed_seconds" => round(elapsed; digits=1),
        "final_grey_rho" => grey_rho[end], "final_J_BW" => hist.J_BW[end], "achieved_solid_fraction" => mean(ρ_bin))
    run = new_run("S0s", p)
    jldsave(joinpath(run.dir, "data.jld2"); iters=collect(its), chi_slices, rho_slices, grey_chi, grey_rho, betas, rcov, frac_rho,
        J=collect(hist.J), J_C=collect(hist.J_C), J_BW=collect(hist.J_BW), w_BW=collect(hist.weight_black_white),
        step=collect(hist.step_length), rho_final=ρ_final, rho_thresholded=ρ_bin, Ceff=Ceff)
    finish_run!(run, merge(Dict{String,Any}("variant" => name, "fraction" => f, "seed" => 1, "family" => "C-Kai"), res); resolution_profile=:prod)
    rebuild_index!()
    @printf("[S0s %s] kappa=%.3f (%.1f%% HS+) mu=%.3f delta=%.4f r_cov=%.4f iters=%d grey(rho)=%.3f J_BW=%.1e t=%.0fs\n",
        name, kappa, 100kappa/kHS, mu, delta_iso(Ceff), res["r_cov"], its[end], grey_rho[end], hist.J_BW[end], elapsed)
end

main(ARGS)
