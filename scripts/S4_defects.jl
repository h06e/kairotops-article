# S4 -- robustness to material and geometrical defects, method in force (OQ-41).
# Design and defect definitions: configs/S4.toml. Structures: S1 production runs at f = 0.3
# (C-Kai, O-Kai, RSA seeds 1-5, gyroid); defect scales normalized by each structure's S1 r_cov.
#
# One stored run per (structure, defect type, amplitude) holding all realizations: effective
# tensors, kappa, mu, solid fraction after the defect, mean modulus factor of the solid
# (material), mid-plane slices of realization 1, and the full defected field of realization 1
# for seed-1 structures (renders).
#
#   julia --project=. scripts/S4_defects.jl main  [--smoke]     # SI protocol, all structures
#   julia --project=. scripts/S4_defects.jl sweep [--smoke]     # amplitude sweep, seed 1 (nearest-voxel warp)
#   julia --project=. scripts/S4_defects.jl main_smooth|lsweep|asweep [--smoke]   # fraction-preserving warp (OQ-42)
#   julia --project=. scripts/S4_defects.jl best [--smoke]      # article configuration on seeds 2-5 (OQ-44)
#   julia --project=. scripts/S4_defects.jl article [--smoke]   # article configuration on all structures in force, existing skipped (OQ-54)
# --smoke: S1 structures subsampled to 64^3, 2 realizations, study S4_smoke, dev profile.

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
using JLD2, JSON3, Printf, Statistics

const EPS = phase_soft.E / phase_stiff.E     # 1e-3: void / solid modulus ratio

"""
    apply_material_defect(ρ, e) -> χ

Solid voxels (ρ = 1) get χ = (e - EPS) / (1 - EPS), so that the linear mixture rule
C(χ) = C_v + χ (C_s - C_v) gives exactly e * C_s (both phases have ν = 0.3, so their
isotropic tensors are proportional, C_v = EPS * C_s); void voxels stay 0 (OQ-41).
Test: test/test_S4_material_defect.jl.
"""
apply_material_defect(ρ, e) = @. ρ * (e - EPS) / (1 - EPS)

function source_structures(c)
    dir = joinpath(CAMPAIGN_ROOT, "results", "S1")
    out = []; s1cfg = resolve_config(load_config(joinpath(CAMPAIGN_ROOT, "configs", "S1.toml")), :prod)
    for h in readdir(dir)
        isfile(joinpath(dir, h, "meta.json")) || continue
        m = JSON3.read(read(joinpath(dir, h, "meta.json"), String), Dict{String,Any})
        e = m["extra"]
        e["fraction"] ≈ c["fraction"] || continue
        fam = e["family"]; fam in c["families"] || continue
        get(e, "failed", false) && continue
        fam == "RSA" && m["config"]["rsa_max_radius"] != c["rsa_max_radius"] && continue
        s1_in_force(m, s1cfg) || continue            # kairotop recipe in force (O-Kai f <= 0.3 revised, OQ-54)
        seed = fam == "Gyroid" ? 0 : e["seed"]
        fam == "Gyroid" || seed in c["seeds"] || continue
        push!(out, (family=fam, seed=seed, hash=h, r_cov=e["r_cov"], kappa_S1=e["kappa_eff"], mu_S1=e["mu_eff"]))
    end
    return sort(out; by=s -> (findfirst(==(s.family), c["families"]), s.seed))
end

function load_structure(s, smoke)
    d = load(joinpath(CAMPAIGN_ROOT, "results", "S1", s.hash, "data.jld2"))
    χ = Float64.(haskey(d, "rho_thresholded") ? d["rho_thresholded"] : d["chi"])
    all(x -> x == 0 || x == 1, χ) || (χ = threshold_chi(χ))       # S1 reference fields are binary already
    return smoke ? χ[1:4:end, 1:4:end, 1:4:end] : χ
end

report_objectives(f) = Objectives(elastic=isotropic_elastic_target(f), solid_fraction=f, diffusion=open_diffusion_target())

function case_config(s, n, defect, amp, c, smoke, lf, rs)
    cfg = Dict{String,Any}("N" => n, "phase_contrast" => c["phase_contrast"], "fraction" => c["fraction"], "family" => s.family,
        "seed" => s.seed, "source_study" => "S1", "source_run" => s.hash, "r_cov" => s.r_cov, "defect_type" => defect,
        "n_realizations" => c["n_realizations"], "defect_seeds" => "1000*r", "length_factor" => lf,
        (defect == "material" ? "material_amplitude" : "amplitude_factor") => amp,
        "defect_application" => Dict("material" => "mixture_rule_exact", "geometric" => "nearest_voxel", "geometric_smooth" => "smooth_trilinear_exact_fraction")[defect])
    defect == "geometric_smooth" && (cfg["smoothing_radius"] = rs)
    smoke && (cfg["smoke_test"] = "v2")   # v1: two material runs stored before the Δf / config-key fixes
    return cfg
end

function run_case(s, χ, defect, amp, c, smoke; length_factor=nothing, skip_existing=false)
    n = size(χ, 1); f = c["fraction"]; R = c["n_realizations"]; f0 = mean(χ)   # actual solid fraction of the structure
    rs = defect == "geometric_smooth" ? c["smooth"]["smoothing_radius"] : 0.0
    lf = length_factor === nothing ? c[defect == "material" ? "material_length_factor" : "geometric_length_factor"] : length_factor
    cfg = case_config(s, n, defect, amp, c, smoke, lf, rs)
    if skip_existing && isfile(joinpath(CAMPAIGN_ROOT, "results", smoke ? "S4_smoke" : "S4", config_hash(cfg), "meta.json"))
        println("[S4] exists, skipped: $(s.family) seed $(s.seed) $defect"); return
    end
    # reference: the structure itself, or for the fraction-preserving defect its smoothed and
    # re-binarized version (δ = 0), so that only the warp is measured (OQ-42)
    χref = defect == "geometric_smooth" ? apply_geometric_defect_smooth(χ, nothing; smoothing_radius=rs, fraction=f0) : χ
    Cb, _ = homogenize_only(χref, report_objectives(f))
    κb, μb = bulk_shear_moduli(Cb)
    Co, _ = defect == "geometric_smooth" ? homogenize_only(χ, report_objectives(f)) : (Cb, nothing)
    if defect in ("geometric", "geometric_smooth")
        ℓ = lf * s.r_cov; a = amp * s.r_cov
    else
        ℓ = lf * s.r_cov; a = amp
    end
    Cs = zeros(6, 6, R); κ = zeros(R); μ = zeros(R); fdef = zeros(R); emean = fill(NaN, R)
    k = n ÷ 2; slice_def = nothing; field_r1 = nothing
    t = @elapsed for r in 1:R
        if defect == "geometric"
            χd = apply_geometric_defect(χ, geometric_distortion_field(n, ℓ, a; seed=1000r))
        elseif defect == "geometric_smooth"
            χd = apply_geometric_defect_smooth(χ, geometric_distortion_field(n, ℓ, a; seed=1000r); smoothing_radius=rs, fraction=f0)
        else
            e = material_defect_field(n, ℓ; amplitude=a, seed=1000r)
            χd = apply_material_defect(χ, e)
            emean[r] = sum(e .* χ) / sum(χ)
        end
        C, _ = homogenize_only(χd, report_objectives(f))
        Cs[:, :, r] = C; κ[r], μ[r] = bulk_shear_moduli(C)
        fdef[r] = defect == "material" ? mean(χ) : mean(χd)
        if r == 1
            slice_def = Float32.(χd[:, :, k])
            s.seed in (0, 1) && (field_r1 = defect == "material" ? Float32.(χd) : UInt8.(χd))
        end
    end
    run = new_run(smoke ? "S4_smoke" : "S4", cfg)
    jldsave(joinpath(run.dir, "data.jld2"); Ceff=Cs, Ceff_base=Cb, kappa=κ, mu=μ, fraction_defected=fdef, e_mean_solid=emean,
        slice_base=Float32.(χ[:, :, k]), slice_defected=slice_def, field_r1)
    dκ = 100 .* (κ ./ κb .- 1); dμ = 100 .* (μ ./ μb .- 1)
    res = Dict{String,Any}("family" => s.family, "seed" => s.seed, "fraction" => f, "defect_type" => defect,
        "amplitude" => amp, "l_defect" => ℓ, "a_defect" => defect == "material" ? NaN : a, "length_factor" => lf,
        "kappa_reference_vs_original" => κb / bulk_shear_moduli(Co)[1] - 1, "mu_reference_vs_original" => μb / bulk_shear_moduli(Co)[2] - 1,
        "kappa_base" => κb, "mu_base" => μb, "kappa_base_vs_S1" => κb / s.kappa_S1 - 1,
        "kappa_change_pct_mean" => mean(dκ), "kappa_change_pct_std" => std(dκ), "mu_change_pct_mean" => mean(dμ), "mu_change_pct_std" => std(dμ),
        "fraction_base" => f0, "fraction_change_mean" => mean(fdef) - f0, "fraction_change_std" => std(fdef), "e_mean_solid" => mean(emean),
        "elapsed_seconds" => round(t; digits=1))
    finish_run!(run, res; resolution_profile=smoke ? :dev : :prod)
    @printf("[S4%s] %-6s seed %d %-9s amp=%-5g l=%.4f | kappa %.3f: %+.2f ± %.2f %% | mu %.3f: %+.2f ± %.2f %% | Δf %+.4f | e_solid %.3f | base/S1-1 %.1e | %.0fs\n",
        smoke ? "_smoke" : "", s.family, s.seed, defect, amp, ℓ, κb, mean(dκ), std(dκ), μb, mean(dμ), std(dμ), mean(fdef) - f0,
        mean(emean), κb / s.kappa_S1 - 1, t)
end

function main(args)
    smoke = "--smoke" in args; mode = first(filter(!=("--smoke"), args))
    c = resolve_config(load_config(joinpath(CAMPAIGN_ROOT, "configs", "S4.toml")), smoke ? :dev : :prod)
    structs = source_structures(c)
    if mode == "main"
        for s in structs
            χ = load_structure(s, smoke)
            run_case(s, χ, "material", c["material_amplitude"], c, smoke)
            run_case(s, χ, "geometric", c["geometric_amplitude_factor"], c, smoke)
        end
    elseif mode == "sweep"
        for s in structs
            s.seed in (0, c["sweep"]["seeds"]...) || continue
            χ = load_structure(s, smoke)
            for a in c["sweep"]["material_amplitudes"]; run_case(s, χ, "material", a, c, smoke); end
            for a in c["sweep"]["geometric_amplitude_factors"]; run_case(s, χ, "geometric", a, c, smoke); end
        end
    elseif mode == "article"
        # article configuration on every structure in force, existing runs skipped (OQ-54: new O-Kai at f = 0.3)
        lf = c["best"]["length_factor"]
        for s in structs
            χ = load_structure(s, smoke)
            run_case(s, χ, "material", c["material_amplitude"], c, smoke; skip_existing=true)
            run_case(s, χ, "geometric_smooth", c["geometric_amplitude_factor"], c, smoke; length_factor=lf, skip_existing=true)
        end
    elseif mode == "best"
        # article configuration (OQ-44): l_delta = best.length_factor * r_cov on every structure;
        # seed 1 and the gyroid already exist from lsweep (identical config), so they are skipped
        lf = c["best"]["length_factor"]
        for s in structs
            s.seed in (0, 1) && continue
            run_case(s, load_structure(s, smoke), "geometric_smooth", c["geometric_amplitude_factor"], c, smoke; length_factor=lf)
        end
    elseif mode in ("main_smooth", "lsweep", "asweep")
        # fraction-preserving geometric defect (OQ-42): SI point on every structure; ℓ and a sweeps on seed 1
        sm = c["smooth"]
        for s in structs
            mode == "main_smooth" || s.seed in (0, 1) || continue
            χ = load_structure(s, smoke)
            if mode == "main_smooth"
                run_case(s, χ, "geometric_smooth", c["geometric_amplitude_factor"], c, smoke)
            elseif mode == "lsweep"
                for lf in sm["length_factors"]; lf == c["geometric_length_factor"] && continue
                    run_case(s, χ, "geometric_smooth", c["geometric_amplitude_factor"], c, smoke; length_factor=lf); end
            else
                for a in sm["amplitude_factors"]; a == c["geometric_amplitude_factor"] && continue
                    run_case(s, χ, "geometric_smooth", a, c, smoke); end
            end
        end
    else
        error("mode must be main | sweep | main_smooth | lsweep | asweep | best | article")
    end
    rebuild_index!()
end

main(ARGS)
