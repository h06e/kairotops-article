# Choose the per-fraction C-Kai / O-Kai parameters for S1 from the S1p tuning
# runs (seed 1, contrast 1000) and write them into configs/S1.toml.
#
# Rules (Robin, 2026-09-27: "same internal length as the latest generated",
# "good generation ... stiffness performances"):
#   eligible:  r_cov in [RCOV_MIN, RCOV_MAX] (internal length ~0.07)
#   C-Kai:     highest kappa_eff among eligible runs
#   O-Kai:     void percolating along 3 axes and stiffness not below the gyroid's
#              (kappa_eff >= kappa_Gyroid and mu_eff >= mu_Gyroid, Robin 2026-09-27),
#              then highest D_eff; if no run qualifies, the percolating eligible run
#              with the highest kappa_eff is taken and flagged.
#   julia --project=. scripts/S1_select_params.jl [--write]

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
using JSON3, Printf

const RCOV_MIN, RCOV_MAX = 0.055, 0.10
const TUNED_KEYS = ("init_smoothing_radius", "density_filter_radius", "projection_beta_init", "weight_BW_init", "weight_D", "init_contrast")

readmeta(study, h) = JSON3.read(read(joinpath(CAMPAIGN_ROOT, "results", study, h, "meta.json"), String), Dict{String,Any})
fkey(f) = @sprintf("f%02d", round(Int, 100f))

function main(args)
    s1 = [readmeta("S1", h) for h in readdir(joinpath(CAMPAIGN_ROOT, "results", "S1"))]
    gyrD = Dict(Float64(m["extra"]["fraction"]) => m["extra"]["D_eff"] for m in s1 if m["extra"]["family"] == "Gyroid")
    gyrK = Dict(Float64(m["extra"]["fraction"]) => m["extra"]["kappa_eff"] for m in s1 if m["extra"]["family"] == "Gyroid")
    gyrM = Dict(Float64(m["extra"]["fraction"]) => m["extra"]["mu_eff"] for m in s1 if m["extra"]["family"] == "Gyroid")
    runs = [(h, readmeta("S1p", h)) for h in readdir(joinpath(CAMPAIGN_ROOT, "results", "S1p"))]
    runs = [(h, m) for (h, m) in runs if get(m["config"], "phase_contrast", 0) == 1000 && m["config"]["seed"] == 1]
    choice = Dict{Tuple{String,Float64},Any}()
    for fam in ("C-Kai", "O-Kai"), f in sort(collect(keys(gyrD)))
        cand = [(h, m) for (h, m) in runs if m["config"]["family"] == fam && isapprox(m["config"]["fraction"], f)]
        isempty(cand) && continue
        println("\n== $fam f=$f  (gyroid kappa=$(round(gyrK[f]; digits=3)) mu=$(round(gyrM[f]; digits=3)) D=$(round(gyrD[f]; digits=2)))")
        elig(m) = RCOV_MIN <= m["extra"]["r_cov"] <= RCOV_MAX
        ok(m) = elig(m) && (fam == "C-Kai" || (m["extra"]["void_percolates_xyz"] == 3 &&
                                               m["extra"]["kappa_eff"] >= gyrK[f] && m["extra"]["mu_eff"] >= gyrM[f]))
        for (h, m) in sort(cand; by=x -> -x[2]["extra"]["kappa_eff"])
            e, c = m["extra"], m["config"]
            @printf("  %s %-5s kappa=%7.3f D=%7.2f r_cov=%.3f perc=%d delta=%.3f | %s\n", h, ok(m) ? "OK" : "--", e["kappa_eff"], e["D_eff"],
                e["r_cov"], e["void_percolates_xyz"], e["delta_iso"], join(["$k=$(c[k])" for k in TUNED_KEYS], " "))
        end
        good = [(h, m) for (h, m) in cand if ok(m)]
        flag = ""
        if isempty(good)
            good = [(h, m) for (h, m) in cand if elig(m) && (fam == "C-Kai" || m["extra"]["void_percolates_xyz"] == 3)]
            isempty(good) && (good = cand)
            best = argmax(x -> x[2]["extra"]["kappa_eff"], good)
            flag = "FALLBACK (no candidate meets the rule)"
        else
            best = argmax(x -> fam == "C-Kai" ? x[2]["extra"]["kappa_eff"] : x[2]["extra"]["D_eff"], good)
        end
        choice[(fam, f)] = (best..., flag)
        println("  -> chosen ", best[1], " ", flag)
    end
    "--write" in args || return
    block = IOBuffer()
    println(block, "# Per-fraction choices (scripts/S1_select_params.jl, rules in its header; OQ-32).")
    for (fam, tbl) in (("C-Kai", "ckai"), ("O-Kai", "okai"))
        println(block, "[prod.$tbl]")
        for f in sort([f for (fa, f) in keys(choice) if fa == fam])
            h, m, flag = choice[(fam, f)]
            println(block, "[prod.$tbl.$(fkey(f))]  # from results/S1p/$h: kappa=$(round(m["extra"]["kappa_eff"]; digits=3)), r_cov=$(round(m["extra"]["r_cov"]; digits=3)) $flag")
            for k in TUNED_KEYS
                println(block, "$k = ", m["config"][k])
            end
            println(block, "tuning_run = \"$h\"   # results/S1p/$h (seed 1): the one-seed S1 result")
        end
    end
    path = joinpath(CAMPAIGN_ROOT, "configs", "S1.toml")
    txt = read(path, String)
    i = findfirst("# Per-fraction choices", txt)
    write(path, txt[1:first(i)-1] * String(take!(block)))
    println("\nWrote per-fraction tables into configs/S1.toml")
end

main(ARGS)
