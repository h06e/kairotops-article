# Numbers of SI §3.1 from an S6 run (history every iteration): julia --project=. scripts/S6_numbers.jl <hash>   (validated on f951205fce46 against the previous SI text, OQ-54)
include(joinpath(@__DIR__, "..", "src", "common.jl")); include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
using JLD2, JSON3, Statistics, Printf
h = ARGS[1]; dir = joinpath(CAMPAIGN_ROOT, "results", "S6", h)
d = load(joinpath(dir, "data.jld2")); m = JSON3.read(read(joinpath(dir, "meta.json"), String), Dict{String,Any}); e = m["extra"]
f = Float64(e["fraction"]); kHS, mHS = hashin_shtrikman_bounds(phase_soft, phase_stiff, f)
dv, ds = phase_soft.d, phase_stiff.d; DHS = dv + f / (1 / (ds - dv) + (1 - f) / (3dv))
it = d["iters"]; K = it[end]; ix(k) = findfirst(==(k), it)
β = d["betas"]; J = d["J"]; JL = d["J_L"]; JK = d["J_K"]; JB = d["J_Bin"]; wB = d["w_Bin"]; tr = d["ls_trials"]
kt = d["kappa_thr"]; mt = d["mu_thr"]; Dt = d["D_thr"]; rc = d["rcov_thr"]; gc = d["grey_chi"]; gr = d["grey_rho"]
# gyroid at f
s1 = joinpath(CAMPAIGN_ROOT, "results", "S1")
gy = only(x for x in (JSON3.read(read(joinpath(s1, hh, "meta.json"), String), Dict{String,Any})["extra"] for hh in readdir(s1) if isfile(joinpath(s1, hh, "meta.json")))
          if x["family"] == "Gyroid" && x["fraction"] ≈ f)
dbl = [it[i] for i in 2:length(it) if β[i] != β[i-1]]
@printf("K=%d stop=%s J_Bin_final=%.2e beta=%.0f t=%.0fs trials=%d S1diff=%s S1ref=%s\n", K, e["stop_reason"], JB[end], β[end], e["elapsed_seconds"], sum(tr), get(e, "S1_seed1_differing_voxels", "?"), get(e, "S1_reference_run", "?"))
@printf("beta doublings at k = %s\n", string(dbl))
trk(k) = tr[k]   # ls_trials[k] = trials of optimization iteration k = 1..K; stages by the beta at the start of iteration k
for b in unique(β); ks = [it[i] for i in eachindex(it) if it[i] > 0 && β[i-1] == b]; @printf("  beta %.0f: %d iterations, %.1f trials/it\n", b, length(ks), mean(trk.(ks))); end
@printf("outcomes: %s\n", string(unique(d["ls_outcome"])))
@printf("k=0: grey chi %.2f rho %.2f | thr kappa %.1f%% mu %.1f%% HS+ | D_thr %.1f | rcov %.3f\n", gc[1], gr[1], 100kt[1]/kHS, 100mt[1]/mHS, Dt[1], rc[1])
for k in (10, 20, 30, 40); i = ix(k); i === nothing && continue
    @printf("k=%d: thr kappa %.1f%% of final, mu %.1f%% of final | rcov %.3f | grey chi %.2f rho %.2f | J_Bin %.2e\n", k, 100kt[i]/kt[end], 100mt[i]/mt[end], rc[i], gc[i], gr[i], JB[i]); end
fin = rc[end]; kfirst = it[findfirst(x -> x ≈ fin, rc)]; klast = it[findlast(i -> !(rc[i] ≈ fin), eachindex(rc)) + 1]
@printf("rcov final %.4f first reached k=%d, constant from k=%d\n", fin, kfirst, klast)
for k in dbl; i = ix(k); @printf("doubling at k=%d: J_Bin %.2e -> %.2e (ratio %.2f), grey rho %.2f -> %.2f, grey chi %.2f -> %.2f\n", k, JB[i-1], JB[i], JB[i-1]/JB[i], gr[i-1], gr[i], gc[i-1], gc[i]); end
# abrupt binarization of chi: largest 2-iteration drop
drop = [(gc[i] - gc[i+2], it[i]) for i in 1:length(gc)-2]; dmax = maximum(drop)
@printf("largest 2-it drop of grey chi: %.2f -> %.2f from k=%d to %d\n", gc[ix(dmax[2])], gc[ix(dmax[2])+2], dmax[2], dmax[2]+2)
kg = findfirst(i -> kt[i] > gy["kappa_eff"] && all(kt[i:end] .> gy["kappa_eff"]), eachindex(kt)); mg = findfirst(i -> mt[i] > gy["mu_eff"] && all(mt[i:end] .> gy["mu_eff"]), eachindex(mt))
@printf("stiffer than gyroid from k=%s (bulk), k=%s (shear)\n", kg === nothing ? "-" : it[kg], mg === nothing ? "-" : it[mg])
i40 = ix(dbl[end]); @printf("after last doubling (k=%d): %d iterations, thr props change < %.2f%%\n", dbl[end], K - dbl[end], 100maximum(abs.(kt[i40:end] ./ kt[end] .- 1)))
nw = count(i -> wB[i] > wB[i-1] * (1 + 1e-6), 2:length(wB)); kw = it[findfirst(==(wB[end]), wB)]
@printf("w_Bin increased %d times from %.2f to %.2f, reached at k=%d\n", nw, wB[1], wB[end], kw)
iL = argmin(JL); @printf("J_L %.3f -> min %.3f (k=%d) -> %.3f | J_K %.3f -> %.3f | J %.2f -> max %.2f (k=%d) -> %.2f\n", JL[1], JL[iL], it[iL], JL[end], JK[1], JK[end], J[1], maximum(J), it[argmax(J)], J[end])
@printf("thr factors final/initial: kappa %.1f mu %.1f\n", kt[end]/kt[1], mt[end]/mt[1])
@printf("FINAL (rho>=0.5): f=%.5f kappa %.2f (%.1f%% HS+) mu %.2f (%.1f%%) | gyroid %.1f%% %.1f%% | vs gyroid +%.1f%% +%.1f%% | K %.1f (%.1f%% HS+) gyroid %.1f (%.0f%% below) | perc %d | delta %.1f%% | rcov %.3f | D_thr0 %.1f\n",
    e["achieved_solid_fraction"], e["kappa_eff"], 100e["kappa_eff"]/kHS, e["mu_eff"], 100e["mu_eff"]/mHS, 100gy["kappa_eff"]/kHS, 100gy["mu_eff"]/mHS,
    100(e["kappa_eff"]/gy["kappa_eff"]-1), 100(e["mu_eff"]/gy["mu_eff"]-1), e["D_eff"], 100e["D_eff"]/DHS, gy["D_eff"], 100(1-e["D_eff"]/gy["D_eff"]), e["void_percolates_xyz"], 100e["delta_iso"], e["r_cov"], Dt[1])
@printf("rho0 thresholded at 0.5 gives f = %.3f\n", mean(d["rho0"] .>= 0.5))
