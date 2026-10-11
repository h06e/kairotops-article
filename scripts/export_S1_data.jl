# Export of the S1 microstructures in force (data/S1): one HDF5 file (deflate-compressed) per run with the binary
# microstructure `chi` (UInt8, 1 = solid, N x N x N, the physical density thresholded at 0.5 as for the reported
# properties), plus data/S1/index.csv (one line per run: properties as reported) and data/S1/configs.json (run hash ->
# full generation parameters). Reads results/S1 (run_S1.sh); the published repository ships the result.
#   julia --project=. scripts/export_S1_data.jl [results_root]
# Reading: Julia h5read("data/S1/<file>.h5", "chi"); Python h5py.File("data/S1/<file>.h5")["chi"][...] (axes in reverse
# order, as for any array written column-major; the microstructures are periodic cubes, so only the axis labels change).

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
using JLD2, JSON3, Printf, HDF5

root = isempty(ARGS) ? CAMPAIGN_ROOT : ARGS[1]
src = joinpath(root, "results", "S1"); out = joinpath(CAMPAIGN_ROOT, "data", "S1"); mkpath(out)
cfg = resolve_config(load_config(joinpath(CAMPAIGN_ROOT, "configs", "S1.toml")), :prod)
rows = String[]; configs = Dict{String,Any}()
push!(rows, "file,run_hash,family,fraction_target,seed,fraction_achieved,kappa_eff,mu_eff,K_eff,delta_iso,r_cov,void_percolating_axes,failed")
for h in sort(readdir(src))
    isfile(joinpath(src, h, "meta.json")) || continue
    m = JSON3.read(read(joinpath(src, h, "meta.json"), String), Dict{String,Any})
    (m["resolution_profile"] == "prod" && s1_in_force(m, cfg)) || continue
    e = m["extra"]; fam = e["family"]; f = e["fraction"]; seed = get(e, "seed", 0)
    d = load(joinpath(src, h, "data.jld2"))
    χ = haskey(d, "rho_thresholded") ? d["rho_thresholded"] : d["chi"]
    name = fam == "Gyroid" ? "$(fam)_f$(f)_$(h).h5" : "$(fam)_f$(f)_seed$(seed)_$(h).h5"
    h5open(joinpath(out, name), "w") do fid
        n = size(χ, 1)
        fid["chi", chunk=(n, n, 16), deflate=9] = UInt8.(χ .>= 0.5)
        HDF5.attributes(fid)["run_hash"] = h; HDF5.attributes(fid)["family"] = fam; HDF5.attributes(fid)["fraction_target"] = f
        HDF5.attributes(fid)["seed"] = seed; HDF5.attributes(fid)["description"] = "1 = solid, 0 = void; periodic unit cell of size 1, N^3 voxels"
    end
    g(k) = (v = get(e, k, nothing); v === nothing ? "" : string(v))
    push!(rows, join([name, h, fam, f, seed, g("achieved_solid_fraction"), g("kappa_eff"), g("mu_eff"), g("D_eff"), g("delta_iso"),
                      g("r_cov"), g("void_percolates_xyz"), get(e, "failed", false)], ","))
    configs[h] = m["config"]
end
write(joinpath(out, "index.csv"), join(rows, "\n") * "\n")
open(joinpath(out, "configs.json"), "w") do io; JSON3.pretty(io, configs); end
@printf("exported %d microstructures to %s (%.1f MB)\n", length(rows) - 1, out, sum(filesize(joinpath(out, f)) for f in readdir(out)) / 1e6)
