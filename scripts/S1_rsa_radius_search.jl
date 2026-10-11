# S1 RSA: per-fraction choice of the maximum void radius r_max (CPU only, no
# homogenization). The SI says r_max "is determined empirically to ensure the
# complete generation depending on f_obj"; Robin asked to keep it small when
# possible. Rule: the smallest r_max of the ladder for which seed 1 reaches the
# target solid fraction within 1e-3 (same r_min = min_distance = 0.015, shrink
# 0.9, 1e6 attempts as configs/S1.toml). Prints a table; the chosen values are
# copied into configs/S1.toml ([prod.rsa_max_radius]).
#   julia --project=. -t 8 scripts/S1_rsa_radius_search.jl

include(joinpath(@__DIR__, "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
using Printf, Statistics

c = resolve_config(load_config(joinpath(@__DIR__, "..", "configs", "S1.toml")), :prod)
ladder = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
fractions = Float64.(c["fractions"])
out = Vector{String}(undef, length(fractions))
Threads.@threads for i in eachindex(fractions)
    f = fractions[i]
    lines = String[]
    for r in ladder
        χv = rsa_initialisation(c["N"], 1 - f, r, c["rsa_min_radius"], c["rsa_min_distance"]; seed=1,
            shrink_factor=c["rsa_shrink_factor"], max_attempts=c["rsa_max_attempts"])
        af = 1 - mean(χv)
        ok = abs(af - f) <= c["rsa_deviation_tolerance"]
        push!(lines, @sprintf("  r_max=%.3f achieved f=%.4f %s", r, af, ok ? "OK" : ""))
        ok && (push!(lines, @sprintf("  -> f=%.1f: r_max = %.3f", f, r)); break)
    end
    out[i] = "f=$f\n" * join(lines, "\n")
end
foreach(println, out)
