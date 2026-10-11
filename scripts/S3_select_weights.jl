# Pick the diffusion weight per S3 case from the S3p seed-1 scan: the weight
# minimizing J_L + J_K (total gap to both targets). Writes the weight table into
# configs/S3.toml with --write.
#   julia --project=. scripts/S3_select_weights.jl [--write]
#
# The table header is matched only at the start of a line (a comment containing
# the header text truncated the config once, HANDOFF.md § 1); only that table's
# key lines are replaced, everything else is kept. Test: test/test_S3_select_weights.jl.

using JSON3, Printf, TOML

"""
    write_weight_table(txt, chosen) -> String

Replace the body of the `[prod.weight_D]` table of the TOML text `txt` by the
weights in `chosen` (case => weight). The header must start a line; the body
ends at the next table header or at the end of the file.
"""
function write_weight_table(txt::AbstractString, chosen::AbstractDict)
    m = match(r"^\[prod\.weight_D\][^\n]*\n?"m, txt)
    m === nothing && error("no [prod.weight_D] table header at the start of a line")
    body_start = m.offset + ncodeunits(m.match)
    nxt = findnext(r"^\["m, txt, body_start)
    rest = nxt === nothing ? "" : "\n" * txt[first(nxt):end]
    table = join(["$k = $(Float64(chosen[k]))" for k in sort(collect(keys(chosen)))], "\n") * "\n"
    head = txt[1:body_start-1]
    endswith(head, "\n") || (head *= "\n")
    return head * table * rest
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "..", "src", "common.jl"))
    include(joinpath(@__DIR__, "..", "src", "provenance.jl"))
    dir = joinpath(CAMPAIGN_ROOT, "results", "S3p")
    ms = [JSON3.read(read(joinpath(dir, h, "meta.json"), String), Dict{String,Any}) for h in readdir(dir)]
    chosen = Dict{String,Float64}()
    for case in ("D", "E", "F")
        rs = [m["extra"] for m in ms if get(m["extra"], "case", "") == case]
        isempty(rs) && continue
        println("== case $case")
        for e in sort(rs; by=e -> e["weight_D"])
            @printf("  wK=%-5g J_L=%.4f J_K=%.4f sum=%.4f\n", e["weight_D"], e["J_L"], e["J_K"], e["J_L"] + e["J_K"])
        end
        best = argmin(e -> e["J_L"] + e["J_K"], rs)
        chosen[case] = best["weight_D"]; println("  -> wK = ", best["weight_D"])
    end
    if "--write" in ARGS
        path = joinpath(CAMPAIGN_ROOT, "configs", "S3.toml")
        new = write_weight_table(read(path, String), chosen)
        # refuse to write anything that does not parse back with every section intact
        before = TOML.parse(read(path, String)); after = TOML.parse(new)
        for k in keys(before["prod"])
            k == "weight_D" || after["prod"][k] == before["prod"][k] || error("write would change [prod].$k; aborted")
        end
        after["prod"]["weight_D"] == chosen || error("written weights do not parse back; aborted")
        write(path, new)
        println("Wrote [prod.weight_D] into configs/S3.toml")
    end
end
