# optitopo5 — shared campaign helpers, included by every S0-S5 study script
#
# Materials, target-tensor builders, the Hashin-Shtrikman bound, and solver
# defaults, built on top of Kairotop's public API. Every `scripts/S*.jl`
# starts with `include(joinpath(@__DIR__, "..", "src", "common.jl"))`.
#
# Resolution: current default is N=64 (dev, GPU iteration speed); production
# is N=256/512 — see `params/README.md` for the per-study dev/prod switch.
# `const N` below is this file's own default, overridden per-study by each
# config's resolved `N` field where applicable.

using Kairotop
using Statistics
using LinearAlgebra
using Printf
using JLD2
using Random

#***************************************************************
#* GRID RESOLUTION — bump to 256 for production runs (see brief).

const N = 64

#***************************************************************
#* MATERIALS
#
# Phase contrast 1000 for both physics (Robin, 2026-09-26: "all the work
# should be done with a contrast of 1000"), matching Ataxiotel/optitopo4
# (E=1e-3/E=1). Until then this was Kairotop's example pair (E=1/100,
# D=100/1, contrast 100). The stiff phase keeps E=100 so moduli stay on the
# same scale as before; the soft phase drops to E=0.1. For diffusion the
# soft/background phase keeps the *high* coefficient (it is the open
# channel; open_diffusion_target() = phase_soft.d = 100 is unchanged) and
# the stiff phase drops to 0.1. phase0 = soft/background (χ=0), phase1 =
# stiff/structural (χ=1).

phase_soft = PhaseMaterial(0.1, 0.3; diffusion=100.0)     # phase0 — background / channel phase
phase_stiff = PhaseMaterial(100.0, 0.3; diffusion=0.1)    # phase1 — structural / stiff phase
materials = TwoPhaseMaterial(phase_soft, phase_stiff)

#***************************************************************
#* RENDERING
#
# Standing requirement (Robin, 2026-09-24): every generated microstructure
# from here on gets a Blender render, grey with a touch of blue. Notebooks
# call it directly on a loaded χ field (not scripts — rendering an
# already-computed result isn't "recomputing", same as any other notebook
# plot) with `color=RENDER_COLOR`.
#
# Use Kairotop.render_microstructure_smoothed_interior (curved cavities,
# exactly flat boundary faces, no periodic-padding+Boolean-clip pipeline —
# see docs/OPEN_QUESTIONS.md OQ-17) — not render_microstructure_surface,
# which has repeatedly missed cavities on some microstructures despite two
# rounds of fixes (OQ-16). render_microstructure_surface remains the right
# choice only for a fully-periodic tiling preview, not this campaign's own
# per-microstructure renders.

const RENDER_COLOR = (0.44, 0.46, 0.51)  # darker grey with a touch of blue -- darkened from (0.58,0.60,0.66) for better contrast against cavity shadows (Robin, 2026-09-25)

#***************************************************************
#* TARGET TENSOR BUILDERS

"""
    isotropic_elastic_target(solid_fraction)

Voigt-average ("rule of mixtures") isotropic stiffness target at the given
solid fraction — same construction as Kairotop's own examples
(`kappa_obj .* PROJ_ISO_J + mu_obj .* PROJ_ISO_K`).
"""
function isotropic_elastic_target(solid_fraction::Real)
    kappa_obj = solid_fraction * phase_stiff.kappa + (1 - solid_fraction) * phase_soft.kappa
    mu_obj = solid_fraction * phase_stiff.mu + (1 - solid_fraction) * phase_soft.mu
    return kappa_obj .* Kairotop.PROJ_ISO_J + mu_obj .* Kairotop.PROJ_ISO_K
end

"""
    open_diffusion_target()

Isotropic conductivity target used by the "open" (permeability-objective)
studies: the soft/background phase's own diffusion coefficient, applied
uniformly — an intentionally aggressive target (as if the whole domain were
the high-diffusivity phase) that pushes the optimizer towards a maximally
connected channel network, rather than a value meant to be achieved exactly.
Matches Kairotop's diffusion_KRTP.jl example value (100) by construction,
since phase_soft.d == 100.
"""
open_diffusion_target() = phase_soft.d * Matrix{Float64}(I, 3, 3)

#***************************************************************
#* HASHIN-SHTRIKMAN UPPER BOUND (reference curve for the plots)

"""
    hashin_shtrikman_bulk_shear(solid_fraction)

Hashin-Shtrikman upper bound for the effective bulk/shear moduli of a
two-phase composite at the given solid fraction of the stiffer phase
(`phase_stiff`) — the classical isotropic bound plotted alongside the
optimized/baseline microstructures in the density-benchmark figure.
"""
function hashin_shtrikman_bulk_shear(solid_fraction::Real)
    k0, mu0 = phase_soft.kappa, phase_soft.mu
    k1, mu1 = phase_stiff.kappa, phase_stiff.mu
    f = solid_fraction

    kappa_hs = k1 + (1 - f) / (1 / (k0 - k1) + (3 * f) / (3 * k1 + 4 * mu1))
    mu_hs = mu1 + (1 - f) / (1 / (mu0 - mu1) + (6 * (k1 + 2 * mu1) * f) / (5 * mu1 * (3 * k1 + 4 * mu1)))

    return kappa_hs, mu_hs
end

#***************************************************************
#* SCORING (Voigt projection of an effective tensor onto kappa/mu)

"""
    tensor_to_kappa_mu(C)

Project a 6x6 Voigt stiffness tensor `C` onto its isotropic
hydrostatic/deviatoric components via `PROJ_ISO_J`/`PROJ_ISO_K` — the same
projection used throughout the optitopo4 plotting notebook's `score`
functions, here shared by every runner/plot script that needs it.
"""
function tensor_to_kappa_mu(C::AbstractMatrix)
    proj_k = Kairotop.PROJ_ISO_J
    proj_m = Kairotop.PROJ_ISO_K
    kappa = sum(C .* proj_k) / sum(proj_k .* proj_k)
    mu = sum(C .* proj_m) / sum(proj_m .* proj_m)
    return kappa, mu
end

#***************************************************************
#* GYROID BASELINE GENERATOR
#
# Kairotop ships three initial-guess generators (random-smoothed, Voronoi,
# RSA) but no TPMS/gyroid one — this is a small, self-contained port of
# Ataxiotel's `generate_gyro` (microstructure_utils.jl), kept local to
# optitopo5 rather than added to Kairotop, since it's only ever used here as
# a *baseline* geometry to homogenize once (no optimization), exactly like
# `rsa_initialisation` is used in `examples/rsa_cost_KRTP.jl`.

"""
    gyroid_microstructure(n, solid_fraction)

Build an `n x n x n` binary (0/1) periodic gyroid microstructure at the
given solid fraction: threshold the gyroid level-set function
`sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)` at the value achieving that
volume fraction exactly.
"""
function gyroid_microstructure(n::Int, solid_fraction::Real)
    0 < solid_fraction < 1 || throw(ArgumentError("solid_fraction must be in (0, 1)."))

    xs = range(0, 2π; length=n)
    ys = range(0, 2π; length=n)
    zs = range(0, 2π; length=n)

    f(x, y, z) = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)
    img = [f(x, y, z) for x in xs, y in ys, z in zs]

    img .-= minimum(img)
    img ./= maximum(img)

    flat = vec(img)
    k = round(Int, (1 - solid_fraction) * length(flat))
    threshold = partialsort(flat, k)

    return Float64.(img .> threshold)
end

#***************************************************************
#* SOLVER DEFAULTS
#
# One shared, single-stage solver configuration for every optimization
# runner (closed/open stiffness sweeps, size study, anisotropic studies) —
# no multi-stage reinitialization loops (see brief). Values follow Kairotop's
# own tuned example (examples/max_stiffness_KRTP.jl) rather than
# optitopo4/Ataxiotel's outdated Dict-based parameters; retune once real
# (N=256) runs are needed.

"""
    default_solver_options(smoothing_radius; maxiter=300, weight_D=1.0)

Shared single-stage `SolverOptions`: `gradient_smoothing_radius` tied to the
initial guess's own feature size (`smoothing_radius`, per Kairotop's own
example guidance), `weight_BW_init`/`weight_update_α`/`weight_update_β` as
in `examples/max_stiffness_KRTP.jl`. `weight_D` only matters when the
problem's `Objectives` sets a `diffusion` target.
"""
function default_solver_options(smoothing_radius::Real; maxiter::Int=300, weight_D::Real=1.0)
    return SolverOptions(
        weight_BW_init=0.001f0,
        maxiter=maxiter,
        gradient_smoothing_radius=Float32(smoothing_radius),
        weight_update_α=0.999,
        weight_update_β=2,
        weight_D=Float32(weight_D),
    )
end

"Shared history-recording mode: cost/weights every iteration, χ snapshot every 10th — same granularity as Kairotop's own max_stiffness_KRTP.jl example."
default_save_mode() = SaveOn(every=10, save_microstructure=true)

#***************************************************************
#* PLAIN HOMOGENIZATION (no optimization) — for the RSA/gyroid baselines

"""
    homogenize_only(χ, objectives)

Build a `Problem`/`CUDAWorkspace` for the (already-final, no optimization
needed) microstructure `χ` and homogenize it once — the same "just
homogenize a fixed geometry" pattern as
`examples/rsa_cost_KRTP.jl`. Returns `(Ceff, Λeff)`.
"""
function homogenize_only(χ::AbstractArray{<:Real,3}, objectives::Objectives)
    problem = Problem(materials, objectives, χ)
    options = SolverOptions()   # only LS_it_max/LS_tol_equi/threads_per_block matter here
    workspace = CUDAWorkspace(problem, options.threads_per_block)
    copyto!(workspace.χ, problem.χ0)
    Ceff, Λeff = homogenize!(workspace, problem, options; save_fields=false)
    return Ceff, Λeff
end

"""
    threshold_chi(χ; threshold=0.5)

Binarize `χ` at `threshold` (`>= threshold` -> `1.0`, else `0.0`).

`homogenize!`/`homogenize_only` treat `χ` as a continuous SIMP-style
material-interpolation weight, not an automatically-binarized field — a
run that hasn't fully converged its black-white penalty (or was
deliberately stopped early, e.g. S0d's speed-tuned variants) can still
have meaningfully grey voxels. Effective properties (`kappa_eff`,
`mu_eff`, `delta_iso`, `Ceff`, `Λeff`) are always reported on this
thresholded, as-manufactured version — a report on the raw, possibly-grey
field doesn't correspond to any physically realizable two-phase
structure. Note this only affects what's fed to `homogenize_only` for
*reporting*; the saved/rendered `χ` itself stays the raw optimizer
output (meshing/rendering already applies its own 0.5 threshold at the
voxel level, and keeping the raw field in slice plots is itself a useful
diagnostic of how close to binary a run actually got).
"""
threshold_chi(χ::AbstractArray{<:Real,3}; threshold::Real=0.5) = Float64.(χ .>= threshold)

#***************************************************************
#* SAVE HELPER

"""
    save_result(path, result::Dict)

Thin wrapper around `JLD2.jldsave`/`save_object` for consistency across
runners — every runner saves a single `Dict` under `data/`.
"""
function save_result(path::AbstractString, result::Dict)
    mkpath(dirname(path))
    save_object(path, result)
    println("Saved: ", abspath(path))
    return path
end

#***************************************************************
#* SHAPE METRICS OF EFFECTIVE TENSORS (S3, OQ-39)

"""
    shape_metrics(Ceff, Keff) -> Dict

Shape (not magnitude) indicators of an effective stiffness (Kelvin 6x6) and diffusion
(3x3) tensor, used to judge the S3 anisotropic cases:
- `delta_iso`: distance of `Ceff` to isotropy (Kairotop.delta_iso, 0 = isotropic);
- `delta_K`: ‖K − (tr K/3) I‖_F / ‖K‖_F (0 = isotropic diffusion);
- `E1`, `E2`, `E3`: Young's moduli along the axes, `Exy_over_Ez` = (E1 + E2) / 2 / E3,
  `E1_over_E2` (in-plane isotropy);
- `K33_over_Kxy` = K33 / ((K11 + K22) / 2); `Kmin_over_Kmax` (eigenvalues of sym(K));
- `channel_angle_deg`: angle between the principal diffusion direction and e3.
"""
function shape_metrics(Ceff::AbstractMatrix, Keff::AbstractMatrix)
    E = [directional_youngs_modulus(Ceff, n) for n in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])]
    Ks = (Keff + Keff') / 2
    λ, V = eigen(Symmetric(Ks))
    dev = Ks - tr(Ks) / 3 * I
    return Dict{String,Any}(
        "delta_iso" => delta_iso(Ceff), "delta_K" => norm(dev) / norm(Ks),
        "E1" => E[1], "E2" => E[2], "E3" => E[3], "Exy_over_Ez" => (E[1] + E[2]) / 2 / E[3], "E1_over_E2" => E[1] / E[2],
        "K33_over_Kxy" => Ks[3, 3] / ((Ks[1, 1] + Ks[2, 2]) / 2), "Kmin_over_Kmax" => λ[1] / λ[end],
        "channel_angle_deg" => acosd(min(abs(V[3, end]), 1.0)),
    )
end

#***************************************************************
#* S1 BENCHMARK SELECTION (OQ-32, OQ-54)

"""
    s1_in_force(m, cfg) -> Bool

Whether the S1 run with metadata `m` belongs to the benchmark in force (`cfg` = resolved configs/S1.toml): RSA with
the per-fraction r_max of `[prod.rsa_max_radius]` (OQ-32); kairotops whose generation parameters match the
per-fraction recipe `[prod.ckai.fXX]` / `[prod.okai.fXX]` (the O-Kai recipe at f = 0.1-0.3 was revised on
2026-10-07, OQ-54; the earlier runs stay in results/S1); the gyroid always. Used by the S1 notebook, the S1 report
tables and the article figures.
"""
function s1_in_force(m, cfg)
    e = m["extra"]; fam = e["family"]; k = @sprintf("f%02d", round(Int, 100 * Float64(e["fraction"])))
    fam == "RSA" && return m["config"]["rsa_max_radius"] == cfg["rsa_max_radius"][k]
    fam in ("C-Kai", "O-Kai") || return true
    t = cfg[fam == "C-Kai" ? "ckai" : "okai"][k]
    return all(isapprox(Float64(m["config"][p]), Float64(v)) for (p, v) in t if p != "tuning_run")
end

using TOML

"""
    s6_current(m, profile) -> Bool

Whether the S6 run with metadata `m` is the detailed example in force for `profile` ("prod"/"dev"): same profile and
the same initialization length and density filter radius as configs/S6.toml (revised with the S1 O-Kai recipe at
f = 0.3 on 2026-10-07, OQ-54; the earlier example stays in results/S6).
"""
function s6_current(m, profile)
    c = TOML.parsefile(joinpath(@__DIR__, "..", "configs", "S6.toml"))[string(profile)]
    m["resolution_profile"] == string(profile) || return false
    return all(isapprox(Float64(m["config"][k]), Float64(c[k])) for k in ("init_smoothing_radius", "density_filter_radius"))
end
