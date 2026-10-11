#!/bin/bash
# S3 -- anisotropic objectives (Fig. 4, SI "Controlling stiffness and diffusion" and the sweep across f and l0).
set -eu; cd "$(dirname "$0")/.."
J="julia --project=."
# diffusion-weight scan (seed 1) and selection (writes [prod.weight_D] of configs/S3.toml; the shipped values are the result)
for spec in "D 0.1" "D 0.3" "D 1.0" "E 1.0" "E 3.0" "E 7.0" "F 0.3" "F 1.0" "F 3.0"; do $J scripts/S3_anisotropic_objectives.jl scan $spec; done
$J scripts/S3_select_weights.jl
for c in D E F; do for s in 1 2 3; do $J scripts/S3_anisotropic_objectives.jl prod $c $s; done; done
# sweep: case E across solid fraction and l0 (r_f = l0/4)
for f in 0.2 0.3 0.4; do for l0 in 0.02 0.03 0.05; do for s in 1 2 3; do $J scripts/S3_anisotropic_objectives.jl sweep $f $l0 $s; done; done; done
# shape tuning (S3t): the runs of Fig. 4 are D 0.5 0.02, E 50 0.03 and F 1 0.03 pd
for spec in "D 0.5 0.02" "D 0.3 0.03 k33only" "D 1 0.03 k33only" "E 20 0.03" "E 20 0.05" "E 50 0.03" "F 0.3 0.03 pd" "F 1 0.03 pd" "F 3 0.03 pd"; do
  $J scripts/S3_anisotropic_objectives.jl tune $spec; done
