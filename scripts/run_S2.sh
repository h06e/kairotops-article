#!/bin/bash
# S2 -- morphological control via the initialization length l0 (Fig. 3, SI internal length). C-Kai, f = 0.3, 3 seeds.
# Needs the S1 references at f = 0.3 (run_S1.sh, refs).
set -eu; cd "$(dirname "$0")/.."
J="julia --project=."
for l0 in 0.015 0.02 0.03 0.05 0.08 0.12; do for s in 1 2 3; do $J scripts/S2_morphological_control.jl $l0 $s; done; done      # fixed r_f = 0.005
for l0 in 0.015 0.03 0.05 0.08 0.12; do rf=$(awk -v l="$l0" 'BEGIN{printf "%.5g", l/4}')                                       # r_f = l0/4 (l0 = 0.02 shared)
  for s in 1 2 3; do $J scripts/S2_morphological_control.jl $l0 $s $rf; done; done
$J scripts/S2_morphological_control.jl refcov                 # covariograms of the S1 gyroid and RSA at f = 0.3
