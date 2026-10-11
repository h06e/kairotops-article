#!/bin/bash
# S8 -- O-Kai at f = 0.2: initialization length l0 x density filter radius r_f (SI figure), 25 runs, seed 1.
# The O-Kai lengths of configs/S1.toml at f <= 0.3 follow this study. Optional (CKAI=1): the same grid for the C-Kai at
# f = 0.1, 0.2, 0.3 (75 runs), from which the C-Kai lengths at f = 0.1 and 0.3 were chosen.
set -eu; cd "$(dirname "$0")/.."
J="julia --project=."
V="0.005 0.0075 0.01 0.02 0.03"
for rf in $V; do for l0 in $V; do $J scripts/S8_l0_rf.jl $l0 $rf; done; done
if [ "${CKAI:-0}" = 1 ]; then
  for f in 0.1 0.2 0.3; do for rf in $V; do for l0 in $V; do $J scripts/S8_l0_rf.jl $l0 $rf --family C-Kai --fraction $f; done; done; done
fi
