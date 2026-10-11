#!/bin/bash
# S0s -- optimization walkthrough (SI "Role of the initialization", Table and Figure): 4 variants, C-Kai, f = 0.3, seed 1.
set -eu; cd "$(dirname "$0")/.."
for v in A_sharp_start B_weak_start C_beta_only D_penalty_only; do julia --project=. scripts/S0s_optimization_walkthrough.jl $v; done
