#!/bin/bash
# S6 -- detailed optimization example (SI Section 3.1, Fig. 1c): O-Kai, f = 0.3, seed 1, history at every iteration.
# Checks that it reproduces the S1 run voxel by voxel (run_S1.sh first). Numbers of the SI text: scripts/S6_numbers.jl <hash>.
set -eu; cd "$(dirname "$0")/.."
julia --project=. scripts/S6_detailed_example.jl --config configs/S6.toml --profile prod
