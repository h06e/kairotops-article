#!/bin/bash
# S1 -- stiffness benchmark across solid fractions (Fig. 2, SI benchmark). One GPU job at a time.
# Recipes in configs/S1.toml ([prod.kai_base], [prod.ckai.fXX], [prod.okai.fXX], [prod.rsa_max_radius]).
set -eu; cd "$(dirname "$0")/.."
J="julia --project=."
$J scripts/S1_benchmark.jl refs                              # gyroid (1 cell) + RSA (5 seeds) per fraction
for f in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do for s in 1 2 3 4 5; do for fam in C O; do
  $J scripts/S1_benchmark.jl kai $fam $f $s                  # C-Kai / O-Kai, 5 random initializations
done; done; done
