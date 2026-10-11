#!/bin/bash
# S4 -- robustness to material and geometrical defects (Section 2.4, SI). Needs the S1 runs at f = 0.3 (run_S1.sh).
# Article configuration: material defect 10 %, geometrical defect l = 1.5 r_cov, a = 0.2 r_cov (fraction kept), 10 realizations.
set -eu; cd "$(dirname "$0")/.."
julia --project=. scripts/S4_defects.jl article
