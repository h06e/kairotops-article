# kairotops-article

Scripts and data of the article

> R. Valmalette, C. Bellis. *Kairotops as opportune designs of disordered materials.* Submitted to npj Computational Materials.

Kairotops are disordered two-phase microstructures obtained by gradient-based optimization from a smoothed random
initialization, so that their effective (homogenized) elastic and diffusion tensors approach prescribed targets at a
given solid fraction. The generation and the FFT-based homogenization are done by the Julia package
[Kairotop.jl](https://github.com/h06e/Kairotop.jl) (version 0.2.0). This repository contains the scripts of every
numerical study of the article, the configuration files with the parameters used, and the microstructures of the
stiffness benchmark (study S1). Figures, plotting code, notebooks, experimental data and intermediate results are not
included: every numerical result can be regenerated with the scripts below.

## Contents

```
configs/     parameters of each study (TOML; [prod] = article, [dev] = small smoke tests at N = 64)
src/         shared helpers (materials, targets, metrics, run storage with provenance, kairotop runner)
scripts/     study scripts (S*.jl), run scripts (run_S*.sh), data export
data/S1/     microstructures of the stiffness benchmark (see "Data")
```

Every run is stored by the scripts as `results/<study>/<run hash>/` (`meta.json`: full parameters, results,
provenance, timings; `data.jld2`: fields). The run hash is a hash of the full generation parameters: rerunning a
configuration gives the same hash and, on the same hardware and software, the same microstructure voxel by voxel
(the generation is deterministic for a given seed). A finished run is never overwritten.

## Requirements

- Julia 1.12 and an NVIDIA GPU with CUDA (the homogenization runs on the GPU only). The article runs used one NVIDIA
  RTX 6000 Ada (48 GB); one generation at N = 256 needs about 12 GB of GPU memory.
- Installation of the Julia environment (Kairotop.jl v0.2.0 is fetched from GitHub):

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

```
julia --project=. scripts/S8_l0_rf.jl 0.02 0.01 --smoke --family C-Kai --fraction 0.1   # N = 64, about a minute
julia --project=. scripts/S1_benchmark.jl kai C 0.3 1                                   # one article generation (C-Kai, f = 0.3, seed 1, ~20 min)
```

Run times below are those of the article runs (one generation at a time on the GPU above).

## Studies

The studies are numbered as in the internal campaign; the article sections they feed are given for each.
Run one study at a time: each `run_S*.sh` runs its generations one after the other.

### S1 — Stiffness benchmark across solid fractions (Section 2.1, Fig. 2; SI benchmark section and figure)

Isotropic stiffness maximization (target: Voigt bound) at solid fractions 0.1 to 0.9, without (C-Kai, closed
porosity) and with (O-Kai, open porosity) the diffusion objective, compared with the gyroid (one deterministic cell
per fraction) and RSA packings of spherical voids (5 seeds; infeasible below 0.3, the packing stalls at about 0.25).
Kairotops: 5 random initializations per fraction. Shared settings: `configs/S1.toml [prod.kai_base]`; lengths and
diffusion weight per fraction: `[prod.ckai.fXX]`, `[prod.okai.fXX]`; RSA maximum radius: `[prod.rsa_max_radius]`.
Properties are computed on the physical density thresholded at 0.5 (N = 256, contrast 1000).

- `scripts/run_S1.sh`: references then 90 kairotop generations (about 30 h).
- `scripts/S1_benchmark.jl refs | kai C|O <f> <seed>`: one step.
- Parameter selection (already applied in `configs/S1.toml`, history only): `S1p_param_investigation.jl` (single-seed
  candidates), `S1_select_params.jl` (selection rule: covariance range in [0.055, 0.10]; stiffest C-Kai; O-Kai with
  percolating void and moduli not below the gyroid, then highest diffusion), `S1_rsa_radius_search.jl`. The O-Kai
  lengths at f <= 0.3 and the C-Kai lengths at f = 0.1 and 0.3 were then set from study S8.
- `scripts/export_S1_data.jl`: writes `data/S1` from `results/S1`.

### S2 — Morphological control via the initialization length (Section 2.2, Fig. 3; SI internal length)

C-Kai at f = 0.3, initialization length l0 in {0.015, 0.02, 0.03, 0.05, 0.08, 0.12}, 3 seeds, with a fixed density
filter radius r_f = 0.005 and with r_f = l0/4; covariograms of the S1 gyroid and RSA at f = 0.3 as references.
Needs S1 (references). `scripts/run_S2.sh` (33 generations).

### S3 — Anisotropic objectives (Section 2.3, Fig. 4; SI "Controlling stiffness and diffusion")

Three cases at f = 0.3 with a mask on the target tensors (`scripts/S3_case_targets.jl`): D, diffusion along e3 with
isotropic stiffness; E, stiffness along e1 with isotropic diffusion; F, stiffness higher in the (e1, e2) plane and a
diffusion channel at 30 degrees from e3. `scripts/run_S3.sh`: diffusion-weight scan and selection, 3 seeds per case,
the sweep of case E across solid fraction and l0, and the shape-tuning runs (S3t). Fig. 4 and the SI figure use the
S3t runs D (w_K = 0.5, l0 = 0.02), E (w_K = 50) and F (positive-definite target, w_K = 1).

### S4 — Robustness to material and geometrical defects (Section 2.4; SI defects section)

The S1 microstructures at f = 0.3 (C-Kai, O-Kai and RSA, 5 seeds each; gyroid) with defect scales in units of each
microstructure's covariance range r_cov: material defect, solid modulus times e(x) in [0.9, 1] (correlation length
0.5 r_cov); geometrical defect, smooth random displacement of the geometry (correlation length 1.5 r_cov, amplitude
0.2 r_cov) at constant solid fraction; 10 realizations each. Needs S1. `scripts/run_S4.sh` (about 30 min). Other modes
of `S4_defects.jl` (amplitude and length sweeps) were exploratory.

### S6 — Detailed optimization example (SI Section 3.1, Fig. 1c)

The S1 O-Kai at f = 0.3, seed 1, regenerated with the history recorded at every iteration (slices of the design and
physical fields, cost terms, weights, line-search trials, properties of the thresholded field). The script checks
that the result is the S1 run voxel by voxel. Needs S1. `scripts/run_S6.sh` (about 30 min); the numbers quoted in
the SI: `julia --project=. scripts/S6_numbers.jl <run hash>`.

### S8 — Initialization length and density filter radius (SI section and figure)

O-Kai at f = 0.2, seed 1, full grid l0 x r_f in {0.005, 0.0075, 0.01, 0.02, 0.03}^2 (25 generations, about 7 h).
The O-Kai lengths of S1 at f <= 0.3 (l0 = 0.03, r_f = 0.01) follow this study. `CKAI=1 scripts/run_S8.sh` adds the
same grid for the C-Kai at f = 0.1, 0.2, 0.3 (75 generations), from which the C-Kai lengths of S1 at f = 0.1
(l0 = 0.03, r_f = 0.005) and f = 0.3 (l0 = 0.03, r_f = 0.0075) were chosen. True stop reasons from the run logs:
`scripts/S8_stop_reasons.py`.

### S0s — Optimization walkthrough (SI "Role of the initialization", table and figure)

C-Kai, f = 0.3, seed 1, four starts of the optimization (sharp start of the method; weak start; projection
continuation without binarization penalty; no filter and projection). `scripts/run_S0s.sh`.

## Data

`data/S1/` holds the 144 microstructures of S1 in force in the article: 45 C-Kai, 45 O-Kai, 45 RSA (including the 10
infeasible attempts at f = 0.1 and 0.2, flagged `failed`, whose solid fraction stalls at 0.25) and 9 gyroid cells.

- `<family>_f<fraction>_seed<seed>_<hash>.h5` (gyroid: no seed): HDF5, dataset `chi`, UInt8, 256 x 256 x 256,
  1 = solid, 0 = void, periodic unit cell of size 1 (deflate-compressed); attributes `run_hash`, `family`,
  `fraction_target`, `seed`. Julia: `h5read(file, "chi")`; Python: `h5py.File(file)["chi"][...]` (axes in reverse
  order).
- `index.csv`: one line per microstructure with the properties reported in the article (solid fraction achieved,
  effective bulk and shear moduli and diffusion coefficient for a solid of Young's modulus 100 and Poisson ratio 0.3,
  a void phase of modulus 0.1 and diffusivity 100; distance to isotropy; covariance range; number of axes along which
  the void percolates). The figures of the article (Fig. 2, SI benchmark) show the mean and standard deviation of these
  values over the seeds.
- `configs.json`: the full generation parameters of each run, by run hash.

Other studies are not distributed as data; they are regenerated by the scripts.

## Citation and license

Please cite the article and this archive (see `CITATION.cff`), and Kairotop.jl. BSD 3-Clause License with citation
requirement (`LICENSE`), for the scripts and the data.

Comments in the scripts refer to decisions of the campaign log as "OQ-nn"; that log is not part of this repository.
