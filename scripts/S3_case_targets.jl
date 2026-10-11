# S3 target tensors per case (shared by the runner and the analysis). Masked components are NaN.
# `variant` selects an opt-in modification used by the tuning study S3t (OQ-39); "" = the
# production definition (unchanged):
#   D "k33only"  diffusion: only K33 = D_void penalized (in-plane K free, as the SI text says)
#   E "kX"       diffusion target X * D_void * I instead of 0.6 * D_void * I (e.g. "k0.5")
#   F "pd"       positive-definite version of the F stiffness target: congruence A L A with
#                A = diag(1, 1, a, sqrt(a), sqrt(a), 1), a = 1/sqrt(10) (Kelvin order 11,22,33,23,13,12)
#                -> L3333 / 10 as before, L1133 = L2233 / sqrt(10), L2323 = L1313 / sqrt(10);
#                the production F target (L3333 / 10 only) has a negative eigenvalue
using LinearAlgebra

function case_targets(case, f; variant="")
    Cv = isotropic_elastic_target(f); Dv = phase_soft.d
    if case == "D"
        K = variant == "k33only" ? [NaN NaN NaN; NaN NaN NaN; NaN NaN 1.0] : [0.0 NaN NaN; NaN 0.0 NaN; NaN NaN 1.0]
        variant in ("", "k33only") || error("unknown D variant $variant")
        return 0.3 .* Cv, Dv .* K
    elseif case == "E"
        s = variant == "" ? 0.6 : startswith(variant, "k") ? parse(Float64, variant[2:end]) : error("unknown E variant $variant")
        C = fill(NaN, 6, 6); C[1, 1] = Cv[1, 1]; C[2, 2] = C[3, 3] = Cv[1, 1] / 8
        return C, s * Dv .* Matrix{Float64}(I, 3, 3)
    elseif case == "F"
        if variant == ""
            C = 0.2 .* Cv; C[3, 3] /= 10
        elseif variant == "pd"
            a = 1 / sqrt(10); A = Diagonal([1, 1, a, sqrt(a), sqrt(a), 1])
            C = A * (0.2 .* Cv) * A
        else
            error("unknown F variant $variant")
        end
        θ = 30π / 180; R = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
        return C, Dv .* (R * [0.0 0 0; 0 0 0; 0 0 1] * R')
    end
    error("unknown case $case")
end
