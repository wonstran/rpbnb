# Copula dependence for `fit_rpbnb` — design spec

**Date:** 2026-07-13
**Status:** approved (design), pending implementation plan
**Scope:** Add copula-based dependence (Frank / Gaussian / Clayton) to the
random-parameter bivariate NB estimator `fit_rpbnb`, as an alternative to the
existing Famoye/Sarmanov λ dependence. Includes a copula-dependent simulator and
recovery tests. First version is a **correct MVP in R** (numeric gradient).

## 1. Motivation

`fit_rpbnb` currently supports only Famoye/Sarmanov dependence (`z_lambda`,
bounded multiplicative factor). The fixed-coefficient estimator `fit_bnb` already
supports copula dependence via `fit_bnb(dependence = copula("frank"|"normal"|"kimeldorf"))`.
This spec brings the same copula option to the random-parameter model: the
dependence between `y1` and `y2` is modelled by a copula on the two NB margins,
while the marginal means remain random across individuals.

## 2. Model

Simulated log-likelihood (unchanged mixing structure):

    L = sum_i log( (1/R) sum_r P_ir )

Parameter vector order:

    beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2), log_m1, log_m2, z_theta

`z_theta` is the copula dependence parameter on the unconstrained scale;
`theta = z_to_native(family, z_theta)` (Frank: identity; Gaussian: tanh -> rho;
Clayton: exp -> theta>0), exactly as in the fixed-model copula.

Per draw `r`, the random coefficients give per-draw means `mu1_ir = exp(eta1_ir)`,
`mu2_ir = exp(eta2_ir)` (identical machinery to the Famoye RP path via
`rand_realize`). With `r_t = 1/m_t`, the NB CDF corners are

    a  = pnbinom(y1,   r1, mu1_ir)   am = pnbinom(y1-1, r1, mu1_ir)  (0 if y1=0)
    b  = pnbinom(y2,   r2, mu2_ir)   bm = pnbinom(y2-1, r2, mu2_ir)  (0 if y2=0)

and the per-draw joint pmf is the discrete-copula rectangle probability

    P_ir = C(a,b;theta) - C(am,b;theta) - C(a,bm;theta) + C(am,bm;theta)

using the copula CDF `C` from `copula_core.R`. `LL[i,r] = log(max(P_ir, 1e-300))`.
Mix per observation with row-log-sum-exp: `value = sum_i (lse_i - log R)`.

When `q1 = q2 = 0` (no random coefficients), `P_ir` does not depend on `r`, and
`L` reduces exactly to the fixed-model copula log-likelihood
`copula_loglik_vec` — this is the primary correctness check.

## 3. Components

### 3.1 RP copula likelihood — `R/rpbnb_copula_likelihood.R` (new)

`bnbr_rp_copula_ll(par, y1, y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                   Z1, Z2, family, dist1, dist2, sign1, sign2)`

- Unpacks `par` (same layout as the Famoye RP unpacker, with `z_theta` last).
- Loops over draws `r`: builds `mu1_ir, mu2_ir` via `rand_realize` (reused), forms
  the four NB CDF corners, evaluates the copula rectangle pmf via the family CDF,
  fills `LL[,r]`.
- Returns the scalar simulated log-likelihood (via `row_log_sum_exp`).
- No analytic gradient in the MVP (see 3.3).
- Optional cluster path (`cl`) may parallelize the draw loop, mirroring the
  Famoye R fallback; not required for MVP correctness.

Reuses: `rand_realize`, `frank_cdf`/`normal_cdf`/`kimeldorf_cdf`, `z_to_native`,
`row_log_sum_exp`, `.bound_mu`-style capping.

### 3.2 Copula-dependent simulator — `R/simulate_rpbnb_copula.R` (new)

`simulate_rpbnb_copula(n, beta1, beta2, random_1, random_2, dispersion,
                       copula, covariates = NULL, seed = NULL)`

- Draw random coefficients per individual (reuse the realization logic from
  `simulate_rpbnb`) -> `mu1_i, mu2_i`.
- Sample copula uniforms `(u_i, v_i)` for the chosen family:
  - Gaussian: draw bivariate normal with correlation `rho`, map through `pnorm`.
  - Frank: closed-form conditional inversion `v = C_{2|1}^{-1}(w | u)`.
  - Clayton: closed-form conditional inversion.
  - (Implemented directly; no new package dependency.)
- `y1_i = qnbinom(u_i, size = r1, mu = mu1_i)`, `y2_i = qnbinom(v_i, r2, mu2_i)`.
- Returns `list(data, mu, true, ...)` including `copula` family + native parameter
  and (for reference) the implied Kendall's tau.

### 3.3 Gradient & standard errors (MVP)

- **Gradient:** numeric — `maxLik`/BFGS finite-differences `bnbr_rp_copula_ll`.
  Consequence: many LL evaluations per iteration -> slow. Validate at moderate n
  (~1500-2000) with modest `draws`.
- **SEs:** numeric Hessian (frozen nothing — the copula has no lambda-bounds).
  `se_method` for the copula path is "numeric" only in the MVP; OPG/analytic are
  N/A without analytic scores.
- **Documented upgrade path:** the fast route is to mix the existing analytic
  `copula_grad_vec` per draw (Louis formula) for an analytic gradient + OPG SEs.
  Explicitly out of scope for the MVP but noted for a follow-up.

### 3.4 API integration — `R/fit_rpbnb.R` (modified)

- Add argument `dependence = "famoye"`.
  - `dependence == "famoye"` (default): today's C++/Famoye path, unchanged.
  - `inherits(dependence, "rpbnb_copula")`: route to the copula path — build the
    `z_theta`-parameterized start vector, optimize `bnbr_rp_copula_ll` with BFGS +
    numeric gradient, compute numeric-Hessian SEs.
- The `rpbnb_fit` constructor gains `cop_family` (NULL for Famoye). For the copula
  path, `lambda`/`bounds` are NULL and `cop_family` is set, which activates the
  existing copula branch in `.natural_scale_table` (methods.R) — output then shows
  the native copula parameter (theta/rho) and Kendall's tau with delta-method SEs.
- Coefficient naming: `b1:*, b2:*, log_sd1:*, log_sd2:*, log_m1, log_m2, z_theta`.

### 3.5 Print/summary

No new method code: methods.R already has
`if ("z_theta" %in% names(cf) && !is.null(object$cop_family))` producing the
native-parameter and Kendall's-tau rows. Setting `cop_family` on the fit object
is sufficient.

## 4. Testing

`tests/testthat/test-rpbnb-copula.R` (new):

1. **Reduction/consistency:** with `random_1 = random_2 = NULL` (q=0), for each
   family, `bnbr_rp_copula_ll(par, ...)` equals `sum(copula_loglik_vec(par, ...))`
   to ~1e-10. Note the two layouts coincide when q=0 — both are
   `(beta1, beta2, log_m1, log_m2, z_theta)` with no `log_sd` block — so the same
   `par` is passed to both. Proves the copula pmf is wired correctly.
2. **pmf sanity:** for a single draw and fixed params, the copula rectangle pmf
   summed over a `(y1,y2)` grid is ~1.
3. **Recovery (simulator round-trip):** simulate copula-dependent RP data with a
   known parameter (Frank and Gaussian at minimum; Clayton if cheap), fit with the
   matching family, assert the native copula parameter and Kendall's tau are
   recovered within a tolerance / a few SEs of truth.
4. **Independence limit:** copula parameter ~ 0 (Frank theta=0 / Gaussian rho=0)
   reproduces near-independent margins and the fit recovers ~0.

`inst/` demo scripts: `simulate_rpbnb_copula_demo.R` (generate) and
`fit_rpbnb_copula_demo.R` (fit + compare to truth), mirroring the existing
Famoye demos.

## 5. Non-goals (this iteration)

- C++/OpenMP port of the copula likelihood (Famoye path stays the fast one).
- Analytic gradient / OPG / analytic Hessian for the copula path.
- Copula parameter as a function of covariates.
- Families beyond Frank / Gaussian / Clayton.

## 6. Risks

- **Performance:** numeric gradient over a simulated likelihood is slow at large n;
  mitigated by validating at moderate n and documenting the analytic-gradient
  upgrade path.
- **Discrete-copula edge cases:** `pnbinom` corners at `y=0`, copula CDF guards for
  `u,v` near 0/1 — already handled in `copula_core.R`/`copula_likelihood.R`;
  reused rather than re-implemented.
- **Simulator conditional inversion:** Frank/Clayton inversion must be validated
  (the recovery test doubles as this check — wrong sampling would fail recovery).
