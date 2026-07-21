# Exact Poisson m=0 branch for the copula RP path + copula boundary tests

**Date:** 2026-07-20
**Status:** Approved design

## Problem

`rpbnb_boundary_tests()` runs boundary-corrected likelihood-ratio tests for a
fitted random-parameter bivariate NB model's boundary parameters — the
random-coefficient standard deviations (`sd1:*`, `sd2:*`) and the NB2
dispersions (`m1`, `m2`). It currently **rejects copula fits** outright
(`R/boundary_tests.R:90-94`):

```r
if (!is.null(fit$cop_family)) {
  stop("rpbnb_boundary_tests() supports famoye fits only; copula fits are ",
       "not supported (Poisson-limit margins are unavailable there).")
}
```

Two independent blockers sit behind that guard:

1. **SD tests** need the restricted-refit plumbing (`.fixed`, `.opt_draws`) that
   `.fit_rpbnb_copula()` never received — but they need **no** Poisson limit.
2. **Dispersion tests** refit a margin at its Poisson limit (`m = 0`), which the
   copula likelihood cannot currently evaluate: `fit_rpbnb.R:122-124` hard-blocks
   `poisson_1`/`poisson_2` for copula dependence.

The goal is full copula support in `rpbnb_boundary_tests()` — both SD and
dispersion — by giving the copula RP stack an **exact** `m = 0` (true Poisson)
branch, mirroring the Famoye/RP Poisson work already merged
(`feat(poisson): exact m=0 ... 3c21b80`), and wiring the restricted-refit
machinery through the copula fit function.

## Approach

Reuse the established **in-band signal**: a Poisson margin is `m = 0`, i.e.
`r = 1/m = Inf`. `r1`/`r2` already flow through the entire copula stack as
doubles, so no new signatures are needed in the C++ core — each NB evaluation
site gains a `!is.finite(r)` → Poisson branch. In the R fit function we also
carry explicit `pois1`/`pois2` flags (to set `r = Inf` and to zero the pinned
`log_m` gradient columns cleanly), exactly the hybrid the Famoye path uses.

Three NB → Poisson swaps are required wherever a margin is Poisson (each because
`pnbinom`/`dnbinom` at `size = Inf` return NaN or segfault — the same failure hit
in the Famoye residual work):

| Quantity | NB2 form | Poisson (`r = Inf`) form |
|----------|----------|--------------------------|
| CDF corner | `pnbinom(y, size=r, mu)` | `ppois(y, mu)` |
| mu-score | `dnbinom(y+1, size=r, mu)` | `dpois(y+1, mu)` |
| dispersion score `s_logm` | `(-r)·(…)` | `0` (pinned parameter; NB form is `(-Inf)·0 = NaN`) |

The `log_m` of a Poisson margin is a **fixed** parameter: pinned in `start`, held
by `maxLik(fixed=)`, dropped from `npar`, and excluded from the free-parameter
covariance — so the fit nests inside the full NB model for the 1-df
boundary-corrected LR test.

## Components

### 1. Shared pmf/score helpers — `R/copula_likelihood.R`
The single source of the four NB-CDF corners and the per-observation copula
scores, reused by both the value and gradient paths.

- `.copula_pmf(y1, y2, mu1, mu2, r1, r2, theta, family)`: derive
  `pois1 = !is.finite(r1)`, `pois2 = !is.finite(r2)`; compute each margin's
  corners (`a`/`am` for margin 1, `b`/`bm` for margin 2) with `ppois` when that
  margin is Poisson, else `pnbinom`.
- `.copula_score_scalars(...)`: `da_dmu1`/`dam_dmu1` (resp. margin 2) use `dpois`
  when the margin is Poisson; set `s_logm1 = 0` (resp. `s_logm2 = 0`) for a
  Poisson margin.
- `.dnb_cdf_dr` already returns `0` for non-finite `r` — no change.

### 2. RP copula likelihood — `R/rpbnb_copula_likelihood.R`
- `bnbr_rp_copula_ll` and `bnbr_rp_copula_ll_grad`: add `pois1 = FALSE,
  pois2 = FALSE`; set `r1 <- if (pois1) Inf else exp(-log_m1)` (same for `r2`).
  In the gradient, force the `im1`/`im2` gradient (and score) columns to `0` for
  a Poisson margin.
- `bnbr_rp_copula_ll_grad_cpp` wrapper: pass `r1`/`r2 = Inf` when the margin is
  Poisson (in-band signal to the C++ core).

### 3. C++ core — `src/copula_parallel.cpp`
- Detect `bool pois1 = !R_finite(r1)`, `bool pois2 = !R_finite(r2)` once.
- Four `pnbinom` corners in Pass 1 (~L202-205) and Pass 2 (~L244-247) →
  `R::ppois(y, mu, 1, 0)` when Poisson.
- mu-score `dnbinom_mu` (~L267-273) → `R::dpois(y, mu, 0)` when Poisson.
- `s_logm1`/`s_logm2` (~L279, L282) and their gradient/score accumulation into
  the `log_m` columns → skipped (left `0`) when the margin is Poisson.
- No change to the `.Call` signature: `r1`/`r2` already arrive as doubles.

### 4. Fit function — `R/fit_rpbnb_copula.R`
- New params: `poisson_1 = FALSE, poisson_2 = FALSE, .fixed = NULL,
  .opt_draws = NULL` (matching the Famoye `fit_rpbnb` internal contract).
- `.opt_draws`: when supplied, reuse its `Z1`/`Z2` (common random numbers across
  restricted refits) instead of generating fresh Halton draws.
- Fixed parameters:
  `fixed_names <- c(if (pois1) "log_m1", if (pois2) "log_m2", names(.fixed))`;
  pin `start[fixed_names]`; pass `fixed = fixed_names` to `maxLik`;
  `npar <- length(par_hat) - length(fixed_names)`; use `free` /
  `.free_index_vcov` in the OPG and numeric SE paths (skipped entirely under the
  boundary-test default `compute_se = FALSE`).
- Thread `pois1`/`pois2` into `ll_fun`; store `poisson_1`/`poisson_2` on the fit.

### 5. Router — `R/fit_rpbnb.R`
- Delete the hard block (L122-125) that rejects `poisson_*` with copula.
- Forward `poisson_1, poisson_2, .fixed, .opt_draws` into the
  `.fit_rpbnb_copula(...)` call.

### 6. Boundary tests — `R/boundary_tests.R`
- Remove the copula guard (L90-94).
- `refit()`: use the fit's actual dependence —
  `dependence = if (!is.null(fit$cop_family)) copula(fit$cop_family) else "famoye"`.
- Update roxygen: `@param fit` no longer "famoye only"; note copula support.

### 7. Tests (TDD) — new `tests/testthat/test-copula-poisson.R`
- `.copula_pmf` / `.copula_score_scalars` Poisson branch vs a hand-written
  `ppois`-corner + `dpois`-score oracle.
- RP copula value: `bnbr_rp_copula_ll(pois1=TRUE)` == reference; C++ == R for
  value, gradient, and scores.
- Gradient: the `log_m` column of a Poisson margin is exactly `0`; every free
  column matches a numeric gradient of the frozen objective.
- `.fit_rpbnb_copula(poisson_1=TRUE)`: `npar` drops by 1, `log_m1` is fixed at
  its pinned value with `NA` SE, logLik finite.
- `rpbnb_boundary_tests(copula_fit, data)`: returns both `sd` and `dispersion`
  rows with finite `LR`/`p.value`; `df` correct.

### 8. Demo — `inst/fit_rpbnb_diff_copula.R`
- The boundary-test section added earlier now runs; verify end-to-end.

## Out of scope (flagged)

- The **fixed-coefficient** copula path (`copula_loglik_vec` / `copula_grad_vec`,
  used by `fit_bnb(dependence = copula())`) keeps its own Poisson block. It is
  not exercised by `rpbnb_boundary_tests()`, which is RP-only.
- **Residuals** on a directly-constructed Poisson copula fit are not wired
  through the copula residual CDF path. Boundary-test restricted fits are
  transient (created → `logLik`/`npar` read → discarded) and never have
  residuals computed, so this is deferred as a follow-up note, not a blocker.

## Testing & verification

- TDD throughout (RED → GREEN → REFACTOR) for every new branch.
- Oracle checks: Poisson-branch value/score computed independently with
  `ppois`/`dpois`.
- C++ == R parity on value, gradient, and scores.
- End-to-end: a copula fit's `rpbnb_boundary_tests()` produces a full table; the
  demo script runs clean.
- Full package suite green (fast tier; slow tier for the end-to-end copula fit).
