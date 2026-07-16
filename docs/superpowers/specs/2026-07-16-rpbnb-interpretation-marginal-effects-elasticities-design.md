# RP marginal effects & elasticities — design

**Date:** 2026-07-16
**Status:** Approved (brainstorming)
**Scope:** Add marginal-effects and elasticity interpretation for the
random-parameter model class `rpbnb_fit`.

## Problem

`bnb_marginal_effects()` and `bnb_elasticities()` (in `R/diagnostics.R`) both
begin with `stopifnot(inherits(fit, "bnb_fit"))` and compute the conditional
mean as `mu = exp(X %*% beta)`. That is correct only for the fixed-coefficient
`bnb_fit`. For the random-parameter `rpbnb_fit`, the population mean is the
Monte-Carlo–integrated

```
mu_i = E_beta[ exp(x_i' beta) ]
```

which `predict.rpbnb_fit` already estimates by averaging `exp(lp)` over the
stored optimization draws `Z` (with distribution-aware realized coefficients).
The interpretation quantities for RP fits must be built on that same integrated
mean, so they are consistent with `predict()`.

## Math

For equation `e`, design `X` (n x p), mean coefficients `b` (length p), random
column indices `rand_idx`, per-column distribution/sign metadata, native scales
`s = exp(log-scale)`, and stored standardized draws `Z` (R rows):

- Per-draw deviations: `dev = rand_realize(Z, dist, sign, b[rand_idx], s)$dev`
  (R x q), realized coefficients `coef` (R x q) from the same call.
- Per-draw linear predictor: `lp_ir = x_i' b + sum_{j in rand} X_ij * dev_rj`.
- Per-draw mean contribution: `g_ir = pmin(exp(lp_ir), RP_PRED_CAP)`.
- Integrated mean: `mu_i = mean_r g_ir`.

### Marginal effects

- **Continuous k:** `d mu_i / d x_ik = mean_r [ coef_rk * g_ir ]`, where
  `coef_rk` is the realized coefficient for column k on draw r:
  - fixed column: `coef_rk = b_k` (constant across draws),
  - random column: `coef_rk = rand_realize(...)$coef[r, .]`.

  This is the exact generalization of the fixed-coefficient `beta_k * mu` and
  collapses to it when k is non-random or the equation is fully fixed.

- **Binary k (values in {0,1}):** discrete difference
  `mu_i(x_ik = 1) - mu_i(x_ik = 0)`, recomputing `lp` with the column forced to
  1 and to 0. Because the random contribution `X_ik * dev_rk` also depends on
  `x_ik`, forcing the column captures both the location and the random-coefficient
  channels.

### Elasticities

- **Continuous k:** pointwise elasticity `x_ik * (d mu_i/d x_ik) / mu_i`
  (reduces to `beta_k * x_ik` under fixed coefficients).
- **Binary k:** semi-elasticity `mu_i(1)/mu_i(0) - 1`
  (reduces to `exp(beta_k) - 1` under fixed coefficients).

### AME vs MEM

- `type = "AME"` (default): average the pointwise quantity over the sample.
- `type = "MEM"`: evaluate at the mean design row `xbar` (using the same
  draws Z). For binary variables MEM forces the mean row's column to 0 / 1.

## Standard errors — numeric delta method

The population mean `E[exp(x'beta)]` depends only on equation e's mean
coefficients `b_e` and its log-scale parameters (`log_sd` / `log_w` / `log_s`),
NOT on the NB dispersion `log_m` or the dependence parameter
(`z_lambda` / `z_theta`). Therefore:

1. Build a deterministic, vector-valued estimand `f(theta_e)` returning the
   AME/MEM/elasticity for all selected variables at once. Inside `f`, reconstruct
   `b` from the mean-coef part of `theta_e`, `s = exp(log-scale part)`, re-derive
   `dev`/`coef` with the draws `Z` held fixed (so `f` is smooth and
   deterministic), and compute the quantity.
2. One `numDeriv::jacobian(f, theta_e)` per equation gives `G` (n_vars x n_par).
3. `SE_k = sqrt( g_k' V_e g_k )`, where `V_e` is the vcov sub-block over the
   `theta_e` parameter names (mean coefs + log-scales for equation e).

`theta_e` names: `b{e}:col` for every design column, plus `{scale_label}{e}:col`
for each random column (scale_label is `log_sd`/`log_w`/`log_s` per distribution).

Holding `Z` fixed mirrors the existing binary-case `numDeriv::jacobian` in
`bnb_marginal_effects`. If `compute_se = FALSE` (the fit's vcov is NA-filled),
the relevant sub-block is non-finite: return `NA` standard errors (and `z`/`p`)
with a single warning.

## API

Parallel, non-generic functions matching the existing `bnb_*` diagnostics
naming (the package uses plain functions, not generics, for `bnb_gof`,
`bnb_marginal_effects`, `bnb_elasticities`):

```r
rpbnb_marginal_effects(fit,
                       which = c("y1", "y2", "both", "all"),
                       type  = c("AME", "MEM"),
                       vars  = NULL,
                       include_intercept = FALSE,
                       digits = 4,
                       print_output = TRUE)

rpbnb_elasticities(fit,
                   which = c("y1", "y2", "both"),
                   type  = c("AME", "MEM"),
                   vars  = NULL,
                   include_intercept = FALSE,
                   digits = 4,
                   print_output = TRUE)
```

- Same output columns as the `bnb_*` pair:
  `Name, Estimate, StdErr, z, p, Signif, var_type`.
- `var_type` annotates random columns, e.g. `"continuous (random)"` /
  `"binary (0->1, random)"`, since random columns use the integrated formula.
- `which = "both"`/`"all"` returns a named list `list(y1 = ., y2 = .)`; a single
  margin returns one data frame (invisibly). Matches `bnb_*` return shapes.
- New code lives in `R/diagnostics.R` alongside the `bnb_*` versions and shares
  the existing `.bnb_me_tidy` / `.bnb_me_print` helpers.
- Touches no existing `bnb_fit` behavior.

**Alternative considered and rejected:** converting to S3 generics
`marginal_effects()` / `elasticities()` with methods for both classes. Cleaner
dispatch, but it renames the existing public API and diverges from the package's
established non-generic diagnostic style. Not worth the churn.

## Edge cases

- **Lognormal analytic infinities** (`sign_j * X_ij > 0`): mirror `predict()` —
  emit a warning and propagate `Inf` for the affected rows (which flows into the
  AME average). MEM at a mean row is affected only if the mean row triggers the
  condition.
- **Fully-fixed equation** (`q = 0`): the integrated formula reduces exactly to
  the `bnb_*` result — used as an equivalence check in tests.
- Reuses `fit$rp_meta$Z1` / `Z2`, so results are consistent with `predict()` and
  reproducible.

## Testing

- **Fast tier** (fixture-based, no optimization; extend `make_rp_fixture()` in
  `helper-slow.R` with a plausible `vcov`/`se`):
  - RP-vs-fixed equivalence on the fully-fixed equation (`q = 0`) against
    `bnb_marginal_effects` / `bnb_elasticities` math.
  - Monte-Carlo marginal-effect and elasticity values on the random equation
    against a direct brute-force recomputation from the fixture draws.
  - Lognormal `Inf` propagation + warning.
  - Delta-method SE has correct shape and finite positive values; `z`/`p`
    consistent.
  - `NA`-vcov path returns `NA` SEs with a warning.
- **Slow tier** (`RPBNB_RUN_SLOW=1`): one real `fit_rpbnb` end-to-end sanity
  check (finite estimates and SEs, sensible signs).

## Docs

- Roxygen `@export` with `@examples` for both functions.
- Short interpretation note in `vignettes/rpbnb-intro.Rmd`.
- `NEWS.md` entry.
