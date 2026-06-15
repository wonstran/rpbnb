# Design: `simulate_bnb()` — Famoye Bivariate NB Simulator

**Date:** 2026-06-15
**Status:** Approved

## Context

The `rpbnb` package fits the Famoye/Sarmanov bivariate negative binomial (BNB) model via
`fit_bnb(..., dependence = "famoye")`. The existing `simulate_rpbnb()` generates data for the
random-parameter variant but blocks `lambda ≠ 0`. There is no simulator for the fixed-parameter
Famoye BNB joint distribution. `simulate_bnb()` fills this gap.

## Signature

```r
simulate_bnb <- function(n, beta1, beta2,
                         dispersion = c(m1 = 0.5, m2 = 0.5),
                         lambda = 0,
                         covariates = NULL,
                         seed = 1234)
```

### Parameters

| Parameter    | Type                  | Description |
|--------------|-----------------------|-------------|
| `n`          | integer               | Number of observations |
| `beta1`      | named numeric vector  | Coefficients for equation 1; must include `"(Intercept)"` |
| `beta2`      | named numeric vector  | Coefficients for equation 2; must include `"(Intercept)"` |
| `dispersion` | named numeric vector  | NB2 overdispersions `c(m1 = ., m2 = .)`; variance = mu + m·mu² |
| `lambda`     | numeric scalar        | Famoye/Sarmanov dependence parameter |
| `covariates` | data frame or NULL    | Pre-supplied covariates; auto-generated as standard-normal if NULL |
| `seed`       | integer               | Passed to `set.seed()` for reproducibility |

## Sampling Algorithm (vectorized matrix, Approach B)

The Famoye/Sarmanov joint PMF is:

```
P(Y1=y1, Y2=y2) = p1(y1) · p2(y2) · [1 + λ · (exp(-y1) − c1) · (exp(-y2) − c2)]
```

where `c_k = E[exp(−Yk)]` under NB2(mu_k, m_k), computed via `c_val()` from `famoye_core.R`.

**Steps:**

1. Build design matrices X1, X2 from `beta1`, `beta2`, `covariates` (same logic as `simulate_rpbnb()`).
2. Compute `mu1 = exp(X1 %*% beta1)`, `mu2 = exp(X2 %*% beta2)`.
3. Draw `y1 <- rnbinom(n, size = 1/m1, mu = mu1)` (marginal NB2).
4. Compute per-obs `c1 <- c_val(mu1, m1)`, `c2 <- c_val(mu2, m2)`.
5. Validate `lambda` against `lambda_bounds_vec(c1, c2)`; error if out of range.
6. Set `ymax <- ceiling(max(qnbinom(0.9999, size = 1/m2, mu = mu2)))`.
7. Build `n × (ymax+1)` marginal probability matrix `P2` via `dnbinom`.
8. Build weight matrix `W[i,j] = 1 + lambda * (exp(-y1[i]) - c1[i]) * (exp(-y2_grid[j]) - c2[i])`.
9. `Q <- P2 * W`; normalize rows to sum to 1.
10. Draw `y2[i] <- sample(0:ymax, 1, prob = Q[i,])` via `apply(Q, 1, ...)`.

## Validation Errors

- `"(Intercept)"` missing from `beta1` or `beta2`
- `dispersion` not named `c(m1, m2)`
- `lambda` outside `[max(lam_min), min(lam_max)]` from `lambda_bounds_vec(c1, c2)`
- `covariates` provided but missing required columns

## Return Value

```r
list(
  data     = data.frame(y1, y2, <covariate columns>),
  mu       = data.frame(mu1, mu2),
  true     = list(beta1, beta2, dispersion, lambda),
  settings = list(n, seed),
  meta     = list(seed, r_version = R.version.string)
)
```

No `coef_realized` field (no random effects).

## File Placement

| File | Action |
|------|--------|
| `R/simulate_bnb.R` | New — implementation |
| `tests/testthat/test-simulate-bnb.R` | New — tests |
| `NAMESPACE` | Updated by `devtools::document()` via `@export` |
| Existing files | No changes |

## Tests

- Reproducibility: same seed → identical output
- Returns documented list structure
- `$data` has columns `y1`, `y2`, and all covariate names
- Marginal means of y1, y2 match `exp(beta_intercept)` for intercept-only model (large n)
- Lambda = 0 gives independent margins (cross-tab correlation ≈ 0)
- Lambda out of bounds errors correctly
- Missing covariates column errors correctly
- `dispersion` without `m1`/`m2` names errors correctly
