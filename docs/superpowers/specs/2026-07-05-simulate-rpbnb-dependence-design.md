# Simulate RPBNB with Dependence — Design

**Date:** 2026-07-05
**Status:** Approved
**Scope:** Add Famoye/Sarmanov and copula-based dependence to `simulate_rpbnb()`, lifting the current `lambda = 0` restriction.

## Goal

Currently `simulate_rpbnb()` hard-stops at `lambda != 0`. This phase adds:

- **Famoye/Sarmanov dependence** via λ (reuses the conditional sampler from `simulate_bnb`)
- **Copula dependence** via `dependence = copula(...)` matching `fit_bnb`'s API (Gaussian, Frank, Clayton)

## API

```r
simulate_rpbnb(n, beta1, beta2,
  random_1 = NULL, random_2 = NULL,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  lambda = 0,                    # unchanged — Famoye dependence
  dependence = NULL,             # NEW — copula("normal", par = 0.3), etc.
  covariates = NULL, seed = NULL)
```

### Behavior matrix

| `lambda` | `dependence` | Result |
|----------|-------------|--------|
| `0` | `NULL` (default) | Independent margins _(backward compat)_ |
| non-zero | `NULL` | Famoye/Sarmanov conditional sampling |
| any | `copula(...)` | Copula-based sampling, λ ignored |

### Copula parameterization

Uses the same `copula()` factory as `fit_bnb`:

```r
simulate_rpbnb(..., dependence = copula("normal",  par = 0.3))
simulate_rpbnb(..., dependence = copula("frank",   par = 2.0))
simulate_rpbnb(..., dependence = copula("clayton", par = 1.5))
```

- `copula("normal")`: par = ρ ∈ (-1, 1)
- `copula("frank")`: par = θ ∈ (-∞, ∞), θ=0 → independence
- `copula("clayton")`: par = θ > 0, θ→0 → independence

## Architecture

### File changes

| File | Change |
|------|--------|
| `R/simulate_rpbnb.R` | Remove λ=0 guard, insert dependence dispatch, update return values |
| `R/simulate_bnb.R` | Extract Famoye conditional sampler to shared internal function |
| `R/copula_core.R` | Add 3 internal copula simulators |
| `tests/testthat/test-copula-sim.R` | New: unit tests for copula samplers |
| `tests/testthat/test-simulate-rpbnb.R` | Extend: Famoye λ tests, copula tests, backward compat |
| `tests/testthat/test-fit-rpbnb.R` | Extend: simulate-then-fit roundtrip with λ>0 |

### Flow (inside `simulate_rpbnb`)

```
1. Parse random spec, validate inputs
2. Realize random coefficients → B1, B2 (per-obs)
3. Compute mu1 = exp(rowSums(X1 * B1)), mu2 = exp(rowSums(X2 * B2))
4. Compute c1 = c_val(mu1, m1), c2 = c_val(mu2, m2)

5. DEPENDENCE DISPATCH:
   ├─ dependence is copula:
   │    u = .sim_copula_<family>(n, par)
   │    y1 = qnbinom(u1, size=1/m1, mu=mu1)
   │    y2 = qnbinom(u2, size=1/m2, mu=mu2)
   │
   ├─ lambda != 0 (Famoye):
   │    y1 = rnbinom(n, size=1/m1, mu=mu1)          ← marginal NB2
   │    validate lambda ∈ bounds(c1, c2)
   │    y2 = .sim_famoye_conditional(y1, mu2, c1, c2, m2, lambda)  ← Sarmanov-weighted
   │
   └─ lambda == 0 (independence):
        y1 = rnbinom(n, size=1/m1, mu=mu1)
        y2 = rnbinom(n, size=1/m2, mu=mu2)
```

### Shared Famoye sampler (extracted from `simulate_bnb`)

```r
# Internal, in simulate_bnb.R (or a shared utility)
# Given y1 already drawn from marginal NB2, draws y2 from the
# Sarmanov-weighted conditional: P(y2) ∝ NB2(y2|mu2) × W(y1, y2)
.sim_famoye_conditional <- function(y1, mu2, c1, c2, m2, lambda) {
  ymax <- ceiling(max(qnbinom(0.9999, size = 1 / m2, mu = mu2)))
  y2_grid <- 0:ymax
  P2 <- outer(mu2, y2_grid, function(mu, y) dnbinom(y, size = 1 / m2, mu = mu))
  row_factor <- lambda * (exp(-y1) - c1)
  c_per_row  <- row_factor * c2
  W <- 1 + outer(row_factor, exp(-y2_grid)) - c_per_row
  Q <- pmax(P2 * W, 0)
  Q <- Q / rowSums(Q)
  cum <- Q
  for (j in seq_len(ncol(cum))[-1]) cum[, j] <- cum[, j - 1L] + cum[, j]
  u <- runif(length(y1)) * cum[, ncol(cum)]
  y2_grid[max.col(cum >= u, ties.method = "first")]
}
```

Both `simulate_bnb` (lines 83-116) and `simulate_rpbnb` call this function. No code duplication.

### Copula simulators (new in `copula_core.R`)

Three internal functions returning `list(u1, u2)` — uniform margins ready for `qnbinom` inversion:

```r
# Gaussian copula — Cholesky decomposition
.sim_copula_normal <- function(n, rho) {
  z1 <- rnorm(n)
  z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)
  list(u1 = pnorm(z1), u2 = pnorm(z2))
}

# Frank copula — conditional inversion
.sim_copula_frank <- function(n, theta) {
  u1 <- runif(n); t <- runif(n)
  A <- exp(-theta * u1); D <- exp(-theta) - 1
  u2 <- -log1p(t * D / pmax(A * (1 - t) + t, 1e-300)) / theta
  u2 <- pmin(pmax(u2, 1e-10), 1 - 1e-10)
  list(u1 = u1, u2 = u2)
}

# Clayton copula — conditional inversion
.sim_copula_clayton <- function(n, theta) {
  u1 <- runif(n); t <- runif(n)
  exp1 <- -theta / (theta + 1)
  inner <- u1^(-theta) * (t^exp1 - 1) + 1
  u2 <- pmax(inner, 1e-300)^(-1 / theta)
  u2 <- pmin(pmax(u2, 1e-10), 1 - 1e-10)
  list(u1 = u1, u2 = u2)
}
```

Ported from `inst/simulate_rwm1984_dgp.R` (lines 108-117, 126-137, 145-156).

### Lambda bounds validation

In the Famoye path, lambda must fall within data-adaptive bounds computed per observation:

```r
bnds <- lambda_bounds_vec(c1, c2)  # returns c(max(λ_lo_i), min(λ_hi_i))
if (lambda < bnds[1] || lambda > bnds[2])
  stop("lambda outside valid bounds [", bnds[1], ", ", bnds[2], "]")
```

`lambda_bounds_vec` already computes global bounds from per-observation c₁_i, c₂_i — works identically for RP-BNB where c₁_i, c₂_i vary per-observation from realized μ₁_i, μ₂_i.

Copula parameters have no such data-adaptive bounds (ρ ∈ (-1,1), θ unconstrained), so no validation needed.

### Return value additions

`$true` gains:
- `dependence` — `"independence"`, `"famoye"`, or copula family (`"normal"`, `"frank"`, `"clayton"`)
- `dependence_par` — λ, ρ, or θ as appropriate

`$settings` gains:
- `dependence_type` — same as `$true$dependence`
- `dependence_par` — the effective parameter value

## Testing

### New test file: `test-copula-sim.R`

| Test | Method |
|------|--------|
| Normal copula recovers ρ | `cor(qnorm(u1), qnorm(u2)) ≈ ρ` for n=5000 |
| Frank copula Kendall's τ | `cor(u1, u2, method="kendall") ≈ frank_tau(θ)` |
| Clayton copula Kendall's τ | `cor(u1, u2, method="kendall") ≈ θ/(θ+2)` |
| Frank θ=0 → independence | τ ≈ 0 for large n |
| Clayton θ→0 → independence | τ ≈ 0 for large n |

### Extend: `test-simulate-rpbnb.R`

| Test | Method |
|------|--------|
| λ=0.1 produces positive correlation | `cor(y1, y2) > 0.01` for n=500 |
| λ=0.1 is seed-reproducible | two calls with same seed give identical `$data` |
| λ outside bounds errors | provide λ >> valid → expect error with "outside the valid bounds" |
| copula("normal", 0.3) produces valid counts | all y1, y2 are non-negative integers |
| copula("frank", 2) is seed-reproducible | two calls with same seed → identical |
| copula("clayton", 1) produces valid output structure | all `$data`, `$mu`, `$coef_realized` present |
| Backward compat: λ=0, no dependence | output identical to previous version (same seed) |
| λ=0 with random coefficients | `$coef_realized` still present and correct |

### Extend: `test-fit-rpbnb.R`

| Test | Method |
|------|--------|
| Simulate with λ=0.1, fit with λ start | λ_hat within 0.05 of true for n=600, draws=200 |
| Simulate with normal copula ρ=0.3, fit with `fit_bnb(copula("normal"))` | ρ_hat ≈ 0.3 |

### Backward compatibility

- Default call `simulate_rpbnb(n, beta1, beta2)` with no `lambda` or `dependence` → identical behavior and output structure
- `lambda = 0` (explicit) → identical
- All 17 existing test files must pass unchanged

## Non-goals (deferred)

- Simulation with λ≠0 and copula simultaneously (nonsensical — pick one dependence structure)
- Johnson S_B or bounded distributions for copula parameters
- `simulate_bnb` copula support (separate change)
- Rejection-sampling exact Famoye sampler (grid approximation is inherited from `simulate_bnb`; same caveat applies)
