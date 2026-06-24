# RPBNB Random-Coefficient Distributions — Design

**Date:** 2026-06-24
**Status:** Approved (design); pending implementation plan
**Scope:** Add Log-Normal, Uniform, and Triangular random-coefficient
distributions to `fit_rpbnb()` and `simulate_rpbnb()`, alongside the existing
Normal. Johnson S_B / zero-censored (bounded) distributions are deferred to a
later phase.

## Goal

Today every random coefficient in the random-parameter bivariate NB model is
**Normal**: `β = β̄ + sd·z`, `z ~ N(0,1)` via Halton draws. This phase lets the
user choose a distribution **per coefficient**, so a coefficient can be:

- **Normal** — both signs; default when direction is unknown.
- **Log-Normal** — strictly one sign across the population (e.g. a price/cost
  coefficient that must always be negative).
- **Uniform** — continuous variation between bounds, no peak.
- **Triangular** — symmetric, peaked at the center, tapering to the bounds.

Per-distribution **analytic** gradients are required (decision: keep the fast
exact-gradient BFGS path; no numeric fallback). Both fitting and simulation get
the new distributions so the user can simulate-then-recover for validation.

## Key structural insight

All four distributions express the realized coefficient for random column *j*
as a one-scale-parameter transform of a single base draw:

| Dist        | base draw   | params (1 scale each)        | realized β        | mean = location? |
|-------------|-------------|------------------------------|-------------------|------------------|
| Normal      | z = Φ⁻¹(u)  | loc `b`, `log_sd` → s        | `b + s·z`         | yes (linear)     |
| Uniform     | u ~ U(0,1)  | center `b`, `log_w` → w      | `b + w·(2u−1)`    | yes (linear)     |
| Triangular  | u ~ U(0,1)  | center `b`, `log_w` → w      | `b + w·Tri(u)`    | yes (linear)     |
| Log-Normal  | z = Φ⁻¹(u)  | log-loc `b`, `log_s` → s, σ  | `σ·exp(b + s·z)`  | no (nonlinear)   |

`Tri(u)` is the inverse-CDF of the symmetric triangular on `[−1, 1]`:
`u < 0.5 → −1 + √(2u)`, `u ≥ 0.5 → 1 − √(2(1−u))`.

**Consequence — parameter vector unchanged.** Every distribution has exactly
**one** scale parameter per random coefficient, so the existing layout

```
[ β1 (k1) ] [ β2 (k2) ] [ scale1 (q1) ] [ scale2 (q2) ] [ log_m1 ] [ log_m2 ] [ z_lambda ]
```

is preserved. Only the *interpretation*, the parameter *labels* (`log_sd` vs
`log_w` vs `log_s`), the *transform*, and the *gradient terms* differ. The
lognormal sign σ is **fixed by the user, not estimated**.

## Gradient impact

The random coefficient enters only through `μ_r = exp(η_r)`,
`η_r = X·β + XR %*% d_r`, where `d_r = β(u) − b` is the per-draw deviation.

- **Linear dists** (Normal, Uniform, Triangular): `∂η/∂loc = XR` (unchanged from
  today). Scale score replaces the current `sd·z` term with the dist's `dscale`.
- **Log-Normal** (nonlinear): `∂η/∂b = XR·β` and `∂η/∂log_s = XR·(s·z·β)`. The
  location-score column itself becomes draw-dependent.

So the analytic gradient gains a per-distribution **location factor**
(`dloc_factor`: `1` for linear dists, `β` for lognormal) applied to the β-score
columns of random coefficients, plus a per-distribution **scale-score** term
(`dscale`) replacing the hard-coded `XR·(sd·z)`.

## Components

### New: `R/rand_dist.R` — distribution registry

Each distribution is a small module; all per-distribution math lives here.

```r
rand_dist_registry[[name]] = list(
  base        = "normal" | "uniform",       # which Halton base draw
  u_to_base   = function(u) ...,            # qnorm(u) or identity(u)
  coef        = function(b, s, base, sign), # realized β per draw
  dev         = function(b, s, base, sign), # deviation d = β − b (η contribution)
  dloc_factor = function(b, s, base, beta), # ∂η/∂loc multiplier: 1, or β (lognormal)
  dscale      = function(b, s, base, beta), # ∂η/∂log_scale per draw
  scale_label = "log_sd" | "log_w" | "log_s",
  start_scale = log(0.2)                     # default start on log scale
)
```

Adding Johnson S_B later = one new registry entry plus multi-param handling.

### `R/simulation_draws.R`

Add `halton_uniform(n_draws, d, burn)` returning the rotated `U` matrix.
`halton_normal` becomes `qnorm(halton_uniform(...))` — **no behavior change** for
the Normal-only path. The fitter generates **one** uniform Halton matrix of
dimension `q1+q2`, then applies each column's `u_to_base` (qnorm for
normal/lognormal, identity for uniform/triangular). Mixed base types coexist in
one draw matrix.

### `R/fit_rpbnb.R`

Parse the named-list spec into a per-random-column vector of dist names (+ signs)
via the shared `parse_rand_spec()` helper. Drives `par_names` labels, start
values, and is passed through to the likelihood, `rebuild_bounds()`, and the
fitted-means block — all of which currently hard-code
`exp(xb + XR·(sd·z))` and must route through `registry$coef`.

### `R/rpbnb_likelihood.R`

`bnbr_rp_ll_and_grad` and `bnbr_rp_ll_fixed_bounds` build `μ_r` via
`registry$coef`/`dev`. The gradient applies `dloc_factor` (modifying the β-score
columns for random coefficients) and `dscale` (replacing the current `sd·z`
term).

### `R/simulate_rpbnb.R`

`random_1`/`random_2` accept the same named-list spec; `realize()` applies
`registry$coef` with `rnorm`/`runif` base draws. The λ = 0 restriction is
unchanged (out of scope here).

### Tests — `tests/testthat/`

- Simulate-and-recover for each of the four distributions (params recovered
  within tolerance).
- Backward-compat: a character-vector spec produces numbers **identical** to the
  current Normal-only path.

## API

`random_1` / `random_2` accept (in `fit_rpbnb` and `simulate_rpbnb`):

- `NULL` → no random coefficients.
- character vector `c("x1", "x2")` → all `"normal"`, sign `+1` (**backward
  compatible**, identical numbers to today).
- named list whose value is **either**:
  - a string: `list(x1 = "uniform", x2 = "triangular")`, or
  - a list for options: `list(price = list(dist = "lognormal", sign = -1))`.

### Shared parsing helper

```r
parse_rand_spec(spec, valid_cols) -> list(names=, dist=, sign=)
```

**Errors (fail fast, `call. = FALSE`):**

- unknown column name (already enforced today);
- unknown distribution name;
- `sign` supplied for a non-lognormal distribution;
- `sign ∉ {−1, +1}`.

## Data flow & numerical notes

- **Start values:** location starts at 0 (as today); scale starts at
  `registry$start_scale`. For lognormal the location slot is a *log*-location,
  so start 0 → median coefficient ±1 — a neutral start.
- **λ-bounds & fitted means:** `rebuild_bounds()` and the `mu1_hat`/`mu2_hat`
  blocks in `fit_rpbnb.R` route through `registry$coef`, so bounds and fitted
  means are correct for every distribution. The bounds-freezing logic itself is
  unchanged.
- **Numerical guards:** lognormal `exp(b + s·z)` can overflow for large draws —
  clamp the realized contribution to `μ` with `pmin(..., 1e15)`, consistent with
  the copula-likelihood clamp. Triangular uses the exact piecewise `√` inverse
  (no division), numerically stable.
- **SE/Hessian:** unchanged mechanism — `bnbr_rp_ll_fixed_bounds` needs only the
  transform (not the gradient), so the existing numDeriv + ridge + ginv path
  works for all distributions.

## Verification gate

Before claiming complete:

- `R CMD check` clean (no new NOTEs/WARNINGs from the changed files);
- all `testthat` green, including the new simulate-recover tests and the
  backward-compat identity test.

## Out of scope (later phases)

- Johnson S_B and zero-censored / bounded distributions (extra shape params,
  bounded-support gradients).
- Dependence (λ ≠ 0) in `simulate_rpbnb`.
