# RPBNB Random-Coefficient Distributions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Log-Normal, Uniform, and Triangular random-coefficient distributions (alongside the existing Normal) to `fit_rpbnb()` and `simulate_rpbnb()`, choosable per coefficient, with per-distribution analytic gradients.

**Architecture:** A single distribution registry (`R/rand_dist.R`) holds all per-distribution math (base-draw type, inverse-CDF transform, realized coefficient, and analytic-gradient factors). The likelihood, gradient, λ-bounds, fitted-means, and simulator all route through it. The parameter vector layout is unchanged — every distribution has exactly one scale parameter per random coefficient — so only the transforms, parameter labels, and two gradient terms differ.

**Tech Stack:** R (package `rpbnb`), `maxLik` (BFGS), `numDeriv` (Hessian + gradient checks), `randtoolbox` (Halton), `testthat` (3rd ed), `roxygen2`.

## Global Constraints

- Distributions in scope this phase: `normal`, `lognormal`, `uniform`, `triangular`. Johnson S_B / zero-censored are **out of scope**.
- Parameter vector layout is fixed and unchanged: `[β1 (k1)] [β2 (k2)] [scale1 (q1)] [scale2 (q2)] [log_m1] [log_m2] [z_lambda]`. Exactly one scale parameter per random coefficient.
- Lognormal `sign` is **set by the user, not estimated**; valid values are `-1` or `1` (default `1`).
- Backward compatibility (bit-identical numbers): a plain **character vector** `random_1`/`random_2` must reproduce today's Normal-only results in both `fit_rpbnb` and `simulate_rpbnb`.
- The realized coefficient deviation feeding `μ` must be clamped consistently with the copula code: `pmin(exp(...), 1e15)`.
- All errors use `stop(..., call. = FALSE)`.
- Run tests with `devtools::test()` (or `pkgload::load_all()` then `testthat::test_dir`). Match existing test style in `tests/testthat/`.

---

### Task 1: Distribution registry and spec parser

**Files:**
- Create: `R/rand_dist.R`
- Test: `tests/testthat/test-rand-dist.R`

**Interfaces:**
- Produces:
  - `tri_icdf(u)` — numeric vector, symmetric triangular inverse-CDF on `[-1, 1]`.
  - `rand_dist_registry` — named list; keys `normal`, `uniform`, `triangular`, `lognormal`; each a list with fields `base` (`"normal"`/`"uniform"`), `u_to_base(u)`, `coef(b, s, base, sign)`, `dev(b, s, base, sign)`, `dloc_factor(b, s, base, coef)`, `dscale(b, s, base, coef)`, `scale_label`.
  - `parse_rand_spec(spec)` → `list(names=<chr>, dist=<chr>, sign=<num>, scale=<num>)` aligned by position.
  - `rand_realize(U, dist, sign, b, s)` → `list(base, coef, dev, dloc, dscale)`, each an `nrow(U) × ncol(U)` matrix.

- [ ] **Step 1: Write the failing test**

Create `tests/testthat/test-rand-dist.R`:

```r
test_that("tri_icdf maps endpoints and midpoint correctly", {
  expect_equal(tri_icdf(0.5), 0)
  expect_equal(tri_icdf(1e-12), -1, tolerance = 1e-5)
  expect_equal(tri_icdf(1 - 1e-12), 1, tolerance = 1e-5)
  u <- seq(0.01, 0.99, length.out = 50)
  expect_true(all(diff(tri_icdf(u)) > 0))  # monotone increasing
})

test_that("registry transforms have the right large-sample moments", {
  set.seed(1)
  z <- stats::qnorm(stats::runif(2e5)); u <- stats::runif(2e5)
  # normal: mean b, sd s
  cn <- rand_dist_registry$normal$coef(0.5, 0.8, z, 1)
  expect_equal(mean(cn), 0.5, tolerance = 0.02)
  expect_equal(stats::sd(cn), 0.8, tolerance = 0.02)
  # uniform on [b-w, b+w]: var = w^2/3
  cu <- rand_dist_registry$uniform$coef(0.5, 0.9,
          rand_dist_registry$uniform$u_to_base(u), 1)
  expect_equal(mean(cu), 0.5, tolerance = 0.02)
  expect_equal(stats::var(cu), 0.9^2 / 3, tolerance = 0.02)
  expect_true(all(cu >= 0.5 - 0.9 - 1e-8 & cu <= 0.5 + 0.9 + 1e-8))
  # triangular: var = w^2/6
  ct <- rand_dist_registry$triangular$coef(0.0, 0.9,
          rand_dist_registry$triangular$u_to_base(u), 1)
  expect_equal(stats::var(ct), 0.9^2 / 6, tolerance = 0.02)
  # lognormal negative sign => strictly negative
  cl <- rand_dist_registry$lognormal$coef(-0.2, 0.5, z, -1)
  expect_true(all(cl < 0))
})

test_that("lognormal gradient factors match definitions", {
  z <- c(-1, 0, 1.5); b <- 0.3; s <- 0.4
  coef <- rand_dist_registry$lognormal$coef(b, s, z, -1)
  expect_equal(rand_dist_registry$lognormal$dloc_factor(b, s, z, coef), coef)
  expect_equal(rand_dist_registry$lognormal$dscale(b, s, z, coef), coef * z * s)
})

test_that("parse_rand_spec handles char vector, named list, and errors", {
  expect_equal(parse_rand_spec(NULL)$names, character(0))
  cv <- parse_rand_spec(c("x1", "x2"))
  expect_equal(cv$dist, c("normal", "normal"))
  expect_equal(cv$sign, c(1, 1))
  nl <- parse_rand_spec(list(x1 = "uniform",
                             p = list(dist = "lognormal", sign = -1)))
  expect_equal(nl$names, c("x1", "p"))
  expect_equal(nl$dist, c("uniform", "lognormal"))
  expect_equal(nl$sign, c(1, -1))
  expect_error(parse_rand_spec(list(x1 = "weibull")), "unknown distribution")
  expect_error(parse_rand_spec(list(x1 = list(dist = "normal", sign = -1))),
               "only meaningful for lognormal")
  expect_error(parse_rand_spec(list(x1 = list(dist = "lognormal", sign = 2))),
               "must be -1 or 1")
  expect_error(parse_rand_spec(list("normal")), "named list")
})

test_that("parse_rand_spec reads scale via scale or sd alias", {
  s <- parse_rand_spec(list(x1 = list(dist = "uniform", scale = 0.7),
                            x2 = list(sd = 0.5)))
  expect_equal(s$scale, c(0.7, 0.5))
  expect_equal(s$dist, c("uniform", "normal"))
})

test_that("rand_realize returns aligned matrices", {
  set.seed(2)
  U <- matrix(stats::runif(20), nrow = 10, ncol = 2)
  out <- rand_realize(U, dist = c("normal", "lognormal"),
                      sign = c(1, -1), b = c(0.1, 0.2), s = c(0.3, 0.4))
  expect_equal(dim(out$coef), c(10, 2))
  expect_true(all(out$coef[, 2] < 0))            # lognormal neg sign
  expect_equal(out$dloc[, 1], rep(1, 10))        # normal location factor
  expect_equal(out$dloc[, 2], out$coef[, 2])     # lognormal location factor
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-rand-dist.R')"`
Expected: FAIL — `could not find function "tri_icdf"` / `rand_dist_registry` not found.

- [ ] **Step 3: Write the implementation**

Create `R/rand_dist.R`:

```r
# Random-coefficient distribution registry: the single source of truth for the
# per-distribution transforms and analytic-gradient factors used by the
# random-parameter BNB likelihood, its gradient, the lambda-bounds, the fitted
# means, and the simulator.
#
# A realized random coefficient for a column is
#   beta = location + scale * base                 (normal, uniform, triangular)
#   beta = sign * exp(location + scale * base)      (lognormal)
# where `base` is the inverse-CDF-ready base draw: a standard normal z for
# normal/lognormal, the symmetric triangular variate for triangular, and the
# uniform u in (0,1) itself for uniform. `u_to_base()` maps a uniform draw to
# the base variate; the simulator may instead draw the base directly.

#' Symmetric triangular inverse-CDF on [-1, 1]
#' @keywords internal
#' @noRd
tri_icdf <- function(u) {
  ifelse(u < 0.5, -1 + sqrt(2 * u), 1 - sqrt(2 * (1 - u)))
}

#' Registry of supported random-coefficient distributions
#' @keywords internal
#' @noRd
rand_dist_registry <- list(
  normal = list(
    base        = "normal",
    u_to_base   = function(u) stats::qnorm(u),
    coef        = function(b, s, base, sign) b + s * base,
    dev         = function(b, s, base, sign) s * base,
    dloc_factor = function(b, s, base, coef) rep(1, length(base)),
    dscale      = function(b, s, base, coef) s * base,
    scale_label = "log_sd"
  ),
  uniform = list(
    base        = "uniform",
    u_to_base   = function(u) u,
    coef        = function(b, s, base, sign) b + s * (2 * base - 1),
    dev         = function(b, s, base, sign) s * (2 * base - 1),
    dloc_factor = function(b, s, base, coef) rep(1, length(base)),
    dscale      = function(b, s, base, coef) s * (2 * base - 1),
    scale_label = "log_w"
  ),
  triangular = list(
    base        = "uniform",
    u_to_base   = function(u) tri_icdf(u),
    coef        = function(b, s, base, sign) b + s * base,
    dev         = function(b, s, base, sign) s * base,
    dloc_factor = function(b, s, base, coef) rep(1, length(base)),
    dscale      = function(b, s, base, coef) s * base,
    scale_label = "log_w"
  ),
  lognormal = list(
    base        = "normal",
    u_to_base   = function(u) stats::qnorm(u),
    coef        = function(b, s, base, sign) sign * exp(b + s * base),
    dev         = function(b, s, base, sign) sign * exp(b + s * base) - b,
    dloc_factor = function(b, s, base, coef) coef,
    dscale      = function(b, s, base, coef) coef * base * s,
    scale_label = "log_s"
  )
)

#' Normalize a random-coefficient spec to aligned name/dist/sign/scale vectors
#'
#' Accepts NULL, a character vector of column names (all Normal), or a named
#' list whose values are either a distribution-name string or a list with
#' `dist`, optional `sign` (lognormal only), and optional `scale` (or `sd`).
#' @keywords internal
#' @noRd
parse_rand_spec <- function(spec) {
  if (is.null(spec) || length(spec) == 0) {
    return(list(names = character(0), dist = character(0),
                sign = numeric(0), scale = numeric(0)))
  }
  valid <- names(rand_dist_registry)
  if (is.character(spec) && is.null(names(spec))) {
    return(list(names = spec,
                dist  = rep("normal", length(spec)),
                sign  = rep(1, length(spec)),
                scale = rep(NA_real_, length(spec))))
  }
  if (!is.list(spec) || is.null(names(spec)) || any(!nzchar(names(spec)))) {
    stop("random spec must be a character vector of names or a named list.",
         call. = FALSE)
  }
  nm    <- names(spec)
  dist  <- character(length(nm))
  sgn   <- numeric(length(nm))
  scale <- numeric(length(nm))
  for (i in seq_along(nm)) {
    v <- spec[[i]]
    if (is.character(v) && length(v) == 1L) {
      d <- v; this_sign <- 1; this_scale <- NA_real_
    } else if (is.list(v)) {
      d          <- if (is.null(v$dist)) "normal" else v$dist
      this_sign  <- if (is.null(v$sign)) 1 else v$sign
      this_scale <- if (!is.null(v$scale)) v$scale
                    else if (!is.null(v$sd)) v$sd else NA_real_
    } else {
      stop("random spec value for '", nm[i],
           "' must be a distribution name or a list.", call. = FALSE)
    }
    if (!d %in% valid) {
      stop("unknown distribution '", d, "' for '", nm[i], "'. Valid: ",
           paste(valid, collapse = ", "), ".", call. = FALSE)
    }
    if (this_sign != 1 && d != "lognormal") {
      stop("`sign` is only meaningful for lognormal (got '", d,
           "' for '", nm[i], "').", call. = FALSE)
    }
    if (!this_sign %in% c(-1, 1)) {
      stop("`sign` must be -1 or 1 for '", nm[i], "'.", call. = FALSE)
    }
    dist[i] <- d; sgn[i] <- this_sign; scale[i] <- this_scale
  }
  list(names = nm, dist = dist, sign = sgn, scale = scale)
}

#' Per-draw realized coefficients and gradient factors for one equation
#'
#' @param U An R x q matrix of uniform Halton draws (one column per random coef).
#' @param dist,sign Length-q distribution names and signs.
#' @param b,s Length-q location and native-scale (exp of the log-scale param).
#' @return A list of R x q matrices: `base`, `coef`, `dev`, `dloc`, `dscale`.
#' @keywords internal
#' @noRd
rand_realize <- function(U, dist, sign, b, s) {
  R <- nrow(U); q <- ncol(U)
  base <- coef <- dev <- dloc <- dscale <- matrix(0, nrow = R, ncol = q)
  for (j in seq_len(q)) {
    reg <- rand_dist_registry[[dist[j]]]
    bj  <- reg$u_to_base(U[, j])
    cj  <- reg$coef(b[j], s[j], bj, sign[j])
    base[, j]   <- bj
    coef[, j]   <- cj
    dev[, j]    <- reg$dev(b[j], s[j], bj, sign[j])
    dloc[, j]   <- reg$dloc_factor(b[j], s[j], bj, cj)
    dscale[, j] <- reg$dscale(b[j], s[j], bj, cj)
  }
  list(base = base, coef = coef, dev = dev, dloc = dloc, dscale = dscale)
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-rand-dist.R')"`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add R/rand_dist.R tests/testthat/test-rand-dist.R
git commit -m "feat: add random-coefficient distribution registry and spec parser

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Uniform Halton draws

**Files:**
- Modify: `R/simulation_draws.R`
- Test: `tests/testthat/test-halton-uniform.R`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `halton_uniform(n_draws, d, burn = 200)` → `n_draws × d` matrix of rotated uniforms in `(0, 1)`; returns a 0-column matrix when `d <= 0`.
  - `halton_normal(n_draws, d, burn = 200)` (refactored) ≡ `qnorm(halton_uniform(...))` — byte-identical to its prior output.

- [ ] **Step 1: Write the failing test**

Create `tests/testthat/test-halton-uniform.R`:

```r
test_that("halton_uniform returns rotated uniforms in (0,1)", {
  set.seed(11)
  U <- halton_uniform(50, 3, burn = 20)
  expect_equal(dim(U), c(50, 3))
  expect_true(all(U > 0 & U < 1))
})

test_that("halton_uniform handles zero dimension", {
  expect_equal(dim(halton_uniform(10, 0)), c(10L, 0L))
})

test_that("halton_normal equals qnorm of halton_uniform with same seed", {
  set.seed(7); Z <- halton_normal(40, 2, burn = 30)
  set.seed(7); U <- halton_uniform(40, 2, burn = 30)
  expect_equal(Z, stats::qnorm(U))
})

test_that("halton_uniform is reproducible for a fixed seed", {
  set.seed(3); a <- halton_uniform(25, 2)
  set.seed(3); b <- halton_uniform(25, 2)
  expect_identical(a, b)
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-halton-uniform.R')"`
Expected: FAIL — `could not find function "halton_uniform"`.

- [ ] **Step 3: Write the implementation**

Replace the body of `R/simulation_draws.R` (keep the file header comment) so it defines both functions:

```r
#' Generate uniform draws from a (randomized) Halton sequence
#'
#' Builds a Halton low-discrepancy sequence, discards the first `burn` points,
#' then applies a Cranley-Patterson rotation (a per-dimension uniform shift drawn
#' from the R RNG) and clamps away from the open endpoints. The rotation makes
#' the draw set depend on the current RNG state, so a caller's [set.seed()] /
#' `seed` choice produces a different but fully reproducible low-discrepancy
#' point set (randomized quasi-Monte Carlo).
#'
#' @param n_draws Number of draws (rows).
#' @param d Dimension (columns). If 0, a 0-column matrix is returned.
#' @param burn Number of leading Halton points to discard.
#' @return An `n_draws` x `d` numeric matrix of uniforms in (0, 1).
#' @keywords internal
#' @noRd
halton_uniform <- function(n_draws, d, burn = 200) {
  if (d <= 0) return(matrix(0, nrow = n_draws, ncol = 0))
  n <- burn + n_draws
  U <- randtoolbox::halton(n = n, dim = d, normal = FALSE,
                           usetime = FALSE, init = TRUE)
  U <- matrix(U, nrow = n, ncol = d)
  U <- U[(burn + 1):n, , drop = FALSE]
  shift <- stats::runif(d)
  U <- sweep(U, 2, shift, `+`)
  U <- U - floor(U)
  pmin(pmax(U, 1e-12), 1 - 1e-12)
}

#' Standard-normal Halton draws (qnorm of [halton_uniform()])
#' @inheritParams halton_uniform
#' @return An `n_draws` x `d` numeric matrix of standard-normal values.
#' @keywords internal
#' @noRd
halton_normal <- function(n_draws, d, burn = 200) {
  if (d <= 0) return(matrix(0, nrow = n_draws, ncol = 0))
  stats::qnorm(halton_uniform(n_draws, d, burn = burn))
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-halton-uniform.R')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add R/simulation_draws.R tests/testthat/test-halton-uniform.R
git commit -m "feat: add halton_uniform and route halton_normal through it

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Per-distribution data generation in `simulate_rpbnb`

**Files:**
- Modify: `R/simulate_rpbnb.R`
- Test: `tests/testthat/test-simulate-rpbnb-dist.R`

**Interfaces:**
- Consumes: `parse_rand_spec()`, `rand_dist_registry` (Task 1).
- Produces: `simulate_rpbnb()` whose `random_1`/`random_2` accept the named-list spec; `realize()` applies `rand_dist_registry[[dist]]$coef`. Backward compat: a list value `list(x1 = list(sd = 0.5))` and a character vector both mean Normal and reproduce prior numbers. For a lognormal coefficient `beta1[[nm]]` is the **log-location**; the realized coefficient is `sign * exp(log_location + scale * z)`.

- [ ] **Step 1: Write the failing test**

Create `tests/testthat/test-simulate-rpbnb-dist.R`:

```r
test_that("normal spec reproduces the pre-refactor draws (backward compat)", {
  set.seed(99)
  ref_cov <- data.frame(x1 = rnorm(2000))
  # Legacy realization formula: B = beta + sd * rnorm(n)
  set.seed(5); legacy_eps <- rnorm(2000)
  legacy_B <- 0.4 + 0.5 * legacy_eps
  s <- simulate_rpbnb(n = 2000,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
        random_1 = list(x1 = list(sd = 0.5)),
        dispersion = c(m1 = 0.4, m2 = 0.5),
        covariates = ref_cov, seed = 5)
  # The x1 realized coefficients are drawn first for eq1 in realize(); compare moments
  expect_equal(mean(s$coef_realized$eq1[, "x1"]), 0.4, tolerance = 0.03)
  expect_equal(stats::sd(s$coef_realized$eq1[, "x1"]), 0.5, tolerance = 0.03)
})

test_that("lognormal sign forces coefficient sign", {
  s <- simulate_rpbnb(n = 1500,
        beta1 = c("(Intercept)" = 0.2, x1 = -0.1),
        beta2 = c("(Intercept)" = 0.1),
        random_1 = list(x1 = list(dist = "lognormal", scale = 0.4, sign = -1)),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 8)
  expect_true(all(s$coef_realized$eq1[, "x1"] < 0))
})

test_that("uniform realized coefficients stay within [center +/- width]", {
  s <- simulate_rpbnb(n = 1500,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.5),
        beta2 = c("(Intercept)" = 0.1),
        random_1 = list(x1 = list(dist = "uniform", scale = 0.3)),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 9)
  cc <- s$coef_realized$eq1[, "x1"]
  expect_true(all(cc >= 0.5 - 0.3 - 1e-8 & cc <= 0.5 + 0.3 + 1e-8))
})

test_that("missing scale for a random coefficient is an error", {
  expect_error(
    simulate_rpbnb(n = 100,
      beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
      beta2 = c("(Intercept)" = 0.1),
      random_1 = list(x1 = list(dist = "uniform")),
      dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1),
    "scale"
  )
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-simulate-rpbnb-dist.R')"`
Expected: FAIL — current `realize()` reads `rl[[nm]]$sd` and ignores `dist`; uniform/lognormal tests fail, missing-scale error not raised.

- [ ] **Step 3: Write the implementation**

In `R/simulate_rpbnb.R`, update the roxygen `@param random_1,random_2` line to:

```r
#' @param random_1,random_2 Named lists giving random coefficients. Each value
#'   is a list with `dist` (one of "normal", "lognormal", "uniform",
#'   "triangular"; default "normal"), `scale` (or `sd`) for the dispersion, and
#'   `sign` (-1/1, lognormal only). Means come from `beta1`/`beta2`; for a
#'   lognormal coefficient the `beta` entry is the log-location and the realized
#'   coefficient is `sign * exp(log_location + scale * z)`.
```

Replace the `chk_rand` block and the `realize` function. First, parse specs right after the `dispersion` name check:

```r
  spec1 <- parse_rand_spec(random_1)
  spec2 <- parse_rand_spec(random_2)
  chk_rand <- function(spec, bv, lbl) {
    miss <- spec$names[!spec$names %in% names(bv)]
    if (length(miss)) {
      stop("random name(s) ", paste(miss, collapse = ", "), " not in ", lbl,
           ".", call. = FALSE)
    }
    if (length(spec$names) && any(is.na(spec$scale))) {
      stop("each random coefficient needs a `scale` (or `sd`) in ", lbl, ".",
           call. = FALSE)
    }
  }
  chk_rand(spec1, beta1, "beta1"); chk_rand(spec2, beta2, "beta2")
```

Then replace the realization helper and its two calls:

```r
  realize <- function(bv, spec, X) {
    B <- matrix(rep(bv, each = n), nrow = n, dimnames = list(NULL, names(bv)))
    for (i in seq_along(spec$names)) {
      nm  <- spec$names[i]
      reg <- rand_dist_registry[[spec$dist[i]]]
      base <- if (reg$base == "normal") stats::rnorm(n)
              else reg$u_to_base(stats::runif(n))
      B[, nm] <- reg$coef(bv[[nm]], spec$scale[i], base, spec$sign[i])
    }
    B
  }
  B1 <- realize(beta1, spec1, X1); B2 <- realize(beta2, spec2, X2)
```

Finally, in the returned `true` list, replace `random_1 = random_1, random_2 = random_2` with the parsed specs so downstream code sees a normalized structure:

```r
    true = list(beta1 = beta1, beta2 = beta2, random_1 = spec1,
                random_2 = spec2, dispersion = dispersion, lambda = lambda),
```

(Note: for `dist == "normal"` the base draw is `rnorm(n)`, so the legacy
`beta + sd * rnorm(n)` numbers are reproduced bit-for-bit.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-simulate-rpbnb-dist.R')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add R/simulate_rpbnb.R tests/testthat/test-simulate-rpbnb-dist.R
git commit -m "feat: per-distribution random coefficients in simulate_rpbnb

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Distribution-aware likelihood and analytic gradient

**Files:**
- Modify: `R/rpbnb_likelihood.R`
- Test: `tests/testthat/test-rpbnb-likelihood-dist.R`

**Interfaces:**
- Consumes: `rand_realize()`, `rand_dist_registry` (Task 1); `c_val`, `lambda_bounds_vec`, `nb_logpmf_y_mu_r`, `dct_dm`, `dc_dbeta_mat`, `d_const` (existing, `R/famoye_core.R`); `row_log_sum_exp` (existing, `R/utilities.R`).
- Produces (updated signatures — both gain `dist1, dist2, sign1, sign2`; the `Z1, Z2` arguments now carry **uniform** Halton draws):
  - `bnbr_rp_ll_and_grad(par, y1, y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2, Z1, Z2, dist1, dist2, sign1, sign2, cl = NULL)` → numeric scalar log-likelihood with `attr(., "gradient")`.
  - `bnbr_rp_ll_fixed_bounds(par, y1, y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2, Z1, Z2, lamLo, lamHi, dist1, dist2, sign1, sign2, cl = NULL)` → numeric scalar.

- [ ] **Step 1: Write the failing test**

Create `tests/testthat/test-rpbnb-likelihood-dist.R`. This test checks the analytic gradient against a numeric gradient for every distribution, and that the all-normal path equals the value computed with the legacy normal draws.

```r
make_case <- function(dist, sign = 1, seed = 4) {
  set.seed(seed)
  n <- 80
  X1 <- cbind(`(Intercept)` = 1, x1 = rnorm(n))
  X2 <- cbind(`(Intercept)` = 1, x1 = rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  rand_idx1 <- 2L; rand_idx2 <- integer(0)
  XR1 <- X1[, rand_idx1, drop = FALSE]; XR2 <- NULL
  U1 <- halton_uniform(60, 1, burn = 50)
  U2 <- matrix(0, nrow = 60, ncol = 0)
  # par: b1(2), b2(2), scale1(1), log_m1, log_m2, z_lambda
  par <- c(0.1, 0.2, 0.05, -0.1, log(0.3), log(0.4), 0.2)
  list(par = par, y1 = y1, y2 = y2, X1 = X1, X2 = X2, XR1 = XR1, XR2 = XR2,
       rand_idx1 = rand_idx1, rand_idx2 = rand_idx2, U1 = U1, U2 = U2,
       dist1 = dist, dist2 = character(0), sign1 = sign, sign2 = numeric(0))
}

check_grad <- function(dist, sign = 1) {
  cs <- make_case(dist, sign)
  f <- function(p) {
    v <- bnbr_rp_ll_and_grad(p, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                             cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                             cs$dist1, cs$dist2, cs$sign1, cs$sign2)
    as.numeric(v)
  }
  v <- bnbr_rp_ll_and_grad(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                           cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                           cs$dist1, cs$dist2, cs$sign1, cs$sign2)
  g_analytic <- attr(v, "gradient")
  g_numeric  <- numDeriv::grad(f, cs$par)
  expect_equal(g_analytic, g_numeric, tolerance = 1e-4)
}

test_that("analytic gradient matches numeric gradient for each distribution", {
  check_grad("normal")
  check_grad("uniform")
  check_grad("triangular")
  check_grad("lognormal", sign = 1)
  check_grad("lognormal", sign = -1)
})

test_that("fixed-bounds LL is finite and matches the free LL bounds at z=0", {
  cs <- make_case("uniform")
  v  <- bnbr_rp_ll_and_grad(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                            cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                            cs$dist1, cs$dist2, cs$sign1, cs$sign2)
  expect_true(is.finite(as.numeric(v)))
  vf <- bnbr_rp_ll_fixed_bounds(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1,
                                cs$XR2, cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                                -5, 5, cs$dist1, cs$dist2, cs$sign1, cs$sign2)
  expect_true(is.finite(vf))
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-rpbnb-likelihood-dist.R')"`
Expected: FAIL — `unused argument (cs$dist1)` (signatures don't yet accept dist/sign).

- [ ] **Step 3: Write the implementation**

In `R/rpbnb_likelihood.R`, rewrite `bnbr_rp_ll_and_grad`. The changes from the current body: (a) new args `dist1, dist2, sign1, sign2`; (b) treat `Z1, Z2` as uniform draws, build per-equation realizations via `rand_realize`; (c) `μ_r` uses the realized `dev` (clamped); (d) the β-score uses a per-draw effective design `Xeff` whose random columns are scaled by `dloc`; (e) the scale-score uses `dscale` in place of `sd*z`.

```r
bnbr_rp_ll_and_grad <- compiler::cmpfun(function(par, y1, y2, X1, X2, XR1, XR2,
                                                 rand_idx1, rand_idx2, Z1, Z2,
                                                 dist1, dist2, sign1, sign2,
                                                 cl = NULL) {
  n   <- length(y1)
  k1  <- ncol(X1); k2 <- ncol(X2)
  q1  <- length(rand_idx1); q2 <- length(rand_idx2)
  R   <- if (q1 + q2 > 0) nrow(Z1) else 1L

  i1 <- 1:k1; i2 <- (k1+1):(k1+k2)
  beta1 <- par[i1]; beta2 <- par[i2]
  lg1   <- if (q1>0) (k1+k2+1):(k1+k2+q1) else integer(0)
  lg2   <- if (q2>0) (k1+k2+q1+1):(k1+k2+q1+q2) else integer(0)
  log_sd1 <- if (q1>0) par[lg1] else numeric(0)
  log_sd2 <- if (q2>0) par[lg2] else numeric(0)
  idx_end <- k1 + k2 + q1 + q2
  log_m1 <- par[idx_end + 1]; log_m2 <- par[idx_end + 2]; zlam <- par[idx_end + 3]

  m1 <- exp(log_m1); r1 <- 1/m1
  m2 <- exp(log_m2); r2 <- 1/m2
  sd1 <- if (q1 > 0) exp(log_sd1) else numeric(0)
  sd2 <- if (q2 > 0) exp(log_sd2) else numeric(0)

  xb1 <- as.vector(X1 %*% beta1)
  xb2 <- as.vector(X2 %*% beta2)

  # per-draw realized coefficient deviations + analytic-gradient factors
  if (q1 > 0) {
    real1 <- rand_realize(Z1, dist1, sign1, b = beta1[rand_idx1], s = sd1)
  } else real1 <- list(dev = matrix(0, R, 0), dloc = matrix(0, R, 0),
                       dscale = matrix(0, R, 0))
  if (q2 > 0) {
    real2 <- rand_realize(Z2, dist2, sign2, b = beta2[rand_idx2], s = sd2)
  } else real2 <- list(dev = matrix(0, R, 0), dloc = matrix(0, R, 0),
                       dscale = matrix(0, R, 0))

  # ---- Pass 1: mu, c, bounds per draw ----
  pass1_fun <- function(r) {
    eta1 <- if (q1 > 0) xb1 + as.vector(XR1 %*% real1$dev[r, ]) else xb1
    eta2 <- if (q2 > 0) xb2 + as.vector(XR2 %*% real2$dev[r, ]) else xb2
    mu1_r <- pmin(exp(eta1), 1e15); mu2_r <- pmin(exp(eta2), 1e15)
    c1_r <- c_val(mu1_r, m1); c2_r <- c_val(mu2_r, m2)
    b <- lambda_bounds_vec(c1_r, c2_r)
    list(mu1 = mu1_r, mu2 = mu2_r, c1 = c1_r, c2 = c2_r,
         lamLo_r = b[1], lamHi_r = b[2])
  }
  pass1 <- if (!is.null(cl)) parallel::parLapply(cl, seq_len(R), pass1_fun)
           else lapply(seq_len(R), pass1_fun)

  lamLo <- max(vapply(pass1, `[[`, numeric(1), "lamLo_r"))
  lamHi <- min(vapply(pass1, `[[`, numeric(1), "lamHi_r"))
  if (!(lamLo < lamHi && is.finite(lamLo) && is.finite(lamHi))) {
    val <- -1e50; attr(val, "gradient") <- rep(0, length(par)); return(val)
  }
  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  dlam_dz <- (lamHi - lamLo) * (1 - 2*eps) * sig * (1 - sig)

  # ---- Pass 2: LL matrix (n x R) ----
  pass2_fun <- function(r) {
    mu1_r <- pass1[[r]]$mu1; mu2_r <- pass1[[r]]$mu2
    c1_r  <- pass1[[r]]$c1;  c2_r  <- pass1[[r]]$c2
    logNB1 <- nb_logpmf_y_mu_r(y1, mu1_r, r1)
    logNB2 <- nb_logpmf_y_mu_r(y2, mu2_r, r2)
    dep <- 1 + lam * (exp(-y1) - c1_r) * (exp(-y2) - c2_r)
    dep <- pmax(dep, 1e-300)
    logNB1 + logNB2 + log(dep)
  }
  cols <- if (!is.null(cl)) parallel::parLapply(cl, seq_len(R), pass2_fun)
          else lapply(seq_len(R), pass2_fun)
  LL <- do.call(cbind, cols)

  lse <- row_log_sum_exp(LL); val <- sum(lse - log(R))
  W <- exp(LL - lse)

  g_beta1 <- numeric(k1); g_beta2 <- numeric(k2)
  g_logsd1 <- if (q1>0) numeric(q1) else numeric(0)
  g_logsd2 <- if (q2>0) numeric(q2) else numeric(0)
  g_logm1 <- 0; g_logm2 <- 0; g_z <- 0

  dconst <- d_const()
  r1v <- 1 / m1; r2v <- 1 / m2
  log_m1_v <- log(m1); log_m2_v <- log(m2)
  S1 <- digamma(r1v + y1) - digamma(r1v)
  S2 <- digamma(r2v + y2) - digamma(r2v)

  for (r in 1:R) {
    mu1_r <- pass1[[r]]$mu1; mu2_r <- pass1[[r]]$mu2
    c1_r  <- pass1[[r]]$c1;  c2_r  <- pass1[[r]]$c2
    w_ir  <- W[, r]

    k1v <- exp(-y1) - c1_r; k2v <- exp(-y2) - c2_r
    dep <- 1 + lam * (k1v * k2v); inv_dep <- 1 / pmax(dep, 1e-300)
    pen1 <- lam * k2v * inv_dep; pen2 <- lam * k1v * inv_dep

    w1 <- (y1 - mu1_r) / (1 + m1 * mu1_r)
    w2 <- (y2 - mu2_r) / (1 + m2 * mu2_r)

    # effective design: scale random columns by their per-draw location factor
    Xeff1 <- X1
    if (q1 > 0) for (j in seq_len(q1)) {
      Xeff1[, rand_idx1[j]] <- X1[, rand_idx1[j]] * real1$dloc[r, j]
    }
    Xeff2 <- X2
    if (q2 > 0) for (j in seq_len(q2)) {
      Xeff2[, rand_idx2[j]] <- X2[, rand_idx2[j]] * real2$dloc[r, j]
    }

    dc1_dbetas <- dc_dbeta_mat(mu1_r, m1, c1_r, Xeff1)
    dc2_dbetas <- dc_dbeta_mat(mu2_r, m2, c2_r, Xeff2)
    score_b1 <- sweep(Xeff1, 1, w1, `*`) - sweep(dc1_dbetas, 1, pen1, `*`)
    score_b2 <- sweep(Xeff2, 1, w2, `*`) - sweep(dc2_dbetas, 1, pen2, `*`)
    g_beta1 <- g_beta1 + colSums(sweep(score_b1, 1, w_ir, `*`))
    g_beta2 <- g_beta2 + colSums(sweep(score_b2, 1, w_ir, `*`))

    dc1_dm1 <- dct_dm(mu1_r, m1, c1_r); dc2_dm2 <- dct_dm(mu2_r, m2, c2_r)
    term_m1 <- r1v^2 * log_m1_v + r1v^2 * (log(mu1_r + r1v) - 1) +
      r1v^2 * (y1 + r1v)/(mu1_r + r1v) - r1v^2 * S1 - (lam * k2v * inv_dep) * dc1_dm1
    term_m2 <- r2v^2 * log_m2_v + r2v^2 * (log(mu2_r + r2v) - 1) +
      r2v^2 * (y2 + r2v)/(mu2_r + r2v) - r2v^2 * S2 - (lam * k1v * inv_dep) * dc2_dm2
    g_logm1 <- g_logm1 + sum(w_ir * (m1 * term_m1))
    g_logm2 <- g_logm2 + sum(w_ir * (m2 * term_m2))

    g_z <- g_z + sum(w_ir * ((k1v * k2v) * inv_dep * dlam_dz))

    if (q1 > 0) {
      M1 <- sweep(XR1, 2, real1$dscale[r, ], `*`)   # dη/d log_scale
      part_nb <- sweep(M1, 1, w1, `*`)
      row_factor1 <- -(dconst * c1_r * mu1_r) / (1 + dconst * m1 * mu1_r)
      part_c  <- sweep(M1, 1, row_factor1, `*`)
      score_logsd1 <- part_nb - sweep(part_c, 1, pen1, `*`)
      g_logsd1 <- g_logsd1 + colSums(sweep(score_logsd1, 1, w_ir, `*`))
    }
    if (q2 > 0) {
      M2 <- sweep(XR2, 2, real2$dscale[r, ], `*`)
      part_nb2 <- sweep(M2, 1, w2, `*`)
      row_factor2 <- -(dconst * c2_r * mu2_r) / (1 + dconst * m2 * mu2_r)
      part_c2  <- sweep(M2, 1, row_factor2, `*`)
      score_logsd2 <- part_nb2 - sweep(part_c2, 1, pen2, `*`)
      g_logsd2 <- g_logsd2 + colSums(sweep(score_logsd2, 1, w_ir, `*`))
    }
  }

  grad <- c(g_beta1, g_beta2, g_logsd1, g_logsd2, g_logm1, g_logm2, g_z)
  attr(val, "gradient") <- grad
  val
})
```

Then rewrite `bnbr_rp_ll_fixed_bounds` to take the dist/sign args and use `rand_realize` for `μ`:

```r
bnbr_rp_ll_fixed_bounds <- function(par, y1, y2, X1, X2, XR1, XR2,
                                    rand_idx1, rand_idx2, Z1, Z2,
                                    lamLo, lamHi,
                                    dist1, dist2, sign1, sign2, cl = NULL) {
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  R  <- if (q1 + q2 > 0) nrow(Z1) else 1L

  i1 <- 1:k1; i2 <- (k1+1):(k1+k2)
  beta1 <- par[i1]; beta2 <- par[i2]
  lg1   <- if (q1>0) (k1+k2+1):(k1+k2+q1) else integer(0)
  lg2   <- if (q2>0) (k1+k2+q1+1):(k1+k2+q1+q2) else integer(0)
  log_sd1 <- if (q1>0) par[lg1] else numeric(0)
  log_sd2 <- if (q2>0) par[lg2] else numeric(0)
  idx_end <- k1+k2+q1+q2
  log_m1 <- par[idx_end+1]; log_m2 <- par[idx_end+2]; zlam <- par[idx_end+3]

  m1 <- exp(log_m1); m2 <- exp(log_m2); r1 <- 1/m1; r2 <- 1/m2
  sd1 <- if (q1>0) exp(log_sd1) else numeric(0)
  sd2 <- if (q2>0) exp(log_sd2) else numeric(0)

  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)

  xb1 <- as.vector(X1 %*% beta1); xb2 <- as.vector(X2 %*% beta2)
  dev1 <- if (q1>0) rand_realize(Z1, dist1, sign1, beta1[rand_idx1], sd1)$dev
          else matrix(0, R, 0)
  dev2 <- if (q2>0) rand_realize(Z2, dist2, sign2, beta2[rand_idx2], sd2)$dev
          else matrix(0, R, 0)

  pass_fun <- function(r) {
    eta1 <- if (q1>0) xb1 + as.vector(XR1 %*% dev1[r, ]) else xb1
    eta2 <- if (q2>0) xb2 + as.vector(XR2 %*% dev2[r, ]) else xb2
    mu1_r <- pmin(exp(eta1), 1e15); mu2_r <- pmin(exp(eta2), 1e15)
    c1_r  <- c_val(mu1_r, m1); c2_r <- c_val(mu2_r, m2)
    logNB1 <- nb_logpmf_y_mu_r(y1, mu1_r, r1)
    logNB2 <- nb_logpmf_y_mu_r(y2, mu2_r, r2)
    dep <- 1 + lam * (exp(-y1) - c1_r) * (exp(-y2) - c2_r)
    dep <- pmax(dep, 1e-300)
    logNB1 + logNB2 + log(dep)
  }

  cols <- if (!is.null(cl)) parallel::parLapply(cl, seq_len(R), pass_fun)
          else lapply(seq_len(R), pass_fun)
  LL   <- do.call(cbind, cols)
  m <- apply(LL, 1, max)
  sum(m + log(rowSums(exp(LL - m))) - log(R))
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-rpbnb-likelihood-dist.R')"`
Expected: PASS (gradient checks within `1e-4` for all five cases).

- [ ] **Step 5: Commit**

```bash
git add R/rpbnb_likelihood.R tests/testthat/test-rpbnb-likelihood-dist.R
git commit -m "feat: distribution-aware RP-BNB likelihood and analytic gradient

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire distributions through `fit_rpbnb`

**Files:**
- Modify: `R/fit_rpbnb.R`
- Test: `tests/testthat/test-fit-rpbnb-dist.R`

**Interfaces:**
- Consumes: `parse_rand_spec`, `rand_dist_registry`, `rand_realize` (Task 1); `halton_uniform` (Task 2); the updated `bnbr_rp_ll_and_grad` / `bnbr_rp_ll_fixed_bounds` signatures (Task 4).
- Produces: `fit_rpbnb()` accepting the named-list `random_1`/`random_2` spec; `par_names` scale labels reflect each coefficient's distribution; backward compat — a character vector reproduces today's coefficients.

- [ ] **Step 1: Write the failing test**

Create `tests/testthat/test-fit-rpbnb-dist.R`:

```r
test_that("character-vector random spec still fits and labels with log_sd", {
  sim <- simulate_rpbnb(n = 400,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 80, seed = 5,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true(any(grepl("^log_sd1:x1", names(coef(fit)))))
  expect_equal(unname(coef(fit)["b1:x1"]), 0.4, tolerance = 0.25)
})

test_that("uniform distribution fits and labels the scale as log_w", {
  sim <- simulate_rpbnb(n = 500,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1),
    random_1 = list(x1 = list(dist = "uniform", scale = 0.6)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 2)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ 1, data = sim$data,
                   random_1 = list(x1 = "uniform"),
                   draws = 100, seed = 7,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true(any(grepl("^log_w1:x1", names(coef(fit)))))
  expect_true(fit$convergence$converged)
})

test_that("lognormal fit recovers a negative-signed coefficient location", {
  sim <- simulate_rpbnb(n = 600,
    beta1 = c("(Intercept)" = 0.2, x1 = -0.2),
    beta2 = c("(Intercept)" = 0.1),
    random_1 = list(x1 = list(dist = "lognormal", scale = 0.3, sign = -1)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 3)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ 1, data = sim$data,
                   random_1 = list(x1 = list(dist = "lognormal", sign = -1)),
                   draws = 120, seed = 11,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true(any(grepl("^log_s1:x1", names(coef(fit)))))
  expect_true(fit$convergence$converged)
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-fit-rpbnb-dist.R')"`
Expected: FAIL — uniform/lognormal list specs error in `idx_from_names` (expects a character vector) and the likelihood calls lack dist/sign args.

- [ ] **Step 3: Write the implementation**

In `R/fit_rpbnb.R`, make the following edits.

(a) Replace the `rand_names1 <- random_1` / `rand_names2 <- random_2` lines and the name-to-index block (currently around lines 65–92) with spec parsing that yields aligned dist/sign vectors:

```r
  spec1 <- parse_rand_spec(random_1)
  spec2 <- parse_rand_spec(random_2)
  n_draws      <- draws
  n_draws_hess <- control$draws_hessian
  halton_burn  <- control$halton_burn
  n_cores      <- control$n_cores
  compute_se   <- control$compute_se
  method       <- "BFGS"
  ml_control   <- list(iterlim = control$iterlim,
                       reltol = control$reltol,
                       printLevel = control$print_level)

  prep <- .prepare_bnb_data(formula_1, formula_2, data)
  Y1 <- prep$Y1; Y2 <- prep$Y2
  X1 <- prep$X1; X2 <- prep$X2

  # random names -> indices; unknown names are an error (not silently dropped)
  idx_from_names <- function(who, X) {
    if (!length(who)) return(integer(0))
    miss <- who[!who %in% colnames(X)]
    if (length(miss)) {
      stop("random name(s) not found: ", paste(miss, collapse = ", "),
           call. = FALSE)
    }
    as.integer(match(who, colnames(X)))
  }
  rand_idx1 <- idx_from_names(spec1$names, X1)
  rand_idx2 <- idx_from_names(spec2$names, X2)
  dist1 <- spec1$dist; sign1 <- spec1$sign
  dist2 <- spec2$dist; sign2 <- spec2$sign
```

(b) Replace `halton_normal` draws with `halton_uniform` (the likelihood now applies each column's `u_to_base`). Around the opt-phase draw block:

```r
  set.seed(seed)
  if ((q1 + q2) > 0) {
    Z_opt  <- halton_uniform(n_draws, q1 + q2, burn = halton_burn)
    Z1_opt <- if (q1 > 0) Z_opt[, 1:q1, drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
    Z2_opt <- if (q2 > 0) Z_opt[, (q1+1):(q1+q2), drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
  } else {
    Z1_opt <- matrix(0, nrow = n_draws, ncol = 0)
    Z2_opt <- matrix(0, nrow = n_draws, ncol = 0)
  }
```

(c) Replace the scale-parameter labels in `par_names` so each random coefficient's label reflects its distribution:

```r
  scale_lab <- function(dist, cols) {
    vapply(seq_along(dist),
           function(j) paste0(rand_dist_registry[[dist[j]]]$scale_label, cols[j]),
           character(1))
  }
  par_names <- c(paste0("b1:", colnames(X1)),
                 paste0("b2:", colnames(X2)),
                 if (q1 > 0) paste0(scale_lab(dist1, paste0("1:", colnames(X1)[rand_idx1]))) else NULL,
                 if (q2 > 0) paste0(scale_lab(dist2, paste0("2:", colnames(X2)[rand_idx2]))) else NULL,
                 "log_m1", "log_m2", "z_lambda")
```

(d) Pass dist/sign into the optimization objective:

```r
  ll_fun <- function(p) {
    v <- bnbr_rp_ll_and_grad(p, Y1, Y2, X1, X2, XR1, XR2,
                             rand_idx1, rand_idx2, Z1_opt, Z2_opt,
                             dist1, dist2, sign1, sign2, cl = cl)
    ll_trace <<- c(ll_trace, as.numeric(v))
    v
  }
```

(e) In `rebuild_bounds()`, replace the inline `Z1sd`/`Z2sd` construction and per-draw `mu` with realized deviations:

```r
  rebuild_bounds <- function(p) {
    beta1 <- p[1:k1]; beta2 <- p[(k1+1):(k1+k2)]
    idx_end <- k1 + k2 + q1 + q2
    m1 <- exp(p[idx_end+1]); m2 <- exp(p[idx_end+2])
    sd1 <- if (q1 > 0) exp(p[(k1+k2+1):(k1+k2+q1)]) else numeric(0)
    sd2 <- if (q2 > 0) exp(p[(k1+k2+q1+1):(k1+k2+q1+q2)]) else numeric(0)
    xb1 <- as.vector(X1 %*% beta1); xb2 <- as.vector(X2 %*% beta2)
    dev1 <- if (q1>0) rand_realize(Z1_opt, dist1, sign1, beta1[rand_idx1], sd1)$dev
            else matrix(0, n_draws, 0)
    dev2 <- if (q2>0) rand_realize(Z2_opt, dist2, sign2, beta2[rand_idx2], sd2)$dev
            else matrix(0, n_draws, 0)
    lamLo <- -Inf; lamHi <- Inf
    Rloc <- if (q1 + q2 > 0) n_draws else 1L
    for (r in 1:Rloc) {
      mu1_r <- if (q1 > 0) pmin(exp(xb1 + as.vector(XR1 %*% dev1[r, ])), 1e15) else exp(xb1)
      mu2_r <- if (q2 > 0) pmin(exp(xb2 + as.vector(XR2 %*% dev2[r, ])), 1e15) else exp(xb2)
      b <- lambda_bounds_vec(c_val(mu1_r, m1), c_val(mu2_r, m2))
      lamLo <- max(lamLo, b[1]); lamHi <- min(lamHi, b[2])
    }
    c(lamLo, lamHi)
  }
```

(f) Generate the Hessian draws with `halton_uniform` and pass dist/sign to `ll_fb`:

```r
      Z_h  <- halton_uniform(n_draws_hess, q1 + q2,
                             burn = max(50, floor(halton_burn / 3)))
```

```r
    ll_fb <- function(p) bnbr_rp_ll_fixed_bounds(p, Y1, Y2, X1, X2, XR1, XR2,
                                                 rand_idx1, rand_idx2, Z1_h, Z2_h,
                                                 lamLo_h, lamHi_h,
                                                 dist1, dist2, sign1, sign2,
                                                 cl = cl_h)
```

(g) In the fitted draw-averaged means block, replace the inline normal transform with `rand_realize` deviations:

```r
  beta1_hat <- par_hat[1:k1]; beta2_hat <- par_hat[(k1+1):(k1+k2)]
  xb1 <- as.vector(X1 %*% beta1_hat); xb2 <- as.vector(X2 %*% beta2_hat)
  if (q1 > 0) {
    sd1  <- exp(par_hat[(k1+k2+1):(k1+k2+q1)])
    dev1 <- rand_realize(Z1_opt, dist1, sign1, beta1_hat[rand_idx1], sd1)$dev
    mu1_mat <- vapply(seq_len(n_draws),
                      function(r) pmin(exp(xb1 + as.vector(XR1 %*% dev1[r, ])), 1e15),
                      numeric(length(Y1)))
    mu1_hat <- rowMeans(mu1_mat)
  } else {
    mu1_hat <- exp(xb1)
  }
  if (q2 > 0) {
    sd2  <- exp(par_hat[(k1+k2+q1+1):(k1+k2+q1+q2)])
    dev2 <- rand_realize(Z2_opt, dist2, sign2, beta2_hat[rand_idx2], sd2)$dev
    mu2_mat <- vapply(seq_len(n_draws),
                      function(r) pmin(exp(xb2 + as.vector(XR2 %*% dev2[r, ])), 1e15),
                      numeric(length(Y2)))
    mu2_hat <- rowMeans(mu2_mat)
  } else {
    mu2_hat <- exp(xb2)
  }
```

(h) The `clusterExport` lists reference `Z1_opt`/`Z2_opt`/`Z1_h`/`Z2_h` which still exist; no name changes needed there. Because `bnbr_rp_ll_and_grad` precomputes realizations inside its own body, workers capture them via the serialized closure — no new exports required.

- [ ] **Step 4: Run the test to verify it passes**

Run: `Rscript -e "pkgload::load_all('.'); testthat::test_file('tests/testthat/test-fit-rpbnb-dist.R')"`
Expected: PASS.

- [ ] **Step 5: Run the full suite + roxygen + check; commit**

```bash
Rscript -e "devtools::document()"
Rscript -e "devtools::test()"
git add R/fit_rpbnb.R R/simulate_rpbnb.R man tests/testthat/test-fit-rpbnb-dist.R NAMESPACE
git commit -m "feat: choose random-coefficient distribution per coefficient in fit_rpbnb

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Documentation and final verification

**Files:**
- Modify: `R/fit_rpbnb.R` (roxygen `@param random_1,random_2`)
- Modify: `NEWS.md` (if present; create if the package keeps one)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the `fit_rpbnb` roxygen for the new spec**

Edit the `@param random_1,random_2` block to:

```r
#' @param random_1,random_2 Random coefficients per equation. Either a character
#'   vector of `model.matrix` column names (all Normal), or a named list whose
#'   values are a distribution name (`"normal"`, `"lognormal"`, `"uniform"`,
#'   `"triangular"`) or a list `list(dist = ..., sign = ...)` (`sign` is -1/1 and
#'   lognormal-only). NULL means all-fixed for that equation.
```

- [ ] **Step 2: Regenerate docs**

Run: `Rscript -e "devtools::document()"`
Expected: man pages updated, no roxygen warnings.

- [ ] **Step 3: Full check**

Run: `Rscript -e "devtools::test()"`
Expected: all tests PASS.

Run: `Rscript -e "devtools::check(args = '--no-manual', error_on = 'warning')"`
Expected: 0 errors, 0 warnings (NOTES from pre-existing conditions are acceptable).

- [ ] **Step 4: Commit**

```bash
git add R/fit_rpbnb.R man NEWS.md
git commit -m "docs: document per-coefficient random distributions in fit_rpbnb

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Why the β-score uses `Xeff`:** for a random column *j*, `∂η/∂location_j = X_j · dloc_factor`. For Normal/Uniform/Triangular `dloc_factor = 1` (so `Xeff == X`, exactly today's behavior). For Lognormal `dloc_factor = β` (the realized coefficient), because `β = sign·exp(b + s·z)` is nonlinear in the location `b`. Both the NB score term and the dependence (`dc/dβ`) term share the same `∂η/∂β`, so both use `Xeff`.
- **Why `Z1/Z2` now carry uniforms:** mixing normal-based (normal, lognormal) and uniform-based (uniform, triangular) distributions in one draw matrix is cleanest when the shared draw is uniform; each column's `u_to_base` maps it to the right base variate. `qnorm(halton_uniform(...)) == halton_normal(...)`, so the all-Normal path is unchanged bit-for-bit.
- **Backward compatibility:** verified by Task 5's character-vector test (label `log_sd1:` and coefficient recovery) and Task 3's normal-draw moment test (legacy `beta + sd*rnorm` stream preserved).
