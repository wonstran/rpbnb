# Simulate RPBNB with Dependence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the `lambda = 0` restriction on `simulate_rpbnb()`, adding Famoye/Sarmanov and copula-based dependence (Gaussian, Frank, Clayton).

**Architecture:** Extract the Famoye conditional sampler from `simulate_bnb` into a shared internal `.sim_famoye_conditional()`. Add three internal copula simulators to `copula_core.R`. Update `simulate_rpbnb` to dispatch on `lambda`/`dependence`, wiring both dependence paths into the existing random-coefficient realization pipeline.

**Tech Stack:** R ≥ 4.1, base R, stats, randtoolbox (unchanged dependencies)

## Global Constraints

- R ≥ 4.1
- No new package dependencies
- All 17 existing test files must continue to pass
- Default `simulate_rpbnb(n, beta1, beta2)` with no `lambda` or `dependence` → identical behavior and output structure
- `lambda = 0` with no `dependence` → identical to current behavior
- Follow existing code style: 2-space indent, roxygen2 `@keywords internal @noRd` for internals, S3 conventions as applicable

---

## File Map

| File | Role | Task |
|------|------|------|
| `R/copula.R` | Update `copula()` to accept optional `par` | Task 1 |
| `R/simulate_bnb.R` | Extract `.sim_famoye_conditional()` shared sampler | Task 2 |
| `R/copula_core.R` | Add 3 internal copula simulators | Task 3 |
| `R/simulate_rpbnb.R` | Remove λ=0 guard, add dispatch, wire paths, update returns | Task 4 |
| `tests/testthat/test-copula-sim.R` | New: copula sampler unit tests | Task 3 |
| `tests/testthat/test-simulate-rpbnb.R` | Extend: λ>0, copula, backward compat tests | Task 5 |
| `tests/testthat/test-fit-rpbnb.R` | Extend: roundtrip simulate-then-fit | Task 6 |

---

### Task 1: Update `copula()` to accept optional `par`

**Files:**
- Modify: `R/copula.R`

**Interfaces:**
- Produces: `copula(family, par = NULL)` → `rpbnb_copula` list with `family` and `par` fields

The `copula()` factory currently only stores `family`. Simulation needs the dependence parameter value. Add an optional `par` argument that is stored but not used by `fit_bnb`.

- [ ] **Step 1: Update the copula function**

Open `R/copula.R`. Change the function signature and body:

```r
#' Specify a copula dependence structure for fit_bnb()
#'
#' @param family One of `"frank"`, `"normal"`, or `"kimeldorf"` (Clayton).
#' @param par Optional dependence parameter value for simulation (not used
#'   by [fit_bnb()], which estimates it).
#' @return An object of class `rpbnb_copula`.
#' @export
#' @examples
#' copula("frank")
#' copula("normal", par = 0.3)
#' copula("kimeldorf")
copula <- function(family = c("frank", "normal", "kimeldorf"), par = NULL) {
  family <- match.arg(family)
  structure(list(family = family, par = par), class = "rpbnb_copula")
}
```

- [ ] **Step 2: Verify existing code still works**

Run: `devtools::load_all()` then verify `copula("frank")$family == "frank"` and `copula("normal", par = 0.3)$par == 0.3`.

- [ ] **Step 3: Run existing tests to confirm no regressions**

Run: `devtools::test()`

- [ ] **Step 4: Commit**

```bash
git add R/copula.R
git commit -m "feat: add optional par argument to copula() for simulation"
```

---

### Task 2: Extract shared `.sim_famoye_conditional` from `simulate_bnb`

**Files:**
- Modify: `R/simulate_bnb.R`

**Interfaces:**
- Produces: `.sim_famoye_conditional(y1, mu2, c1, c2, m2, lambda)` → `y2` (integer vector)
- Consumed by: `simulate_bnb` (this task), `simulate_rpbnb` (Task 4)

Extract the vectorized grid-based conditional sampler (currently lines 83–116 in `simulate_bnb.R`) into a standalone internal function. `simulate_bnb` then calls it.

- [ ] **Step 1: Add the extracted function to simulate_bnb.R**

Insert the following function BEFORE the `simulate_bnb` function definition (after the roxygen block, before line 38):

```r
#' Famoye/Sarmanov conditional sampler via vectorized grid
#'
#' Given y1 already drawn from its marginal NB2(mu1, m1), draws y2 from the
#' Sarmanov-weighted conditional P(Y2=y2|Y1=y1) = NB2(y2|mu2) * W(y1, y2).
#' Uses a truncated grid 0:ymax where ymax is the 0.9999 quantile of the
#' largest mu2; the extreme upper tail is dropped (consistent with the
#' existing simulate_bnb approximate sampler).
#'
#' @param y1 Integer vector, already drawn from NB2(mu1, m1).
#' @param mu2 Numeric vector of NB2 means for equation 2.
#' @param c1,c2 Vectors c_k = E[exp(-Y_k)] under NB2(mu_k, m_k).
#' @param m2 NB2 dispersion for equation 2.
#' @param lambda Famoye dependence parameter.
#' @return Integer vector of y2 draws (same length as y1).
#' @keywords internal
#' @noRd
.sim_famoye_conditional <- function(y1, mu2, c1, c2, m2, lambda) {
  n <- length(y1)
  ymax <- ceiling(max(qnbinom(0.9999, size = 1 / m2, mu = mu2)))
  y2_grid <- 0:ymax

  # P2[i, j] = dnbinom(y2_grid[j], size=1/m2, mu=mu2[i])
  P2 <- outer(mu2, y2_grid, function(mu, y) dnbinom(y, size = 1 / m2, mu = mu))

  # W[i, j] = 1 + lambda * (exp(-y1[i]) - c1[i]) * (exp(-y2_grid[j]) - c2[i])
  row_factor <- lambda * (exp(-y1) - c1)   # length n
  c_per_row  <- row_factor * c2            # per-obs constant subtracted from each row
  W <- 1 + outer(row_factor, exp(-y2_grid)) - c_per_row
  Q <- pmax(P2 * W, 0)
  Q <- Q / rowSums(Q)

  # Vectorized inverse-CDF sampling
  cum <- Q
  for (j in seq_len(ncol(cum))[-1]) cum[, j] <- cum[, j - 1L] + cum[, j]
  u  <- runif(n) * cum[, ncol(cum)]
  y2_grid[max.col(cum >= u, ties.method = "first")]
}
```

- [ ] **Step 2: Refactor `simulate_bnb` to call the extracted function**

Replace lines 78–116 in `simulate_bnb.R` (the `if (lambda == 0) { ... } else { ... }` block) with:

```r
  if (lambda == 0) {
    # The Sarmanov weight W = 1 + lambda*(...) is identically 1 when
    # lambda == 0, so the conditional collapses exactly to the NB2 marginal —
    # draw it directly instead of building the truncated conditional grid.
    y2 <- rnbinom(n, size = 1 / m2, mu = mu2)
  } else {
    # Draw Y2 from conditional P(Y2=y2|Y1=y1) using the Sarmanov-weighted grid.
    y2 <- .sim_famoye_conditional(y1, mu2, c1, c2, m2, lambda)
  }
```

The y1 drawing on line 76 (`y1 <- rnbinom(n, size = 1 / m1, mu = mu1)`) stays unchanged.

- [ ] **Step 3: Verify `simulate_bnb` produces identical output**

Run in R console:

```r
devtools::load_all()
set.seed(42)
a <- simulate_bnb(n = 500, beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
                  beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
                  dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 1)
# Spot check: all y1, y2 are non-negative integers
stopifnot(all(a$data$y1 >= 0), all(a$data$y2 >= 0))
stopifnot(is.integer(a$data$y1) || all(a$data$y1 == floor(a$data$y1)))
cat("OK: y1 mean =", mean(a$data$y1), "y2 mean =", mean(a$data$y2),
    "cor =", cor(a$data$y1, a$data$y2), "\n")
```

- [ ] **Step 4: Run all tests**

Run: `devtools::test()`

Expected: all existing tests pass (including test-simulate-bnb.R).

- [ ] **Step 5: Commit**

```bash
git add R/simulate_bnb.R
git commit -m "refactor: extract .sim_famoye_conditional from simulate_bnb"
```

---

### Task 3: Add copula simulators to `copula_core.R` + unit tests

**Files:**
- Modify: `R/copula_core.R`
- Create: `tests/testthat/test-copula-sim.R`

**Interfaces:**
- Produces:
  - `.sim_copula_normal(n, rho)` → `list(u1, u2)` of uniforms in (0, 1)
  - `.sim_copula_frank(n, theta)` → `list(u1, u2)` of uniforms in (0, 1)
  - `.sim_copula_clayton(n, theta)` → `list(u1, u2)` of uniforms in (0, 1)

- [ ] **Step 1: Write the test file**

Create `tests/testthat/test-copula-sim.R`:

```r
test_that(".sim_copula_normal recovers rho", {
  set.seed(42)
  n <- 5000; rho <- 0.5
  u <- rpbnb:::.sim_copula_normal(n, rho)
  z1 <- qnorm(u$u1); z2 <- qnorm(u$u2)
  expect_equal(cor(z1, z2), rho, tolerance = 0.04)
  expect_true(all(u$u1 > 0 & u$u1 < 1))
  expect_true(all(u$u2 > 0 & u$u2 < 1))
})

test_that(".sim_copula_frank Kendall's tau matches theoretical value", {
  n <- 5000; theta <- 2.0
  u <- rpbnb:::.sim_copula_frank(n, theta)
  tau_emp <- cor(u$u1, u$u2, method = "kendall")
  tau_theo <- rpbnb:::frank_tau(theta)
  expect_equal(tau_emp, tau_theo, tolerance = 0.04)
})

test_that(".sim_copula_frank theta=0 produces independence", {
  n <- 3000
  u <- rpbnb:::.sim_copula_frank(n, 0)
  tau_emp <- cor(u$u1, u$u2, method = "kendall")
  expect_equal(tau_emp, 0, tolerance = 0.05)
})

test_that(".sim_copula_clayton Kendall's tau matches theoretical value", {
  n <- 5000; theta <- 1.5
  u <- rpbnb:::.sim_copula_clayton(n, theta)
  tau_emp <- cor(u$u1, u$u2, method = "kendall")
  tau_theo <- theta / (theta + 2)
  expect_equal(tau_emp, tau_theo, tolerance = 0.04)
})

test_that(".sim_copula_clayton small theta approaches independence", {
  n <- 3000; theta <- 0.01
  u <- rpbnb:::.sim_copula_clayton(n, theta)
  tau_emp <- cor(u$u1, u$u2, method = "kendall")
  expect_equal(tau_emp, 0, tolerance = 0.06)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `testthat::test_file("tests/testthat/test-copula-sim.R")`

Expected: FAIL — `.sim_copula_normal`, `.sim_copula_frank`, `.sim_copula_clayton` not found.

- [ ] **Step 3: Add the three copula simulators to `copula_core.R`**

Append to `R/copula_core.R` (the file currently ends at copula likelihood/gradient code):

```r
# Copula simulators — internal ------------------------------------------------

#' Gaussian (Normal) copula simulator via Cholesky decomposition
#'
#' @param n Number of observations.
#' @param rho Correlation parameter in (-1, 1).
#' @return A list with `u1` and `u2`, uniform margins in (0, 1).
#' @keywords internal
#' @noRd
.sim_copula_normal <- function(n, rho) {
  z1 <- stats::rnorm(n)
  z2 <- rho * z1 + sqrt(1 - rho^2) * stats::rnorm(n)
  list(u1 = stats::pnorm(z1), u2 = stats::pnorm(z2))
}

#' Frank copula simulator via conditional inversion
#'
#' @param n Number of observations.
#' @param theta Frank dependence parameter.
#' @return A list with `u1` and `u2`, uniform margins in (0, 1).
#' @keywords internal
#' @noRd
.sim_copula_frank <- function(n, theta) {
  u1 <- stats::runif(n)
  t  <- stats::runif(n)
  A  <- exp(-theta * u1)
  D  <- exp(-theta) - 1
  u2 <- -log1p(t * D / pmax(A * (1 - t) + t, 1e-300)) / theta
  u2 <- pmin(pmax(u2, 1e-10), 1 - 1e-10)
  list(u1 = u1, u2 = u2)
}

#' Clayton (Kimeldorf-Sampson) copula simulator via conditional inversion
#'
#' @param n Number of observations.
#' @param theta Clayton dependence parameter (> 0).
#' @return A list with `u1` and `u2`, uniform margins in (0, 1).
#' @keywords internal
#' @noRd
.sim_copula_clayton <- function(n, theta) {
  u1   <- stats::runif(n)
  t    <- stats::runif(n)
  exp1 <- -theta / (theta + 1)
  inner <- u1^(-theta) * (t^exp1 - 1) + 1
  u2 <- pmax(inner, 1e-300)^(-1 / theta)
  u2 <- pmin(pmax(u2, 1e-10), 1 - 1e-10)
  list(u1 = u1, u2 = u2)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `testthat::test_file("tests/testthat/test-copula-sim.R")`

Expected: PASS (all 5 tests).

- [ ] **Step 5: Run full test suite**

Run: `devtools::test()`

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add R/copula_core.R tests/testthat/test-copula-sim.R
git commit -m "feat: add internal copula simulators (normal, frank, clayton)"
```

---

### Task 4: Update `simulate_rpbnb` core — remove guard, add dispatch, wire paths

**Files:**
- Modify: `R/simulate_rpbnb.R`

**Interfaces:**
- Modifies: `simulate_rpbnb(n, beta1, beta2, random_1, random_2, dispersion, lambda, dependence, covariates, seed)`
- Consumes: `.sim_famoye_conditional` (Task 2), `.sim_copula_*` (Task 3), `copula()` (Task 1)

- [ ] **Step 1: Update the function signature and roxygen**

Replace lines 1–29 (the roxygen block and function signature up to the opening brace) with:

```r
#' Simulate data from a random-parameter bivariate NB process
#'
#' @param n Number of observations.
#' @param beta1,beta2 Named numeric vectors of fixed coefficient means per
#'   equation; must include "(Intercept)".
#' @param random_1,random_2 Named lists giving random coefficients. Each value
#'   is a list with `dist` (one of "normal", "lognormal", "uniform",
#'   "triangular"; default "normal"), `scale` (or `sd`) for the dispersion, and
#'   `sign` (-1/1, lognormal only). Means come from `beta1`/`beta2`; for a
#'   lognormal coefficient the `beta` entry is the log-location and the realized
#'   coefficient is `sign * exp(log_location + scale * z)`.
#' @param dispersion Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersions.
#' @param lambda Famoye/Sarmanov dependence parameter (0 = independent margins).
#'   Ignored when `dependence` is set.
#' @param dependence Optional [copula()] object specifying a copula dependence
#'   structure. When provided, `lambda` is ignored. Supported families:
#'   `"normal"`, `"frank"`, `"kimeldorf"` (Clayton).
#' @param covariates Optional data frame of covariates; if NULL, standard-normal
#'   columns are generated for every non-intercept name.
#' @param seed Optional random seed. If `NULL` (default) the RNG is left
#'   untouched and draws continue from the caller's current stream, so repeated
#'   calls yield distinct datasets; supply an integer for reproducible output.
#' @return A list with `data` (y1, y2, covariates), `coef_realized`
#'   (per-obs coefficients per equation), `mu` (conditional means), `true`
#'   (parameters, including `dependence` and `dependence_par`), `settings`
#'   (`dependence_type`, `dependence_par` added), and `meta`.
#' @export
#' @examples
#' # Famoye dependence
#' sim <- simulate_rpbnb(n = 500,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.15, seed = 1)
#' head(sim$data)
#'
#' # Copula dependence
#' sim2 <- simulate_rpbnb(n = 500,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5),
#'   dependence = copula("normal", par = 0.3), seed = 1)
#' head(sim2$data)
simulate_rpbnb <- function(n, beta1, beta2,
                           random_1 = NULL, random_2 = NULL,
                           dispersion = c(m1 = 0.5, m2 = 0.5),
                           lambda = 0, dependence = NULL,
                           covariates = NULL, seed = NULL) {
```

- [ ] **Step 2: Remove the lambda=0 guard**

Delete lines 52–55 (the `if (lambda != 0) { stop("Phase 1 ...") }` block). In the current file these are the 4 lines after `chk_rand(spec1, beta1, "beta1"); chk_rand(spec2, beta2, "beta2")`.

- [ ] **Step 3: Replace the independent-draw section with the dependence dispatch**

Replace lines 86–87 (the two `rnbinom` calls and the line before them):

```r
  m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]
  y1 <- rnbinom(n, size = 1 / m1, mu = mu1)
  y2 <- rnbinom(n, size = 1 / m2, mu = mu2)
```

Replace with:

```r
  m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]

  # Dependence dispatch
  if (!is.null(dependence)) {
    # ── Copula path ───────────────────────────────────────────────────
    if (!inherits(dependence, "rpbnb_copula")) {
      stop("`dependence` must be a copula() object.", call. = FALSE)
    }
    family <- dependence$family
    dep_par <- dependence$par
    if (is.null(dep_par)) {
      stop("copula `par` must be provided for simulation.", call. = FALSE)
    }
    u <- switch(family,
      normal    = .sim_copula_normal(n, dep_par),
      frank     = .sim_copula_frank(n, dep_par),
      kimeldorf = .sim_copula_clayton(n, dep_par),
      stop("Unknown copula family: ", family, call. = FALSE))
    y1 <- qnbinom(u$u1, size = 1 / m1, mu = mu1)
    y2 <- qnbinom(u$u2, size = 1 / m2, mu = mu2)
    dep_type <- family
  } else if (lambda != 0) {
    # ── Famoye/Sarmanov path ──────────────────────────────────────────
    c1 <- c_val(mu1, m1)
    c2 <- c_val(mu2, m2)
    bnds <- lambda_bounds_vec(c1, c2)
    if (lambda < bnds[1] || lambda > bnds[2]) {
      stop("lambda (", lambda, ") is outside the valid bounds [",
           round(bnds[1], 4), ", ", round(bnds[2], 4), "].", call. = FALSE)
    }
    y1 <- rnbinom(n, size = 1 / m1, mu = mu1)
    y2 <- .sim_famoye_conditional(y1, mu2, c1, c2, m2, lambda)
    dep_type <- "famoye"
    dep_par  <- lambda
  } else {
    # ── Independence path ─────────────────────────────────────────────
    y1 <- rnbinom(n, size = 1 / m1, mu = mu1)
    y2 <- rnbinom(n, size = 1 / m2, mu = mu2)
    dep_type <- "independence"
    dep_par <- 0
  }
```

- [ ] **Step 4: Update the return value**

Replace the `$true` and `$settings` list construction (lines 94–97):

```r
    true = list(beta1 = beta1, beta2 = beta2, random_1 = spec1,
                random_2 = spec2, dispersion = dispersion, lambda = lambda),
    settings = list(n = n, seed = seed),
```

Replace with:

```r
    true = list(beta1 = beta1, beta2 = beta2, random_1 = spec1,
                random_2 = spec2, dispersion = dispersion, lambda = lambda,
                dependence = dep_type, dependence_par = dep_par),
    settings = list(n = n, seed = seed,
                    dependence_type = dep_type, dependence_par = dep_par),
```

- [ ] **Step 5: Verify basic functionality**

Run in R console:

```r
devtools::load_all()

# Independence (backward compat)
s0 <- simulate_rpbnb(n = 200, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                     beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                     dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
stopifnot("data" %in% names(s0), "coef_realized" %in% names(s0))
stopifnot(s0$true$dependence == "independence")

# Famoye
s1 <- simulate_rpbnb(n = 200, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                     beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                     random_1 = list(x1 = list(sd = 0.3)),
                     dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 2)
stopifnot(s1$true$dependence == "famoye", s1$true$dependence_par == 0.1)
cat("Indep cor:", cor(s0$data$y1, s0$data$y2),
    "  Famoye cor:", cor(s1$data$y1, s1$data$y2), "\n")

# Copula
s2 <- simulate_rpbnb(n = 200, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                     beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                     dependence = copula("normal", par = 0.3),
                     dispersion = c(m1 = 0.4, m2 = 0.5), seed = 3)
stopifnot(s2$true$dependence == "normal", s2$true$dependence_par == 0.3)
cat("Copula cor:", cor(s2$data$y1, s2$data$y2), "\n")
```

- [ ] **Step 6: Run full test suite**

Run: `devtools::test()`

Some tests may fail because they inspect `$true$lambda` or `$true` structure. Fix any failures in the next task.

- [ ] **Step 7: Commit**

```bash
git add R/simulate_rpbnb.R
git commit -m "feat: add Famoye and copula dependence to simulate_rpbnb"
```

---

### Task 5: Extend `test-simulate-rpbnb.R` with dependence tests

**Files:**
- Modify: `tests/testthat/test-simulate-rpbnb.R`

**Interfaces:**
- Consumes: updated `simulate_rpbnb` (Task 4)

- [ ] **Step 1: Write new tests**

Append to `tests/testthat/test-simulate-rpbnb.R`:

```r
# ── Famoye dependence ────────────────────────────────────────────────────

test_that("simulate_rpbnb with lambda=0.1 produces positive dependence", {
  s <- simulate_rpbnb(n = 500,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 10)
  expect_gt(cor(s$data$y1, s$data$y2), 0.01)
})

test_that("simulate_rpbnb with lambda is seed-reproducible", {
  a <- simulate_rpbnb(n = 200,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        random_1 = list(x1 = list(sd = 0.5)),
        dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 55)
  b <- simulate_rpbnb(n = 200,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        random_1 = list(x1 = list(sd = 0.5)),
        dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 55)
  expect_identical(a$data, b$data)
})

test_that("simulate_rpbnb errors when lambda is outside valid bounds", {
  expect_error(
    simulate_rpbnb(n = 100,
      beta1 = c("(Intercept)" = 0.0, x1 = 0.0),
      beta2 = c("(Intercept)" = 0.0, x1 = 0.0),
      dispersion = c(m1 = 0.1, m2 = 0.1), lambda = 100, seed = 1),
    "outside the valid bounds"
  )
})

# ── Copula dependence ────────────────────────────────────────────────────

test_that("simulate_rpbnb with copula('normal') produces valid counts", {
  s <- simulate_rpbnb(n = 300,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        dependence = copula("normal", par = 0.3),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 12)
  expect_true(all(s$data$y1 >= 0))
  expect_true(all(s$data$y2 >= 0))
  expect_true(all(s$data$y1 == floor(s$data$y1)))
  expect_true(all(s$data$y2 == floor(s$data$y2)))
  expect_gt(cor(s$data$y1, s$data$y2), 0.02)
})

test_that("simulate_rpbnb with copula('frank') is seed-reproducible", {
  a <- simulate_rpbnb(n = 200,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        dependence = copula("frank", par = 2.0),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 77)
  b <- simulate_rpbnb(n = 200,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        dependence = copula("frank", par = 2.0),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 77)
  expect_identical(a$data, b$data)
})

test_that("simulate_rpbnb with copula('clayton') returns correct structure", {
  s <- simulate_rpbnb(n = 100,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1),
        dependence = copula("kimeldorf", par = 1.0),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 13)
  expect_true(all(c("data", "coef_realized", "mu", "true", "settings", "meta")
                  %in% names(s)))
  expect_equal(s$true$dependence, "kimeldorf")
  expect_equal(s$true$dependence_par, 1.0)
  expect_equal(s$settings$dependence_type, "kimeldorf")
})

test_that("simulate_rpbnb copula errors without par", {
  expect_error(
    simulate_rpbnb(n = 100,
      beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
      beta2 = c("(Intercept)" = 0.1),
      dependence = copula("normal"),
      dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1),
    "par"
  )
})

# ── Backward compatibility ───────────────────────────────────────────────

test_that("simulate_rpbnb with lambda=0 and no dependence matches old output", {
  s <- simulate_rpbnb(n = 200,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 99)
  expect_equal(s$true$dependence, "independence")
  expect_equal(s$settings$dependence_type, "independence")
  expect_true(all(c("data", "coef_realized", "mu", "true", "settings", "meta")
                  %in% names(s)))
})

test_that("simulate_rpbnb lambda=0 with random coefs still has coef_realized", {
  s <- simulate_rpbnb(n = 100,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
        random_1 = list(x1 = list(sd = 0.5)),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 14)
  expect_true("coef_realized" %in% names(s))
  expect_true("eq1" %in% names(s$coef_realized))
  expect_true("x1" %in% colnames(s$coef_realized$eq1))
})
```

- [ ] **Step 2: Run the new tests**

Run: `testthat::test_file("tests/testthat/test-simulate-rpbnb.R")`

Expected: all new tests pass. Fix any issues from old tests that inspect `$true` structure (may need to reference `$true$dependence` instead of depending on exact `$true` shape — but old tests shouldn't break since we only added fields).

- [ ] **Step 3: Run full test suite**

Run: `devtools::test()`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/testthat/test-simulate-rpbnb.R
git commit -m "test: add dependence tests for simulate_rpbnb"
```

---

### Task 6: Integration roundtrip tests + backward compat verification

**Files:**
- Modify: `tests/testthat/test-fit-rpbnb.R`
- Modify: (none — verify `test-fit-rpbnb-dist.R` still passes)

**Interfaces:**
- Consumes: updated `simulate_rpbnb` (Task 4), `fit_rpbnb` (existing), `fit_bnb` (existing)

- [ ] **Step 1: Add roundtrip tests to `test-fit-rpbnb.R`**

Append to `tests/testthat/test-fit-rpbnb.R`:

```r
test_that("simulate-then-fit roundtrip with lambda > 0 recovers parameters", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 600,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 20)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 200, seed = 30,
                   control = rpbnb_control(compute_se = FALSE))
  cf <- coef(fit)
  # Fixed coefficients should be in the ballpark
  expect_equal(unname(cf["b1:x1"]), 0.4, tolerance = 0.3)
  expect_equal(unname(cf["b2:x1"]), -0.3, tolerance = 0.3)
  expect_true(fit$convergence$converged)
})

test_that("roundtrip with copula('normal') via fit_bnb recovers rho", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 400,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    dependence = copula("normal", par = 0.3),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 21)
  fit <- fit_bnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                 dependence = copula("normal"))
  expect_equal(coef(fit)["z_theta"], 0.3, tolerance = 0.3)
  expect_true(fit$convergence$converged)
})
```

- [ ] **Step 2: Run the roundtrip tests**

Run: `testthat::test_file("tests/testthat/test-fit-rpbnb.R")`
(Only the two new tests at the end need to pass; existing tests should still pass.)

- [ ] **Step 3: Verify all 18 test files pass**

Run: `devtools::test()`

Expected: 0 failures, 0 errors, 0 warnings (or only pre-existing warnings).

- [ ] **Step 4: Regenerate documentation**

Run: `devtools::document()`

This regenerates `NAMESPACE` and `man/simulate_rpbnb.Rd` with the updated roxygen.

- [ ] **Step 5: Full R CMD check**

Run: `devtools::check()`

Expected: 0 errors, 0 warnings, 0 notes (or only pre-existing notes).

- [ ] **Step 6: Commit**

```bash
git add tests/testthat/test-fit-rpbnb.R man/ NAMESPACE
git commit -m "test: add simulate-then-fit roundtrip tests for RP-BNB dependence"
```
