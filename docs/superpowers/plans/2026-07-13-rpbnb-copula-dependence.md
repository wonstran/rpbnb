# RP-BNB Copula Dependence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `fit_rpbnb` model the dependence between the two count outcomes with a copula (Frank / Gaussian / Clayton), as an alternative to the Famoye/Sarmanov λ.

**Architecture:** The random-coefficient mixing loop is unchanged; only the per-draw dependence factor becomes the discrete-copula joint pmf (finite-difference of the copula CDF over the two NB CDFs). A new R simulated-likelihood function is optimized with BFGS + numeric gradient; SEs via the numeric Hessian. `fit_rpbnb` gains a `dependence` argument mirroring `fit_bnb`. A companion copula-dependent simulator enables end-to-end recovery tests.

**Tech Stack:** R, `maxLik` (BFGS), `numDeriv` (Hessian), `pbivnorm` (Gaussian copula), the package's existing `copula_core.R` / `copula_likelihood.R` / `rand_realize` / `row_log_sum_exp`.

## Global Constraints

- Parameter vector order for the copula RP model: `beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2), log_m1, log_m2, z_theta`. `z_theta` is last; `theta = z_to_native(family, z_theta)`.
- NB parameterization: dispersion `m_t`, `log_m_t = log(m_t)`, NB `size = r_t = 1/m_t = exp(-log_m_t)`. Match `.bound_mu` capping `pmin(pmax(exp(eta), 1e-300), 1e15)` for the per-draw means.
- Copula families are exactly `"frank"`, `"normal"` (Gaussian), `"kimeldorf"` (Clayton) — the strings used by `copula()` and `copula_core.R`. Do not invent new names.
- Reuse existing internals; do NOT re-implement copula CDFs, `rand_realize`, `row_log_sum_exp`, or the NB-CDF corner logic.
- Coefficient names: `b1:<col>`, `b2:<col>`, `log_sd1:<col>`, `log_sd2:<col>`, `log_m1`, `log_m2`, `z_theta`.
- Internal (non-exported) functions get `#' @keywords internal` + `#' @noRd`. `simulate_rpbnb_copula` is `@export`.
- Run R via `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "..."` on this machine.
- Tests are testthat; run a single file with `testthat::test_file("tests/testthat/<file>.R")` inside `devtools::load_all()`.

---

### Task 1: RP copula simulated log-likelihood

**Files:**
- Create: `R/rpbnb_copula_likelihood.R`
- Test: `tests/testthat/test-rpbnb-copula.R`

**Interfaces:**
- Consumes: `rand_realize` (R/rand_dist.R), `frank_cdf`/`normal_cdf`/`kimeldorf_cdf` and `z_to_native` (R/copula_core.R), `row_log_sum_exp` (R/utilities.R), `copula_loglik_vec` (R/copula_likelihood.R, for the reduction test).
- Produces: `bnbr_rp_copula_ll(par, y1, y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2, Z1, Z2, family, dist1=NULL, dist2=NULL, sign1=NULL, sign2=NULL)` → scalar simulated log-likelihood (numeric length 1).

- [ ] **Step 1: Write the failing reduction + pmf tests**

Create `tests/testthat/test-rpbnb-copula.R`:

```r
# With no random coefficients (q=0), the RP copula simulated log-likelihood must
# reduce EXACTLY to the fixed-model discrete-copula log-likelihood, for every
# family. Layouts coincide when q=0: (beta1, beta2, log_m1, log_m2, z_theta).

make_fixed_case <- function(n = 60, seed = 3) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1); X2 <- cbind(`(Intercept)` = 1, x2 = x2)
  y1 <- rnbinom(n, mu = exp(0.3 + 0.2 * x1), size = 2)
  y2 <- rnbinom(n, mu = exp(0.2 - 0.1 * x2), size = 2)
  list(y1 = y1, y2 = y2, X1 = X1, X2 = X2)
}

test_that("RP copula LL reduces to fixed-model copula LL when q=0", {
  cs <- make_fixed_case()
  # layout: b1(2), b2(2), log_m1, log_m2, z_theta
  par <- c(0.3, 0.2, 0.2, -0.1, log(0.5), log(0.6), 0.4)
  Z0 <- matrix(0, 1, 0)
  for (fam in c("frank", "normal", "kimeldorf")) {
    rp <- bnbr_rp_copula_ll(par, cs$y1, cs$y2, cs$X1, cs$X2, NULL, NULL,
                            integer(0), integer(0), Z0, Z0, family = fam)
    fx <- sum(copula_loglik_vec(par, cs$y1, cs$y2, cs$X1, cs$X2, family = fam))
    expect_equal(rp, fx, tolerance = 1e-9, info = fam)
  }
})

test_that("per-draw copula pmf sums to ~1 over a count grid", {
  # single obs, single draw, moderate params -> rectangle pmf over grid sums to 1
  theta <- 5
  r1 <- 1 / 0.5; r2 <- 1 / 0.6; mu1 <- 2.0; mu2 <- 1.5
  grid <- expand.grid(y1 = 0:80, y2 = 0:80)
  a  <- pnbinom(grid$y1,     size = r1, mu = mu1)
  am <- ifelse(grid$y1 > 0, pnbinom(grid$y1 - 1, size = r1, mu = mu1), 0)
  b  <- pnbinom(grid$y2,     size = r2, mu = mu2)
  bm <- ifelse(grid$y2 > 0, pnbinom(grid$y2 - 1, size = r2, mu = mu2), 0)
  p <- frank_cdf(a, b, theta) - frank_cdf(am, b, theta) -
       frank_cdf(a, bm, theta) + frank_cdf(am, bm, theta)
  expect_equal(sum(p), 1, tolerance = 1e-3)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-copula.R')"`
Expected: FAIL — `could not find function "bnbr_rp_copula_ll"` (the pmf test should already pass).

- [ ] **Step 3: Implement `bnbr_rp_copula_ll`**

Create `R/rpbnb_copula_likelihood.R`:

```r
# Random-parameter BNB simulated log-likelihood with COPULA dependence.
# The per-draw dependence factor is the discrete-copula joint pmf
# (finite difference of the copula CDF over the two NB CDFs). Marginal means are
# random across individuals exactly as in the Famoye RP path. Internal.

#' Simulated log-likelihood for the copula RP-BNB model
#'
#' Parameter order: beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2),
#' log_m1, log_m2, z_theta. family is one of "frank", "normal", "kimeldorf".
#' @keywords internal
#' @noRd
bnbr_rp_copula_ll <- function(par, y1, y2, X1, X2, XR1, XR2,
                              rand_idx1, rand_idx2, Z1, Z2, family,
                              dist1 = NULL, dist2 = NULL,
                              sign1 = NULL, sign2 = NULL) {
  n  <- length(y1)
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  R  <- if (q1 + q2 > 0) nrow(Z1) else 1L

  beta1 <- par[1:k1]; beta2 <- par[(k1 + 1):(k1 + k2)]
  lg1 <- if (q1 > 0) (k1 + k2 + 1):(k1 + k2 + q1) else integer(0)
  lg2 <- if (q2 > 0) (k1 + k2 + q1 + 1):(k1 + k2 + q1 + q2) else integer(0)
  sd1 <- if (q1 > 0) exp(par[lg1]) else numeric(0)
  sd2 <- if (q2 > 0) exp(par[lg2]) else numeric(0)
  idx_end <- k1 + k2 + q1 + q2
  log_m1 <- par[idx_end + 1]; log_m2 <- par[idx_end + 2]; z_theta <- par[idx_end + 3]
  r1 <- exp(-log_m1); r2 <- exp(-log_m2)
  theta <- z_to_native(family, z_theta)

  if (is.null(dist1) && q1 > 0) dist1 <- rep("normal", q1)
  if (is.null(dist2) && q2 > 0) dist2 <- rep("normal", q2)
  if (is.null(sign1) && q1 > 0) sign1 <- rep(1, q1)
  if (is.null(sign2) && q2 > 0) sign2 <- rep(1, q2)

  xb1 <- as.vector(X1 %*% beta1); xb2 <- as.vector(X2 %*% beta2)
  real1 <- if (q1 > 0) rand_realize(Z1, dist1, sign1, beta1[rand_idx1], sd1) else NULL
  real2 <- if (q2 > 0) rand_realize(Z2, dist2, sign2, beta2[rand_idx2], sd2) else NULL
  XR1m <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2m <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else NULL

  cop_cdf <- switch(family, frank = frank_cdf, normal = normal_cdf,
                    kimeldorf = kimeldorf_cdf)

  LL <- matrix(0, n, R)
  for (r in seq_len(R)) {
    eta1 <- xb1 + if (q1 > 0) as.vector(XR1m %*% real1$dev[r, ]) else 0
    eta2 <- xb2 + if (q2 > 0) as.vector(XR2m %*% real2$dev[r, ]) else 0
    mu1 <- pmin(pmax(exp(eta1), 1e-300), 1e15)
    mu2 <- pmin(pmax(exp(eta2), 1e-300), 1e15)
    a  <- pnbinom(y1, size = r1, mu = mu1)
    am <- ifelse(y1 > 0, pnbinom(y1 - 1, size = r1, mu = mu1), 0)
    b  <- pnbinom(y2, size = r2, mu = mu2)
    bm <- ifelse(y2 > 0, pnbinom(y2 - 1, size = r2, mu = mu2), 0)
    p_obs <- cop_cdf(a, b, theta) - cop_cdf(am, b, theta) -
             cop_cdf(a, bm, theta) + cop_cdf(am, bm, theta)
    LL[, r] <- log(pmax(p_obs, 1e-300))
  }
  lse <- row_log_sum_exp(LL)
  sum(lse - log(R))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-copula.R')"`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add R/rpbnb_copula_likelihood.R tests/testthat/test-rpbnb-copula.R
git commit -m "feat(rpbnb): add copula simulated log-likelihood + reduction tests"
```

---

### Task 2: Copula-dependent RP-BNB simulator

**Files:**
- Create: `R/simulate_rpbnb_copula.R`
- Test: `tests/testthat/test-rpbnb-copula.R` (append)

**Interfaces:**
- Consumes: `parse_rand_spec`, `rand_dist_registry` (R/rand_dist.R), `.build_sim_X`, `.check_sim_covariates`, `.sim_default_covariates` (R/simulation_draws.R), `pbivnorm::pbivnorm` (Gaussian sampling uses base `rnorm`; only the CDF path needs pbivnorm elsewhere).
- Produces: `simulate_rpbnb_copula(n, beta1, beta2, random_1=NULL, random_2=NULL, dispersion=c(m1=.5,m2=.5), copula, covariates=NULL, seed=NULL)` → `list(data, mu, true, settings)` where `true$copula` = family string, `true$theta` = native param, `true$tau` = Kendall's tau.

- [ ] **Step 1: Write the failing simulator tests**

Append to `tests/testthat/test-rpbnb-copula.R`:

```r
test_that("copula simulator produces correct marginals and dependence sign", {
  sim <- simulate_rpbnb_copula(
    n = 4000,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.6),   # rho = 0.6 -> positive dependence
    seed = 11
  )
  expect_setequal(names(sim$data), c("y1", "y2", "x1"))
  expect_equal(nrow(sim$data), 4000)
  # marginal means approximately match the model means
  expect_equal(mean(sim$data$y1), mean(sim$mu$mu1), tolerance = 0.15)
  expect_equal(mean(sim$data$y2), mean(sim$mu$mu2), tolerance = 0.15)
  # positive dependence built in -> positive Spearman correlation
  expect_gt(cor(sim$data$y1, sim$data$y2, method = "spearman"), 0.15)
})

test_that("copula simulator with rho=0 yields near-independent margins", {
  sim <- simulate_rpbnb_copula(
    n = 4000,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.0), seed = 12
  )
  expect_lt(abs(cor(sim$data$y1, sim$data$y2, method = "spearman")), 0.05)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-copula.R')"`
Expected: FAIL — `could not find function "simulate_rpbnb_copula"`.

- [ ] **Step 3: Implement `simulate_rpbnb_copula`**

Create `R/simulate_rpbnb_copula.R`:

```r
#' Sample copula uniforms (u, v) for a given family and native parameter.
#'
#' Conditional-inversion sampling: draw independent u, w ~ U(0,1); set
#' v = C_{2|1}^{-1}(w | u). Gaussian uses a bivariate-normal draw directly.
#' @keywords internal
#' @noRd
.rcopula_uv <- function(n, family, theta) {
  u <- stats::runif(n)
  if (family == "normal") {
    rho <- theta
    z1 <- stats::qnorm(u)
    z2 <- rho * z1 + sqrt(1 - rho^2) * stats::rnorm(n)
    return(list(u = u, v = stats::pnorm(z2)))
  }
  w <- stats::runif(n)
  if (family == "frank") {
    if (abs(theta) < 1e-8) return(list(u = u, v = w))
    et  <- exp(-theta)
    etu <- exp(-theta * u)
    v <- -log(1 + w * (et - 1) / (1 + (1 - w) * (etu - 1))) / theta
    v <- pmin(pmax(v, 1e-12), 1 - 1e-12)
    return(list(u = u, v = v))
  }
  if (family == "kimeldorf") {          # Clayton, theta > 0
    if (theta < 1e-8) return(list(u = u, v = w))
    v <- (u^(-theta) * (w^(-theta / (theta + 1)) - 1) + 1)^(-1 / theta)
    v <- pmin(pmax(v, 1e-12), 1 - 1e-12)
    return(list(u = u, v = v))
  }
  stop("unknown copula family: ", family, call. = FALSE)
}

#' Simulate data from a copula RP-BNB process
#'
#' @param n Number of observations.
#' @param beta1,beta2 Named coefficient means; must include "(Intercept)".
#' @param random_1,random_2 Random-coefficient specs (see [simulate_rpbnb()]).
#' @param dispersion Named `c(m1=, m2=)` NB2 dispersions.
#' @param copula An [copula()] object giving the family and native parameter `par`.
#' @param covariates Optional covariate data frame; NULL -> standard-normal columns.
#' @param seed Optional RNG seed.
#' @return list(data, mu, true, settings).
#' @export
simulate_rpbnb_copula <- function(n, beta1, beta2,
                                  random_1 = NULL, random_2 = NULL,
                                  dispersion = c(m1 = 0.5, m2 = 0.5),
                                  copula, covariates = NULL, seed = NULL) {
  stopifnot(inherits(copula, "rpbnb_copula"), !is.null(copula$par),
            "(Intercept)" %in% names(beta1), "(Intercept)" %in% names(beta2))
  spec1 <- parse_rand_spec(random_1); spec2 <- parse_rand_spec(random_2)
  if (!is.null(seed)) set.seed(seed)
  if (is.null(covariates)) covariates <- .sim_default_covariates(beta1, beta2, n)
  .check_sim_covariates(covariates, beta1, beta2, n)

  d1 <- .build_sim_X(beta1, covariates, n); X1 <- d1$X; b1 <- d1$beta
  d2 <- .build_sim_X(beta2, covariates, n); X2 <- d2$X; b2 <- d2$beta
  realize <- function(bv, spec, X) {
    B <- matrix(rep(bv, each = n), n, dimnames = list(NULL, names(bv)))
    for (i in seq_along(spec$names)) {
      reg  <- rand_dist_registry[[spec$dist[i]]]
      base <- if (reg$base == "normal") stats::rnorm(n) else reg$u_to_base(stats::runif(n))
      B[, spec$names[i]] <- reg$coef(bv[[spec$names[i]]], spec$scale[i], base, spec$sign[i])
    }
    B
  }
  B1 <- realize(b1, spec1, X1); B2 <- realize(b2, spec2, X2)
  mu1 <- exp(rowSums(X1 * B1)); mu2 <- exp(rowSums(X2 * B2))
  m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]

  uv <- .rcopula_uv(n, copula$family, copula$par)
  y1 <- stats::qnbinom(uv$u, size = 1 / m1, mu = mu1)
  y2 <- stats::qnbinom(uv$v, size = 1 / m2, mu = mu2)

  td <- copula_tau_and_deriv(copula$family, switch(copula$family,
        frank = copula$par, normal = atanh(copula$par), kimeldorf = log(copula$par)))
  list(
    data = data.frame(y1 = y1, y2 = y2, covariates),
    mu   = data.frame(mu1 = mu1, mu2 = mu2),
    true = list(beta1 = beta1, beta2 = beta2, random_1 = spec1, random_2 = spec2,
                dispersion = dispersion, copula = copula$family,
                theta = copula$par, tau = td$tau),
    settings = list(n = n, seed = seed)
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-copula.R')"`
Expected: PASS (all tests so far).

- [ ] **Step 5: Regenerate docs (NAMESPACE export) and commit**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::document()"`
Expected: `NAMESPACE` gains `export(simulate_rpbnb_copula)`; `man/simulate_rpbnb_copula.Rd` created.

```bash
git add R/simulate_rpbnb_copula.R tests/testthat/test-rpbnb-copula.R NAMESPACE man/simulate_rpbnb_copula.Rd
git commit -m "feat(rpbnb): add copula-dependent RP-BNB simulator"
```

---

### Task 3: Copula estimator + `dependence` dispatch in `fit_rpbnb`

**Files:**
- Create: `R/fit_rpbnb_copula.R`
- Modify: `R/fit_rpbnb.R` (add `dependence` arg + dispatch; add `cop_family` to `new_rpbnb_fit`)
- Test: `tests/testthat/test-rpbnb-copula.R` (append)

**Interfaces:**
- Consumes: `bnbr_rp_copula_ll` (Task 1), `.prepare_bnb_data` (R/data_prep.R), `parse_rand_spec` (R/rand_dist.R), `halton_uniform` (R/simulation_draws.R), `z_to_native`/`copula_tau_and_deriv` (R/copula_core.R), `maxLik::maxLik`, `numDeriv::hessian`, `MASS::ginv`.
- Produces: `.fit_rpbnb_copula(formula_1, formula_2, data, random_1, random_2, draws, draw_type, seed, start, control, family)` → object of class `rpbnb_fit` with `cop_family = family`, `lambda = NULL`, `bounds = NULL`, and a `z_theta` coefficient. `fit_rpbnb` gains parameter `dependence = "famoye"`.

- [ ] **Step 1: Write the failing dispatch test**

Append to `tests/testthat/test-rpbnb-copula.R`:

```r
test_that("fit_rpbnb dispatches to the copula path and returns a copula fit", {
  sim <- simulate_rpbnb_copula(
    n = 800,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.5), seed = 21)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   dependence = copula("normal"),
                   draws = 50, seed = 1,
                   control = rpbnb_control(compute_se = FALSE))
  expect_s3_class(fit, "rpbnb_fit")
  expect_identical(fit$cop_family, "normal")
  expect_true("z_theta" %in% names(fit$coef))
  expect_true(is.null(fit$lambda))
  # print() must use the copula branch (native param + Kendall's tau)
  out <- paste(capture.output(print(fit)), collapse = "\n")
  expect_match(out, "rho|tau|Gaussian")
})

test_that("fit_rpbnb default dependence is unchanged (famoye) and has z_lambda", {
  sim <- simulate_rpbnb_copula(
    n = 400, beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.3), seed = 31)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, draws = 50, seed = 1,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true("z_lambda" %in% names(fit$coef))
  expect_null(fit$cop_family)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-copula.R')"`
Expected: FAIL — `unused argument (dependence = ...)`.

- [ ] **Step 3a: Add `cop_family` to the fit constructor**

In `R/fit_rpbnb.R`, modify `new_rpbnb_fit` to accept and store `cop_family = NULL`. Change the signature line to add `cop_family = NULL` before `call`, and add `cop_family = cop_family` to the `structure(list(...))` body. Exact edit:

Signature — add `cop_family = NULL,` immediately before `call)`:
```r
new_rpbnb_fit <- function(coef, vcov, se, logLik, nobs, npar,
                          m1, m2, lambda, bounds, mu1, mu2, X1, X2, Y1, Y2,
                          rand_idx1, rand_idx2, formula_1, formula_2,
                          draws, draw_type, seed, ll_trace, convergence,
                          cop_family = NULL, call) {
```
List body — add `cop_family = cop_family,` next to `convergence = convergence,`:
```r
         ll_trace = ll_trace, convergence = convergence,
         cop_family = cop_family,
```

- [ ] **Step 3b: Add the `dependence` arg + dispatch to `fit_rpbnb`**

In `R/fit_rpbnb.R`, add `dependence = "famoye"` to the `fit_rpbnb` argument list (after `control = rpbnb_control()`), and at the very top of the function body (after `stopifnot(is.data.frame(data))`), insert:
```r
  if (inherits(dependence, "rpbnb_copula")) {
    return(.fit_rpbnb_copula(formula_1, formula_2, data, random_1, random_2,
                             draws, draw_type, seed, start, control,
                             family = dependence$family))
  }
```
Also add a roxygen line for the new param:
```r
#' @param dependence Dependence structure: "famoye" (default; Famoye/Sarmanov,
#'   the multithreaded C++ path) or an [copula()] object for copula dependence
#'   (Frank / Gaussian / Clayton; estimated on a slower R path).
```

- [ ] **Step 3c: Implement the copula estimator**

Create `R/fit_rpbnb_copula.R`:

```r
# Estimator for the copula random-parameter BNB model. Optimizes the copula
# simulated log-likelihood (R/rpbnb_copula_likelihood.R) with BFGS + numeric
# gradient; standard errors from the numeric Hessian. Internal.

#' @keywords internal
#' @noRd
.fit_rpbnb_copula <- function(formula_1, formula_2, data,
                              random_1, random_2, draws, draw_type,
                              seed, start, control, family) {
  draw_type <- match.arg(draw_type, "halton")
  spec1 <- parse_rand_spec(random_1); spec2 <- parse_rand_spec(random_2)

  prep <- .prepare_bnb_data(formula_1, formula_2, data)
  Y1 <- prep$Y1; Y2 <- prep$Y2; X1 <- prep$X1; X2 <- prep$X2
  k1 <- ncol(X1); k2 <- ncol(X2)

  idx_from_names <- function(who, X) {
    if (!length(who)) return(integer(0))
    miss <- who[!who %in% colnames(X)]
    if (length(miss)) stop("random name(s) not found: ",
                           paste(miss, collapse = ", "), call. = FALSE)
    as.integer(match(who, colnames(X)))
  }
  rand_idx1 <- idx_from_names(spec1$names, X1)
  rand_idx2 <- idx_from_names(spec2$names, X2)
  dist1 <- spec1$dist; sign1 <- spec1$sign
  dist2 <- spec2$dist; sign2 <- spec2$sign
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  XR1 <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2 <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else NULL

  set.seed(seed)
  if ((q1 + q2) > 0) {
    Z  <- halton_uniform(draws, q1 + q2, burn = control$halton_burn)
    Z1 <- if (q1 > 0) Z[, 1:q1, drop = FALSE] else matrix(0, draws, 0)
    Z2 <- if (q2 > 0) Z[, (q1 + 1):(q1 + q2), drop = FALSE] else matrix(0, draws, 0)
  } else {
    Z1 <- matrix(0, 1, 0); Z2 <- matrix(0, 1, 0)
  }

  scale_lab <- function(dist, cols)
    vapply(seq_along(dist),
           function(j) paste0(rand_dist_registry[[dist[j]]]$scale_label, cols[j]),
           character(1))
  par_names <- c(paste0("b1:", colnames(X1)), paste0("b2:", colnames(X2)),
                 if (q1 > 0) scale_lab(dist1, paste0("1:", colnames(X1)[rand_idx1])),
                 if (q2 > 0) scale_lab(dist2, paste0("2:", colnames(X2)[rand_idx2])),
                 "log_m1", "log_m2", "z_theta")
  if (is.null(start))
    start <- c(rep(0, k1 + k2),
               if (q1 > 0) rep(log(0.2), q1), if (q2 > 0) rep(log(0.2), q2),
               log(0.5), log(0.5), 0)
  names(start) <- par_names

  ll_trace <- numeric(0)
  ll_fun <- function(p) {
    v <- bnbr_rp_copula_ll(p, Y1, Y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                           Z1, Z2, family, dist1, dist2, sign1, sign2)
    ll_trace[[length(ll_trace) + 1L]] <<- v
    v
  }
  fit <- maxLik::maxLik(logLik = ll_fun, start = start, method = "BFGS",
                        control = list(iterlim = control$iterlim,
                                       reltol = control$reltol,
                                       printLevel = control$print_level))
  par_hat <- stats::coef(fit); names(par_hat) <- par_names
  npar <- length(par_hat)

  if (isTRUE(control$compute_se)) {
    H <- numDeriv::hessian(ll_fun, par_hat,
                           method.args = list(r = control$hess_r, eps = control$hess_eps))
    info <- -(H + t(H)) / 2
    vc <- try(solve(info), silent = TRUE)
    if (inherits(vc, "try-error")) vc <- MASS::ginv(info)
    se <- sqrt(pmax(diag(vc), 0))
  } else {
    vc <- matrix(NA_real_, npar, npar); se <- rep(NA_real_, npar)
  }
  dimnames(vc) <- list(par_names, par_names); names(se) <- par_names

  idx_end <- k1 + k2 + q1 + q2
  m1_hat <- unname(exp(par_hat[idx_end + 1]))
  m2_hat <- unname(exp(par_hat[idx_end + 2]))

  new_rpbnb_fit(
    coef = par_hat, vcov = vc, se = se, logLik = as.numeric(logLik(fit)),
    nobs = length(Y1), npar = npar, m1 = m1_hat, m2 = m2_hat,
    lambda = NULL, bounds = NULL, mu1 = NULL, mu2 = NULL,
    X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
    rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
    formula_1 = formula_1, formula_2 = formula_2,
    draws = draws, draw_type = draw_type, seed = seed,
    ll_trace = ll_trace, convergence = fit$code,
    cop_family = family, call = match.call()
  )
}
```

Note: verify `new_rpbnb_fit`'s existing arg names against R/fit_rpbnb.R before running — if `logLik` on a maxLik object needs `stats::logLik`, use `as.numeric(stats::logLik(fit))`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-copula.R')"`
Expected: PASS. If `print()` errors on `cop_family`, confirm methods.R's `.natural_scale_table` copula branch reads `object$cop_family` (it does) and that `new_rpbnb_fit` stored it.

- [ ] **Step 5: Regenerate docs and commit**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::document()"`

```bash
git add R/fit_rpbnb_copula.R R/fit_rpbnb.R tests/testthat/test-rpbnb-copula.R man/ NAMESPACE
git commit -m "feat(rpbnb): dispatch fit_rpbnb to copula estimator via dependence arg"
```

---

### Task 4: End-to-end recovery tests

**Files:**
- Test: `tests/testthat/test-rpbnb-copula.R` (append)

**Interfaces:**
- Consumes: `simulate_rpbnb_copula` (Task 2), `fit_rpbnb` with `dependence` (Task 3), `z_to_native` (R/copula_core.R).

- [ ] **Step 1: Write the recovery tests**

Append to `tests/testthat/test-rpbnb-copula.R`:

```r
recover_copula <- function(fam, par, n = 2000, draws = 200, seed = 7) {
  sim <- simulate_rpbnb_copula(
    n = n,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    random_1 = list(x1 = list(sd = 0.3)),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula(fam, par = par), seed = seed)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   random_1 = "x1", random_2 = "x1",
                   dependence = copula(fam),
                   draws = draws, seed = 1,
                   control = rpbnb_control(compute_se = TRUE, print_level = 0))
  z <- fit$coef[["z_theta"]]
  list(true = par, est = z_to_native(fam, z),
       se_z = fit$se[["z_theta"]], sim_tau = sim$true$tau)
}

test_that("Gaussian copula parameter is recovered", {
  r <- recover_copula("normal", par = 0.5)
  expect_equal(r$est, r$true, tolerance = 0.12)
})

test_that("Frank copula parameter is recovered", {
  r <- recover_copula("frank", par = 4)
  expect_equal(r$est, r$true, tolerance = 1.2)   # theta on its native (wide) scale
})

test_that("Clayton copula parameter is recovered", {
  r <- recover_copula("kimeldorf", par = 1.0)
  expect_equal(r$est, r$true, tolerance = 0.5)
})
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' -e "devtools::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-copula.R')"`
Expected: PASS. If a family's recovery is off, first check `.rcopula_uv`'s conditional-inverse formula for that family (wrong sampling is the likely cause), then widen `draws` before loosening tolerance.

- [ ] **Step 3: Commit**

```bash
git add tests/testthat/test-rpbnb-copula.R
git commit -m "test(rpbnb): end-to-end copula parameter recovery (frank/normal/clayton)"
```

---

### Task 5: Demo scripts

**Files:**
- Create: `inst/simulate_rpbnb_copula_demo.R`
- Create: `inst/fit_rpbnb_copula_demo.R`

**Interfaces:**
- Consumes: `simulate_rpbnb_copula`, `fit_rpbnb` with `dependence`.

- [ ] **Step 1: Write the generate demo**

Create `inst/simulate_rpbnb_copula_demo.R`:

```r
#!/usr/bin/env Rscript
# Generate a copula-dependent RP-BNB dataset (Gaussian copula, rho = 0.6).
devtools::load_all(quiet = TRUE)
sim <- simulate_rpbnb_copula(
  n = 3000,
  beta1 = c("(Intercept)" = 0.4, x_age = 0.20, x_income = 0.15),
  beta2 = c("(Intercept)" = 0.3, x_age = -0.10, x_income = 0.25),
  random_1 = list(x_age = list(sd = 0.30)),
  random_2 = list(x_income = list(sd = 0.25)),
  dispersion = c(m1 = 0.5, m2 = 0.6),
  copula = copula("normal", par = 0.6), seed = 707)
cat(sprintf("n=%d  true rho=%.2f  Kendall tau=%.3f  Spearman(y1,y2)=%.3f\n",
            nrow(sim$data), sim$true$theta, sim$true$tau,
            cor(sim$data$y1, sim$data$y2, method = "spearman")))
dir.create("data", showWarnings = FALSE)
write.csv(sim$data, "data/simulated_rpbnb_copula.csv", row.names = FALSE)
saveRDS(sim$true, "data/simulated_rpbnb_copula_truth.rds")
cat("Saved data/simulated_rpbnb_copula.csv + truth rds\n")
```

- [ ] **Step 2: Write the fit demo**

Create `inst/fit_rpbnb_copula_demo.R`:

```r
#!/usr/bin/env Rscript
# Fit the Gaussian-copula RP-BNB dataset and compare to truth.
devtools::load_all(quiet = TRUE)
data  <- read.csv("data/simulated_rpbnb_copula.csv")
truth <- readRDS("data/simulated_rpbnb_copula_truth.rds")
fit <- fit_rpbnb(y1 ~ x_age + x_income, y2 ~ x_age + x_income, data = data,
                 random_1 = "x_age", random_2 = "x_income",
                 dependence = copula("normal"),
                 draws = 300, seed = 20240712,
                 control = rpbnb_control(print_level = 1))
print(fit)
rho_hat <- tanh(fit$coef[["z_theta"]])
cat(sprintf("\nCopula rho: true %.3f  estimated %.3f\n", truth$theta, rho_hat))
```

- [ ] **Step 3: Run both demos to verify they execute**

Run: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' inst/simulate_rpbnb_copula_demo.R`
Then: `& 'C:/Program Files/R/R-4.5.1/bin/Rscript' inst/fit_rpbnb_copula_demo.R`
Expected: both run without error; the fit demo prints `rho` estimate near 0.6.

- [ ] **Step 4: Commit**

```bash
git add inst/simulate_rpbnb_copula_demo.R inst/fit_rpbnb_copula_demo.R data/simulated_rpbnb_copula.csv data/simulated_rpbnb_copula_truth.rds
git commit -m "docs(rpbnb): copula simulate + fit demo scripts"
```

---

## Notes for the implementer

- **Verify `new_rpbnb_fit`'s real argument list** in `R/fit_rpbnb.R` before Task 3 Step 3c — the constructor call must match it exactly (names/order). The plan's call assumes the current arg set plus the new `cop_family`; adjust field-by-field if the constructor differs.
- **`row_log_sum_exp` on a single-column matrix** returns that column, so the `q=0` reduction is exact — do not special-case it.
- **Performance:** numeric gradient over the draw loop is slow; keep test `n`/`draws` modest (as in the tasks). Do not raise them to "be safe" — that only slows CI.
- **Do not touch the Famoye C++ path.** The dispatch returns early for copula; everything else in `fit_rpbnb` is unchanged.
