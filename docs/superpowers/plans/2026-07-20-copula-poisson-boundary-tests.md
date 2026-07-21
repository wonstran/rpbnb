# Copula Poisson m=0 Branch + Copula Boundary Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the copula random-parameter BNB stack an exact Poisson (`m = 0`) branch and wire the restricted-refit machinery through the copula fit, so `rpbnb_boundary_tests()` supports copula fits for both SD and dispersion parameters.

**Architecture:** Reuse the established in-band signal — a Poisson margin is `m = 0`, i.e. `r = 1/m = Inf`. `r1`/`r2` already flow through the copula stack as doubles, so each NB evaluation site gains a `!is.finite(r)` → Poisson branch (`pnbinom`→`ppois`, `dnbinom`→`dpois`, dispersion score→0). The fit function pins the Poisson margin's `log_m` via `maxLik(fixed=)` and gains `.fixed`/`.opt_draws` for restricted refits, mirroring the Famoye `fit_rpbnb` contract.

**Tech Stack:** R, Rcpp/RcppParallel (OpenMP), maxLik, testthat, devtools.

## Global Constraints

- R package `rpbnb`; run all R via `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"`.
- Load the package for tests with `suppressMessages(devtools::load_all("."))`; C++ changes require a recompile, which `load_all()` triggers automatically.
- Parameter order for the copula RP model is fixed: `beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2), log_m1, log_m2, z_theta`.
- In-band Poisson signal: `r = Inf` means the margin is Poisson (`m = 0`). Never call `pnbinom`/`dnbinom` with `size = Inf` (returns NaN / can segfault).
- TDD throughout: write the failing test, watch it fail, minimal implementation, watch it pass, commit.
- Slow end-to-end tests guard with `skip_slow()` (see `tests/testthat/helper-slow.R`); run them with `Sys.setenv(RPBNB_RUN_SLOW = "1")`.

---

### Task 1: Poisson branch in shared pmf/score helpers

The single source of the NB-CDF corners and per-observation copula scores, reused by both the value and gradient paths. A margin with `r = Inf` must use `ppois`/`dpois`; its `log_m` score is forced to 0.

**Files:**
- Modify: `R/copula_likelihood.R` (`.copula_pmf`, `.copula_score_scalars`)
- Test: `tests/testthat/test-copula-poisson.R` (create)

**Interfaces:**
- Consumes: existing `.copula_pmf(y1, y2, mu1, mu2, r1, r2, theta, family)` and `.copula_score_scalars(y1, y2, mu1, mu2, r1, r2, theta, dth_dz, family)`.
- Produces: same signatures; when `r1` (resp. `r2`) is `Inf`, corners use `ppois` and `da_dmu`/`s_logm` use the Poisson forms. No signature change.

- [ ] **Step 1: Write the failing test**

Create `tests/testthat/test-copula-poisson.R`:

```r
test_that(".copula_pmf uses ppois corners for a Poisson (r = Inf) margin", {
  y1 <- c(0L, 2L, 5L); y2 <- c(1L, 0L, 3L)
  mu1 <- c(1.5, 4.0, 2.0); mu2 <- c(0.8, 2.0, 5.0)
  r2 <- 3.0; theta <- 0.5; family <- "frank"

  pm <- rpbnb:::.copula_pmf(y1, y2, mu1, mu2, Inf, r2, theta, family)

  # Margin 1 corners must equal the exact Poisson CDF, not an NB approximation.
  expect_equal(pm$a,  stats::ppois(y1, mu1))
  expect_equal(pm$am, ifelse(y1 > 0L, stats::ppois(y1 - 1L, mu1), 0))
  # Margin 2 stays NB2.
  expect_equal(pm$b,  stats::pnbinom(y2, size = r2, mu = mu2))
  expect_true(all(pm$ok))
})

test_that(".copula_score_scalars zeroes s_logm and uses dpois for a Poisson margin", {
  y1 <- c(0L, 2L, 5L); y2 <- c(1L, 0L, 3L)
  mu1 <- c(1.5, 4.0, 2.0); mu2 <- c(0.8, 2.0, 5.0)
  r2 <- 3.0; theta <- 0.5; dth_dz <- 1.0; family <- "frank"

  sc <- rpbnb:::.copula_score_scalars(y1, y2, mu1, mu2, Inf, r2, theta, dth_dz, family)

  # The pinned Poisson dispersion contributes no score.
  expect_equal(sc$s_logm1, rep(0, length(y1)))
  # Margin-1 mean score is finite (dpois path), not NaN from dnbinom(size=Inf).
  expect_true(all(is.finite(sc$s_eta1)))
  # Margin 2's dispersion score is unaffected.
  expect_true(any(sc$s_logm2 != 0))
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: FAIL — `.copula_pmf` calls `pnbinom(size = Inf)` for margin 1, so `pm$a` is `NaN`/mismatched and `ok` is FALSE; `s_logm1` is `NaN` not 0.

- [ ] **Step 3: Implement the Poisson branch in `.copula_pmf`**

In `R/copula_likelihood.R`, replace the corner block in `.copula_pmf` (currently lines ~82-85):

```r
.copula_pmf <- function(y1, y2, mu1, mu2, r1, r2, theta, family) {
  # A Poisson-restricted margin is signalled in-band by r = Inf (m = 0); its
  # CDF corners use ppois, not pnbinom(size = Inf) which returns NaN.
  pois1 <- !is.finite(r1); pois2 <- !is.finite(r2)
  a  <- if (pois1) stats::ppois(y1, mu1) else pnbinom(y1, size = r1, mu = mu1)
  am <- if (pois1) ifelse(y1 > 0L, stats::ppois(y1 - 1L, mu1), 0)
        else       ifelse(y1 > 0L, pnbinom(y1 - 1L, size = r1, mu = mu1), 0)
  b  <- if (pois2) stats::ppois(y2, mu2) else pnbinom(y2, size = r2, mu = mu2)
  bm <- if (pois2) ifelse(y2 > 0L, stats::ppois(y2 - 1L, mu2), 0)
        else       ifelse(y2 > 0L, pnbinom(y2 - 1L, size = r2, mu = mu2), 0)
  ok <- is.finite(a) & is.finite(am) & is.finite(b) & is.finite(bm)
  cop_cdf <- switch(family, frank = frank_cdf, normal = normal_cdf, kimeldorf = kimeldorf_cdf)
  p_obs <- cop_cdf(a, b, theta) - cop_cdf(am, b, theta) -
           cop_cdf(a, bm, theta) + cop_cdf(am, bm, theta)
  ok <- ok & is.finite(p_obs)
  underflow <- !(p_obs > 1e-300)
  list(a = a, am = am, b = b, bm = bm, p_obs = pmax(p_obs, 1e-300),
       ok = ok, underflow = underflow)
}
```

- [ ] **Step 4: Implement the Poisson branch in `.copula_score_scalars`**

In `R/copula_likelihood.R`, `.copula_score_scalars`: derive the flags after the `pm <- .copula_pmf(...)` line, branch the mu-score `dnbinom` calls to `dpois`, and zero the dispersion scores for a Poisson margin. Replace lines ~113-141:

```r
  pm <- .copula_pmf(y1, y2, mu1, mu2, r1, r2, theta, family)
  a <- pm$a; am <- pm$am; b <- pm$b; bm <- pm$bm; p_obs <- pm$p_obs; ok <- pm$ok
  pois1 <- !is.finite(r1); pois2 <- !is.finite(r2)

  cu_ab   <- .cop_du(a,  b,  theta, family); cu_amb  <- .cop_du(am, b,  theta, family)
  cu_abm  <- .cop_du(a,  bm, theta, family); cu_ambm <- .cop_du(am, bm, theta, family)
  cv_ab   <- .cop_dv(a,  b,  theta, family); cv_amb  <- .cop_dv(am, b,  theta, family)
  cv_abm  <- .cop_dv(a,  bm, theta, family); cv_ambm <- .cop_dv(am, bm, theta, family)
  ct_rect <- .cop_dtheta(a, b, theta, family) - .cop_dtheta(am, b, theta, family) -
             .cop_dtheta(a, bm, theta, family) + .cop_dtheta(am, bm, theta, family)

  # mu-score: dpois for a Poisson margin (dnbinom(size = Inf) is NaN).
  da_dmu1  <- if (pois1) -(y1 + 1L) * stats::dpois(y1 + 1L, mu1) / mu1
              else       -(y1 + 1L) * dnbinom(y1 + 1L, size = r1, mu = mu1) / mu1
  dam_dmu1 <- if (pois1) ifelse(y1 > 0L, -y1 * stats::dpois(y1, mu1) / mu1, 0)
              else       ifelse(y1 > 0L, -y1 * dnbinom(y1, size = r1, mu = mu1) / mu1, 0)
  delta_u_a  <- cu_ab - cu_abm
  delta_u_am <- -cu_amb + cu_ambm
  s_eta1 <- (delta_u_a * da_dmu1 * mu1 + delta_u_am * dam_dmu1 * mu1) / p_obs

  db_dmu2  <- if (pois2) -(y2 + 1L) * stats::dpois(y2 + 1L, mu2) / mu2
              else       -(y2 + 1L) * dnbinom(y2 + 1L, size = r2, mu = mu2) / mu2
  dbm_dmu2 <- if (pois2) ifelse(y2 > 0L, -y2 * stats::dpois(y2, mu2) / mu2, 0)
              else       ifelse(y2 > 0L, -y2 * dnbinom(y2, size = r2, mu = mu2) / mu2, 0)
  delta_v_b  <- cv_ab - cv_amb
  delta_v_bm <- -cv_abm + cv_ambm
  s_eta2 <- (delta_v_b * db_dmu2 * mu2 + delta_v_bm * dbm_dmu2 * mu2) / p_obs

  # Dispersion score: a Poisson margin's log_m is pinned -> zero score (the NB2
  # form (-r)*(...) would be (-Inf)*0 = NaN at r = Inf).
  if (pois1) {
    s_logm1 <- rep(0, length(y1))
  } else {
    da_dr1  <- mapply(.dnb_cdf_dr, y1,      mu1, r1)
    dam_dr1 <- ifelse(y1 > 0L, mapply(.dnb_cdf_dr, y1 - 1L, mu1, r1), 0)
    s_logm1 <- (-r1) * (delta_u_a * da_dr1 + delta_u_am * dam_dr1) / p_obs
  }
  if (pois2) {
    s_logm2 <- rep(0, length(y2))
  } else {
    db_dr2  <- mapply(.dnb_cdf_dr, y2,      mu2, r2)
    dbm_dr2 <- ifelse(y2 > 0L, mapply(.dnb_cdf_dr, y2 - 1L, mu2, r2), 0)
    s_logm2 <- (-r2) * (delta_v_b * db_dr2 + delta_v_bm * dbm_dr2) / p_obs
  }

  s_ztheta <- ct_rect * dth_dz / p_obs
```

(The `bad`-masking block and the returned `list(...)` below it are unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add R/copula_likelihood.R tests/testthat/test-copula-poisson.R
git commit -m "feat(copula): exact Poisson m=0 branch in shared pmf/score helpers"
```

---

### Task 2: Poisson branch in the RP copula R likelihood

Thread `pois1`/`pois2` through the R value and gradient functions so a restricted margin uses `r = Inf` and its `log_m` gradient column is exactly zero.

**Files:**
- Modify: `R/rpbnb_copula_likelihood.R` (`bnbr_rp_copula_ll`, `bnbr_rp_copula_ll_grad`, `bnbr_rp_copula_ll_grad_cpp`)
- Test: `tests/testthat/test-copula-poisson.R` (append)

**Interfaces:**
- Consumes: `.copula_pmf`, `.copula_score_scalars` (Task 1).
- Produces:
  - `bnbr_rp_copula_ll(par, ..., family, dist1, dist2, sign1, sign2, pois1 = FALSE, pois2 = FALSE)`
  - `bnbr_rp_copula_ll_grad(par, ..., want_scores = FALSE, pois1 = FALSE, pois2 = FALSE)`
  - `bnbr_rp_copula_ll_grad_cpp(par, ..., want_scores = FALSE, n_threads = 0L, pois1 = FALSE, pois2 = FALSE)`

- [ ] **Step 1: Write the failing test**

Append to `tests/testthat/test-copula-poisson.R`:

```r
test_that("bnbr_rp_copula_ll(pois1=TRUE) matches a ppois-corner reference", {
  set.seed(11)
  n <- 40
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rnbinom(n, size = 2, mu = 1.5)
  # No random coefficients: R = 1, so the RP value reduces to the fixed pmf.
  # par order: beta1(2), beta2(2), log_m1, log_m2, z_theta. log_m1 (index 5) is
  # inert here because pois1 = TRUE forces r1 = Inf.
  par <- c(0.3, 0.1, 0.2, -0.1, 0, 0.4, 0.5)
  # Build the reference directly from the Poisson-margin pmf.
  mu1 <- as.vector(exp(X1 %*% par[1:2])); mu2 <- as.vector(exp(X2 %*% par[3:4]))
  r2 <- exp(-par[6]); theta <- rpbnb:::z_to_native("frank", par[7])
  pm <- rpbnb:::.copula_pmf(y1, y2, mu1, mu2, Inf, r2, theta, "frank")
  ref <- sum(log(pm$p_obs))

  val <- rpbnb:::bnbr_rp_copula_ll(
    par, y1, y2, X1, X2, NULL, NULL, integer(0), integer(0),
    matrix(0, 1, 0), matrix(0, 1, 0), "frank", pois1 = TRUE)
  expect_equal(as.numeric(val), ref, tolerance = 1e-8)
})

test_that("bnbr_rp_copula_ll_grad zeroes the log_m1 column for a Poisson margin", {
  set.seed(12)
  n <- 50
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rnbinom(n, size = 2, mu = 1.5)
  par <- c(0.3, 0.1, 0.2, -0.1, 0, 0.4, 0.5)  # index 5 = log_m1
  g <- attr(rpbnb:::bnbr_rp_copula_ll_grad(
    par, y1, y2, X1, X2, NULL, NULL, integer(0), integer(0),
    matrix(0, 1, 0), matrix(0, 1, 0), "frank", pois1 = TRUE), "gradient")
  expect_equal(g[5], 0)                    # log_m1 pinned -> zero gradient
  # Free columns match a numeric gradient of the frozen (pois1) objective.
  f <- function(p) as.numeric(rpbnb:::bnbr_rp_copula_ll(
    p, y1, y2, X1, X2, NULL, NULL, integer(0), integer(0),
    matrix(0, 1, 0), matrix(0, 1, 0), "frank", pois1 = TRUE))
  gnum <- numDeriv::grad(f, par)
  free <- c(1, 2, 3, 4, 6, 7)
  expect_equal(unname(g[free]), gnum[free], tolerance = 1e-5)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: FAIL — `bnbr_rp_copula_ll` has no `pois1` argument (`unused argument`).

- [ ] **Step 3: Add `pois1`/`pois2` to `bnbr_rp_copula_ll`**

In `R/rpbnb_copula_likelihood.R`, extend the signature and the `r1`/`r2` lines:

```r
bnbr_rp_copula_ll <- function(par, y1, y2, X1, X2, XR1, XR2,
                              rand_idx1, rand_idx2, Z1, Z2, family,
                              dist1 = NULL, dist2 = NULL,
                              sign1 = NULL, sign2 = NULL,
                              pois1 = FALSE, pois2 = FALSE) {
```

and change (was `r1 <- exp(-log_m1); r2 <- exp(-log_m2)`):

```r
  r1 <- if (pois1) Inf else exp(-log_m1)
  r2 <- if (pois2) Inf else exp(-log_m2)
```

- [ ] **Step 4: Add `pois1`/`pois2` to `bnbr_rp_copula_ll_grad` and zero the log_m columns**

Extend the signature (add `pois1 = FALSE, pois2 = FALSE` after `want_scores = FALSE`), change the `r1`/`r2` lines identically, and after the Pass-2 loop that fills `grad`/`S`, force the pinned columns to 0:

```r
  if (pois1) { grad[im1] <- 0; if (want_scores) S[, im1] <- 0 }
  if (pois2) { grad[im2] <- 0; if (want_scores) S[, im2] <- 0 }

  out <- value
  attr(out, "gradient") <- grad
  if (want_scores) attr(out, "scores") <- S
  out
```

(`im1`/`im2` are already defined near the top of the function as `idx_end + 1`/`idx_end + 2`.)

- [ ] **Step 5: Add `pois1`/`pois2` to the C++ wrapper (in-band r = Inf)**

In `bnbr_rp_copula_ll_grad_cpp`, extend the signature (add `pois1 = FALSE, pois2 = FALSE`) and change the `r1`/`r2` lines:

```r
  r1 <- if (pois1) Inf else exp(-log_m1)
  r2 <- if (pois2) Inf else exp(-log_m2)
```

(The `.Call` already passes `r1`, `r2` as doubles — `Inf` flows through unchanged.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: PASS (4 tests). The C++ wrapper test is added in Task 3.

- [ ] **Step 7: Commit**

```bash
git add R/rpbnb_copula_likelihood.R tests/testthat/test-copula-poisson.R
git commit -m "feat(copula): thread Poisson pois1/pois2 through the RP copula R likelihood"
```

---

### Task 3: Poisson branch in the C++ copula core

Add the `r = Inf` → Poisson branch to `src/copula_parallel.cpp` so the multithreaded path matches the R path exactly.

**Files:**
- Modify: `src/copula_parallel.cpp`
- Test: `tests/testthat/test-copula-poisson.R` (append a C++ == R parity test)

**Interfaces:**
- Consumes: the C++ entry point already receives `double r1, double r2`; `Inf` is the in-band Poisson signal.
- Produces: identical value/gradient/scores to the R path when `r1`/`r2` are `Inf`.

- [ ] **Step 1: Write the failing parity test**

Append to `tests/testthat/test-copula-poisson.R`:

```r
test_that("C++ copula Poisson path matches the R path (value, gradient, scores)", {
  skip_if_not(rpbnb:::rpbnb_copula_cpp_available(), "C++ copula core not compiled")
  set.seed(21)
  n <- 60
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rnbinom(n, size = 2, mu = 1.5)
  par <- c(0.3, 0.1, 0.2, -0.1, 0, 0.4, 0.5)
  args <- list(par, y1, y2, X1, X2, NULL, NULL, integer(0), integer(0),
               matrix(0, 1, 0), matrix(0, 1, 0), "frank")

  rR <- do.call(rpbnb:::bnbr_rp_copula_ll_grad,
                c(args, list(want_scores = TRUE, pois1 = TRUE)))
  rC <- do.call(rpbnb:::bnbr_rp_copula_ll_grad_cpp,
                c(args, list(want_scores = TRUE, pois1 = TRUE)))
  expect_equal(as.numeric(rC), as.numeric(rR), tolerance = 1e-8)
  expect_equal(attr(rC, "gradient"), attr(rR, "gradient"), tolerance = 1e-6)
  expect_equal(attr(rC, "scores"),   attr(rR, "scores"),   tolerance = 1e-6)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: FAIL — the C++ core computes `pnbinom(size = Inf)` corners → NaN, so `rC` value is `-Inf` / gradient mismatched.

- [ ] **Step 3: Detect the Poisson flags in the C++ core**

In `src/copula_parallel.cpp`, immediately after the parameters `r1`, `r2` are in scope in the main entry function (near line 172-177, before Pass 1), add:

```cpp
  const bool pois1 = !R_finite(r1);
  const bool pois2 = !R_finite(r2);
```

- [ ] **Step 4: Branch the CDF corners (Pass 1 and Pass 2)**

Replace the four-corner blocks at ~L202-205 and ~L244-247 (identical in both passes) with a Poisson branch. For Pass 1:

```cpp
      double a  = pois1 ? R::ppois(py1[i], m1i, 1, 0)   : R::pnbinom(py1[i], r1, r1/(r1+m1i), 1, 0);
      double am = (py1[i]>0) ? (pois1 ? R::ppois(py1[i]-1, m1i, 1, 0) : R::pnbinom(py1[i]-1, r1, r1/(r1+m1i), 1, 0)) : 0.0;
      double b  = pois2 ? R::ppois(py2[i], m2i, 1, 0)   : R::pnbinom(py2[i], r2, r2/(r2+m2i), 1, 0);
      double bm = (py2[i]>0) ? (pois2 ? R::ppois(py2[i]-1, m2i, 1, 0) : R::pnbinom(py2[i]-1, r2, r2/(r2+m2i), 1, 0)) : 0.0;
```

Apply the same replacement to the Pass-2 corner block (~L244-247).

- [ ] **Step 5: Branch the mu-score (Pass 2)**

Replace the mu-score lines at ~L267-273:

```cpp
        double da_dmu1 = pois1 ? -(py1[i]+1.0)*R::dpois(py1[i]+1.0, m1i, 0)/m1i
                               : -(py1[i]+1.0)*R::dnbinom_mu(py1[i]+1.0, r1, m1i, 0)/m1i;
        double dam_dmu1= (py1[i]>0)? (pois1 ? -py1[i]*R::dpois(py1[i], m1i, 0)/m1i
                                            : -py1[i]*R::dnbinom_mu(py1[i], r1, m1i, 0)/m1i) : 0.0;
        double db_dmu2 = pois2 ? -(py2[i]+1.0)*R::dpois(py2[i]+1.0, m2i, 0)/m2i
                               : -(py2[i]+1.0)*R::dnbinom_mu(py2[i]+1.0, r2, m2i, 0)/m2i;
        double dbm_dmu2= (py2[i]>0)? (pois2 ? -py2[i]*R::dpois(py2[i], m2i, 0)/m2i
                                            : -py2[i]*R::dnbinom_mu(py2[i], r2, m2i, 0)/m2i) : 0.0;
```

- [ ] **Step 6: Zero the dispersion score for a Poisson margin (Pass 2)**

Replace the `s_logm1`/`s_logm2` computations at ~L277-282:

```cpp
        double s_logm1;
        if (pois1) { s_logm1 = 0.0; }
        else {
          double da_dr1 = dnb_cdf_dr((int)py1[i], m1i, r1);
          double dam_dr1= (py1[i]>0)? dnb_cdf_dr((int)py1[i]-1, m1i, r1):0.0;
          s_logm1 = (-r1)*(du_a*da_dr1 + du_am*dam_dr1)/p;
        }
        double s_logm2;
        if (pois2) { s_logm2 = 0.0; }
        else {
          double db_dr2 = dnb_cdf_dr((int)py2[i], m2i, r2);
          double dbm_dr2= (py2[i]>0)? dnb_cdf_dr((int)py2[i]-1, m2i, r2):0.0;
          s_logm2 = (-r2)*(dv_b*db_dr2 + dv_bm*dbm_dr2)/p;
        }
```

The existing `bad`-masking (L285-287) and the `gloc[im1] += wv*s_logm1; ...` / `psc[...] += wv*s_logm1` accumulation (L300, L311) then carry the zeros through unchanged.

- [ ] **Step 7: Run test to verify it passes (recompiles C++)**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: PASS (5 tests). `load_all()` recompiles `copula_parallel.cpp`.

- [ ] **Step 8: Commit**

```bash
git add src/copula_parallel.cpp tests/testthat/test-copula-poisson.R
git commit -m "feat(copula): exact Poisson m=0 branch in the C++ copula core"
```

---

### Task 4: `.fit_rpbnb_copula` — poisson / .fixed / .opt_draws support

Give the copula fit function the restricted-refit contract the Famoye `fit_rpbnb` already has: pin Poisson `log_m`, accept arbitrary fixed parameters, and reuse supplied optimization draws.

**Files:**
- Modify: `R/fit_rpbnb_copula.R`
- Test: `tests/testthat/test-copula-poisson.R` (append)

**Interfaces:**
- Consumes: `.resolve_start`, `opg_vcov`, `.observed_info_vcov`, `.free_index_vcov`, `halton_uniform`, `new_rpbnb_fit` (all existing).
- Produces: `.fit_rpbnb_copula(formula_1, formula_2, data, random_1, random_2, draws, draw_type, seed, start, control, family, poisson_1 = FALSE, poisson_2 = FALSE, .fixed = NULL, .opt_draws = NULL)` returning an `rpbnb_fit` with `poisson_1`/`poisson_2` stored and `npar` reduced by the number of fixed parameters.

- [ ] **Step 1: Write the failing test**

Append to `tests/testthat/test-copula-poisson.R`:

```r
test_that(".fit_rpbnb_copula(poisson_1=TRUE) pins log_m1 and drops it from npar", {
  set.seed(31)
  n <- 200
  d <- data.frame(x = rnorm(n))
  d$y1 <- rpois(n, exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(n, size = 2, mu = exp(0.1 - 0.1 * d$x))
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)

  full <- rpbnb:::.fit_rpbnb_copula(y1 ~ x, y2 ~ x, d, character(0), character(0),
    draws = 50, draw_type = "halton", seed = 1, start = NULL,
    control = ctrl, family = "frank")
  pois <- rpbnb:::.fit_rpbnb_copula(y1 ~ x, y2 ~ x, d, character(0), character(0),
    draws = 50, draw_type = "halton", seed = 1, start = NULL,
    control = ctrl, family = "frank", poisson_1 = TRUE)

  expect_equal(pois$npar, full$npar - 1L)      # one fewer free parameter
  expect_true(isTRUE(pois$poisson_1))          # flag stored
  expect_true(is.finite(as.numeric(logLik(pois))))
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: FAIL — `.fit_rpbnb_copula` has no `poisson_1` argument.

- [ ] **Step 3: Extend the signature and draw handling**

In `R/fit_rpbnb_copula.R`, change the signature:

```r
.fit_rpbnb_copula <- function(formula_1, formula_2, data,
                              random_1, random_2, draws, draw_type,
                              seed, start, control, family,
                              poisson_1 = FALSE, poisson_2 = FALSE,
                              .fixed = NULL, .opt_draws = NULL) {
```

Replace the draw-generation block (currently `set.seed(seed); if ((q1+q2) > 0) { Z <- halton_uniform(...) ... }`) so it honours `.opt_draws`:

```r
  if (!is.null(.opt_draws)) {
    Z1 <- .opt_draws$Z1; Z2 <- .opt_draws$Z2
  } else {
    set.seed(seed)
    if ((q1 + q2) > 0) {
      Z  <- halton_uniform(draws, q1 + q2, burn = control$halton_burn)
      Z1 <- if (q1 > 0) Z[, 1:q1, drop = FALSE] else matrix(0, draws, 0)
      Z2 <- if (q2 > 0) Z[, (q1 + 1):(q1 + q2), drop = FALSE] else matrix(0, draws, 0)
    } else {
      Z1 <- matrix(0, 1, 0); Z2 <- matrix(0, 1, 0)
    }
  }
```

- [ ] **Step 4: Build `fixed_names`, pin starts, pass Poisson flags to `ll_fun`**

After `start` is resolved (after the `.resolve_start(...)` call), add:

```r
  fixed_names <- c(if (isTRUE(poisson_1)) "log_m1", if (isTRUE(poisson_2)) "log_m2")
  if (length(fixed_names)) start[fixed_names] <- log(POISSON_M)
  if (!is.null(.fixed)) {
    if (is.null(names(.fixed)) || any(!nzchar(names(.fixed))))
      stop("`.fixed` must be a fully named numeric vector.", call. = FALSE)
    unknown <- setdiff(names(.fixed), par_names)
    if (length(unknown))
      stop("`.fixed` names not in the model: ", paste(unknown, collapse = ", "), call. = FALSE)
    start[names(.fixed)] <- .fixed
    fixed_names <- union(fixed_names, names(.fixed))
  }
  free <- !(par_names %in% fixed_names)
```

Change the `ll_fun` body to pass the flags to both branches:

```r
    v <- if (use_cpp)
      bnbr_rp_copula_ll_grad_cpp(p, Y1, Y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                                 Z1, Z2, family, dist1, dist2, sign1, sign2,
                                 n_threads = cpp_threads, pois1 = poisson_1, pois2 = poisson_2)
    else
      bnbr_rp_copula_ll_grad(p, Y1, Y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                             Z1, Z2, family, dist1, dist2, sign1, sign2,
                             pois1 = poisson_1, pois2 = poisson_2)
```

- [ ] **Step 5: Pass `fixed=` to maxLik and reduce npar**

Change the `maxLik::maxLik(...)` call to add `fixed = if (length(fixed_names)) fixed_names else NULL`, and change the npar line:

```r
  fit <- maxLik::maxLik(logLik = ll_fun, start = start, method = "BFGS",
                        control = list(iterlim = control$iterlim,
                                       reltol = control$reltol,
                                       printLevel = control$print_level),
                        fixed = if (length(fixed_names)) fixed_names else NULL)
  par_hat <- stats::coef(fit); names(par_hat) <- par_names
  npar <- length(par_hat) - length(fixed_names)
```

- [ ] **Step 6: Pass Poisson flags to the SE scorers and use free-index vcov**

In the `compute_se` block, pass `pois1`/`pois2` to the `want_scores`/numeric scorers, and use the free-index variants for fixed parameters. Replace the OPG and numeric branches:

```r
    } else if (identical(se_method, "opg")) {
      res <- if (use_cpp)
        bnbr_rp_copula_ll_grad_cpp(par_hat, Y1, Y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                                   Z1, Z2, family, dist1, dist2, sign1, sign2,
                                   want_scores = TRUE, n_threads = cpp_threads,
                                   pois1 = poisson_1, pois2 = poisson_2)
      else
        bnbr_rp_copula_ll_grad(par_hat, Y1, Y2, X1, X2, XR1, XR2,
                               rand_idx1, rand_idx2, Z1, Z2, family,
                               dist1, dist2, sign1, sign2, want_scores = TRUE,
                               pois1 = poisson_1, pois2 = poisson_2)
      inv <- .free_index_vcov(crossprod(attr(res, "scores")), par_names, free,
                              label = "copula RP-BNB (OPG)")
      vc <- inv$vcov; se <- inv$se; hdiag <- inv$diag
    } else {  # "numeric"
      H <- numDeriv::hessian(function(p) bnbr_rp_copula_ll(p, Y1, Y2, X1, X2, XR1, XR2,
                             rand_idx1, rand_idx2, Z1, Z2, family, dist1, dist2, sign1, sign2,
                             pois1 = poisson_1, pois2 = poisson_2),
                             par_hat,
                             method.args = list(r = control$hess_r, eps = control$hess_eps))
      inv <- .free_index_vcov(-H, par_names, free,
                              label = paste0(family, " copula RP-BNB (numeric Hessian)"))
      vc <- inv$vcov; se <- inv$se; hdiag <- inv$diag
    }
```

- [ ] **Step 7: Store the Poisson flags on the fit**

Change the `new_rpbnb_fit(...)` call to pass the flags (add as the final arguments):

```r
    rp_meta = list(dist1 = dist1, dist2 = dist2, sign1 = sign1, sign2 = sign2,
                   Z1 = Z1, Z2 = Z2),
    poisson_1 = poisson_1, poisson_2 = poisson_2
  )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: PASS (6 tests).

- [ ] **Step 9: Commit**

```bash
git add R/fit_rpbnb_copula.R tests/testthat/test-copula-poisson.R
git commit -m "feat(copula): poisson/.fixed/.opt_draws support in .fit_rpbnb_copula"
```

---

### Task 5: Router — forward the copula restricted-refit args

Remove the hard block and forward the new arguments from `fit_rpbnb()` into `.fit_rpbnb_copula()`.

**Files:**
- Modify: `R/fit_rpbnb.R:121-129`
- Test: `tests/testthat/test-copula-poisson.R` (append)

**Interfaces:**
- Consumes: `.fit_rpbnb_copula(..., poisson_1, poisson_2, .fixed, .opt_draws)` (Task 4).
- Produces: `fit_rpbnb(..., dependence = copula(...), poisson_1 = TRUE)` returns a Poisson-restricted copula fit instead of erroring.

- [ ] **Step 1: Write the failing test**

Append to `tests/testthat/test-copula-poisson.R`:

```r
test_that("fit_rpbnb copula path accepts poisson_1 via the public API", {
  set.seed(41)
  n <- 200
  d <- data.frame(x = rnorm(n))
  d$y1 <- rpois(n, exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(n, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fit <- fit_rpbnb(y1 ~ x, y2 ~ x, data = d, dependence = copula("frank"),
                   draws = 50, seed = 1, poisson_1 = TRUE,
                   control = rpbnb_control(print_level = 0, compute_se = FALSE))
  expect_true(isTRUE(fit$poisson_1))
  expect_false(is.null(fit$cop_family))
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: FAIL — the current block stops with "poisson_1 / poisson_2 ... not supported with a copula() dependence."

- [ ] **Step 3: Replace the block with forwarding**

In `R/fit_rpbnb.R`, replace lines 121-129:

```r
  if (inherits(dependence, "rpbnb_copula")) {
    return(.fit_rpbnb_copula(formula_1, formula_2, data, random_1, random_2,
                             draws, draw_type, seed, start, control,
                             family = dependence$family,
                             poisson_1 = poisson_1, poisson_2 = poisson_2,
                             .fixed = .fixed, .opt_draws = .opt_draws))
  }
```

- [ ] **Step 4: Update the roxygen note on `poisson_1,poisson_2`**

In `R/fit_rpbnb.R`, the `@param poisson_1,poisson_2` block currently ends by describing Famoye only. Replace the final sentence so it no longer says copula is unsupported:

```r
#'   fixed-dispersion approximation. Supported with both Famoye/Sarmanov and
#'   [copula()] dependence (the copula path uses the same exact m = 0 branch).
```

- [ ] **Step 5: Run test to verify it passes**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add R/fit_rpbnb.R tests/testthat/test-copula-poisson.R
git commit -m "feat(copula): forward poisson/.fixed/.opt_draws from fit_rpbnb to the copula path"
```

---

### Task 6: `rpbnb_boundary_tests` — copula support

Remove the copula guard and make the restricted refits use the fit's actual dependence family.

**Files:**
- Modify: `R/boundary_tests.R:90-94`, `:120-128` (the `refit` closure), roxygen `@param fit`
- Test: `tests/testthat/test-copula-poisson.R` (append, guarded by `skip_slow()`)

**Interfaces:**
- Consumes: `fit_rpbnb(..., dependence = copula(family), poisson_1/2, .fixed, .opt_draws)` (Task 5); `copula()`.
- Produces: `rpbnb_boundary_tests(copula_fit, data)` returns an `rpbnb_boundary_tests` table with SD and dispersion rows.

- [ ] **Step 1: Write the failing test**

Append to `tests/testthat/test-copula-poisson.R`:

```r
test_that("rpbnb_boundary_tests supports a copula fit (sd + dispersion)", {
  skip_slow()
  set.seed(51)
  n <- 400
  d <- data.frame(x = rnorm(n))
  d$y1 <- rpois(n, exp(0.3 + 0.4 * d$x))
  d$y2 <- rnbinom(n, size = 1.5, mu = exp(0.1 - 0.2 * d$x))
  fit <- fit_rpbnb(y1 ~ x, y2 ~ x, data = d, random_1 = "x",
                   dependence = copula("frank"), draws = 100, seed = 1,
                   control = rpbnb_control(print_level = 0, se_method = "opg"))
  bt <- rpbnb_boundary_tests(fit, d)
  expect_s3_class(bt, "rpbnb_boundary_tests")
  expect_true("sd1:x" %in% bt$Parameter)      # SD test row present
  expect_true(all(c("m1", "m2") %in% bt$Parameter))  # dispersion rows present
  expect_true(all(is.finite(bt$p.value)))
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `Sys.setenv(RPBNB_RUN_SLOW = "1")` then `testthat::test_file(...)`:
`"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "Sys.setenv(RPBNB_RUN_SLOW='1'); suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: FAIL — `rpbnb_boundary_tests()` stops with "supports famoye fits only".

- [ ] **Step 3: Remove the copula guard**

In `R/boundary_tests.R`, delete the block at lines 90-94:

```r
  if (!is.null(fit$cop_family)) {
    stop("rpbnb_boundary_tests() supports famoye fits only; copula fits are ",
         "not supported (Poisson-limit margins are unavailable there).",
         call. = FALSE)
  }
```

- [ ] **Step 4: Make `refit()` use the fit's actual dependence**

In the `refit` closure, replace the hard-coded `dependence = "famoye"`:

```r
  full_dep <- if (!is.null(fit$cop_family)) copula(fit$cop_family) else "famoye"
  refit <- function(poisson_1 = FALSE, poisson_2 = FALSE, fixed = NULL,
                    opt_draws = full_draws) {
    fit_rpbnb(fit$formula_1, fit$formula_2, data = data,
              random_1 = full1, random_2 = full2,
              draws = fit$draws, draw_type = fit$draw_type, seed = fit$seed,
              start = fit$coef, control = control, dependence = full_dep,
              poisson_1 = poisson_1, poisson_2 = poisson_2,
              .fixed = fixed, .opt_draws = opt_draws)
  }
```

- [ ] **Step 5: Update the roxygen `@param fit`**

Replace the `@param fit` line (currently "A famoye `rpbnb_fit` ... Copula fits are not supported ..."):

```r
#' @param fit A converged `rpbnb_fit` (the full model), from a Famoye or a
#'   [copula()] dependence. Both paths use the exact `m = 0` branch for the
#'   dispersion tests.
```

- [ ] **Step 6: Run test to verify it passes**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "Sys.setenv(RPBNB_RUN_SLOW='1'); suppressMessages(devtools::load_all('.')); testthat::test_file('tests/testthat/test-copula-poisson.R')"`
Expected: PASS (8 tests).

- [ ] **Step 7: Regenerate docs and commit**

```bash
"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(roxygen2::roxygenise('.'))"
git add R/boundary_tests.R man/rpbnb_boundary_tests.Rd man/fit_rpbnb.Rd tests/testthat/test-copula-poisson.R
git commit -m "feat(copula): rpbnb_boundary_tests() supports copula fits (sd + dispersion)"
```

---

### Task 7: Demo verification and full-suite regression

Confirm the demo section runs and nothing else regressed.

**Files:**
- Verify: `inst/fit_rpbnb_diff_copula.R` (no change expected)
- Verify: full test suite

**Interfaces:**
- Consumes: everything above.
- Produces: a green full suite and a working demo.

- [ ] **Step 1: Run the full fast suite**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); r <- as.data.frame(testthat::test_dir('tests/testthat', reporter='silent', stop_on_failure=FALSE)); cat('failed:', sum(r[['failed']]), ' passed:', sum(r[['passed']]), '\n')"`
Expected: `failed: 0`.

- [ ] **Step 2: Run the boundary + copula slow tier**

Run: `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "Sys.setenv(RPBNB_RUN_SLOW='1'); suppressMessages(devtools::load_all('.')); r <- as.data.frame(testthat::test_dir('tests/testthat', filter='copula-poisson|boundary-tests', reporter='silent', stop_on_failure=FALSE)); cat('failed:', sum(r[['failed']]), ' passed:', sum(r[['passed']]), '\n')"`
Expected: `failed: 0`.

- [ ] **Step 3: Smoke-run the demo's boundary section on a small fit**

Run this inline smoke (mirrors the demo, small/fast):

```r
"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "suppressMessages(devtools::load_all('.')); set.seed(7); n<-300; d<-data.frame(x=rnorm(n)); d\$y1<-rpois(n, exp(0.3+0.3*d\$x)); d\$y2<-rnbinom(n, size=1.5, mu=exp(0.1-0.2*d\$x)); fit<-fit_rpbnb(y1~x, y2~x, data=d, random_1='x', dependence=copula('normal'), draws=80, seed=1, control=rpbnb_control(print_level=0, se_method='opg')); print(rpbnb_boundary_tests(fit, d)); cat('DEMO-OK\n')"
```

Expected: a printed boundary-test table with `sd1:x`, `m1`, `m2` rows and `DEMO-OK`.

- [ ] **Step 4: Commit any doc/regen leftovers**

```bash
git status --short
# if only expected files changed:
git add -A && git commit -m "test(copula): full-suite + demo verification for copula boundary tests"
```

(If `git status` is clean, skip the commit.)

---

## Notes for the implementer

- **Do not** touch the fixed-coefficient copula path (`copula_loglik_vec` / `copula_grad_vec` in `R/copula_likelihood.R`) — it is out of scope and keeps its own Poisson block.
- **Residuals** on a Poisson-restricted copula fit are not wired; boundary tests never compute residuals on the transient restricted fits, so this is a known follow-up, not a blocker.
- If a C++ compile error appears, read the `load_all()` output — `copula_parallel.cpp` uses `R::ppois(x, lambda, lower_tail, log)` and `R::dpois(x, lambda, log)` (Rmath signatures).
