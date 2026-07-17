# RP Interpretation Multithreading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the delta-method standard-error computation in `rpbnb_marginal_effects()` and `rpbnb_elasticities()` across an R-level cluster via a new `n_cores` parameter, with results numerically identical to the sequential path.

**Architecture:** `numDeriv::jacobian()`'s single call is replaced by an exact per-parameter-column decomposition (`.rp_jac_col()`), dispatched via `parallel::parLapply()` across a `parallel::makeCluster()` PSOCK cluster built once per top-level call and reused across every equation. `.rp_diag_one()` gains a `cl = NULL` parameter (default preserves today's exact sequential code path); a new `.rp_make_cluster()` helper centralizes cluster creation, the required `clusterExport()` call, and the `parallel`-unavailable fallback.

**Tech Stack:** R (>= 4.1), base `parallel` package (already a base-R package, no new dependency), `numDeriv` (already in Imports), roxygen2 docs, testthat 3e.

## Global Constraints

- `n_cores = 1L` (default) on both exported functions MUST leave the existing sequential code path in `.rp_diag_one()` completely unchanged — zero behavior change for every existing caller.
- Worker-dispatched closures MUST use BARE (unqualified) calls to internal helpers (`.rp_estimand`, `.rp_g_matrix`, `.rp_inf_rows`, `rand_realize`, `rand_dist_registry`), never `rpbnb:::`-qualified calls. This project's dev workflow loads the package via `pkgload::load_all()`, not a formal install; a `parallel::makeCluster()` PSOCK worker has no access to that in-memory namespace, so `rpbnb:::foo(...)` fails on a worker. A bare call resolves via `parallel::clusterExport()` populating the worker's `.GlobalEnv` with the required names. This was verified empirically (four separate probes, including the exact realistic case: a namespace-scoped closure passed directly as `parLapply`'s `FUN` argument) — see the design spec's "Worker resolution" section for the full account. **Do not deviate from the bare-call + clusterExport-by-name pattern**, and do not "simplify" by using `rpbnb:::` inside any function that will run on a worker.
- The parallel per-parameter-column jacobian decomposition MUST be mathematically identical to `numDeriv::jacobian()`'s single-call result. This was verified bit-for-bit (`max abs diff = 0`) in the design phase. Tests in this plan assert **exact** equality (`tolerance = 0`), not approximate closeness — if any test needs a nonzero tolerance to pass, that indicates a real implementation bug, not a numerical-noise issue to paper over.
- `clusterExport()` calls MUST use `envir = environment()` from *within* the exporting function (mirrors the exact working pattern already used in `R/fit_rpbnb.R`'s `clusterExport(cl, c(...), envir = environment())` calls).
- New parameter `n_cores` goes in trailing position on both `rpbnb_marginal_effects()` and `rpbnb_elasticities()` signatures (after `print_output`), so it does not disturb any existing positional call site.
- Use PSOCK clusters (`parallel::makeCluster()`'s default type) — FORK clusters do not work on Windows, and this project's `fit_rpbnb.R` R-fallback path already exclusively uses PSOCK for the same reason.
- Windows dev commands (adjust the R version path if it changes):
  - Run one test file: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "pkgload::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-interpretation.R')"`
  - Regenerate docs (only needed when a roxygen `@param`/`@export` block changes): `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "roxygen2::roxygenise()"`
  - Fast tier: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" tools/test-tiers.R fast`
- **Committing:** stage and commit only the files each task's step explicitly names. Before running `git commit`, run an UNFILTERED `git status --short` and confirm the staged set matches exactly what the step lists — this repository routinely has a large number of unrelated pre-existing staged/unstaged changes from other in-progress work, and a broad `git add`/unchecked commit will sweep them in.

---

## File Structure

- **Modify** `R/diagnostics.R` — add `.rp_jac_col()` (new internal helper), add `cl = NULL` to `.rp_diag_one()`'s signature with a parallel branch for the SE jacobian, add `.rp_make_cluster()` (new internal helper), add `n_cores = 1L` to both `rpbnb_marginal_effects()` and `rpbnb_elasticities()`.
- **Modify** `tests/testthat/test-rpbnb-interpretation.R` — add an internal-mechanism exact-equivalence test (Task 1) and two public-API exact-equivalence tests (Task 2), inserted before the existing slow end-to-end test at the end of the file.
- **Modify** `inst/fit_rpbnb_complex.R`, `inst/fit_rpbnb_copula_complex.R` — pass `n_cores = rpbnb_threads()` to the Step 7 `rpbnb_marginal_effects()`/`rpbnb_elasticities()` calls added in the prior feature.
- **Modify** `NEWS.md` — changelog entry.
- **Modify** `NAMESPACE`, `man/rpbnb_marginal_effects.Rd`, `man/rpbnb_elasticities.Rd` — regenerated by roxygen2 (documents the new `n_cores` param; no new exports).

---

## Task 1: Internal mechanism — `.rp_jac_col()` and the `cl` parameter on `.rp_diag_one()`

**Files:**
- Modify: `R/diagnostics.R` (the `.rp_diag_one()` function and its preceding comment, currently around lines 575-609; the `.rp_estimand()` function that precedes it is unchanged)
- Test: `tests/testthat/test-rpbnb-interpretation.R` (insert a new test before the final slow-gated test)

**Interfaces:**
- Consumes: `.rp_estimand(theta, meta, quantity, mark_inf)` (existing, unchanged), `numDeriv::jacobian`, `parallel::parLapply`/`makeCluster`/`clusterExport`/`stopCluster`, `make_rp_fixture()` (test fixture, from `tests/testthat/helper-slow.R`).
- Produces: `.rp_jac_col(k, theta_hat, theta_names, meta, quantity)` returning an `nvars x 1` matrix (one column of the delta-method jacobian). `.rp_diag_one(fit, eq, quantity, type, vars, include_intercept, digits, print_output, resp_name, cl = NULL)` — same return shape as before (a data frame), with a new trailing `cl` parameter that later tasks (and later calls) can pass a `parallel` cluster object into.

- [ ] **Step 1: Write the failing test**

Open `tests/testthat/test-rpbnb-interpretation.R`. Find this existing test near the end of the file:

```r
test_that("type='MEM' runs on both functions and returns finite results", {
```

Insert a new test immediately AFTER that test's closing `})` and BEFORE the final test in the file (`test_that("interpretation runs end-to-end on a real fit (slow)", {`):

```r
test_that(".rp_diag_one: parallel (cl=<cluster>) SEs match sequential (cl=NULL) exactly", {
  skip_if_not_installed("parallel")
  f <- make_rp_fixture("normal")

  seq_tab <- rpbnb:::.rp_diag_one(f, 1L, "me", "AME", NULL, FALSE, 4, FALSE, "y1")

  cl <- parallel::makeCluster(2)
  on.exit(parallel::stopCluster(cl))
  parallel::clusterExport(cl,
    c(".rp_estimand", ".rp_g_matrix", ".rp_inf_rows", "rand_realize", "rand_dist_registry"),
    envir = asNamespace("rpbnb"))
  par_tab <- rpbnb:::.rp_diag_one(f, 1L, "me", "AME", NULL, FALSE, 4, FALSE, "y1", cl = cl)

  expect_equal(par_tab$Estimate, seq_tab$Estimate, tolerance = 0)
  expect_equal(par_tab$StdErr, seq_tab$StdErr, tolerance = 0)
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "pkgload::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-interpretation.R')"`

Expected: FAIL — `unused argument (cl = cl)` (`.rp_diag_one()` does not yet accept a `cl` parameter).

- [ ] **Step 3: Implement `.rp_jac_col()` and the `cl` branch in `.rp_diag_one()`**

In `R/diagnostics.R`, find this exact block (the comment, then the full `.rp_diag_one` function through the closing brace of its `if (any(!is.finite(V))) { ... } else { ... }` block):

```r
# Compute point estimates + delta-method SEs for one equation and assemble the
# tidy output frame (Name/Estimate/StdErr/z/p/Signif/var_type).
.rp_diag_one <- function(fit, eq, quantity, type, vars, include_intercept,
                         digits, print_output, resp_name) {
  meta <- .rp_diag_meta(fit, eq, type, vars, include_intercept)

  est <- .rp_estimand(fit$coef, meta, quantity, mark_inf = TRUE)

  # Warn once if any reported estimate is a lognormal analytic infinity.
  if (any(!is.finite(est))) {
    warning(sum(!is.finite(est)), " selected variable(s) in ", resp_name,
            " have an analytically infinite estimate (a lognormal random ",
            "coefficient with sign * covariate > 0); reporting Inf.",
            call. = FALSE)
  }

  theta_names <- c(meta$b_names, meta$scale_names)
  V <- fit$vcov[theta_names, theta_names, drop = FALSE]
  if (any(!is.finite(V))) {
    warning("vcov is unavailable (fit made with compute_se = FALSE?); ",
            "standard errors set to NA for ", resp_name, ".", call. = FALSE)
    se <- rep(NA_real_, length(est))
  } else {
    theta_hat <- fit$coef[theta_names]
    G <- numDeriv::jacobian(
      function(t) { names(t) <- theta_names
                    .rp_estimand(t, meta, quantity, mark_inf = FALSE) },
      theta_hat)
    se <- vapply(seq_len(nrow(G)), function(m) {
      g <- G[m, ]
      sqrt(as.numeric(t(g) %*% V %*% g))
    }, numeric(1))
    # A non-finite point estimate (analytic Inf) has no meaningful SE.
    se[!is.finite(est)] <- NA_real_
  }
```

Replace it with:

```r
# One column of the delta-method jacobian: the Richardson-extrapolated
# derivative of .rp_estimand() with respect to theta_hat[[k]], holding every
# other parameter fixed. Mathematically identical to column k of
# numDeriv::jacobian(f_vector, theta_hat) -- verified bit-for-bit against it
# (max abs diff 0) -- splitting the jacobian into these independent
# per-parameter calls is what makes the standard-error step embarrassingly
# parallel across a cluster.
#
# Uses a BARE (unqualified) call to .rp_estimand, NOT rpbnb:::.rp_estimand:
# on a parallel::makeCluster() worker the rpbnb namespace is not loaded (this
# project's dev workflow runs off pkgload::load_all(), not an installed
# package), so a `:::`-qualified call fails on a worker. A bare call resolves
# via ordinary lexical/global-env lookup, which parallel::clusterExport()
# (see .rp_make_cluster()) satisfies by placing .rp_estimand and its own
# transitive callees into each worker's .GlobalEnv.
.rp_jac_col <- function(k, theta_hat, theta_names, meta, quantity) {
  fk <- function(tk) {
    t <- theta_hat; t[[k]] <- tk
    names(t) <- theta_names
    .rp_estimand(t, meta, quantity, mark_inf = FALSE)
  }
  numDeriv::jacobian(fk, theta_hat[[k]])
}

# Compute point estimates + delta-method SEs for one equation and assemble the
# tidy output frame (Name/Estimate/StdErr/z/p/Signif/var_type). `cl` is an
# optional parallel cluster (see .rp_make_cluster()); when non-NULL, the SE
# jacobian is computed as independent per-parameter columns dispatched across
# the cluster instead of one sequential numDeriv::jacobian() call -- the two
# paths are numerically identical (see .rp_jac_col()).
.rp_diag_one <- function(fit, eq, quantity, type, vars, include_intercept,
                         digits, print_output, resp_name, cl = NULL) {
  meta <- .rp_diag_meta(fit, eq, type, vars, include_intercept)

  est <- .rp_estimand(fit$coef, meta, quantity, mark_inf = TRUE)

  # Warn once if any reported estimate is a lognormal analytic infinity.
  if (any(!is.finite(est))) {
    warning(sum(!is.finite(est)), " selected variable(s) in ", resp_name,
            " have an analytically infinite estimate (a lognormal random ",
            "coefficient with sign * covariate > 0); reporting Inf.",
            call. = FALSE)
  }

  theta_names <- c(meta$b_names, meta$scale_names)
  V <- fit$vcov[theta_names, theta_names, drop = FALSE]
  if (any(!is.finite(V))) {
    warning("vcov is unavailable (fit made with compute_se = FALSE?); ",
            "standard errors set to NA for ", resp_name, ".", call. = FALSE)
    se <- rep(NA_real_, length(est))
  } else {
    theta_hat <- fit$coef[theta_names]
    G <- if (!is.null(cl)) {
      do.call(cbind, parallel::parLapply(cl, seq_along(theta_hat), .rp_jac_col,
                                         theta_hat = theta_hat, theta_names = theta_names,
                                         meta = meta, quantity = quantity))
    } else {
      numDeriv::jacobian(
        function(t) { names(t) <- theta_names
                      .rp_estimand(t, meta, quantity, mark_inf = FALSE) },
        theta_hat)
    }
    se <- vapply(seq_len(nrow(G)), function(m) {
      g <- G[m, ]
      sqrt(as.numeric(t(g) %*% V %*% g))
    }, numeric(1))
    # A non-finite point estimate (analytic Inf) has no meaningful SE.
    se[!is.finite(est)] <- NA_real_
  }
```

Do not modify anything below this block (the `tab <- .bnb_me_tidy(...)` line and everything after it in `.rp_diag_one` stays exactly as-is).

- [ ] **Step 4: Run the test to verify it passes**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "pkgload::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-interpretation.R')"`

Expected: all tests PASS, including the new `.rp_diag_one: parallel (cl=<cluster>) SEs match sequential (cl=NULL) exactly` test.

- [ ] **Step 5: Run the full existing test file once more to confirm nothing else regressed**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "pkgload::load_all(quiet=TRUE); r<-testthat::test_file('tests/testthat/test-rpbnb-interpretation.R', reporter='minimal'); d<-as.data.frame(r); cat('PASS=',sum(d\`$\`passed),' FAIL=',sum(d\`$\`failed),' SKIP=',sum(d\`$\`skipped),'\n', sep='')"`

Expected: `FAIL=0` (skip count should be 1, the slow end-to-end test).

- [ ] **Step 6: Commit**

Run `git status --short` first and confirm ONLY these two files show as modified before staging:

```bash
git status --short -- R/diagnostics.R tests/testthat/test-rpbnb-interpretation.R
git add R/diagnostics.R tests/testthat/test-rpbnb-interpretation.R
git commit -m "feat: parallel-ready jacobian decomposition for RP interpretation SEs"
```

---

## Task 2: Public API — `n_cores` on `rpbnb_marginal_effects()` and `rpbnb_elasticities()`

**Files:**
- Modify: `R/diagnostics.R` (add `.rp_make_cluster()`; modify both exported functions' signatures, roxygen blocks, and bodies)
- Test: `tests/testthat/test-rpbnb-interpretation.R` (insert two new tests, after Task 1's test and before the final slow-gated test)

**Interfaces:**
- Consumes: `.rp_diag_one(..., cl = NULL)` (Task 1), `parallel::makeCluster`/`clusterExport`/`stopCluster`, `requireNamespace`.
- Produces: `.rp_make_cluster(n_cores)` returning `NULL` (n_cores <= 1, or `parallel` unavailable) or a ready-to-use exported cluster object. `rpbnb_marginal_effects(fit, which, type, vars, include_intercept, digits, print_output, n_cores = 1L)` and `rpbnb_elasticities(fit, which, type, vars, include_intercept, digits, print_output, n_cores = 1L)` — same return shapes as before, with the new trailing `n_cores` parameter.

- [ ] **Step 1: Write the failing tests**

In `tests/testthat/test-rpbnb-interpretation.R`, insert these two tests immediately AFTER the `.rp_diag_one: parallel (cl=<cluster>) SEs match sequential (cl=NULL) exactly` test added in Task 1, and BEFORE the final slow-gated test:

```r
test_that("rpbnb_marginal_effects: n_cores > 1 matches n_cores = 1 exactly", {
  skip_if_not_installed("parallel")
  f <- make_rp_fixture("normal")
  seq_res <- rpbnb_marginal_effects(f, which = "both", type = "AME",
                                    print_output = FALSE, n_cores = 1L)
  par_res <- rpbnb_marginal_effects(f, which = "both", type = "AME",
                                    print_output = FALSE, n_cores = 3L)
  expect_equal(par_res$y1$Estimate, seq_res$y1$Estimate, tolerance = 0)
  expect_equal(par_res$y1$StdErr,   seq_res$y1$StdErr,   tolerance = 0)
  expect_equal(par_res$y2$Estimate, seq_res$y2$Estimate, tolerance = 0)
  expect_equal(par_res$y2$StdErr,   seq_res$y2$StdErr,   tolerance = 0)
})

test_that("rpbnb_elasticities: n_cores > 1 matches n_cores = 1 exactly", {
  skip_if_not_installed("parallel")
  f <- make_rp_fixture("normal")
  seq_res <- rpbnb_elasticities(f, which = "both", type = "AME",
                                print_output = FALSE, n_cores = 1L)
  par_res <- rpbnb_elasticities(f, which = "both", type = "AME",
                                print_output = FALSE, n_cores = 3L)
  expect_equal(par_res$y1$Estimate, seq_res$y1$Estimate, tolerance = 0)
  expect_equal(par_res$y1$StdErr,   seq_res$y1$StdErr,   tolerance = 0)
  expect_equal(par_res$y2$Estimate, seq_res$y2$Estimate, tolerance = 0)
  expect_equal(par_res$y2$StdErr,   seq_res$y2$StdErr,   tolerance = 0)
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "pkgload::load_all(quiet=TRUE); testthat::test_file('tests/testthat/test-rpbnb-interpretation.R')"`

Expected: FAIL — `unused argument (n_cores = 1L)` (or `n_cores = 3L`) on both new tests.

- [ ] **Step 3: Implement `.rp_make_cluster()` and wire `n_cores` into both exported functions**

In `R/diagnostics.R`, find the closing brace of `.rp_diag_one` (the line that is just `}` immediately after the `if (print_output) { ... }` block and the final `tab` line — the LAST line of the `.rp_diag_one` function you edited in Task 1), followed by a blank line, followed by the roxygen block starting `#' Marginal effects for a random-parameter bivariate NB model`. Insert this new helper in that blank line, right before the roxygen block:

```r

# Build (if n_cores > 1) and export-configure a PSOCK cluster for the parallel
# delta-method jacobian in .rp_diag_one() (see .rp_jac_col()). Returns NULL for
# n_cores <= 1 (the caller then uses .rp_diag_one()'s unchanged sequential
# path) or when the `parallel` package is unavailable (with a warning,
# matching the fallback message used in fit_rpbnb.R). The caller owns the
# cluster's lifetime and must parallel::stopCluster() it when done (typically
# via on.exit()), and should reuse the SAME cluster across every equation
# computed in one call (y1/y2/both/all) rather than building one per equation.
.rp_make_cluster <- function(n_cores) {
  if (n_cores <= 1) return(NULL)
  if (!requireNamespace("parallel", quietly = TRUE)) {
    warning("Package 'parallel' not available; running sequentially.", call. = FALSE)
    return(NULL)
  }
  cl <- parallel::makeCluster(as.integer(n_cores))
  parallel::clusterExport(cl,
    c(".rp_estimand", ".rp_g_matrix", ".rp_inf_rows", "rand_realize", "rand_dist_registry"),
    envir = environment())
  cl
}
```

Next, find this exact block (the `rpbnb_marginal_effects` roxygen `@param`/`@return` tail and the full function body):

```r
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @return A data frame (single margin, invisibly) or a named list of data frames
#'   (`both`/`all`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
#'   `Signif`, `var_type`.
#' @seealso [bnb_marginal_effects()] for fixed-coefficient `bnb_fit` models.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 400,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 100)
#' rpbnb_marginal_effects(fit, which = "y1", type = "AME")
rpbnb_marginal_effects <- function(fit,
                                   which = c("y1", "y2", "both", "all"),
                                   type  = c("AME", "MEM"),
                                   vars  = NULL,
                                   include_intercept = FALSE,
                                   digits = 4,
                                   print_output = TRUE) {
  stopifnot(inherits(fit, "rpbnb_fit"))
  which <- match.arg(which)
  type  <- match.arg(type)
  if (which %in% c("both", "all")) {
    return(invisible(list(
      y1 = .rp_diag_one(fit, 1L, "me", type, vars, include_intercept,
                        digits, print_output, "y1"),
      y2 = .rp_diag_one(fit, 2L, "me", type, vars, include_intercept,
                        digits, print_output, "y2"))))
  }
  eq <- if (which == "y1") 1L else 2L
  invisible(.rp_diag_one(fit, eq, "me", type, vars, include_intercept,
                         digits, print_output, which))
}
```

Replace it with:

```r
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @param n_cores Number of worker processes for the delta-method standard-error
#'   jacobian (1 = sequential, the default). When `n_cores > 1`, the jacobian's
#'   independent per-parameter columns are dispatched across a
#'   `parallel::makeCluster()` cluster (one cluster per call, shared across
#'   every equation `which` computes); results are numerically identical to the
#'   sequential path. Falls back to sequential with a warning if the `parallel`
#'   package is unavailable.
#' @return A data frame (single margin, invisibly) or a named list of data frames
#'   (`both`/`all`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
#'   `Signif`, `var_type`.
#' @seealso [bnb_marginal_effects()] for fixed-coefficient `bnb_fit` models.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 400,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 100)
#' rpbnb_marginal_effects(fit, which = "y1", type = "AME")
rpbnb_marginal_effects <- function(fit,
                                   which = c("y1", "y2", "both", "all"),
                                   type  = c("AME", "MEM"),
                                   vars  = NULL,
                                   include_intercept = FALSE,
                                   digits = 4,
                                   print_output = TRUE,
                                   n_cores = 1L) {
  stopifnot(inherits(fit, "rpbnb_fit"))
  which <- match.arg(which)
  type  <- match.arg(type)
  cl <- .rp_make_cluster(n_cores)
  on.exit(if (!is.null(cl)) parallel::stopCluster(cl), add = TRUE)
  if (which %in% c("both", "all")) {
    return(invisible(list(
      y1 = .rp_diag_one(fit, 1L, "me", type, vars, include_intercept,
                        digits, print_output, "y1", cl = cl),
      y2 = .rp_diag_one(fit, 2L, "me", type, vars, include_intercept,
                        digits, print_output, "y2", cl = cl))))
  }
  eq <- if (which == "y1") 1L else 2L
  invisible(.rp_diag_one(fit, eq, "me", type, vars, include_intercept,
                         digits, print_output, which, cl = cl))
}
```

Next, find this exact block (the `rpbnb_elasticities` roxygen `@param`/`@return` tail and the full function body):

```r
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @return A data frame (single margin, invisibly) or a named list of data frames
#'   (`both`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
#'   `Signif`, `var_type`.
#' @seealso [bnb_elasticities()] for fixed-coefficient `bnb_fit` models.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 400,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 100)
#' rpbnb_elasticities(fit, which = "both", type = "AME")
rpbnb_elasticities <- function(fit,
                               which = c("y1", "y2", "both"),
                               type  = c("AME", "MEM"),
                               vars  = NULL,
                               include_intercept = FALSE,
                               digits = 4,
                               print_output = TRUE) {
  stopifnot(inherits(fit, "rpbnb_fit"))
  which <- match.arg(which)
  type  <- match.arg(type)
  if (which == "both") {
    return(invisible(list(
      y1 = .rp_diag_one(fit, 1L, "elas", type, vars, include_intercept,
                        digits, print_output, "y1"),
      y2 = .rp_diag_one(fit, 2L, "elas", type, vars, include_intercept,
                        digits, print_output, "y2"))))
  }
  eq <- if (which == "y1") 1L else 2L
  invisible(.rp_diag_one(fit, eq, "elas", type, vars, include_intercept,
                         digits, print_output, which))
}
```

Replace it with:

```r
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @param n_cores Number of worker processes for the delta-method standard-error
#'   jacobian (1 = sequential, the default). When `n_cores > 1`, the jacobian's
#'   independent per-parameter columns are dispatched across a
#'   `parallel::makeCluster()` cluster (one cluster per call, shared across
#'   every equation `which` computes); results are numerically identical to the
#'   sequential path. Falls back to sequential with a warning if the `parallel`
#'   package is unavailable.
#' @return A data frame (single margin, invisibly) or a named list of data frames
#'   (`both`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
#'   `Signif`, `var_type`.
#' @seealso [bnb_elasticities()] for fixed-coefficient `bnb_fit` models.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 400,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 100)
#' rpbnb_elasticities(fit, which = "both", type = "AME")
rpbnb_elasticities <- function(fit,
                               which = c("y1", "y2", "both"),
                               type  = c("AME", "MEM"),
                               vars  = NULL,
                               include_intercept = FALSE,
                               digits = 4,
                               print_output = TRUE,
                               n_cores = 1L) {
  stopifnot(inherits(fit, "rpbnb_fit"))
  which <- match.arg(which)
  type  <- match.arg(type)
  cl <- .rp_make_cluster(n_cores)
  on.exit(if (!is.null(cl)) parallel::stopCluster(cl), add = TRUE)
  if (which == "both") {
    return(invisible(list(
      y1 = .rp_diag_one(fit, 1L, "elas", type, vars, include_intercept,
                        digits, print_output, "y1", cl = cl),
      y2 = .rp_diag_one(fit, 2L, "elas", type, vars, include_intercept,
                        digits, print_output, "y2", cl = cl))))
  }
  eq <- if (which == "y1") 1L else 2L
  invisible(.rp_diag_one(fit, eq, "elas", type, vars, include_intercept,
                         digits, print_output, which, cl = cl))
}
```

- [ ] **Step 4: Regenerate docs**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "roxygen2::roxygenise()"`

Expected: updates `man/rpbnb_marginal_effects.Rd` and `man/rpbnb_elasticities.Rd` with the new `n_cores` parameter documentation. No new exports (`NAMESPACE` should be unchanged, since both functions were already exported). If `roxygenise()` incidentally touches unrelated `.Rd` files, do NOT stage those.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "pkgload::load_all(quiet=TRUE); r<-testthat::test_file('tests/testthat/test-rpbnb-interpretation.R', reporter='minimal'); d<-as.data.frame(r); cat('PASS=',sum(d\`$\`passed),' FAIL=',sum(d\`$\`failed),' SKIP=',sum(d\`$\`skipped),'\n', sep='')"`

Expected: `FAIL=0`, `SKIP=1` (only the slow end-to-end test skips).

- [ ] **Step 6: Commit**

Run `git status --short` first and confirm the staged set matches exactly this list (plus `man/rpbnb-package.Rd` or other `.Rd` files ONLY if `roxygenise()` actually changed them — verify with `git diff man/rpbnb-package.Rd` before adding it; if unchanged, leave it out):

```bash
git status --short -- R/diagnostics.R tests/testthat/test-rpbnb-interpretation.R NAMESPACE man/rpbnb_marginal_effects.Rd man/rpbnb_elasticities.Rd
git add R/diagnostics.R tests/testthat/test-rpbnb-interpretation.R NAMESPACE man/rpbnb_marginal_effects.Rd man/rpbnb_elasticities.Rd
git commit -m "feat: n_cores parameter for RP marginal effects and elasticities"
```

---

## Task 3: Update demo scripts to use `n_cores`

**Files:**
- Modify: `inst/fit_rpbnb_complex.R`
- Modify: `inst/fit_rpbnb_copula_complex.R`

**Interfaces:** none new (consumes `rpbnb_threads()`, already used elsewhere in both scripts, and the `n_cores` parameter added in Task 2).

- [ ] **Step 1: Update `inst/fit_rpbnb_complex.R`**

Find this exact block (the end of the file, Step 7):

```r
cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("AVERAGE MARGINAL EFFECTS (AME)\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
me <- rpbnb_marginal_effects(fit, which = "both", type = "AME")

cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("ELASTICITIES / SEMI-ELASTICITIES (AME)\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
el <- rpbnb_elasticities(fit, which = "both", type = "AME")
```

Replace it with:

```r
cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("AVERAGE MARGINAL EFFECTS (AME)\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
me <- rpbnb_marginal_effects(fit, which = "both", type = "AME",
                             n_cores = rpbnb_threads())

cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("ELASTICITIES / SEMI-ELASTICITIES (AME)\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
el <- rpbnb_elasticities(fit, which = "both", type = "AME",
                         n_cores = rpbnb_threads())
```

- [ ] **Step 2: Update `inst/fit_rpbnb_copula_complex.R`**

Find the identical block in this file (same text as Step 1's "find" block) and apply the identical replacement (same "replace" text as Step 1).

- [ ] **Step 3: Parse-check both files**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" -e "invisible(parse('inst/fit_rpbnb_complex.R')); invisible(parse('inst/fit_rpbnb_copula_complex.R')); cat('BOTH PARSE OK\n')"`

Expected: `BOTH PARSE OK`.

- [ ] **Step 4: Run the Famoye demo end-to-end**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" inst/fit_rpbnb_complex.R`

Expected: completes successfully (exit code 0) and the two new sections print `Marginal effects (RP integrated mean) for y1/y2 (AME)` and `Elasticities (RP integrated mean) for y1/y2 (AME)` tables with finite, statistically sensible estimates (same shape as the run already verified in the prior feature — this run additionally exercises `n_cores = rpbnb_threads()`, which on most dev machines is > 1). This may take a minute or two; if it hangs or errors, investigate before proceeding (a hang here would indicate the cluster is not shutting down correctly — check that `on.exit()` in `rpbnb_marginal_effects`/`rpbnb_elasticities` fires).

Do NOT run `inst/fit_rpbnb_copula_complex.R` end-to-end — its own header comment documents a 10-20 minute runtime; the `n_cores` mechanism is already covered by Task 1/2's tests and this task's parse-check plus the Famoye run.

- [ ] **Step 5: Commit**

Run `git status --short` first and confirm ONLY these two files show as modified before staging:

```bash
git status --short -- inst/fit_rpbnb_complex.R inst/fit_rpbnb_copula_complex.R
git add inst/fit_rpbnb_complex.R inst/fit_rpbnb_copula_complex.R
git commit -m "demo: use n_cores for RP marginal effects / elasticities SEs"
```

---

## Task 4: NEWS entry and full fast-tier regression

**Files:**
- Modify: `NEWS.md`

**Interfaces:** none (documentation + verification only).

- [ ] **Step 1: Add a NEWS entry**

In `NEWS.md`, find this exact block (the bullet added by the prior feature, currently the second bullet under the development header):

```markdown
* `rpbnb_marginal_effects()` and `rpbnb_elasticities()`: interpretation for
  random-parameter (`rpbnb_fit`) models, built on the Monte-Carlo integrated
  population mean `E[exp(x'beta)]` (consistent with `predict.rpbnb_fit()`).
  Continuous marginal effects use the per-draw realized coefficient
  (`mean_r coef_rj * exp(lp_r)`); binary effects use the integrated discrete
  difference. Standard errors use a numeric delta method over each equation's
  mean and log-scale parameters. Mirrors the existing `bnb_marginal_effects()` /
  `bnb_elasticities()` for fixed-coefficient models.
```

Insert this new bullet immediately AFTER it (still before whatever section/bullet follows):

```markdown
* `rpbnb_marginal_effects()` and `rpbnb_elasticities()` gain an `n_cores`
  parameter (default `1`, sequential) that parallelizes the delta-method
  standard-error computation across a `parallel` cluster for `n_cores > 1`.
  The jacobian is decomposed into independent per-parameter columns dispatched
  via `parallel::parLapply()`; results are numerically identical to the
  sequential path (verified exactly, `tolerance = 0`, in
  `tests/testthat/test-rpbnb-interpretation.R`).
```

- [ ] **Step 2: Run the full fast tier**

Run: `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" tools/test-tiers.R fast`

Expected: `[fast         ] pass=<N> fail=0 warn=<w> skip=<s>` — zero failures. `<N>` should be at least 3 higher than the pre-Task-1 count (the three new tests added across Tasks 1 and 2).

- [ ] **Step 3: If any failure, debug before proceeding**

Use superpowers:systematic-debugging. Do not mark the plan complete with a non-zero `fail` count.

- [ ] **Step 4: Commit**

Run `git status --short` first and confirm ONLY `NEWS.md` shows as modified before staging:

```bash
git status --short -- NEWS.md
git add NEWS.md
git commit -m "docs: NEWS entry for RP interpretation multithreading"
```

---

## Self-Review Notes

- **Spec coverage:** mathematical decomposition + exact-equivalence verification (Task 1's test), bare-call/clusterExport worker-resolution requirement (Global Constraints + Task 1/2 code comments), `n_cores` API on both functions with cluster shared across equations (Task 2), `parallel`-unavailable fallback (`.rp_make_cluster`, matches spec's "not independently testable without mocking" — no separate test added, consistent with the spec's own testing section), demo script updates (Task 3), NEWS entry (Task 4). All covered.
- **Type consistency:** `.rp_jac_col(k, theta_hat, theta_names, meta, quantity)`, `.rp_diag_one(fit, eq, quantity, type, vars, include_intercept, digits, print_output, resp_name, cl = NULL)`, `.rp_make_cluster(n_cores)` — signatures are identical everywhere they are defined and called across Tasks 1-2.
- **No placeholders:** every code step shows complete code; every run step gives the exact command and expected result.
