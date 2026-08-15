# Covers fit_rpbnb_tmb()'s recovery from stats::nlminb() raising a hard R
# error ("NA/NaN function evaluation" / "NA/NaN gradient evaluation") on its
# FIRST call, before the restart loop (which already tolerates this via its
# own try()) ever gets a chance to run. Reproduced on the truck data under a
# Kimeldorf (Clayton) copula after nearly an hour of otherwise productive
# optimization -- too expensive to reproduce here, so these tests force the
# same failure deterministically by mocking stats::nlminb() itself.

recovery_data <- function(n = 60, seed = 11) {
  set.seed(seed)
  x1 <- rnorm(n)
  mu1 <- exp(0.3 + 0.2 * x1)
  mu2 <- exp(0.1 - 0.1 * x1)
  data.frame(y1 = rpois(n, mu1), y2 = rpois(n, mu2), x1 = x1)
}

test_that("a first-call nlminb() NA/NaN abort recovers instead of propagating", {
  skip_on_cran()
  d <- recovery_data()
  # restarts = 0 keeps the outcome deterministic: with the while loop never
  # entered, fit$optimizer is exactly what the tryCatch recovery produced.
  ctrl <- rpbnb_tmb_control(n_cores = 1, restarts = 0L)

  testthat::local_mocked_bindings(
    nlminb = function(...) stop("NA/NaN gradient evaluation"),
    .package = "stats"
  )

  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                       dependence = copula("kimeldorf"), draws = 10,
                       seed = 3, control = ctrl)

  expect_identical(fit$optimizer$convergence, 1L)
  expect_identical(fit$optimizer$restarts, 0L)
  expect_match(fit$optimizer$message, "recovered", fixed = TRUE)
  expect_match(fit$optimizer$message, "last.par.best", fixed = TRUE)
  expect_true(is.finite(fit$optimizer$objective))
  expect_true(all(is.finite(fit$optimizer$par)))
})

test_that("the restart loop can still make progress after a recovered first call", {
  skip_on_cran()
  d <- recovery_data()
  ctrl <- rpbnb_tmb_control(n_cores = 1)  # default restarts, real nlminb after the first throw

  real_nlminb <- stats::nlminb
  call_count <- 0L
  testthat::local_mocked_bindings(
    nlminb = function(...) {
      call_count <<- call_count + 1L
      if (call_count == 1L) stop("NA/NaN function evaluation")
      real_nlminb(...)
    },
    .package = "stats"
  )

  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                       dependence = copula("kimeldorf"), draws = 10,
                       seed = 3, control = ctrl)

  # call_count > 1 confirms the mocked nlminb() was invoked again after the
  # first-call recovery -- i.e. the restart loop picked up from the
  # recovered point rather than the fit stopping dead on the first abort.
  expect_gt(call_count, 1L)
  expect_true(is.finite(fit$optimizer$objective))
  expect_true(all(is.finite(fit$optimizer$par)))
})

test_that("an nlminb() error unrelated to NA/NaN is not swallowed", {
  skip_on_cran()
  d <- recovery_data()
  ctrl <- rpbnb_tmb_control(n_cores = 1, restarts = 0L)

  testthat::local_mocked_bindings(
    nlminb = function(...) stop("some unrelated optimizer failure"),
    .package = "stats"
  )

  expect_error(
    fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                 dependence = copula("kimeldorf"), draws = 10,
                 seed = 3, control = ctrl),
    "some unrelated optimizer failure"
  )
})

test_that("an unrecoverable abort inside one restricted refit becomes NA, not a lost batch", {
  skip_on_cran()
  # A restricted refit's warm start (the full fit's coefficients with one
  # boundary parameter pinned) can itself be non-finite at every point
  # nlminb() tries, leaving fit_rpbnb_tmb() with no finite point to recover
  # to (obj$env$last.par.best is never updated) -- so it correctly re-raises
  # rather than fabricate a result (see the first test above: recovery only
  # ever succeeds when some finite point was actually seen). Observed on the
  # truck data's `m1` dispersion test under a Kimeldorf copula. This must not
  # take the whole boundary-test batch down with it -- rpbnb_tmb_boundary_tests()
  # wraps each restricted refit and funnels an unrecoverable error into the
  # same "did not converge" -> NA path an ordinary non-convergent refit
  # already takes.
  d <- recovery_data()
  ctrl <- rpbnb_tmb_control(n_cores = 1)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, random_1 = "x1",
                       dependence = copula("kimeldorf"), draws = 10,
                       seed = 4, control = ctrl)
  expect_identical(fit$optimizer$convergence, 0L)

  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(...) stop("NA/NaN gradient evaluation")
  )

  # Every restricted refit warns (one per boundary parameter), so
  # expect_warning()'s single-warning capture doesn't fit here -- collect
  # them by hand instead, without disturbing the call's actual return value.
  warnings_seen <- character(0)
  bt <- withCallingHandlers(
    rpbnb_tmb_boundary_tests(fit, d, control = ctrl),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  expect_true(length(warnings_seen) > 0L)
  expect_true(all(grepl("did not converge", warnings_seen, fixed = TRUE)))
  expect_s3_class(bt, "rpbnb_boundary_tests")
  expect_gt(nrow(bt), 0L)
  expect_true(all(is.na(bt$LR)))
  expect_true(all(is.na(bt$p.value)))
})

test_that("the exact-Poisson branch survives a mu vastly larger than the observed count", {
  skip_on_cran()
  # This is the actual root cause behind the truck data's m1 boundary test
  # going NaN on its very first evaluation (see NEWS.md): the template's
  # exact m = 0 (poisson_1/poisson_2 = TRUE) branch used to compute
  # ppois()/dpois() in LINEAR space and log() the result. Once mu is large
  # enough -- mu > ~750 already does it for a small y, since dpois(0, mu) =
  # exp(-mu) and ppois(0, mu) = exp(-mu) underflow to an exact double 0.0
  # there -- BOTH the CDF-at-(y-1) and the PMF-at-y can underflow at once,
  # so clayton_cell_prob()'s log_ratio() computes log(0) - log(0), an
  # Inf - Inf subtraction that is NaN by construction and poisons the whole
  # objective/gradient (every free parameter's gradient goes NaN at once,
  # since NaN propagates through the sum-of-observations reduction wherever
  # it appears -- not just parameters tied to the offending observation).
  #
  # A large covariate value on a random-coefficient carrier reaches mu > 750
  # easily: with the default starting sd = 0.2, X1 = 25 needs only
  # qnorm(halton) > 1.324 (the upper ~9% of a normal) to push eta1 = 0.2 *
  # qnorm(u) * 25 past log(750) = 6.62 -- routine among 30 Halton points.
  # y1 is fixed (not simulated) at both 0 and positive values on the
  # high-leverage rows so the Inf - Inf combination (y1 > 0, huge mu, so both
  # la1m and lpmf1 underflow) is deterministically exercised rather than left
  # to chance.
  set.seed(21)
  n <- 40
  x1 <- c(rep(25, 5), rnorm(n - 5))
  y1 <- c(1L, 2L, 0L, 1L, 3L, rpois(n - 5, 1))
  y2 <- rpois(n, 1)
  d <- data.frame(y1 = y1, y2 = y2, x1 = x1)

  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, random_1 = "x1",
                       dependence = copula("kimeldorf"), draws = 30,
                       seed = 5, poisson_1 = TRUE,
                       control = rpbnb_tmb_control(n_cores = 1, restarts = 0L))

  expect_true(is.finite(fit$optimizer$objective))
  expect_true(is.finite(fit$optimizer$max_abs_gradient))
  expect_identical(fit$optimizer$convergence, 0L)
})
