# rpbnb_tmb_boundary_tests(): the TMB engine's counterpart of
# rpbnb_boundary_tests() (R/boundary_tests.R), covering both the
# random-coefficient scales and the NB2 dispersions.

.tmb_boundary_fixture <- function(n = 250, seed = 42) {
  set.seed(seed)
  x1 <- rnorm(n, mean = 50, sd = 10)
  mu1 <- exp(0.5 + 0.02 * x1)
  mu2 <- exp(0.2 + 0.01 * x1)
  data.frame(y1 = rnbinom(n, mu = mu1, size = 2),
             y2 = rnbinom(n, mu = mu2, size = 2), x1 = x1)
}

# Two random-coefficient carriers in equation 1: x1 genuinely varies
# (sd = 0.8), x2 does not (sd = 0). A boundary test that works must separate
# them -- a test that merely runs would pass a smoke check either way.
.tmb_scale_fixture <- function(n = 300, seed = 42) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  u <- rnorm(n, 0, 0.8)
  mu1 <- exp(0.5 + (0.4 + u) * x1 + 0.1 * x2)
  mu2 <- exp(0.2 + 0.2 * x1)
  data.frame(y1 = rnbinom(n, mu = mu1, size = 2),
             y2 = rnbinom(n, mu = mu2, size = 2), x1 = x1, x2 = x2)
}

test_that("rpbnb_tmb_boundary_tests() tests m1 and m2 and matches a manual lr_test()", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  bt <- rpbnb_tmb_boundary_tests(fit, d)
  expect_s3_class(bt, "rpbnb_boundary_tests")
  expect_identical(sort(bt$Parameter), c("m1", "m2"))
  expect_true(all(is.finite(bt$LR)))
  expect_true(all(bt$LR >= 0))
  expect_identical(bt$df, c(1L, 1L))
  expect_true(all(bt$p.value >= 0 & bt$p.value <= 1))

  # The result must reproduce a hand-built restricted refit + lr_test() --
  # rpbnb_tmb_boundary_tests() is a convenience wrapper around exactly that.
  fit_p1 <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                          poisson_1 = TRUE, start = fit$coef,
                          control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L,
                                                      max_workload = Inf),
                          inference = "none")
  manual <- lr_test(fit_p1, fit, boundary = TRUE)
  pkg_row <- bt[bt$Parameter == "m1", ]
  expect_equal(manual$statistic, pkg_row$LR, tolerance = 1e-6)
  expect_equal(manual$p.value, pkg_row$p.value, tolerance = 1e-6)
})

test_that("scale tests separate genuine heterogeneity from none", {
  skip_on_cran()
  d <- .tmb_scale_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1 + x2, y2 ~ x1, data = d,
                       random_1 = c("x1", "x2"), draws = 40, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  bt <- rpbnb_tmb_boundary_tests(fit, d)
  expect_true(all(c("sd1:x1", "sd1:x2", "m1", "m2") %in% bt$Parameter))
  expect_true(all(bt$df == 1L, na.rm = TRUE))

  # The whole point: x1 carries a real random slope (sd = 0.8 in the DGP),
  # x2 carries none. A test that cannot tell them apart is not a test.
  p_real <- bt$p.value[bt$Parameter == "sd1:x1"]
  p_null <- bt$p.value[bt$Parameter == "sd1:x2"]
  expect_lt(p_real, 0.01)
  expect_gt(p_null, p_real)
  expect_gt(p_null, 0.05)
})

test_that("a scale restriction pins log_sd, drops one df, and keeps the other draws", {
  skip_on_cran()
  d <- .tmb_scale_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1 + x2, y2 ~ x1, data = d,
                       random_1 = c("x1", "x2"), draws = 40, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  rest <- fit_rpbnb_tmb(y1 ~ x1 + x2, y2 ~ x1, data = d,
                        random_1 = c("x1", "x2"), draws = 40, seed = 7,
                        start = fit$coef, inference = "none",
                        control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L),
                        .fixed = c("log_sd1:x2" = -20))

  # Pinned at the parameterization's zero, and out of the free-parameter count.
  expect_equal(unname(coef(rest)[["log_sd1:x2"]]), -20)
  expect_identical(rest$npar, fit$npar - 1L)

  # Common random numbers: pinning x2's scale must not move x1's estimate
  # materially. Dropping x2 from random_1 instead would renumber x1's Halton
  # dimension and this would not hold.
  expect_equal(coef(rest)[["log_sd1:x1"]], coef(fit)[["log_sd1:x1"]],
               tolerance = 0.05)
})

test_that("which = filters scale vs dispersion rows", {
  skip_on_cran()
  d <- .tmb_scale_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1 + x2, y2 ~ x1, data = d,
                       random_1 = c("x1", "x2"), draws = 40, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  bt_sd <- rpbnb_tmb_boundary_tests(fit, d, which = "sd")
  expect_identical(sort(bt_sd$Parameter), c("sd1:x1", "sd1:x2"))

  bt_disp <- rpbnb_tmb_boundary_tests(fit, d, which = "dispersion")
  expect_identical(sort(bt_disp$Parameter), c("m1", "m2"))
})

test_that("summary() shows the LR test on scale rows once boundary tests are attached", {
  skip_on_cran()
  d <- .tmb_scale_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1 + x2, y2 ~ x1, data = d,
                       random_1 = c("x1", "x2"), draws = 40, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  before <- capture.output(summary(fit))
  expect_true(any(grepl("No Wald z/p or boundary LR test", before)))

  fit$boundary_tests <- rpbnb_tmb_boundary_tests(fit, d)
  after <- capture.output(summary(fit))
  expect_true(any(grepl("boundary-corrected LR test \\(H0: scale = 0", after)))
  # The scale row must carry a real statistic, not NA.
  sd_line <- grep("^sd1:x1", after, value = TRUE)
  expect_gt(length(sd_line), 0L)
  expect_false(any(grepl("\\bNA\\b", sd_line)))
})

test_that("a fixed-coefficient model rejects which = \"sd\" instead of returning nothing", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_error(rpbnb_tmb_boundary_tests(fit, d, which = "sd"),
               "No boundary parameters to test")
})

test_that(".fixed validates its input", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  expect_error(
    fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 10, seed = 7,
                  control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L),
                  .fixed = c(nonesuch = 1)),
    "not in the model's parameter vector"
  )
  expect_error(
    fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 10, seed = 7,
                  control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L),
                  .fixed = 1),
    "named numeric vector"
  )
})

test_that("a Poisson-pinned margin is skipped, not re-tested", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       poisson_1 = TRUE,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  bt <- rpbnb_tmb_boundary_tests(fit, d)
  expect_identical(bt$Parameter, "m2")
})

test_that("a fixed-coefficient, fully-Poisson model has nothing to test", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  # No random coefficients AND both dispersions pinned: neither category has
  # a free boundary parameter, so this errors rather than returning an empty
  # table. (With random coefficients present, the scale rows are still
  # testable even when both margins are Poisson -- see the `which` test.)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       poisson_1 = TRUE, poisson_2 = TRUE,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "No boundary parameters to test")
})

test_that("a non-converged full fit is rejected before any refit runs", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  # A normally-converged fit with the convergence record overwritten
  # afterward -- exercises the pre-refit check without needing a genuinely
  # under-optimized (and warning-noisy) fixture.
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  fit$optimizer$convergence <- 1L
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "did not converge")
})

test_that("rpbnb(engine = \"tmb\", boundary_tests = TRUE) attaches $boundary_tests", {
  skip_on_cran()
  # Fixed-coefficient model: dispersions are the only boundary parameters.
  d <- .tmb_boundary_fixture()
  fit <- rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb",
              draws = 20, seed = 7, boundary_tests = TRUE,
              control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_false(is.null(fit$boundary_tests))
  expect_identical(sort(fit$boundary_tests$Parameter), c("m1", "m2"))

  # summary()'s dispersion block must merge the LR/df/p in, not just carry
  # $boundary_tests without displaying it.
  out <- capture.output(summary(fit))
  disp_line <- grep("^1\\s+m1", out, value = TRUE)
  expect_length(disp_line, 1L)
  expect_false(grepl("\\bNA\\b", disp_line))
})

test_that("rpbnb(engine = \"tmb\", boundary_tests = TRUE) covers scales too", {
  skip_on_cran()
  d <- .tmb_scale_fixture()
  fit <- rpbnb(y1 ~ x1 + x2, y2 ~ x1, data = d, engine = "tmb",
              random_1 = c("x1", "x2"), draws = 40, seed = 7,
              boundary_tests = TRUE,
              control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_true(all(c("sd1:x1", "sd1:x2", "m1", "m2") %in%
                    fit$boundary_tests$Parameter))
  out <- capture.output(summary(fit))
  expect_true(any(grepl("boundary-corrected LR test \\(H0: scale = 0", out)))
})

test_that("rpbnb(boundary_tests = TRUE) works for both engines without a hard error", {
  skip_on_cran()
  # Historical note: engine = "tmb" used to be a documented hard error for
  # boundary_tests = TRUE (rpbnb_boundary_tests() is classic-only). This
  # guards against that error coming back now that rpbnb_tmb_boundary_tests()
  # exists to serve engine = "tmb" instead.
  d <- .tmb_boundary_fixture()
  expect_no_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb",
         draws = 20, seed = 7, boundary_tests = TRUE,
         control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  )
})

.empty_boundary_tests <- structure(
  data.frame(Parameter = character(0), LR = numeric(0), df = integer(0),
             p.value = numeric(0), Signif = character(0),
             stringsAsFactors = FALSE),
  class = c("rpbnb_boundary_tests", "data.frame"))

test_that("rpbnb(engine = \"tmb\") passes boundary_draws through to rpbnb_tmb_boundary_tests()", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()

  # Mocked one level above the restricted refits (rpbnb_tmb_boundary_tests()
  # itself, not fit_rpbnb_tmb()) so the main fit -- a real fit_rpbnb_tmb()
  # call -- is unaffected and this only observes what rpbnb() forwards.
  captured <- NULL
  testthat::local_mocked_bindings(
    rpbnb_tmb_boundary_tests = function(fit, data, ..., draws = fit$draws) {
      captured <<- draws
      .empty_boundary_tests
    }
  )
  fit <- rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb",
              draws = 20, seed = 7, boundary_tests = TRUE, boundary_draws = 77,
              control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_identical(captured, 77)
  expect_identical(fit$boundary_tests, .empty_boundary_tests)
})

test_that("rpbnb(engine = \"tmb\", boundary_draws = NULL) defaults to the fit's own draws", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()

  captured <- NULL
  testthat::local_mocked_bindings(
    rpbnb_tmb_boundary_tests = function(fit, data, ..., draws = fit$draws) {
      captured <<- draws
      .empty_boundary_tests
    }
  )
  fit <- rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb",
              draws = 20, seed = 7, boundary_tests = TRUE,
              control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_identical(fit$draws, 20L)
  expect_identical(captured, 20L)
})

test_that("rpbnb(engine = \"classic\", boundary_draws = ) errors: no draws knob to honor", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
         draws = 20, seed = 7, boundary_tests = TRUE, boundary_draws = 50),
    "only supported for engine = \"tmb\""
  )
})

test_that("$boundary_tests is absent when boundary_tests = FALSE (the default)", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_null(fit$boundary_tests)
  out <- capture.output(summary(fit))
  expect_true(any(grepl("no boundary LR test", out)))
})

test_that("default control (control = NULL) reuses the original fit's n_cores", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  # Simulate a fit that was originally requested with n_cores = 3, without
  # actually needing 3 cores to be available in the test environment.
  fit$parallel$requested <- 3L

  captured <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., control) {
      captured <<- control$n_cores
      stop("stop-early-for-test")
    }
  )
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "stop-early-for-test")
  expect_identical(captured, 3L)
})

test_that("default control falls back to n_cores = 1 when $parallel is absent", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  fit$parallel <- NULL  # simulate an older fit predating the stored field

  captured <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., control) {
      captured <<- control$n_cores
      stop("stop-early-for-test")
    }
  )
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "stop-early-for-test")
  expect_identical(captured, 1L)
})

test_that("force_parallel_gaussian defaults to FALSE and is forwarded to every refit", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  captured <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., force_parallel_gaussian) {
      captured <<- force_parallel_gaussian
      stop("stop-early-for-test")
    }
  )
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "stop-early-for-test")
  expect_identical(captured, FALSE)
})

test_that("force_parallel_gaussian = TRUE is forwarded to every refit", {
  skip_on_cran()
  # This is the flag's actual reason for existing: a Gaussian-copula fit's
  # own force_parallel_gaussian = TRUE does NOT propagate here on its own
  # (fit does not record it), so rpbnb_tmb_boundary_tests() needs it passed
  # again -- otherwise every restricted refit silently re-caps to one thread.
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  captured <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., force_parallel_gaussian) {
      captured <<- force_parallel_gaussian
      stop("stop-early-for-test")
    }
  )
  expect_error(
    rpbnb_tmb_boundary_tests(fit, d, force_parallel_gaussian = TRUE),
    "stop-early-for-test"
  )
  expect_identical(captured, TRUE)
})

test_that("draws defaults to fit$draws and is overridable", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  captured <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., draws) {
      captured <<- draws
      stop("stop-early-for-test")
    }
  )
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "stop-early-for-test")
  expect_identical(captured, fit$draws)

  captured2 <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., draws) {
      captured2 <<- draws
      stop("stop-early-for-test")
    }
  )
  expect_error(rpbnb_tmb_boundary_tests(fit, d, draws = 50), "stop-early-for-test")
  expect_identical(captured2, 50)
})

test_that("a message announces each restricted refit, unless print_level = 0", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  msgs <- testthat::capture_messages(
    bt <- rpbnb_tmb_boundary_tests(
      fit, d,
      control = rpbnb_tmb_control(print_level = 1L, n_cores = 1L, max_workload = Inf))
  )
  expect_true(any(grepl("Boundary LR test: m1", msgs, fixed = TRUE)))
  expect_true(any(grepl("Boundary LR test: m2", msgs, fixed = TRUE)))

  expect_no_message(
    rpbnb_tmb_boundary_tests(
      fit, d,
      control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L, max_workload = Inf)),
    message = "Boundary LR test"
  )
})
