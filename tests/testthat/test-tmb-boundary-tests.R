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

test_that("default control propagates max_workload from a chunked original fit, not Inf", {
  # Without this, every boundary refit of an auto-chunked fit would rebuild
  # one full unchunked tape (max_workload = Inf), recreating the exact OOM
  # the draw-chunking feature exists to prevent -- see
  # docs/TMB_SML_large_draws_OOM_guide.md and the reviewed plan's
  # boundary-refit propagation section.
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  fit$tape_integration <- list(chunks = 4L, chunked = 1L,
                               max_workload = 12345, draws_requested = 20L,
                               draws_effective = 20L)

  captured <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., control) {
      captured <<- control$max_workload
      stop("stop-early-for-test")
    }
  )
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "stop-early-for-test")
  expect_identical(captured, 12345)
})

test_that("default control stays Inf for an unchunked original fit or one predating $tape_integration", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))

  captured <- NULL
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., control) {
      captured <<- control$max_workload
      stop("stop-early-for-test")
    }
  )

  # $tape_integration absent (older fit).
  fit_old <- fit
  fit_old$tape_integration <- NULL
  expect_error(rpbnb_tmb_boundary_tests(fit_old, d), "stop-early-for-test")
  expect_identical(captured, Inf)

  # $tape_integration present but chunked = 0L (this fit did not chunk).
  captured <- NULL
  fit_unchunked <- fit
  fit_unchunked$tape_integration <- list(chunks = 1L, chunked = 0L,
                                         max_workload = 999,
                                         draws_requested = 20L,
                                         draws_effective = 20L)
  expect_error(rpbnb_tmb_boundary_tests(fit_unchunked, d), "stop-early-for-test")
  expect_identical(captured, Inf)
})

test_that("default control also propagates a PINNED tape_chunks when max_workload was Inf", {
  # Regression for a real crash: a fit built with control$tape_chunks set
  # explicitly (bypassing the workload budget entirely) alongside
  # max_workload = Inf -- the exact pattern inst/rpbnb_truck_open_v2.R uses
  # deliberately, because the workload calibration under-estimates that
  # data's per-draw cost. Propagating max_workload alone (Inf) gives the
  # refit's resolver no budget to derive a layout from, so it silently fell
  # back to C = 1 and rebuilt one full tape -- reproducing
  # `std::bad_alloc` in MakeADFunObject() during a boundary LR test refit.
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  fit$tape_integration <- list(chunks = 10L, chunked = 1L,
                               max_workload = Inf, draws_requested = 20L,
                               draws_effective = 20L)

  captured <- list()
  testthat::local_mocked_bindings(
    fit_rpbnb_tmb = function(..., control) {
      captured$max_workload <<- control$max_workload
      captured$tape_chunks <<- control$tape_chunks
      stop("stop-early-for-test")
    }
  )

  # draws unspecified -> defaults to fit$draws (20): the pinned layout
  # propagates alongside max_workload.
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "stop-early-for-test")
  expect_identical(captured$max_workload, Inf)
  expect_identical(captured$tape_chunks, 10L)

  # An explicit draws EQUAL to fit$draws by value (a plain double, not the
  # same object) must still count as "unchanged" -- identical() would say
  # no here even though nothing about the workload changed.
  captured <- list()
  expect_error(rpbnb_tmb_boundary_tests(fit, d, draws = 20),
               "stop-early-for-test")
  expect_identical(captured$tape_chunks, 10L)

  # draws DIFFERS from fit$draws -> the stale chunk count must not be
  # reused; only max_workload propagates, and the refit re-resolves fresh.
  captured <- list()
  expect_error(rpbnb_tmb_boundary_tests(fit, d, draws = 40),
               "stop-early-for-test")
  expect_identical(captured$max_workload, Inf)
  expect_null(captured$tape_chunks)
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

# ---- sml_fallback -----------------------------------------------------------
# A Laplace fit's restricted refit can have no optimum for the inner Newton to
# find (pinning a margin to Poisson can drive the dependence strong enough
# that the cell probability is non-log-concave in the random effects -- see
# the sml_fallback argument doc). The fallback re-runs that one test with
# BOTH sides estimated by SML. These tests pin the mechanics with mocked
# refits: which estimator each side used, that the full-model SML refit is
# built once and cached, that a Laplace logLik is never paired with an SML
# one, and that every failure path still reports NA with its own warning.

.tmb_laplace_anchor <- function(n = 120, seed = 9) {
  set.seed(seed)
  x1 <- rnorm(n)
  u <- rnorm(n, 0, 0.4)
  d <- data.frame(y1 = rnbinom(n, mu = exp(0.4 + (0.3 + u) * x1), size = 2),
                  y2 = rnbinom(n, mu = exp(0.1 + 0.2 * x1), size = 2),
                  x1 = x1)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, random_1 = "x1",
                       method = "laplace", draws = 10, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L,
                                                   n_cores = 1L))
  list(d = d, fit = fit)
}

.stub_tmb_fit <- function(logLik, npar, code = 0L,
                          msg = "relative convergence (4)") {
  structure(list(logLik = logLik, npar = npar, nobs = 120L,
                 optimizer = list(convergence = code, message = msg)),
            class = "rpbnb_tmb_fit")
}

test_that("sml_fallback rescues a Laplace-unfittable restriction with an SML pair", {
  skip_on_cran()
  a <- .tmb_laplace_anchor()
  skip_if_not(identical(a$fit$optimizer$convergence, 0L),
              "Laplace anchor fit did not converge on this platform")

  full_sml_calls <- 0L
  fake <- function(...) {
    args <- list(...)
    restricted <- isTRUE(args$poisson_1) || isTRUE(args$poisson_2) ||
      !is.null(args$.fixed)
    if (identical(args$method, "laplace")) {
      return(.stub_tmb_fit(NA_real_, 8L, code = 1L,
                           msg = "false convergence (8)"))
    }
    if (!restricted) {
      full_sml_calls <<- full_sml_calls + 1L
      return(.stub_tmb_fit(-100, 9L))
    }
    .stub_tmb_fit(-110, 8L)
  }
  testthat::local_mocked_bindings(fit_rpbnb_tmb = fake)

  msgs <- testthat::capture_messages(
    bt <- rpbnb_tmb_boundary_tests(
      a$fit, a$d, which = "dispersion",
      control = rpbnb_tmb_control(print_level = 1L, n_cores = 1L,
                                  max_workload = Inf))
  )
  expect_true(any(grepl("retrying the test with both sides", msgs)))
  expect_true(any(grepl("refitting the FULL model", msgs)))

  # Both dispersion rows fell back, and each carries the SML pair's LR:
  # 2 * (-100 - -110) = 20 on 1 df, halved by the boundary mixture.
  expect_identical(attr(bt, "sml_fallback"), c("m1", "m2"))
  expect_equal(bt$LR, rep(20, 2L))
  expect_identical(bt$df, rep(1L, 2L))
  expect_equal(bt$p.value,
               rep(0.5 * pchisq(20, 1L, lower.tail = FALSE), 2L),
               tolerance = 1e-12)
  # One full-model SML refit serves both tests.
  expect_identical(full_sml_calls, 1L)

  # print() footnotes the fallback rows (and only prints them for TMB
  # results -- the classic engine never sets the attribute).
  out <- capture.output(print(bt))
  expect_true(any(grepl("Estimated by an SML pair", out)))
  expect_true(any(grepl("m1, m2", out)))
})

test_that("sml_fallback = FALSE keeps the NA-with-warning behaviour", {
  skip_on_cran()
  a <- .tmb_laplace_anchor()
  skip_if_not(identical(a$fit$optimizer$convergence, 0L),
              "Laplace anchor fit did not converge on this platform")

  sml_calls <- 0L
  fake <- function(...) {
    args <- list(...)
    if (identical(args$method, "sml")) sml_calls <<- sml_calls + 1L
    .stub_tmb_fit(NA_real_, 8L, code = 1L, msg = "false convergence (8)")
  }
  testthat::local_mocked_bindings(fit_rpbnb_tmb = fake)

  warns <- testthat::capture_warnings(
    bt <- rpbnb_tmb_boundary_tests(
      a$fit, a$d, which = "dispersion", sml_fallback = FALSE,
      control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L,
                                  max_workload = Inf))
  )
  expect_true(all(grepl("did not converge", warns)))
  expect_true(all(is.na(bt$LR)))
  expect_identical(attr(bt, "sml_fallback"), character(0))
  # The off switch means SML was never tried.
  expect_identical(sml_calls, 0L)
})

test_that("a failed SML restricted fit reports NA without paying for the full refit", {
  skip_on_cran()
  a <- .tmb_laplace_anchor()
  skip_if_not(identical(a$fit$optimizer$convergence, 0L),
              "Laplace anchor fit did not converge on this platform")

  full_sml_calls <- 0L
  fake <- function(...) {
    args <- list(...)
    restricted <- isTRUE(args$poisson_1) || isTRUE(args$poisson_2) ||
      !is.null(args$.fixed)
    if (identical(args$method, "sml") && !restricted) {
      full_sml_calls <<- full_sml_calls + 1L
      return(.stub_tmb_fit(-100, 9L))
    }
    .stub_tmb_fit(NA_real_, 8L, code = 1L, msg = "false convergence (8)")
  }
  testthat::local_mocked_bindings(fit_rpbnb_tmb = fake)

  warns <- testthat::capture_warnings(suppressMessages(
    bt <- rpbnb_tmb_boundary_tests(
      a$fit, a$d, which = "dispersion",
      control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L,
                                  max_workload = Inf))
  ))
  expect_true(any(grepl("SML fallback restricted fit did not converge",
                        warns)))
  expect_true(all(is.na(bt$LR)))
  expect_identical(attr(bt, "sml_fallback"), character(0))
  # The restricted side failed first, so the full-model SML refit -- whose
  # only purpose is to pair with it -- was never built.
  expect_identical(full_sml_calls, 0L)
})

test_that("a failed full-model SML refit reports NA with its own warning", {
  skip_on_cran()
  a <- .tmb_laplace_anchor()
  skip_if_not(identical(a$fit$optimizer$convergence, 0L),
              "Laplace anchor fit did not converge on this platform")

  fake <- function(...) {
    args <- list(...)
    restricted <- isTRUE(args$poisson_1) || isTRUE(args$poisson_2) ||
      !is.null(args$.fixed)
    if (identical(args$method, "sml") && restricted) {
      return(.stub_tmb_fit(-110, 8L))
    }
    .stub_tmb_fit(NA_real_, 8L, code = 1L, msg = "false convergence (8)")
  }
  testthat::local_mocked_bindings(fit_rpbnb_tmb = fake)

  warns <- testthat::capture_warnings(suppressMessages(
    bt <- rpbnb_tmb_boundary_tests(
      a$fit, a$d, which = "dispersion",
      control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L,
                                  max_workload = Inf))
  ))
  expect_true(any(grepl("FULL-model refit did not converge", warns)))
  expect_true(all(is.na(bt$LR)))
  expect_identical(attr(bt, "sml_fallback"), character(0))
})

test_that("an SML fit never engages the fallback (nothing different to try)", {
  skip_on_cran()
  d <- .tmb_boundary_fixture(n = 120)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L,
                                                   n_cores = 1L))
  fake <- function(...) {
    .stub_tmb_fit(NA_real_, 8L, code = 1L, msg = "false convergence (8)")
  }
  testthat::local_mocked_bindings(fit_rpbnb_tmb = fake)

  msgs <- testthat::capture_messages(warns <- testthat::capture_warnings(
    bt <- rpbnb_tmb_boundary_tests(
      fit, d, which = "dispersion",
      control = rpbnb_tmb_control(print_level = 1L, n_cores = 1L,
                                  max_workload = Inf))
  ))
  expect_false(any(grepl("retrying the test", msgs)))
  expect_true(all(grepl("did not converge", warns)))
  expect_true(all(is.na(bt$LR)))
})

test_that("sml_fallback must be one non-missing logical", {
  skip_on_cran()
  d <- .tmb_boundary_fixture(n = 120)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L,
                                                   n_cores = 1L))
  expect_error(rpbnb_tmb_boundary_tests(fit, d, sml_fallback = NA),
               "sml_fallback")
  expect_error(rpbnb_tmb_boundary_tests(fit, d, sml_fallback = "yes"),
               "sml_fallback")
})

test_that("a converged Laplace pair with a NEGATIVE raw LR also falls back to SML", {
  skip_on_cran()
  a <- .tmb_laplace_anchor()
  skip_if_not(identical(a$fit$optimizer$convergence, 0L),
              "Laplace anchor fit did not converge on this platform")

  # The restricted Laplace refits report clean convergence (code 0) but a
  # logLik ABOVE the full fit's -- the truck data's -3838 failure mode, where
  # the Laplace value itself is wrong (spurious ridge near a singular inner
  # Hessian), scaled down. lr_test() must never see this pair: it would clamp
  # the statistic to 0 and warn, turning a wrong value into "no evidence".
  inflated <- as.numeric(stats::logLik(a$fit)) + 3
  fake <- function(...) {
    args <- list(...)
    restricted <- isTRUE(args$poisson_1) || isTRUE(args$poisson_2) ||
      !is.null(args$.fixed)
    if (identical(args$method, "laplace")) {
      return(.stub_tmb_fit(inflated, 8L))       # code 0, inflated logLik
    }
    if (!restricted) return(.stub_tmb_fit(-100, 9L))
    .stub_tmb_fit(-110, 8L)
  }
  testthat::local_mocked_bindings(fit_rpbnb_tmb = fake)

  warns <- testthat::capture_warnings(
    msgs <- testthat::capture_messages(
      bt <- rpbnb_tmb_boundary_tests(
        a$fit, a$d, which = "dispersion",
        control = rpbnb_tmb_control(print_level = 1L, n_cores = 1L,
                                    max_workload = Inf))
    )
  )
  expect_true(any(grepl("EXCEEDS the full", msgs)))
  # No clamp warning: the inconsistent pair was never handed to lr_test().
  expect_false(any(grepl("higher log-likelihood", warns)))
  expect_identical(attr(bt, "sml_fallback"), c("m1", "m2"))
  expect_equal(bt$LR, rep(20, 2L))
  expect_equal(bt$p.value,
               rep(0.5 * pchisq(20, 1L, lower.tail = FALSE), 2L),
               tolerance = 1e-12)
})

# ---- Dependence (association) test ------------------------------------------

test_that("which = 'dependence' refits at independence and drops one df", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  ctl <- rpbnb_tmb_control(print_level = 0L, n_cores = 1L, max_workload = Inf)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = ctl)

  bt <- rpbnb_tmb_boundary_tests(fit, d, which = "dependence", control = ctl)
  expect_identical(bt$Parameter, "lam")        # famoye dependence
  expect_identical(bt$df, 1L)                  # z_dep mapped out
  expect_true(is.finite(bt$LR) && bt$LR >= 0)

  # Must reproduce the hand-built independence refit + an ORDINARY chi-square
  # LR: the Famoye lambda's null is interior, so no 50:50 mixture here.
  rest <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                        dependence = "independence",
                        start = fit$coef[setdiff(names(fit$coef), "z_dep")],
                        control = ctl, inference = "none")
  manual <- lr_test(rest, fit, boundary = FALSE)
  expect_equal(manual$statistic, bt$LR, tolerance = 1e-6)
  expect_equal(manual$p.value, bt$p.value, tolerance = 1e-6)
  # The boundary-corrected p-value would be exactly half -- assert we did NOT
  # report that, since the whole point is that this null is interior.
  expect_false(isTRUE(all.equal(bt$p.value, manual$p.value / 2)))
})

test_that("summary() replaces the dependence Wald row with the LR test", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  ctl <- rpbnb_tmb_control(print_level = 0L, n_cores = 1L, max_workload = Inf)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = ctl)

  out_wald <- capture.output(summary(fit))
  expect_true(any(grepl("z value", out_wald, fixed = TRUE)))

  fit$boundary_tests <- rpbnb_tmb_boundary_tests(fit, d, which = "dependence",
                                                 control = ctl)
  out_lr <- capture.output(summary(fit))
  expect_true(any(grepl("independence", out_lr, fixed = TRUE)))
  expect_true(any(grepl("Pr(>chisq)", out_lr, fixed = TRUE)))
})

test_that("an independence fit has no dependence parameter to test", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  ctl <- rpbnb_tmb_control(print_level = 0L, n_cores = 1L, max_workload = Inf)
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       dependence = "independence", control = ctl)
  expect_error(
    rpbnb_tmb_boundary_tests(fit, d, which = "dependence", control = ctl),
    "No boundary parameters to test")
})

test_that("rpbnb(boundary_tests = ) switches groups on and off", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  ctl <- rpbnb_tmb_control(print_level = 0L, n_cores = 1L, max_workload = Inf)

  only_disp <- suppressMessages(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", draws = 20, seed = 7,
          control = ctl, boundary_tests = "dispersion"))
  expect_identical(sort(only_disp$boundary_tests$Parameter), c("m1", "m2"))

  disp_dep <- suppressMessages(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", draws = 20, seed = 7,
          control = ctl, boundary_tests = c("dispersion", "dependence")))
  expect_identical(sort(disp_dep$boundary_tests$Parameter),
                   c("lam", "m1", "m2"))

  # TRUE keeps its historical meaning: no dependence row appears.
  hist <- suppressMessages(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", draws = 20, seed = 7,
          control = ctl, boundary_tests = TRUE))
  expect_false("lam" %in% hist$boundary_tests$Parameter)

  expect_null(
    suppressMessages(
      rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", draws = 20, seed = 7,
            control = ctl, boundary_tests = FALSE))$boundary_tests)
})

test_that("a TMB fit reports the control settings its engine did not read", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(
    y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
    control = rpbnb_control(print_level = 0L, n_cores = 1L,
                            max_workload = Inf,
                            se_method = "opg", draws_hessian = 50L))
  expect_setequal(fit$control_ignored, c("se_method", "draws_hessian"))
  expect_identical(fit$control_engine, "tmb")
  out <- capture.output(print(fit))
  expect_true(any(grepl("Control settings ignored", out, fixed = TRUE)))
  expect_true(any(grepl("se_method", out, fixed = TRUE)))
})
