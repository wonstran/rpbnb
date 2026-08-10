# rpbnb_tmb_boundary_tests(): the TMB engine's counterpart of
# rpbnb_boundary_tests() (R/boundary_tests.R), scoped to the NB2 dispersions
# (m1, m2) -- see its documentation for why it does not also test
# random-coefficient SDs the way the classic engine's version does.

.tmb_boundary_fixture <- function(n = 250, seed = 42) {
  set.seed(seed)
  x1 <- rnorm(n, mean = 50, sd = 10)
  mu1 <- exp(0.5 + 0.02 * x1)
  mu2 <- exp(0.2 + 0.01 * x1)
  data.frame(y1 = rnbinom(n, mu = mu1, size = 2),
             y2 = rnbinom(n, mu = mu2, size = 2), x1 = x1)
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

test_that("a Poisson-pinned margin is skipped, not re-tested", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       poisson_1 = TRUE,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  bt <- rpbnb_tmb_boundary_tests(fit, d)
  expect_identical(bt$Parameter, "m2")
})

test_that("both margins Poisson-restricted errors instead of returning an empty table", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       poisson_1 = TRUE, poisson_2 = TRUE,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_error(rpbnb_tmb_boundary_tests(fit, d), "no dispersion parameter left")
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

test_that("$boundary_tests is absent when boundary_tests = FALSE (the default)", {
  skip_on_cran()
  d <- .tmb_boundary_fixture()
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, draws = 20, seed = 7,
                       control = rpbnb_tmb_control(print_level = 0L, n_cores = 1L))
  expect_null(fit$boundary_tests)
  out <- capture.output(summary(fit))
  expect_true(any(grepl("no boundary LR test", out)))
})
