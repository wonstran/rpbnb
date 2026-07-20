# Tests for the Poisson-limit fits (poisson_1/poisson_2) that enable
# overdispersion LR tests on the NB2 dispersions m1/m2.

make_od_data <- function(n = 500, seed = 7) {
  set.seed(seed)
  d <- data.frame(x = rnorm(n))
  # Overdispersed counts (NB size = 1.5) so m1, m2 are well away from 0.
  d$y1 <- rnbinom(n, size = 1.5, mu = exp(0.4 + 0.2 * d$x))
  d$y2 <- rnbinom(n, size = 1.5, mu = exp(0.2 - 0.1 * d$x))
  d
}

test_that("poisson_1 drops exactly one free parameter (df and npar)", {
  d <- make_od_data()
  full <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye")
  p1   <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE)

  expect_equal(p1$npar, full$npar - 1L)
  expect_equal(attr(logLik(p1), "df"), full$npar - 1L)

  p12 <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye",
                 poisson_1 = TRUE, poisson_2 = TRUE)
  expect_equal(p12$npar, full$npar - 2L)
})

test_that("pinned margin has m ~ 0 and NA natural-scale SE", {
  d  <- make_od_data()
  p1 <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE)

  expect_lt(exp(p1$coef[["log_m1"]]), 1e-5)          # m1 pinned at the limit
  s   <- summary(p1)
  m1r <- s$natural[s$natural$Parameter == "m1 (dispersion)", ]
  expect_true(is.na(m1r$StdErr))                     # not estimated -> no SE
})

test_that("default (no poisson args) is unchanged", {
  d <- make_od_data()
  a <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye")
  b <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye",
               poisson_1 = FALSE, poisson_2 = FALSE)
  expect_equal(a$coef, b$coef)
  expect_equal(as.numeric(logLik(a)), as.numeric(logLik(b)))
})

test_that("overdispersion LR test rejects on overdispersed data", {
  d    <- make_od_data()
  full <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye")
  rest <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE)
  res  <- lr_test(rest, full, boundary = TRUE)
  expect_equal(res$df, 1)
  expect_lt(res$p.value, 0.01)                       # strong overdispersion
})

test_that("independence path fits an exact Poisson margin", {
  d  <- make_od_data()
  p1 <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "independence",
                poisson_1 = TRUE)
  # Margin 1 logLik should match a direct Poisson GLM on that margin.
  g  <- stats::glm(y1 ~ x, family = stats::poisson, data = d)
  # p1 logLik = poisson margin 1 + NB margin 2; recover margin-1 piece:
  g2 <- MASS::glm.nb(y2 ~ x, data = d)
  expect_equal(as.numeric(logLik(p1)),
               as.numeric(logLik(g)) + as.numeric(logLik(g2)), tolerance = 1e-4)
  expect_equal(p1$npar, length(coef(g)) + length(coef(g2)) + 1L)  # +1 for m2 only
})

test_that("poisson flags error on a copula dependence", {
  d <- make_od_data()
  expect_error(
    fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = copula("frank"), poisson_1 = TRUE),
    "copula"
  )
  expect_error(
    fit_rpbnb(y1 ~ x, y2 ~ x, data = d, dependence = copula("frank"), poisson_1 = TRUE),
    "copula"
  )
})

test_that("end-to-end: fit_rpbnb Poisson-limit fit nests for an overdispersion LR test", {
  skip_slow()
  sim <- simulate_rpbnb(n = 600,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
  ctrl <- rpbnb_control(print_level = 0)
  full <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                    draws = 200, seed = 1, control = ctrl)
  # A tail draw can push one conditional mean past the Poisson-limit guard's mean
  # tolerance; that diagnostic is exercised separately. Muffle only it, so a
  # genuine convergence warning would still surface.
  p1   <- suppress_poisson_warning(
    fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
              draws = 200, seed = 1, control = ctrl, poisson_1 = TRUE))

  expect_equal(p1$npar, full$npar - 1L)
  expect_lt(exp(p1$coef[["log_m1"]]), 1e-5)          # m1 pinned at the limit
  expect_true(is.na(p1$se[["log_m1"]]))              # fixed -> no SE
  expect_true(is.finite(p1$se[["log_m2"]]))          # other SEs still valid
  res <- lr_test(p1, full, boundary = TRUE)
  expect_equal(res$df, 1)
  expect_lt(res$p.value, 0.01)                       # m1 = 0.4 in the DGP
})

test_that("invalid poisson flags are rejected, not silently ignored (P2)", {
  d <- make_od_data(n = 60)
  bad_vals <- list(one = 1, na = NA, empty = logical(0),
                   two = c(TRUE, FALSE), chr = "yes")
  for (bad in bad_vals) {
    expect_error(fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "independence",
                         poisson_1 = bad), "logical")
    expect_error(fit_rpbnb(y1 ~ x, y2 ~ x, data = d, poisson_2 = bad), "logical")
  }
})

test_that("bnb_gof() preserves a Poisson-restricted margin in the null (P2)", {
  d  <- make_od_data()
  # independence path -> exact Poisson margin 1, NB margin 2
  p1 <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "independence",
                poisson_1 = TRUE)
  g  <- bnb_gof(p1, print_output = FALSE)
  expect_true(isTRUE(g$null_fit$poisson_1))            # null keeps the restriction
  expect_false(isTRUE(g$null_fit$poisson_2))
  expect_lt(exp(g$null_fit$coef[["log_m1"]]), 1e-5)    # null margin 1 still Poisson
  # intercept-only Poisson/NB null: 1 + 1 betas + 1 free dispersion = 3 (not 4)
  expect_equal(g$null_fit$npar, 3L)
})

test_that("bnb_gof() null is unrestricted NB when the fit is unrestricted", {
  d   <- make_od_data()
  fit <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "independence")
  g   <- bnb_gof(fit, print_output = FALSE)
  expect_false(isTRUE(g$null_fit$poisson_1))
  expect_false(isTRUE(g$null_fit$poisson_2))
  expect_equal(g$null_fit$npar, 4L)                    # 2 betas + 2 dispersions
})

# The exact m = 0 (true Poisson) likelihood/CDF branch replaced the old fixed
# POISSON_M = 1e-6 approximation, so the tests that documented that approximation
# and its .warn_poisson_limit range guard (dpois divergence at large mu, the
# warning firing above a mean tolerance, and pinned-m logLik stability) are gone.
# Exact-branch coverage lives in test-poisson-exact.R: c_val/nb_logpmf at m = 0,
# the famoye and RP objectives/gradients/Hessians, the fit-level "no POISSON_M
# warning + exact logLik" regressions, and the residual/CDF/variance branches.
