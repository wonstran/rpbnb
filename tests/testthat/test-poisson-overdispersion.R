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
  p1   <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                    draws = 200, seed = 1, control = ctrl, poisson_1 = TRUE)

  expect_equal(p1$npar, full$npar - 1L)
  expect_lt(exp(p1$coef[["log_m1"]]), 1e-5)          # m1 pinned at the limit
  expect_true(is.na(p1$se[["log_m1"]]))              # fixed -> no SE
  expect_true(is.finite(p1$se[["log_m2"]]))          # other SEs still valid
  res <- lr_test(p1, full, boundary = TRUE)
  expect_equal(res$df, 1)
  expect_lt(res$p.value, 0.01)                       # m1 = 0.4 in the DGP
})

test_that("POISSON_M limit is stable: restricted logLik agrees across 1e-6 and 1e-8", {
  skip_slow()
  d  <- make_od_data(n = 800)
  p6 <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE)
  # At the fitted restricted estimate, the total logLik as a function of the
  # pinned m1 should be flat near the Poisson limit: values at 1e-6 and 1e-8
  # must agree to well under one logLik unit (the LR-statistic tolerance).
  X <- model.matrix(~ x, d)
  ll_at <- function(mval) {
    st <- p6$coef; st[["log_m1"]] <- log(mval)
    sum(rpbnb:::bnb_loglik_vec(st, d$y1, d$y2, X, X))
  }
  expect_lt(abs(ll_at(1e-6) - ll_at(1e-8)), 1)
})
