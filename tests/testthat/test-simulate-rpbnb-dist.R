test_that("normal spec reproduces the pre-refactor draws (backward compat)", {
  set.seed(99)
  ref_cov <- data.frame(x1 = rnorm(2000))
  # Legacy realization formula: B = beta + sd * rnorm(n)
  set.seed(5); legacy_eps <- rnorm(2000)
  legacy_B <- 0.4 + 0.5 * legacy_eps
  s <- simulate_rpbnb(n = 2000,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
        random_1 = list(x1 = list(sd = 0.5)),
        dispersion = c(m1 = 0.4, m2 = 0.5),
        covariates = ref_cov, seed = 5)
  # The x1 realized coefficients are drawn first for eq1 in realize(); compare moments
  expect_equal(mean(s$coef_realized$eq1[, "x1"]), 0.4, tolerance = 0.1)
  expect_equal(stats::sd(s$coef_realized$eq1[, "x1"]), 0.5, tolerance = 0.06)
})

test_that("lognormal sign forces coefficient sign", {
  s <- simulate_rpbnb(n = 1500,
        beta1 = c("(Intercept)" = 0.2, x1 = -0.1),
        beta2 = c("(Intercept)" = 0.1),
        random_1 = list(x1 = list(dist = "lognormal", scale = 0.4, sign = -1)),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 8)
  expect_true(all(s$coef_realized$eq1[, "x1"] < 0))
})

test_that("uniform realized coefficients stay within [center +/- width]", {
  s <- simulate_rpbnb(n = 1500,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.5),
        beta2 = c("(Intercept)" = 0.1),
        random_1 = list(x1 = list(dist = "uniform", scale = 0.3)),
        dispersion = c(m1 = 0.4, m2 = 0.5), seed = 9)
  cc <- s$coef_realized$eq1[, "x1"]
  expect_true(all(cc >= 0.5 - 0.3 - 1e-8 & cc <= 0.5 + 0.3 + 1e-8))
})

test_that("missing scale for a random coefficient is an error", {
  expect_error(
    simulate_rpbnb(n = 100,
      beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
      beta2 = c("(Intercept)" = 0.1),
      random_1 = list(x1 = list(dist = "uniform")),
      dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1),
    "scale"
  )
})
