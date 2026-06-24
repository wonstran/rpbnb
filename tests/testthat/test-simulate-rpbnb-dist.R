test_that("normal spec reproduces the pre-refactor draws (backward compat)", {
  set.seed(99)
  ref_cov <- data.frame(x1 = rnorm(2000))
  # Legacy realization formula: B = beta + sd * rnorm(n)
  # Inside simulate_rpbnb(seed=5): set.seed(5) runs, then covariates are provided (no RNG),
  # then realize() is called for eq1, which draws rnorm(n) for the x1 random coefficient.
  set.seed(5)
  legacy_eq1_x1 <- 0.4 + 0.5 * rnorm(2000)
  s <- simulate_rpbnb(n = 2000,
        beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
        beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
        random_1 = list(x1 = list(sd = 0.5)),
        dispersion = c(m1 = 0.4, m2 = 0.5),
        covariates = ref_cov, seed = 5)
  # Assert exact bit-identity reproduction of the legacy normal stream
  expect_equal(unname(s$coef_realized$eq1[, "x1"]), legacy_eq1_x1, tolerance = 1e-9)
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
