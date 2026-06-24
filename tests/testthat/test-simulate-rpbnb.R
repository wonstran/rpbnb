test_that("simulate_rpbnb is reproducible for a fixed seed", {
  a <- simulate_rpbnb(n = 200, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                      beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                      dispersion = c(m1 = 0.4, m2 = 0.5), seed = 42)
  b <- simulate_rpbnb(n = 200, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                      beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                      dispersion = c(m1 = 0.4, m2 = 0.5), seed = 42)
  expect_identical(a$data, b$data)
  expect_identical(a$coef_realized, b$coef_realized)
})

test_that("simulate_rpbnb returns the documented pieces and true params", {
  s <- simulate_rpbnb(n = 100, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                      beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                      random_1 = list(x1 = list(sd = 0.5)),
                      dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0, seed = 1)
  expect_true(all(c("data", "coef_realized", "mu", "true", "settings", "meta") %in% names(s)))
  expect_true(all(c("y1", "y2", "x1") %in% names(s$data)))
  expect_equal(s$true$dispersion[["m1"]], 0.4)
  expect_equal(s$true$random_1$scale[[1]], 0.5)
  expect_equal(nrow(s$data), 100)
})

test_that("realized random coefficients match requested mean and sd", {
  s <- simulate_rpbnb(n = 20000, beta1 = c("(Intercept)" = 0.0, x1 = 0.5),
                      beta2 = c("(Intercept)" = 0.0, x1 = 0.0),
                      random_1 = list(x1 = list(sd = 0.7)),
                      dispersion = c(m1 = 0.4, m2 = 0.4), seed = 2)
  bx1 <- s$coef_realized$eq1[, "x1"]
  expect_equal(mean(bx1), 0.5, tolerance = 0.02)
  expect_equal(sd(bx1),  0.7, tolerance = 0.02)
})

test_that("counts are overdispersed when dispersion > 0", {
  s <- simulate_rpbnb(n = 5000, beta1 = c("(Intercept)" = 1.0, x1 = 0.0),
                      beta2 = c("(Intercept)" = 1.0, x1 = 0.0),
                      dispersion = c(m1 = 0.8, m2 = 0.8), seed = 3)
  expect_gt(var(s$data$y1), mean(s$data$y1))
})

test_that("simulate_rpbnb errors when supplied covariates miss a column", {
  cov <- data.frame(x1 = rnorm(50))
  expect_error(
    simulate_rpbnb(n = 50, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                   beta2 = c("(Intercept)" = 0.1, x2 = -0.2),
                   dispersion = c(m1 = 0.4, m2 = 0.4),
                   covariates = cov, seed = 1),
    "missing required column"
  )
})
