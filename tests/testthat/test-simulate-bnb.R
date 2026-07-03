test_that("simulate_bnb is reproducible for a fixed seed", {
  a <- simulate_bnb(n = 200, beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
                    beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
                    dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 42)
  b <- simulate_bnb(n = 200, beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
                    beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
                    dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 42)
  expect_identical(a$data, b$data)
})

test_that("simulate_bnb does not fix the RNG when no seed is given", {
  a <- simulate_bnb(n = 200, beta1 = c("(Intercept)" = 0.5),
                    beta2 = c("(Intercept)" = 0.2),
                    dispersion = c(m1 = 0.4, m2 = 0.5))
  b <- simulate_bnb(n = 200, beta1 = c("(Intercept)" = 0.5),
                    beta2 = c("(Intercept)" = 0.2),
                    dispersion = c(m1 = 0.4, m2 = 0.5))
  expect_false(isTRUE(all.equal(a$data, b$data)))
})

test_that("simulate_bnb induces dependence with the sign of lambda", {
  base <- list(n = 4e4, beta1 = c("(Intercept)" = 0.6),
               beta2 = c("(Intercept)" = 0.4),
               dispersion = c(m1 = 0.4, m2 = 0.4))
  pos <- do.call(simulate_bnb, c(base, lambda = 3.0, seed = 5))
  neg <- do.call(simulate_bnb, c(base, lambda = -2.5, seed = 5))
  expect_gt(cor(pos$data$y1, pos$data$y2), 0.1)
  expect_lt(cor(neg$data$y1, neg$data$y2), -0.1)
})

test_that("simulate_bnb returns the documented list structure", {
  s <- simulate_bnb(n = 100, beta1 = c("(Intercept)" = 0.5),
                    beta2 = c("(Intercept)" = 0.2),
                    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
  expect_true(all(c("data", "mu", "true", "settings", "meta") %in% names(s)))
  expect_false("coef_realized" %in% names(s))
  expect_true(all(c("y1", "y2") %in% names(s$data)))
  expect_equal(nrow(s$data), 100)
  expect_equal(ncol(s$mu), 2)
  expect_named(s$mu, c("mu1", "mu2"))
})

test_that("simulate_bnb marginal means match exp(intercept) for intercept-only model", {
  s <- simulate_bnb(n = 10000, beta1 = c("(Intercept)" = 1.0),
                    beta2 = c("(Intercept)" = 0.5),
                    dispersion = c(m1 = 0.5, m2 = 0.5), lambda = 0, seed = 7)
  expect_equal(mean(s$data$y1), exp(1.0), tolerance = 0.05)
  expect_equal(mean(s$data$y2), exp(0.5), tolerance = 0.05)
})

test_that("simulate_bnb data includes covariate columns", {
  s <- simulate_bnb(n = 50, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                    beta2 = c("(Intercept)" = 0.1, x2 = -0.2),
                    dispersion = c(m1 = 0.4, m2 = 0.4), seed = 2)
  expect_true(all(c("y1", "y2", "x1", "x2") %in% names(s$data)))
})

test_that("simulate_bnb true parameters are stored correctly", {
  s <- simulate_bnb(n = 50, beta1 = c("(Intercept)" = 0.3),
                    beta2 = c("(Intercept)" = 0.1),
                    dispersion = c(m1 = 0.6, m2 = 0.7), lambda = 0.05, seed = 3)
  expect_equal(s$true$dispersion[["m1"]], 0.6)
  expect_equal(s$true$dispersion[["m2"]], 0.7)
  expect_equal(s$true$lambda, 0.05)
  expect_equal(s$true$beta1[["(Intercept)"]], 0.3)
})

test_that("simulate_bnb errors when (Intercept) missing from beta1", {
  expect_error(
    simulate_bnb(n = 50, beta1 = c(x1 = 0.3),
                 beta2 = c("(Intercept)" = 0.1),
                 dispersion = c(m1 = 0.4, m2 = 0.4), seed = 1),
    "Intercept"
  )
})

test_that("simulate_bnb errors when dispersion names are wrong", {
  expect_error(
    simulate_bnb(n = 50, beta1 = c("(Intercept)" = 0.2),
                 beta2 = c("(Intercept)" = 0.1),
                 dispersion = c(0.4, 0.4), seed = 1),
    "m1.*m2|named"
  )
})

test_that("simulate_bnb errors when supplied covariates miss a column", {
  cov <- data.frame(x1 = rnorm(50))
  expect_error(
    simulate_bnb(n = 50, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                 beta2 = c("(Intercept)" = 0.1, x2 = -0.2),
                 dispersion = c(m1 = 0.4, m2 = 0.4),
                 covariates = cov, seed = 1),
    "missing required column"
  )
})

test_that("simulate_bnb errors when lambda is out of bounds", {
  expect_error(
    simulate_bnb(n = 50, beta1 = c("(Intercept)" = 0.5),
                 beta2 = c("(Intercept)" = 0.5),
                 dispersion = c(m1 = 0.5, m2 = 0.5), lambda = 1e6, seed = 1),
    "lambda"
  )
})

test_that("simulate_bnb marginal means are correct when (Intercept) is not listed first", {
  # beta1/beta2 name-index by position in matrix multiplication is a trap:
  # this must give the same means as the intercept-first ordering below.
  # $mu is a deterministic function of the design matrix given the seed, so
  # a small n gives identical coverage to a large one here.
  s_scrambled <- simulate_bnb(n = 200,
                              beta1 = c(x1 = 0.3, "(Intercept)" = 1.0),
                              beta2 = c(x1 = -0.2, "(Intercept)" = 0.5),
                              dispersion = c(m1 = 0.5, m2 = 0.5), lambda = 0, seed = 7)
  s_ordered <- simulate_bnb(n = 200,
                            beta1 = c("(Intercept)" = 1.0, x1 = 0.3),
                            beta2 = c("(Intercept)" = 0.5, x1 = -0.2),
                            dispersion = c(m1 = 0.5, m2 = 0.5), lambda = 0, seed = 7)
  expect_identical(s_scrambled$mu, s_ordered$mu)
})

test_that("simulate_bnb errors when covariates has the wrong number of rows", {
  cov <- data.frame(x1 = rnorm(30))
  expect_error(
    simulate_bnb(n = 50, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                 beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                 dispersion = c(m1 = 0.4, m2 = 0.4),
                 covariates = cov, seed = 1),
    "n = 50 rows"
  )
})
