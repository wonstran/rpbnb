test_that(".copula_pmf uses ppois corners for a Poisson (r = Inf) margin", {
  y1 <- c(0L, 2L, 5L); y2 <- c(1L, 0L, 3L)
  mu1 <- c(1.5, 4.0, 2.0); mu2 <- c(0.8, 2.0, 5.0)
  r2 <- 3.0; theta <- 0.5; family <- "frank"

  pm <- rpbnb:::.copula_pmf(y1, y2, mu1, mu2, Inf, r2, theta, family)

  # Margin 1 corners must equal the exact Poisson CDF, not an NB approximation.
  expect_equal(pm$a,  stats::ppois(y1, mu1))
  expect_equal(pm$am, ifelse(y1 > 0L, stats::ppois(y1 - 1L, mu1), 0))
  # Margin 2 stays NB2.
  expect_equal(pm$b,  stats::pnbinom(y2, size = r2, mu = mu2))
  expect_true(all(pm$ok))
})

test_that(".copula_score_scalars zeroes s_logm and uses dpois for a Poisson margin", {
  y1 <- c(0L, 2L, 5L); y2 <- c(1L, 0L, 3L)
  mu1 <- c(1.5, 4.0, 2.0); mu2 <- c(0.8, 2.0, 5.0)
  r2 <- 3.0; theta <- 0.5; dth_dz <- 1.0; family <- "frank"

  sc <- rpbnb:::.copula_score_scalars(y1, y2, mu1, mu2, Inf, r2, theta, dth_dz, family)

  # The pinned Poisson dispersion contributes no score.
  expect_equal(sc$s_logm1, rep(0, length(y1)))
  # Margin-1 mean score is finite (dpois path), not NaN from dnbinom(size=Inf).
  expect_true(all(is.finite(sc$s_eta1)))
  # Margin 2's dispersion score is unaffected.
  expect_true(any(sc$s_logm2 != 0))
})

test_that(".copula_score_scalars s_eta1 matches a numeric derivative (Poisson margin)", {
  y1 <- c(0L, 2L, 5L); y2 <- c(1L, 0L, 3L)
  mu1 <- c(1.5, 4.0, 2.0); mu2 <- c(0.8, 2.0, 5.0)
  r2 <- 3.0; theta <- 0.5; dth_dz <- 1.0; family <- "frank"
  sc <- rpbnb:::.copula_score_scalars(y1, y2, mu1, mu2, Inf, r2, theta, dth_dz, family)
  h <- 1e-6
  lp <- function(m1) log(rpbnb:::.copula_pmf(y1, y2, m1, mu2, Inf, r2, theta, family)$p_obs)
  # s_eta1 = d log p / d eta1; an eta1 shift is a multiplicative exp-shift in mu1.
  num <- (lp(mu1 * exp(h)) - lp(mu1 * exp(-h))) / (2 * h)
  expect_equal(sc$s_eta1, num, tolerance = 1e-5)
})

test_that(".copula_pmf/.copula_score_scalars handle a Poisson margin 2 (and both)", {
  y1 <- c(0L, 2L, 5L); y2 <- c(1L, 0L, 3L)
  mu1 <- c(1.5, 4.0, 2.0); mu2 <- c(0.8, 2.0, 5.0)
  r1 <- 2.0; theta <- 0.5; dth_dz <- 1.0; family <- "frank"
  pm <- rpbnb:::.copula_pmf(y1, y2, mu1, mu2, r1, Inf, theta, family)
  expect_equal(pm$b,  stats::ppois(y2, mu2))
  expect_equal(pm$bm, ifelse(y2 > 0L, stats::ppois(y2 - 1L, mu2), 0))
  sc <- rpbnb:::.copula_score_scalars(y1, y2, mu1, mu2, r1, Inf, theta, dth_dz, family)
  expect_equal(sc$s_logm2, rep(0, length(y2)))
  # both margins Poisson
  pm2 <- rpbnb:::.copula_pmf(y1, y2, mu1, mu2, Inf, Inf, theta, family)
  expect_equal(pm2$a, stats::ppois(y1, mu1))
  expect_equal(pm2$b, stats::ppois(y2, mu2))
  expect_true(all(pm2$ok))
})
