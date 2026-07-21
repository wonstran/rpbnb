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

test_that("bnbr_rp_copula_ll(pois1=TRUE) matches a ppois-corner reference", {
  set.seed(11)
  n <- 40
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rnbinom(n, size = 2, mu = 1.5)
  # No random coefficients: R = 1, so the RP value reduces to the fixed pmf.
  # par order: beta1(2), beta2(2), log_m1, log_m2, z_theta. log_m1 (index 5) is
  # inert here because pois1 = TRUE forces r1 = Inf.
  par <- c(0.3, 0.1, 0.2, -0.1, 0, 0.4, 0.5)
  # Build the reference directly from the Poisson-margin pmf.
  mu1 <- as.vector(exp(X1 %*% par[1:2])); mu2 <- as.vector(exp(X2 %*% par[3:4]))
  r2 <- exp(-par[6]); theta <- rpbnb:::z_to_native("frank", par[7])
  pm <- rpbnb:::.copula_pmf(y1, y2, mu1, mu2, Inf, r2, theta, "frank")
  ref <- sum(log(pm$p_obs))

  val <- rpbnb:::bnbr_rp_copula_ll(
    par, y1, y2, X1, X2, NULL, NULL, integer(0), integer(0),
    matrix(0, 1, 0), matrix(0, 1, 0), "frank", pois1 = TRUE)
  expect_equal(as.numeric(val), ref, tolerance = 1e-8)
})

test_that("bnbr_rp_copula_ll_grad zeroes the log_m1 column for a Poisson margin", {
  set.seed(12)
  n <- 50
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rnbinom(n, size = 2, mu = 1.5)
  par <- c(0.3, 0.1, 0.2, -0.1, 0, 0.4, 0.5)  # index 5 = log_m1
  g <- attr(rpbnb:::bnbr_rp_copula_ll_grad(
    par, y1, y2, X1, X2, NULL, NULL, integer(0), integer(0),
    matrix(0, 1, 0), matrix(0, 1, 0), "frank", pois1 = TRUE), "gradient")
  expect_equal(g[5], 0)                    # log_m1 pinned -> zero gradient
  # Free columns match a numeric gradient of the frozen (pois1) objective.
  f <- function(p) as.numeric(rpbnb:::bnbr_rp_copula_ll(
    p, y1, y2, X1, X2, NULL, NULL, integer(0), integer(0),
    matrix(0, 1, 0), matrix(0, 1, 0), "frank", pois1 = TRUE))
  gnum <- numDeriv::grad(f, par)
  free <- c(1, 2, 3, 4, 6, 7)
  expect_equal(unname(g[free]), gnum[free], tolerance = 1e-5)
})
