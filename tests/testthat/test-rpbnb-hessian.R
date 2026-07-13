# The analytic observed-information Hessian bnbr_rp_hessian() must match the
# numeric Hessian of the same frozen-bounds simulated log-likelihood. The
# numeric Hessian is the oracle; finite-difference noise is ~1e-3 absolute, so
# we compare on relative error.

hess_case <- function(random_cols_1, random_cols_2, par,
                      dists1 = NULL, dists2 = NULL, signs1 = NULL, signs2 = NULL,
                      n = 50, R = 64, lo = -0.4, hi = 0.4, seed = 7) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1, x2 = x2); X2 <- X1
  y1 <- rnbinom(n, mu = exp(0.4 + 0.2 * x1), size = 2)
  y2 <- rnbinom(n, mu = exp(0.3 - 0.1 * x2), size = 2)
  ri1 <- match(random_cols_1, colnames(X1)); ri2 <- match(random_cols_2, colnames(X2))
  q1 <- length(ri1); q2 <- length(ri2)
  XR1 <- if (q1) X1[, ri1, drop = FALSE] else NULL
  XR2 <- if (q2) X2[, ri2, drop = FALSE] else NULL
  if (is.null(dists1)) dists1 <- rep("normal", q1)
  if (is.null(dists2)) dists2 <- rep("normal", q2)
  if (is.null(signs1)) signs1 <- rep(1, q1)
  if (is.null(signs2)) signs2 <- rep(1, q2)
  set.seed(99); Z <- halton_uniform(R, q1 + q2, burn = 50)
  Z1 <- if (q1) Z[, 1:q1, drop = FALSE] else matrix(0, R, 0)
  Z2 <- if (q2) Z[, (q1 + 1):(q1 + q2), drop = FALSE] else matrix(0, R, 0)

  Hana <- bnbr_rp_hessian(par, y1, y2, X1, X2, XR1, XR2, ri1, ri2, Z1, Z2,
                          dists1, dists2, signs1, signs2, lamLo = lo, lamHi = hi)
  f <- function(p) bnbr_rp_ll_fixed_bounds(p, y1, y2, X1, X2, XR1, XR2, ri1, ri2,
                                           Z1, Z2, lo, hi, dists1, dists2, signs1, signs2)
  Hnum <- numDeriv::hessian(f, par, method.args = list(r = 6, eps = 1e-4))
  list(ana = Hana, num = Hnum, rel = max(abs(Hana - Hnum)) / max(abs(Hnum)))
}

test_that("analytic Hessian matches numeric (all-normal random coefs)", {
  r <- hess_case(c("(Intercept)", "x1", "x2"), c("(Intercept)", "x1", "x2"),
                 par = c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1,
                         log(0.2), log(0.2), log(0.2), log(0.2), log(0.2), log(0.2),
                         log(0.5), log(0.5), 0.1))
  expect_lt(r$rel, 5e-3)
})

test_that("analytic Hessian matches numeric (subset random coefs)", {
  r <- hess_case(c("x1"), c("(Intercept)", "x2"),
                 par = c(0.5, 0.1, -0.2, 0.2, 0.3, 0.0,
                         log(0.3), log(0.25), log(0.15), log(0.4), log(0.6), -0.2))
  expect_lt(r$rel, 5e-3)
})

test_that("analytic Hessian matches numeric (no random coefs)", {
  r <- hess_case(character(0), character(0),
                 par = c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1, log(0.5), log(0.5), 0.1))
  expect_lt(r$rel, 5e-3)
})

test_that("analytic Hessian matches numeric (lognormal random coef)", {
  r <- hess_case(c("x1"), c("x1"),
                 par = c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1,
                         log(0.2), log(0.2), log(0.5), log(0.5), 0.05),
                 dists1 = "lognormal", dists2 = "normal", signs1 = 1, signs2 = 1)
  expect_lt(r$rel, 5e-3)
})

test_that("analytic Hessian is symmetric", {
  r <- hess_case(c("x1", "x2"), c("x1"),
                 par = c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1,
                         log(0.2), log(0.2), log(0.2), log(0.5), log(0.5), 0.1))
  expect_equal(r$ana, t(r$ana), tolerance = 1e-10)
})
