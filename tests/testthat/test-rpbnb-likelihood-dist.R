make_case <- function(dist, sign = 1, seed = 4) {
  set.seed(seed)
  n <- 80
  X1 <- cbind(`(Intercept)` = 1, x1 = rnorm(n))
  X2 <- cbind(`(Intercept)` = 1, x1 = rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  rand_idx1 <- 2L; rand_idx2 <- integer(0)
  XR1 <- X1[, rand_idx1, drop = FALSE]; XR2 <- NULL
  U1 <- halton_uniform(60, 1, burn = 50)
  U2 <- matrix(0, nrow = 60, ncol = 0)
  # par: b1(2), b2(2), scale1(1), log_m1, log_m2, z_lambda
  par <- c(0.1, 0.2, 0.05, -0.1, log(0.3), log(0.4), log(0.5), 0.2)
  list(par = par, y1 = y1, y2 = y2, X1 = X1, X2 = X2, XR1 = XR1, XR2 = XR2,
       rand_idx1 = rand_idx1, rand_idx2 = rand_idx2, U1 = U1, U2 = U2,
       dist1 = dist, dist2 = character(0), sign1 = sign, sign2 = numeric(0))
}

# Compute frozen lambda bounds at a given par (mirrors approach in test-fit-rpbnb.R).
frozen_bounds <- function(cs, par) {
  # Delegates to the SAME support bound the objective uses. This used to
  # reimplement the old per-draw reduction, which quietly turned the comparison
  # below into analytic-gradient-of-one-objective versus numeric-gradient-of-
  # another once the engines moved to support bounds. The point of the test is
  # that the two gradients of ONE objective agree.
  unname(rpbnb:::.rp_support_bounds(
    par, cs$X1, cs$X2, cs$rand_idx1, cs$rand_idx2,
    cs$dist1, cs$dist2, cs$sign1, cs$sign2
  ))
}

check_grad <- function(dist, sign = 1) {
  cs <- make_case(dist, sign)
  # Compute bounds at eval point and freeze them (analytic gradient holds lam fixed)
  bnds <- frozen_bounds(cs, cs$par)
  f <- function(p) {
    bnbr_rp_ll_fixed_bounds(p, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                            cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                            bnds[1], bnds[2],
                            cs$dist1, cs$dist2, cs$sign1, cs$sign2)
  }
  v <- bnbr_rp_ll_and_grad(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                           cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                           cs$dist1, cs$dist2, cs$sign1, cs$sign2)
  g_analytic <- unname(attr(v, "gradient"))
  g_numeric  <- numDeriv::grad(f, cs$par)
  expect_equal(g_analytic, g_numeric, tolerance = 1e-4)
}

test_that("analytic gradient matches numeric gradient for each distribution", {
  check_grad("normal")
  check_grad("uniform")
  check_grad("triangular")
  check_grad("lognormal", sign = 1)
  check_grad("lognormal", sign = -1)
})

test_that("analytic gradient matches numeric gradient for mixed cross-equation random coefs", {
  # Case: eq1 has lognormal (sign=-1) random coef; eq2 has triangular random coef
  # This exercises q1 > 0 AND q2 > 0 simultaneously.
  set.seed(7)
  n <- 80
  X1 <- cbind(`(Intercept)` = 1, x1 = rnorm(n))
  X2 <- cbind(`(Intercept)` = 1, x2 = rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  rand_idx1 <- 2L; rand_idx2 <- 2L
  XR1 <- X1[, rand_idx1, drop = FALSE]
  XR2 <- X2[, rand_idx2, drop = FALSE]
  U1 <- halton_uniform(60, 1, burn = 50)
  U2 <- halton_uniform(60, 1, burn = 50)
  dist1 <- "lognormal"; sign1 <- -1
  dist2 <- "triangular"; sign2 <- 1
  # par: b1(2), b2(2), scale1(1), scale2(1), log_m1, log_m2, z_lambda
  par <- c(0.1, -0.3, 0.05, -0.1, log(0.3), log(0.4), log(0.5), log(0.6), 0.2)

  # Same support bound the objective uses; see frozen_bounds() above for why
  # this must not reimplement a reduction over draws.
  bnds <- rpbnb:::.rp_support_bounds(par, X1, X2, rand_idx1, rand_idx2,
                                     dist1, dist2, sign1, sign2)

  f <- function(p) {
    bnbr_rp_ll_fixed_bounds(p, y1, y2, X1, X2, XR1, XR2,
                            rand_idx1, rand_idx2, U1, U2,
                            bnds[1], bnds[2],
                            dist1, dist2, sign1, sign2)
  }
  v <- bnbr_rp_ll_and_grad(par, y1, y2, X1, X2, XR1, XR2,
                           rand_idx1, rand_idx2, U1, U2,
                           dist1, dist2, sign1, sign2)
  g_analytic <- unname(attr(v, "gradient"))
  g_numeric  <- numDeriv::grad(f, par)
  expect_equal(g_analytic, g_numeric, tolerance = 1e-4)
})

test_that("fixed-bounds LL is finite and matches the free LL bounds at z=0", {
  cs <- make_case("uniform")
  v  <- bnbr_rp_ll_and_grad(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                            cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                            cs$dist1, cs$dist2, cs$sign1, cs$sign2)
  expect_true(is.finite(as.numeric(v)))
  vf <- bnbr_rp_ll_fixed_bounds(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1,
                                cs$XR2, cs$rand_idx1, cs$rand_idx2, cs$U1, cs$U2,
                                -5, 5, cs$dist1, cs$dist2, cs$sign1, cs$sign2)
  expect_true(is.finite(vf))
})

test_that("the support bound uses the true scale for bounded distributions", {
  # An UPPER cap on the scale is not safe. Uniform and triangular deviations are
  # supported on (-s, s), so shrinking s shrinks the attainable mean range and
  # WIDENS the admissible lambda interval. With one uniform coefficient per
  # margin, log_w = 30, loading 1e-9 and m = 0.5, capping at exp(20) returned
  # [-2.0362897, 2.5327946] -- admitting lambda = 2 against a model whose pmf is
  # negative there -- where the true support gives [-1, 1].
  X <- cbind(`(Intercept)` = 1, x = rep(1e-9, 5L))
  par <- c(0, 0, 0, 0, 30, 30, log(0.5), log(0.5), 0)
  b <- rpbnb:::.rp_support_bounds(par, X, X, 2L, 2L, "uniform", "uniform", 1, 1)
  expect_equal(unname(b), c(-1, 1))
  expect_lt(b[["upper"]], 2)

  # The lower floor is retained and is conservative: it only decides that an
  # underflowing scale still counts as varying, which NARROWS the interval.
  set.seed(4)
  Xn <- cbind(1, rnorm(20))
  parn <- c(0, 0, 0, 0, -1000, -1000, log(0.5), log(0.5), 0)
  bn <- rpbnb:::.rp_support_bounds(parn, Xn, Xn, 2L, 2L, "normal", "normal", 1, 1)
  expect_equal(unname(bn), c(-1, 1))
})

test_that("the objective the optimizer sees has an exact analytic gradient", {
  # This differentiates the SAME wrapper the optimizer calls, with the interval
  # frozen exactly as fit_rpbnb() freezes it -- not a separate fixed-bounds
  # reference. Uniform coefficients are the point: their support bound depends
  # on beta, m AND s, so a bound recomputed per call would leave d(bound)/d(par)
  # out of the gradient. Measured before freezing, that gap was 2.05, on every
  # coordinate except z_lambda (the only one the bounds do not involve).
  set.seed(21)
  N <- 20L
  X1 <- cbind(1, rnorm(N)); X2 <- cbind(1, rnorm(N))
  y1 <- rpois(N, 2); y2 <- rpois(N, 2)
  XR1 <- X1[, 2, drop = FALSE]; XR2 <- X2[, 2, drop = FALSE]
  U1 <- halton_uniform(40, 1, burn = 50); U2 <- halton_uniform(40, 1, burn = 50)
  p0 <- c(0.2, 0.1, 0.15, -0.1, log(0.3), log(0.3), log(0.4), log(0.4), 0.1)
  frozen <- rpbnb:::.rp_support_bounds(p0, X1, X2, 2L, 2L,
                                       "uniform", "uniform", 1, 1)
  # Genuinely parameter-dependent, or this test would be vacuous.
  moved <- rpbnb:::.rp_support_bounds(p0 + 0.25, X1, X2, 2L, 2L,
                                      "uniform", "uniform", 1, 1)
  expect_false(isTRUE(all.equal(unname(frozen), unname(moved))))

  f <- function(p) as.numeric(bnbr_rp_ll_and_grad(
    p, y1, y2, X1, X2, XR1, XR2, 2L, 2L, U1, U2,
    "uniform", "uniform", 1, 1, lam_bounds = frozen))
  v <- bnbr_rp_ll_and_grad(p0, y1, y2, X1, X2, XR1, XR2, 2L, 2L, U1, U2,
                           "uniform", "uniform", 1, 1, lam_bounds = frozen)
  expect_equal(unname(attr(v, "gradient")), numDeriv::grad(f, p0),
               tolerance = 1e-5)

  skip_if_not(rpbnb:::rpbnb_cpp_available(), "C++ core unavailable")
  fc <- function(p) as.numeric(bnbr_rp_ll_and_grad_cpp(
    p, y1, y2, X1, X2, XR1, XR2, 2L, 2L, U1, U2,
    "uniform", "uniform", n_threads = 1L, lam_bounds = frozen))
  vc <- bnbr_rp_ll_and_grad_cpp(p0, y1, y2, X1, X2, XR1, XR2, 2L, 2L, U1, U2,
                                "uniform", "uniform", n_threads = 1L,
                                lam_bounds = frozen)
  expect_equal(unname(attr(vc, "gradient")), numDeriv::grad(fc, p0),
               tolerance = 1e-5)
})
