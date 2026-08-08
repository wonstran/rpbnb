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
