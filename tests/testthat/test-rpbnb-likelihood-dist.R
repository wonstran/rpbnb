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
  k1 <- ncol(cs$X1); k2 <- ncol(cs$X2)
  q1 <- length(cs$rand_idx1); q2 <- length(cs$rand_idx2)
  beta1 <- par[seq_len(k1)]; beta2 <- par[(k1+1):(k1+k2)]
  idx_end <- k1 + k2 + q1 + q2
  sd1 <- if (q1 > 0) exp(par[(k1+k2+1):(k1+k2+q1)]) else numeric(0)
  m1 <- exp(par[idx_end+1]); m2 <- exp(par[idx_end+2])
  xb1 <- as.vector(cs$X1 %*% beta1); xb2 <- as.vector(cs$X2 %*% beta2)
  real1 <- if (q1 > 0) rand_realize(cs$U1, cs$dist1, cs$sign1,
                                    b = beta1[cs$rand_idx1], s = sd1)
           else list(dev = matrix(0, nrow(cs$U1), 0))
  R <- nrow(cs$U1); lo <- -Inf; hi <- Inf
  for (r in seq_len(R)) {
    mu1 <- pmin(exp(xb1 + if (q1 > 0) as.vector(cs$XR1 %*% real1$dev[r, ]) else 0), 1e15)
    mu2 <- pmin(exp(xb2), 1e15)
    b <- lambda_bounds_vec(c_val(mu1, m1), c_val(mu2, m2))
    lo <- max(lo, b[1]); hi <- min(hi, b[2])
  }
  c(lo, hi)
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
