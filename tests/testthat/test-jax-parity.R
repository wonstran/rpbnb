# Value and gradient parity between the TMB engine and the JAX engine.
# Skipped unless the project-local .venv-jax is present (tools/jax-setup.R).
#
# TMB is the reference, not numDeriv: a finite-difference check would agree
# with a ported formula error just as happily as with a correct port.

jax_fixture <- function(family_code, n = 60L, draws = 16L, seed = 11L) {
  set.seed(seed)
  x <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x = x)
  X2 <- cbind(`(Intercept)` = 1, x = x)
  Y1 <- rpois(n, exp(0.4 + 0.2 * x))
  Y2 <- rpois(n, exp(0.3 - 0.1 * x))
  Z <- .tmb_halton_uniform(draws, 2L, burn = 30L)
  list(
    data = .build_tmb_data(
      Y1 = as.numeric(Y1), Y2 = as.numeric(Y2), X1 = X1, X2 = X2,
      rand_idx1 = 2L, rand_idx2 = 2L,
      Z1 = Z[, 1, drop = FALSE], Z2 = Z[, 2, drop = FALSE],
      dist1 = 0L, dist2 = 0L, sign1 = 1L, sign2 = 1L,
      family_code = family_code, pois1 = FALSE, pois2 = FALSE,
      lamLo = -0.9, lamHi = 0.9, est_method = 0L
    ),
    k1 = 2L, k2 = 2L, q1 = 1L, q2 = 1L
  )
}

# Template order: b1(k1), b2(k2), log_sd1(q1), log_sd2(q2), log_m1, log_m2, z_dep
jax_start <- function(z_dep = 0.3) {
  c(0.4, 0.2, 0.3, -0.1, log(0.25), log(0.30), log(0.5), log(0.6), z_dep)
}

expect_jax_parity <- function(family_code, pars, tol = 1e-8) {
  fx <- jax_fixture(family_code)
  start <- jax_start()
  free <- rep(TRUE, length(start))
  if (family_code < 0L) free[length(start)] <- FALSE  # z_dep pinned

  tmb <- .make_rpbnb_tmb_object(
    data = fx$data,
    parameters = list(
      beta1 = start[1:2], beta2 = start[3:4],
      log_sd1 = start[5], log_sd2 = start[6],
      log_m1 = start[7], log_m2 = start[8],
      z_dep = start[9],
      u1 = matrix(0, length(fx$data$Y1), 1L),
      u2 = matrix(0, length(fx$data$Y1), 1L)
    ),
    map = c(
      if (family_code < 0L) list(z_dep = factor(NA)),
      list(u1 = factor(rep(NA_integer_, length(fx$data$Y1))),
           u2 = factor(rep(NA_integer_, length(fx$data$Y1))))
    ),
    random = NULL, silent = TRUE, n_cores = 1L, max_threads = 1L
  )$obj
  jx <- .make_rpbnb_jax_object(fx$data, start, free)

  for (p in pars) {
    pf <- p[free]
    expect_equal(jx$fn(pf), tmb$fn(pf), tolerance = tol)
    expect_equal(jx$gr(pf), as.numeric(tmb$gr(pf)), tolerance = tol)
  }
}

test_that("JAX matches TMB for independence", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(-1L, list(jax_start(), jax_start() + 0.15))
})

test_that("JAX matches TMB for Famoye", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(0L, list(jax_start(0.3), jax_start(-0.8)))
})
