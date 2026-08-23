# Value and gradient parity between the TMB engine and the JAX engine.
# Skipped unless the project-local .venv-jax is present (tools/jax-setup.R).
#
# TMB is the reference, not numDeriv: a finite-difference check would agree
# with a ported formula error just as happily as with a correct port.

# The fixture exposes every switch the objective branches on, because parity
# only covers the branches the data selects. pois*/dist*/sign* default to the
# combination the first two tests used, and the later tests vary them.
jax_fixture <- function(family_code, n = 60L, draws = 16L, seed = 11L,
                        pois1 = FALSE, pois2 = FALSE,
                        dist1 = 0L, dist2 = 0L, sign1 = 1L, sign2 = 1L) {
  set.seed(seed)
  x <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x = x)
  X2 <- cbind(`(Intercept)` = 1, x = x)
  Y1 <- rpois(n, exp(0.4 + 0.2 * x))
  Y2 <- rpois(n, exp(0.3 - 0.1 * x))
  Z <- .tmb_halton_uniform(draws, 2L, burn = 30L)
  .build_tmb_data(
    Y1 = as.numeric(Y1), Y2 = as.numeric(Y2), X1 = X1, X2 = X2,
    rand_idx1 = 2L, rand_idx2 = 2L,
    Z1 = Z[, 1, drop = FALSE], Z2 = Z[, 2, drop = FALSE],
    dist1 = dist1, dist2 = dist2, sign1 = sign1, sign2 = sign2,
    family_code = family_code, pois1 = pois1, pois2 = pois2,
    lamLo = -0.9, lamHi = 0.9, est_method = 0L
  )
}

# Template order: b1(k1), b2(k2), log_sd1(q1), log_sd2(q2), log_m1, log_m2, z_dep
jax_start <- function(z_dep = 0.3) {
  c(0.4, 0.2, 0.3, -0.1, log(0.25), log(0.30), log(0.5), log(0.6), z_dep)
}

# R pins z_dep for family < 0; every other coordinate is free.
jax_free_mask <- function(family_code, start = jax_start()) {
  free <- rep(TRUE, length(start))
  if (family_code < 0L) free[length(start)] <- FALSE
  free
}

# Perturb only the free coordinates. Shifting a pinned one would be ignored
# by both engines, which is exactly the silent no-op expect_jax_parity()
# now refuses.
jax_perturb <- function(start, free, delta) {
  start[free] <- start[free] + delta
  start
}

# `pars` is a list of FULL-length parameter vectors. Pinned coordinates must
# equal those of `start`, since both engines take them from `start`; a
# disagreement is an error rather than a silently ignored value.
expect_jax_parity <- function(family_code, pars, start = jax_start(),
                              fixture = jax_fixture(family_code),
                              obs_chunk = 256L, tol = 1e-8) {
  free <- jax_free_mask(family_code, start)
  n <- length(fixture$Y1)

  tmb <- .make_rpbnb_tmb_object(
    data = fixture,
    parameters = list(
      beta1 = start[1:2], beta2 = start[3:4],
      log_sd1 = start[5], log_sd2 = start[6],
      log_m1 = start[7], log_m2 = start[8],
      z_dep = start[9],
      u1 = matrix(0, n, 1L), u2 = matrix(0, n, 1L)
    ),
    map = c(
      if (family_code < 0L) list(z_dep = factor(NA)),
      list(u1 = factor(rep(NA_integer_, n)),
           u2 = factor(rep(NA_integer_, n)))
    ),
    random = NULL, silent = TRUE, n_cores = 1L, max_threads = 1L
  )$obj
  jx <- .make_rpbnb_jax_object(fixture, start, free, obs_chunk = obs_chunk)

  for (p in pars) {
    # The TMB object bakes the pinned coordinates in at construction and the
    # JAX template takes them from `start`, so a `p` that disagrees on a
    # pinned coordinate would be evaluated at `start`'s value by BOTH engines
    # and pass while testing nothing.
    expect_identical(p[!free], start[!free])
    pf <- p[free]
    expect_equal(jx$fn(pf), tmb$fn(pf), tolerance = tol)
    expect_equal(jx$gr(pf), as.numeric(tmb$gr(pf)), tolerance = tol)
  }
}

test_that("JAX matches TMB for independence", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  free <- jax_free_mask(-1L)
  expect_jax_parity(-1L, list(jax_start(),
                              jax_perturb(jax_start(), free, 0.15)))
})

test_that("JAX matches TMB for Famoye", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(0L, list(jax_start(0.3), jax_start(-0.8)))
})

# --- Branches the two tests above never select -----------------------------
# The fixture above fixes pois1 = pois2 = FALSE and dist = 0 (normal), so it
# exercises neither log_dpois, nor _eta_floor's Poisson form (a bare -35
# rather than the m-dependent floor), nor _famoye_c's Poisson branch, nor
# three of the four _u_to_base/_compute_dev pairs.

test_that("JAX matches TMB with a Poisson first margin", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  # Mixed on purpose: one margin takes the Poisson eta floor and log_dpois,
  # the other stays NB2, so a floor swapped between margins shows up.
  expect_jax_parity(
    0L, list(jax_start(0.3), jax_start(-0.8)),
    fixture = jax_fixture(0L, pois1 = TRUE, pois2 = FALSE))
})

test_that("JAX matches TMB with both margins Poisson", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  free <- jax_free_mask(-1L)
  expect_jax_parity(
    -1L, list(jax_start(), jax_perturb(jax_start(), free, 0.15)),
    fixture = jax_fixture(-1L, pois1 = TRUE, pois2 = TRUE))
})

test_that("JAX matches TMB for lognormal random coefficients", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  # sign * exp(b + s * base) - b is the most mis-portable line in
  # objective.py: it is the only compute_dev form that reads beta as well as
  # the scale, and the only one whose sign code does anything.
  expect_jax_parity(
    0L, list(jax_start(0.3), jax_start(-0.8)),
    fixture = jax_fixture(0L, dist1 = 1L, dist2 = 1L,
                          sign1 = 1L, sign2 = 1L))
})

test_that("JAX matches TMB for a negatively signed lognormal coefficient", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(
    0L, list(jax_start(0.3), jax_start(-0.8)),
    fixture = jax_fixture(0L, dist1 = 1L, dist2 = 1L,
                          sign1 = -1L, sign2 = -1L))
})

test_that("JAX matches TMB for uniform random coefficients", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(
    0L, list(jax_start(0.3), jax_start(-0.8)),
    fixture = jax_fixture(0L, dist1 = 2L, dist2 = 2L))
})

test_that("JAX matches TMB for triangular random coefficients", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(
    0L, list(jax_start(0.3), jax_start(-0.8)),
    fixture = jax_fixture(0L, dist1 = 3L, dist2 = 3L))
})

# --- Copula families ------------------------------------------------------
# The first family to reach the count CDF triples at all: every test above
# takes a closed-form log mass and never builds the (n, R, kmax+1) grid.

test_that("JAX matches TMB for the Frank copula", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  # theta = 35 * tanh(z_dep / 35), so these are z_dep = 0.1 -> 0.0999,
  # 2.0 -> 1.996 and -1.5 -> -1.499. The first two take the kernel's th > 0
  # branch and the third its three-regime th < 0 branch; 0.1 is also the
  # value R starts Frank at (R/fit_rpbnb_tmb.R:358-363), so it is the point
  # a real fit's first gradient is taken at.
  expect_jax_parity(1L, list(jax_start(0.1), jax_start(2.0), jax_start(-1.5)))
})

test_that("JAX matches TMB for Frank at exact independence", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  # z_dep = 0 gives theta = 0 exactly, the copula's removable singularity and
  # the only point that selects frank_log_cell_prob()'s |th| < 1e-5
  # expansion -- a third of the kernel, and untested against the reference by
  # the three points above. R avoids starting a fit here because TMB's tape
  # loses the dependence score at exactly 0 (R/fit_rpbnb_tmb.R:358-363); that
  # is a property of where the fit BEGINS, not of the value at this point,
  # and both engines agree here to 2e-13 on fn and 5e-14 on gr.
  expect_jax_parity(1L, list(jax_start(0), jax_start(1e-6)))
})
