# Famoye/Sarmanov bivariate NB (NB2) likelihood internals.
# Parameter vector: [beta1 (p1), beta2 (p2), log_m1, log_m2, z_lambda].
# Core math (c_val, nb_logpmf_y_mu_r, lambda_bounds_vec, dct_dm, dc_dbeta_mat,
# d_const) lives in famoye_core.R; these functions call those helpers.
#
# Gradient note: the objective (bnb_loglik_vec) recomputes the data-adaptive
# global lambda bounds at every parameter value, while the analytic score
# (bnb_score_mat) treats those bounds as constant -- it omits the derivative of
# the max/min-over-observations bounds, which are non-smooth and contribute only
# through one binding observation. Away from the optimum the two can differ, but
# at the solution the bounds are locally stationary and the analytic gradient
# matches the true objective gradient to ~1e-3 (both ~0); see the regression test
# "analytic (frozen-bounds) gradient agrees with the true objective gradient" in
# tests/testthat/test-fit-bnb.R. The Hessian/SEs use the same frozen-bounds
# objective (bnbr_loglik_fixed_bounds) for consistency.
#
# DECISION -- this is the opposite trade-off from the RP engine, on purpose.
# There are two coherent ways to handle a lambda interval that depends on the
# parameters, and this package uses one on each Famoye fitter:
#
#   * fit_bnb (here): MOVING bounds. The interval is recomputed from the
#     per-observation c values at every evaluation, so the fitted lambda is
#     admissible BY CONSTRUCTION and no post-fit validity check is needed.
#     The price is the gradient inconsistency documented above -- search
#     directions en route are not exact derivatives -- mitigated by the
#     multi-start policy in fit_bnb.R and accepted because the fixed-parameter
#     objective is cheap enough to multi-start.
#   * fit_rpbnb / fit_rpbnb_tmb: FROZEN bounds. The interval is fixed at the
#     starting values, so the analytic gradient is exactly the derivative of
#     the optimized objective (a property the simulated likelihood's expensive
#     evaluations cannot buy back by multi-starting). The price is that the
#     optimizer can escape the region admissible at the optimum, which is
#     detected post fit (lambda_admissible) but not prevented.
#
# Neither guarantee can be had for free with the other; changing either side
# means re-litigating this note, the multi-start policy, and the RP engines'
# post-fit guard together.

#' Per-observation log-likelihood (vector) for the Famoye BNB model
#'
#' `pois1`/`pois2` restrict the corresponding margin to its exact Poisson limit
#' (m = 0): the NB2 dispersion is forced to exactly 0 (r = Inf), so the margin's
#' log-pmf is dpois and its dependence constant c is exp(-d*mu). This is the
#' exact m = 0 restriction, not the fixed-POISSON_M numerical stand-in, so it is
#' accurate at any fitted mean. The pinned log_m in `par` is then only a display
#' placeholder -- the math ignores its value.
#' @keywords internal
#' @noRd
bnb_loglik_vec <- function(par, y1, y2, X1, X2, pois1 = FALSE, pois2 = FALSE,
                           off1 = NULL, off2 = NULL) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]

  m1 <- if (pois1) 0 else exp(log_m1); m2 <- if (pois2) 0 else exp(log_m2)
  r1 <- if (pois1) Inf else 1/m1;      r2 <- if (pois2) Inf else 1/m2
  mu1 <- .bound_mu(X1, beta1, off1)
  mu2 <- .bound_mu(X2, beta2, off2)
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)

  # Data-adaptive bounds and interior logistic map (strictly inside)
  bnds <- lambda_bounds_vec(c1, c2); lamLo <- bnds[1]; lamHi <- bnds[2]
  eps  <- 1e-6
  sig  <- plogis(zlam)
  lam  <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)

  lnb1 <- nb_logpmf_y_mu_r(y1, mu1, r1)
  lnb2 <- nb_logpmf_y_mu_r(y2, mu2, r2)
  dep  <- 1 + lam * (exp(-y1) - c1) * (exp(-y2) - c2)
  dep  <- pmax(dep, 1e-300)

  lnb1 + lnb2 + log(dep)
}

#' Analytic score (per observation) for the Famoye BNB model
#'
#' `pois1`/`pois2` select the exact Poisson (m = 0) margin: the beta/dependence
#' score pieces reduce to their exact Poisson limits (w = y - mu, dc/dbeta =
#' -d*c*mu*x with c = exp(-d*mu)), and the log_m score column -- a fixed,
#' unestimated parameter for a Poisson margin -- is set to 0 (its NB2 value is a
#' 0/0 at m = 0 and is dropped from the free-parameter information anyway).
#' @keywords internal
#' @noRd
bnb_score_mat <- function(par, y1, y2, X1, X2, pois1 = FALSE, pois2 = FALSE,
                          off1 = NULL, off2 = NULL) {
  p1 <- ncol(X1); p2 <- ncol(X2)

  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]

  m1 <- if (pois1) 0 else exp(log_m1); m2 <- if (pois2) 0 else exp(log_m2)
  r1 <- if (pois1) Inf else 1/m1;      r2 <- if (pois2) Inf else 1/m2
  mu1 <- .bound_mu(X1, beta1, off1)
  mu2 <- .bound_mu(X2, beta2, off2)
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)

  bnds <- lambda_bounds_vec(c1, c2); lamLo <- bnds[1]; lamHi <- bnds[2]
  eps  <- 1e-6; sig <- plogis(zlam)
  lam  <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  dlam_dz <- (lamHi - lamLo) * (1 - 2*eps) * sig * (1 - sig)

  k1 <- exp(-y1) - c1
  k2 <- exp(-y2) - c2
  dep <- pmax(1 + lam * (k1 * k2), 1e-300)
  inv_dep <- 1/dep

  # beta blocks
  w1 <- (y1 - mu1) / (1 + m1 * mu1)
  w2 <- (y2 - mu2) / (1 + m2 * mu2)
  dc1_dbetas <- dc_dbeta_mat(mu1, m1, c1, X1)     # n x p1
  dc2_dbetas <- dc_dbeta_mat(mu2, m2, c2, X2)     # n x p2
  pen1 <- lam * k2 * inv_dep
  pen2 <- lam * k1 * inv_dep
  score_beta1_mat <- sweep(X1, 1, w1, `*`) - sweep(dc1_dbetas, 1, pen1, `*`)
  score_beta2_mat <- sweep(X2, 1, w2, `*`) - sweep(dc2_dbetas, 1, pen2, `*`)

  # m blocks (A2-A3) + chain to log m. A Poisson margin's log_m is fixed (not
  # estimated), so its score column is 0 -- the NB2 formula is a 0/0 at m = 0.
  if (pois1) {
    score_logm1 <- numeric(length(y1))
  } else {
    S1 <- digamma(r1 + y1) - digamma(r1)
    dc1_dm1 <- dct_dm(mu1, m1, c1)
    s_m1 <-  (m1^(-2)) * ( -S1 + log(m1) + log(mu1 + r1) - 1 + (y1 + r1)/(mu1 + r1) ) -
      (lam * k2 * inv_dep) * dc1_dm1
    score_logm1 <- m1 * s_m1
  }
  if (pois2) {
    score_logm2 <- numeric(length(y2))
  } else {
    S2 <- digamma(r2 + y2) - digamma(r2)
    dc2_dm2 <- dct_dm(mu2, m2, c2)
    s_m2 <-  (m2^(-2)) * ( -S2 + log(m2) + log(mu2 + r2) - 1 + (y2 + r2)/(mu2 + r2) ) -
      (lam * k1 * inv_dep) * dc2_dm2
    score_logm2 <- m2 * s_m2
  }

  # lambda part (A1) + chain rule z -> lambda
  dL_dlambda <- (k1 * k2) * inv_dep
  score_z <- dL_dlambda * dlam_dz

  cbind(score_beta1_mat, score_beta2_mat, score_logm1, score_logm2, score_z)
}

#' Summed analytic gradient for BFGS
#' @keywords internal
#' @noRd
bnb_grad_vec <- function(par, y1, y2, X1, X2, pois1 = FALSE, pois2 = FALSE,
                         off1 = NULL, off2 = NULL) {
  colSums(bnb_score_mat(par, y1, y2, X1, X2, pois1 = pois1, pois2 = pois2,
                        off1 = off1, off2 = off2))
}

#' Fixed-bounds summed logLik for numeric Hessian (freeze lambda-bounds at optimum)
#' @keywords internal
#' @noRd
bnbr_loglik_fixed_bounds <- function(par, Y1, Y2, X1, X2, lamLo, lamHi,
                                     tiny = 1e-10, pois1 = FALSE, pois2 = FALSE,
                                     off1 = NULL, off2 = NULL) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]

  m1 <- if (pois1) 0 else exp(log_m1); m2 <- if (pois2) 0 else exp(log_m2)
  r1 <- if (pois1) Inf else 1/m1;      r2 <- if (pois2) Inf else 1/m2
  mu1 <- .bound_mu(X1, beta1, off1)
  mu2 <- .bound_mu(X2, beta2, off2)
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)

  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)

  logNB1 <- nb_logpmf_y_mu_r(Y1, mu1, r1)
  logNB2 <- nb_logpmf_y_mu_r(Y2, mu2, r2)
  fac <- 1 + lam * (exp(-Y1) - c1) * (exp(-Y2) - c2)
  fac <- pmax(fac, tiny)

  sum(logNB1 + logNB2 + log(fac))
}

#' Per-obs cumulative sum of Famoye's m-dispersion second-derivative term
#'
#' Returns, for each observation, sum_{j=0}^{y-1} (m^-4 + 2 j m^-3)/(m^-1 + j)^2
#' (the explicit series in Appendix A11/A15). Vectorized over a shared scalar
#' (m, r) by cumulating once over 0..max(y) and indexing by y.
#' @keywords internal
#' @noRd
.disp_hess_series <- function(y, r, m) {
  ymax <- max(y)
  if (ymax == 0L) return(numeric(length(y)))
  j  <- 0:(ymax - 1)
  tj <- (m^(-4) + 2 * j * m^(-3)) / (r + j)^2
  cs <- c(0, cumsum(tj))            # cs[k+1] = sum_{j=0}^{k-1}
  cs[y + 1]
}

#' Analytic Hessian of the frozen-bounds BNB log-likelihood
#'
#' Implements Famoye (2010) Appendix (A6)-(A20) for the natural parameters
#' (beta1, beta2, m1, m2, lambda), then maps to the estimation-scale parameters
#' (beta1, beta2, log_m1, log_m2, z_lambda) via the diagonal reparameterization
#' (m_t = exp(log_m_t); lambda = frozen logistic-bounds map of z). The bounds
#' lamLo/lamHi are held fixed -- matching bnbr_loglik_fixed_bounds and the
#' analytic gradient -- so this is a drop-in replacement for numDeriv::hessian.
#' @keywords internal
#' @noRd
bnb_hessian_fixed_bounds <- function(par, y1, y2, X1, X2, lamLo, lamHi,
                                     pois1 = FALSE, pois2 = FALSE,
                                     off1 = NULL, off2 = NULL) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]

  m1 <- if (pois1) 0 else exp(log_m1); m2 <- if (pois2) 0 else exp(log_m2)
  r1 <- if (pois1) Inf else 1 / m1;    r2 <- if (pois2) Inf else 1 / m2
  mu1 <- .bound_mu(X1, beta1, off1); mu2 <- .bound_mu(X2, beta2, off2)
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)

  # Frozen logistic-bounds map for lambda, with first/second derivatives in z.
  eps <- 1e-6; sig <- plogis(zlam)
  lam     <- lamLo + (lamHi - lamLo) * (eps + (1 - 2 * eps) * sig)
  dlam_dz <- (lamHi - lamLo) * (1 - 2 * eps) * sig * (1 - sig)
  d2lam_dz2 <- (lamHi - lamLo) * (1 - 2 * eps) * sig * (1 - sig) * (1 - 2 * sig)

  k1 <- exp(-y1) - c1; k2 <- exp(-y2) - c2
  dep   <- 1 + lam * (k1 * k2)
  invD  <- 1 / dep; invD2 <- invD^2

  # First derivatives of c (n-vectors / matrices) and the penalty factors.
  dc1m <- dct_dm(mu1, m1, c1); dc2m <- dct_dm(mu2, m2, c2)
  dc1b <- dc_dbeta_mat(mu1, m1, c1, X1)   # n x p1
  dc2b <- dc_dbeta_mat(mu2, m2, c2, X2)   # n x p2
  P1 <- lam * k2 * invD; P2 <- lam * k1 * invD
  Q  <- lam * invD2

  # Second derivatives of c.
  d2c1m <- d2ct_dm2(mu1, m1, c1); d2c2m <- d2ct_dm2(mu2, m2, c2)
  b1f <- d2ct_dbeta2_factor(mu1, m1, c1); b2f <- d2ct_dbeta2_factor(mu2, m2, c2)
  e1f <- d2ct_dmdbeta_factor(mu1, m1, c1); e2f <- d2ct_dmdbeta_factor(mu2, m2, c2)

  # Natural-coordinate gradient pieces needed for the reparameterization
  # second-order corrections: dL/dm_t (A2/A3) and dL/dlambda (A1).
  S1 <- digamma(r1 + y1) - digamma(r1)
  S2 <- digamma(r2 + y2) - digamma(r2)
  g_m1 <- sum(m1^(-2) * (-S1 + log(m1) + log(mu1 + r1) - 1 + (y1 + r1) / (mu1 + r1)) -
                P1 * dc1m)
  g_m2 <- sum(m2^(-2) * (-S2 + log(m2) + log(mu2 + r2) - 1 + (y2 + r2) / (mu2 + r2)) -
                P2 * dc2m)
  g_lam <- sum(k1 * k2 * invD)

  np <- p1 + p2 + 3
  H  <- matrix(0, np, np)
  ib1 <- seq_len(p1); ib2 <- p1 + seq_len(p2)
  im1 <- p1 + p2 + 1; im2 <- p1 + p2 + 2; iz <- p1 + p2 + 3

  ## ---- natural-coord blocks ----
  # beta1-beta1 (A18): xx' part + rank-one c'' part - (P1)^2 c'_b c'_b
  s_b1b1 <- -mu1 * (1 + m1 * y1) / (1 + m1 * mu1)^2 - P1 * b1f
  Hb1b1  <- crossprod(X1, X1 * s_b1b1) - crossprod(dc1b, dc1b * (P1^2))
  # beta2-beta2 (A20)
  s_b2b2 <- -mu2 * (1 + m2 * y2) / (1 + m2 * mu2)^2 - P2 * b2f
  Hb2b2  <- crossprod(X2, X2 * s_b2b2) - crossprod(dc2b, dc2b * (P2^2))
  # beta1-beta2 (A19)
  Hb1b2  <- crossprod(dc1b, dc2b * Q)

  # m1-m1 (A11), m2-m2 (A15)
  Hm1m1 <- sum(
    m1^(-3) * (3 - 2 * log(m1) - 2 * log(mu1 + r1) - r1 / (mu1 + r1)) -
      (m1^(-2) * y1 + 2 * m1^(-3) + 2 * mu1 * r1 * y1 + 3 * mu1 * m1^(-2)) /
        (1 + mu1 * m1)^2 +
      .disp_hess_series(y1, r1, m1) -
      P1 * d2c1m - (P1 * dc1m)^2
  )
  Hm2m2 <- sum(
    m2^(-3) * (3 - 2 * log(m2) - 2 * log(mu2 + r2) - r2 / (mu2 + r2)) -
      (m2^(-2) * y2 + 2 * m2^(-3) + 2 * mu2 * r2 * y2 + 3 * mu2 * m2^(-2)) /
        (1 + mu2 * m2)^2 +
      .disp_hess_series(y2, r2, m2) -
      P2 * d2c2m - (P2 * dc2m)^2
  )
  # m1-m2 (A12)
  Hm1m2 <- sum(Q * dc1m * dc2m)

  # m1-beta1 (A13), m2-beta2 (A17): vectors over the own-equation betas
  s_m1b1 <- m1^(-2) * (mu1 - y1) / (mu1 + r1)^2 * mu1 - P1 * e1f
  Hm1b1  <- colSums(X1 * s_m1b1) - colSums(dc1b * (P1^2 * dc1m))
  s_m2b2 <- m2^(-2) * (mu2 - y2) / (mu2 + r2)^2 * mu2 - P2 * e2f
  Hm2b2  <- colSums(X2 * s_m2b2) - colSums(dc2b * (P2^2 * dc2m))
  # m1-beta2 (A14), m2-beta1 (A16): cross-equation
  Hm1b2 <- colSums(dc2b * (Q * dc1m))
  Hm2b1 <- colSums(dc1b * (Q * dc2m))

  # lambda blocks (A6-A10)
  Hll   <- -sum((k1 * k2 * invD)^2)
  Hlm1  <- sum(-k2 * invD2 * dc1m)
  Hlm2  <- sum(-k1 * invD2 * dc2m)
  Hlb1  <- colSums(dc1b * (-k2 * invD2))
  Hlb2  <- colSums(dc2b * (-k1 * invD2))

  ## ---- assemble in natural coords, then reparameterize ----
  H[ib1, ib1] <- Hb1b1
  H[ib2, ib2] <- Hb2b2
  H[ib1, ib2] <- Hb1b2;            H[ib2, ib1] <- t(Hb1b2)

  # beta-log_m : multiply natural m-column by dm/dlog_m = m
  H[ib1, im1] <- m1 * Hm1b1;       H[im1, ib1] <- H[ib1, im1]
  H[ib2, im2] <- m2 * Hm2b2;       H[im2, ib2] <- H[ib2, im2]
  H[ib1, im2] <- m2 * Hm2b1;       H[im2, ib1] <- H[ib1, im2]
  H[ib2, im1] <- m1 * Hm1b2;       H[im1, ib2] <- H[ib2, im1]

  # beta-z : multiply natural lambda-column by dlam/dz
  H[ib1, iz] <- dlam_dz * Hlb1;    H[iz, ib1] <- H[ib1, iz]
  H[ib2, iz] <- dlam_dz * Hlb2;    H[iz, ib2] <- H[ib2, iz]

  # log_m diagonal: m^2 * H_mm + m * g_m (second-derivative-of-transform term)
  H[im1, im1] <- m1^2 * Hm1m1 + m1 * g_m1
  H[im2, im2] <- m2^2 * Hm2m2 + m2 * g_m2
  H[im1, im2] <- m1 * m2 * Hm1m2;  H[im2, im1] <- H[im1, im2]

  # log_m - z
  H[im1, iz] <- m1 * dlam_dz * Hlm1; H[iz, im1] <- H[im1, iz]
  H[im2, iz] <- m2 * dlam_dz * Hlm2; H[iz, im2] <- H[im2, iz]

  # z - z : (dlam/dz)^2 * H_ll + d2lam/dz2 * g_lam
  H[iz, iz] <- dlam_dz^2 * Hll + d2lam_dz2 * g_lam

  # A Poisson margin's log_m is fixed: its NB2 curvature (the digamma/dispersion
  # series and dct_dm terms) is a 0/0 at m = 0 and enters only its own row/column,
  # which .free_index_vcov() drops from the free-parameter information. Zero that
  # row/column so no 0*Inf placeholder leaks into the returned Hessian; the beta
  # and lambda blocks above already used the exact m = 0 limits.
  if (pois1) { H[im1, ] <- 0; H[, im1] <- 0 }
  if (pois2) { H[im2, ] <- 0; H[, im2] <- 0 }

  H
}
