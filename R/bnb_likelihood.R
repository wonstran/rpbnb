# Famoye/Sarmanov bivariate NB (NB2) likelihood internals.
# Parameter vector: [beta1 (p1), beta2 (p2), log_m1, log_m2, z_lambda].
# Core math (c_val, nb_logpmf_y_mu_r, lambda_bounds_vec, dct_dm, dc_dbeta_mat,
# d_const) lives in famoye_core.R; these functions call those helpers.

#' Per-observation log-likelihood (vector) for the Famoye BNB model
#' @keywords internal
#' @noRd
bnb_loglik_vec <- function(par, y1, y2, X1, X2) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]

  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1/m1;        r2 <- 1/m2
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
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
#' @keywords internal
#' @noRd
bnb_score_mat <- function(par, y1, y2, X1, X2) {
  p1 <- ncol(X1); p2 <- ncol(X2)

  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]

  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1/m1;        r2 <- 1/m2
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
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

  # m blocks (A2-A3) + chain to log m
  S1 <- digamma(r1 + y1) - digamma(r1)
  S2 <- digamma(r2 + y2) - digamma(r2)
  dc1_dm1 <- dct_dm(mu1, m1, c1)
  dc2_dm2 <- dct_dm(mu2, m2, c2)

  s_m1 <-  (m1^(-2)) * ( -S1 + log(m1) + log(mu1 + r1) - 1 + (y1 + r1)/(mu1 + r1) ) -
    (lam * k2 * inv_dep) * dc1_dm1
  s_m2 <-  (m2^(-2)) * ( -S2 + log(m2) + log(mu2 + r2) - 1 + (y2 + r2)/(mu2 + r2) ) -
    (lam * k1 * inv_dep) * dc2_dm2

  score_logm1 <- m1 * s_m1
  score_logm2 <- m2 * s_m2

  # lambda part (A1) + chain rule z -> lambda
  dL_dlambda <- (k1 * k2) * inv_dep
  score_z <- dL_dlambda * dlam_dz

  cbind(score_beta1_mat, score_beta2_mat, score_logm1, score_logm2, score_z)
}

#' Summed analytic gradient for BFGS
#' @keywords internal
#' @noRd
bnb_grad_vec <- function(par, y1, y2, X1, X2) {
  colSums(bnb_score_mat(par, y1, y2, X1, X2))
}

#' Fixed-bounds summed logLik for numeric Hessian (freeze lambda-bounds at optimum)
#' @keywords internal
#' @noRd
bnbr_loglik_fixed_bounds <- function(par, Y1, Y2, X1, X2, lamLo, lamHi, tiny = 1e-10) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]

  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1/m1;        r2 <- 1/m2
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)

  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)

  logNB1 <- nb_logpmf_y_mu_r(Y1, mu1, r1)
  logNB2 <- nb_logpmf_y_mu_r(Y2, mu2, r2)
  fac <- 1 + lam * (exp(-Y1) - c1) * (exp(-Y2) - c2)
  fac <- pmax(fac, tiny)

  sum(logNB1 + logNB2 + log(fac))
}
