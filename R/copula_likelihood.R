# Discrete-copula NB2 log-likelihood and analytic gradient. Internal.

# ∂F(y; mu, r)/∂r = sum_{k=0}^y P(k) [psi(k+r) - psi(r) + log(r/(r+mu)) + 1 - (r+k)/(r+mu)]
# Returns 0 when y < 0.
.dnb_cdf_dr <- function(y, mu, r) {
  if (y < 0L) return(0)
  k  <- 0L:y
  pk <- dnbinom(k, size = r, mu = mu)
  wk <- digamma(k + r) - digamma(r) + log(r / (r + mu)) + 1 - (r + k) / (r + mu)
  sum(pk * wk)
}

# ∂F(y; mu, r)/∂mu = -(y+1) * P(y+1; mu, r) / mu  (telescope identity; 0 when y < 0)
.dnb_cdf_dmu <- function(y, mu, r) {
  if (y < 0L) return(0)
  -(y + 1L) * dnbinom(y + 1L, size = r, mu = mu) / mu
}

#' Per-observation discrete-copula NB2 log-likelihood
#'
#' @param par Parameter vector (b1, b2, log_m1, log_m2, z_theta).
#' @param y1,y2 Response vectors.
#' @param X1,X2 Design matrices.
#' @param family Copula family string: "frank", "normal", or "kimeldorf".
#' @return Numeric vector of length n.
#' @keywords internal
#' @noRd
copula_loglik_vec <- function(par, y1, y2, X1, X2, family) {
  p1 <- NCOL(X1); p2 <- NCOL(X2)
  beta1   <- par[seq_len(p1)]
  beta2   <- par[p1 + seq_len(p2)]
  log_m1  <- par[p1 + p2 + 1L]
  log_m2  <- par[p1 + p2 + 2L]
  z_theta <- par[p1 + p2 + 3L]

  r1  <- exp(-log_m1); r2 <- exp(-log_m2)
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  theta <- z_to_native(family, z_theta)

  .cop_cdf <- switch(family,
    frank     = frank_cdf,
    normal    = normal_cdf,
    kimeldorf = kimeldorf_cdf
  )

  a  <- pnbinom(y1,       size = r1, mu = mu1)
  am <- ifelse(y1 > 0L, pnbinom(y1 - 1L, size = r1, mu = mu1), 0)
  b  <- pnbinom(y2,       size = r2, mu = mu2)
  bm <- ifelse(y2 > 0L, pnbinom(y2 - 1L, size = r2, mu = mu2), 0)

  p_obs <- .cop_cdf(a, b, theta) - .cop_cdf(am, b, theta) -
           .cop_cdf(a, bm, theta) + .cop_cdf(am, bm, theta)

  log(pmax(p_obs, 1e-300))
}

#' Per-observation analytic gradient of the discrete-copula log-likelihood
#'
#' Returns a matrix (n × k) where columns are the per-obs gradient contributions.
#' Summing over rows gives the gradient of the total log-lik.
#'
#' @inheritParams copula_loglik_vec
#' @return Named numeric vector of length k = p1 + p2 + 3.
#' @keywords internal
#' @noRd
copula_grad_vec <- function(par, y1, y2, X1, X2, family) {
  p1 <- NCOL(X1); p2 <- NCOL(X2)
  beta1   <- par[seq_len(p1)]
  beta2   <- par[p1 + seq_len(p2)]
  log_m1  <- par[p1 + p2 + 1L]
  log_m2  <- par[p1 + p2 + 2L]
  z_theta <- par[p1 + p2 + 3L]
  n       <- length(y1)

  r1  <- exp(-log_m1); r2 <- exp(-log_m2)
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  theta   <- z_to_native(family, z_theta)
  dth_dz  <- dnative_dz(family, z_theta)

  a  <- pnbinom(y1,       size = r1, mu = mu1)
  am <- ifelse(y1 > 0L, pnbinom(y1 - 1L, size = r1, mu = mu1), 0)
  b  <- pnbinom(y2,       size = r2, mu = mu2)
  bm <- ifelse(y2 > 0L, pnbinom(y2 - 1L, size = r2, mu = mu2), 0)

  # Copula values for the PMF
  c_ab   <- switch(family, frank=frank_cdf(a,b,theta),   normal=normal_cdf(a,b,theta),   kimeldorf=kimeldorf_cdf(a,b,theta))
  c_amb  <- switch(family, frank=frank_cdf(am,b,theta),  normal=normal_cdf(am,b,theta),  kimeldorf=kimeldorf_cdf(am,b,theta))
  c_abm  <- switch(family, frank=frank_cdf(a,bm,theta),  normal=normal_cdf(a,bm,theta),  kimeldorf=kimeldorf_cdf(a,bm,theta))
  c_ambm <- switch(family, frank=frank_cdf(am,bm,theta), normal=normal_cdf(am,bm,theta), kimeldorf=kimeldorf_cdf(am,bm,theta))

  p_obs <- c_ab - c_amb - c_abm + c_ambm
  p_obs <- pmax(p_obs, 1e-300)

  # dC/du at four corners
  cu_ab   <- .cop_du(a,  b,  theta, family)
  cu_amb  <- .cop_du(am, b,  theta, family)
  cu_abm  <- .cop_du(a,  bm, theta, family)
  cu_ambm <- .cop_du(am, bm, theta, family)

  # dC/dv at four corners
  cv_ab   <- .cop_dv(a,  b,  theta, family)
  cv_amb  <- .cop_dv(am, b,  theta, family)
  cv_abm  <- .cop_dv(a,  bm, theta, family)
  cv_ambm <- .cop_dv(am, bm, theta, family)

  # dC/dtheta (rectangle sum) — boundary zeros handled by .cop_dtheta guards
  ct_rect <- .cop_dtheta(a,b,theta,family) - .cop_dtheta(am,b,theta,family) -
             .cop_dtheta(a,bm,theta,family) + .cop_dtheta(am,bm,theta,family)

  # ---- beta1 gradient ----
  # da/dmu1 = -(y1+1)*P(y1+1)/mu1; dam/dmu1 = -y1*P(y1)/mu1 (when y1>0)
  da_dmu1  <- -(y1 + 1L) * dnbinom(y1 + 1L, size = r1, mu = mu1) / mu1
  dam_dmu1 <- ifelse(y1 > 0L, -y1 * dnbinom(y1, size = r1, mu = mu1) / mu1, 0)

  # Per-obs scalar factor for beta1 (multiply by mu1 * x1_ij since mu1 = exp(X1 b1))
  # d(a)/d(beta1_j) = da/dmu1 * mu1 * x1_ij
  delta_u_a  <- cu_ab - cu_abm     # C_u(a,b) - C_u(a,bm)
  delta_u_am <- -cu_amb + cu_ambm  # -C_u(am,b) + C_u(am,bm)

  # Per-obs scalar contribution to d(loglik)/d(b1) before multiplying by x1_i
  sc_beta1 <- (delta_u_a * da_dmu1 * mu1 + delta_u_am * dam_dmu1 * mu1) / p_obs

  # ---- beta2 gradient ----
  db_dmu2  <- -(y2 + 1L) * dnbinom(y2 + 1L, size = r2, mu = mu2) / mu2
  dbm_dmu2 <- ifelse(y2 > 0L, -y2 * dnbinom(y2, size = r2, mu = mu2) / mu2, 0)

  delta_v_b  <- cv_ab - cv_amb
  delta_v_bm <- -cv_abm + cv_ambm

  sc_beta2 <- (delta_v_b * db_dmu2 * mu2 + delta_v_bm * dbm_dmu2 * mu2) / p_obs

  # ---- log_m1 gradient ----
  # da/dr1 = .dnb_cdf_dr(y1, mu1, r1); dam/dr1 = .dnb_cdf_dr(y1-1, mu1, r1) [y1>0]
  # d(r1)/d(log_m1) = -r1
  da_dr1  <- mapply(.dnb_cdf_dr, y1,      mu1, r1)
  dam_dr1 <- ifelse(y1 > 0L, mapply(.dnb_cdf_dr, y1 - 1L, mu1, r1), 0)

  sc_logm1 <- (-r1) * (delta_u_a * da_dr1 + delta_u_am * dam_dr1) / p_obs

  # ---- log_m2 gradient ----
  db_dr2  <- mapply(.dnb_cdf_dr, y2,      mu2, r2)
  dbm_dr2 <- ifelse(y2 > 0L, mapply(.dnb_cdf_dr, y2 - 1L, mu2, r2), 0)

  sc_logm2 <- (-r2) * (delta_v_b * db_dr2 + delta_v_bm * dbm_dr2) / p_obs

  # ---- z_theta gradient ----
  sc_z <- ct_rect * dth_dz / p_obs

  # ---- Assemble gradient: sum over observations ----
  g_beta1  <- as.vector(t(X1) %*% sc_beta1)
  g_beta2  <- as.vector(t(X2) %*% sc_beta2)
  g_logm1  <- sum(sc_logm1)
  g_logm2  <- sum(sc_logm2)
  g_z      <- sum(sc_z)

  c(g_beta1, g_beta2, g_logm1, g_logm2, g_z)
}
