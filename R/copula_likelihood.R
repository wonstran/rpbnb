# Discrete-copula NB2 log-likelihood and analytic gradient. Internal.

# ∂F(y; mu, r)/∂r = sum_{k=0}^y P(k) [psi(k+r) - psi(r) + log(r/(r+mu)) + 1 - (r+k)/(r+mu)]
# Returns 0 when y < 0. As r -> Inf (the NB2 -> Poisson boundary reached when
# log_m -> -Inf) every term of wk analytically vanishes, so the derivative's
# true limit is 0; guard it explicitly since digamma(Inf) - digamma(Inf) and
# log(r/(r+mu)) both evaluate to NaN in floating point at literal Inf.
.dnb_cdf_dr <- function(y, mu, r) {
  if (y < 0L) return(0)
  if (!is.finite(r)) return(0)
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
  mu1 <- .bound_mu(X1, beta1)
  mu2 <- .bound_mu(X2, beta2)
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

  # A parameter proposal that leaves a/am/b/bm non-finite (e.g. NaN inherited
  # from an earlier non-finite gradient step) is invalid regardless of what
  # the copula CDF does with it; -Inf lets the optimizer reject it outright
  # instead of possibly landing on a finite-looking but bogus plateau. The
  # copula CDF itself can also produce a non-finite p_obs from finite a/am/b/bm
  # (e.g. an unbounded frank theta driving exp(-theta*u) to Inf), so p_obs's
  # own finiteness must be checked too, not just its four finite inputs.
  ok <- is.finite(a) & is.finite(am) & is.finite(b) & is.finite(bm)

  p_obs <- .cop_cdf(a, b, theta) - .cop_cdf(am, b, theta) -
           .cop_cdf(a, bm, theta) + .cop_cdf(am, bm, theta)
  ok <- ok & is.finite(p_obs)

  ll <- log(pmax(p_obs, 1e-300))
  ll[!ok] <- -Inf
  ll
}

#' Discrete-copula joint pmf and finiteness mask for one (per-observation) set of
#' means. The single source of the four NB-CDF corners + rectangle pmf, reused by
#' `.copula_score_scalars`, `bnbr_rp_copula_ll`, and its gradient twin so their
#' masking cannot silently diverge. `p_obs` is floored at 1e-300; `ok` is FALSE
#' where any corner or the raw pmf is non-finite.
#' @keywords internal
#' @noRd
.copula_pmf <- function(y1, y2, mu1, mu2, r1, r2, theta, family) {
  # A Poisson-restricted margin is signalled in-band by r = Inf (m = 0); its
  # CDF corners use ppois, not pnbinom(size = Inf) which returns NaN.
  pois1 <- !is.finite(r1); pois2 <- !is.finite(r2)
  a  <- if (pois1) stats::ppois(y1, mu1) else pnbinom(y1, size = r1, mu = mu1)
  am <- if (pois1) ifelse(y1 > 0L, stats::ppois(y1 - 1L, mu1), 0)
        else       ifelse(y1 > 0L, pnbinom(y1 - 1L, size = r1, mu = mu1), 0)
  b  <- if (pois2) stats::ppois(y2, mu2) else pnbinom(y2, size = r2, mu = mu2)
  bm <- if (pois2) ifelse(y2 > 0L, stats::ppois(y2 - 1L, mu2), 0)
        else       ifelse(y2 > 0L, pnbinom(y2 - 1L, size = r2, mu = mu2), 0)
  ok <- is.finite(a) & is.finite(am) & is.finite(b) & is.finite(bm)
  cop_cdf <- switch(family, frank = frank_cdf, normal = normal_cdf, kimeldorf = kimeldorf_cdf)
  p_obs <- cop_cdf(a, b, theta) - cop_cdf(am, b, theta) -
           cop_cdf(a, bm, theta) + cop_cdf(am, bm, theta)
  ok <- ok & is.finite(p_obs)
  # An extreme count makes both NB CDF corners round to 1 (e.g. pnbinom(120)==
  # pnbinom(119)==1 to machine precision), so the rectangle cancels to ~0 and gets
  # floored. There the log-lik is a clamped constant, so its gradient is 0 -- but
  # score = numerator/floor would instead be a huge *finite* number that the
  # is.finite mask cannot catch. `underflow` flags these for the score path only;
  # it must NOT enter `ok`, or the value would collapse to -Inf instead of the
  # finite log-floor penalty. `ok` observations already dominate the floor test,
  # so only guard where the raw pmf actually failed to clear the floor.
  underflow <- !(p_obs > 1e-300)
  list(a = a, am = am, b = b, bm = bm, p_obs = pmax(p_obs, 1e-300),
       ok = ok, underflow = underflow)
}

#' Per-observation copula score scalars (shared by the fixed and RP estimators)
#'
#' Returns dlogP/deta1, dlogP/deta2, dlogP/dlog_m1, dlogP/dlog_m2, dlogP/dz_theta
#' per observation (bad observations zeroed), given per-observation means and the
#' scalar dispersion/dependence parameters. `copula_grad_vec` contracts these with
#' the design; the RP estimator reuses them per simulation draw.
#' @keywords internal
#' @noRd
.copula_score_scalars <- function(y1, y2, mu1, mu2, r1, r2, theta, dth_dz, family) {
  pm <- .copula_pmf(y1, y2, mu1, mu2, r1, r2, theta, family)
  a <- pm$a; am <- pm$am; b <- pm$b; bm <- pm$bm; p_obs <- pm$p_obs; ok <- pm$ok
  pois1 <- !is.finite(r1); pois2 <- !is.finite(r2)

  cu_ab   <- .cop_du(a,  b,  theta, family); cu_amb  <- .cop_du(am, b,  theta, family)
  cu_abm  <- .cop_du(a,  bm, theta, family); cu_ambm <- .cop_du(am, bm, theta, family)
  cv_ab   <- .cop_dv(a,  b,  theta, family); cv_amb  <- .cop_dv(am, b,  theta, family)
  cv_abm  <- .cop_dv(a,  bm, theta, family); cv_ambm <- .cop_dv(am, bm, theta, family)
  ct_rect <- .cop_dtheta(a, b, theta, family) - .cop_dtheta(am, b, theta, family) -
             .cop_dtheta(a, bm, theta, family) + .cop_dtheta(am, bm, theta, family)

  # mu-score: dpois for a Poisson margin (dnbinom(size = Inf) is NaN).
  da_dmu1  <- if (pois1) -(y1 + 1L) * stats::dpois(y1 + 1L, mu1) / mu1
              else       -(y1 + 1L) * dnbinom(y1 + 1L, size = r1, mu = mu1) / mu1
  dam_dmu1 <- if (pois1) ifelse(y1 > 0L, -y1 * stats::dpois(y1, mu1) / mu1, 0)
              else       ifelse(y1 > 0L, -y1 * dnbinom(y1, size = r1, mu = mu1) / mu1, 0)
  delta_u_a  <- cu_ab - cu_abm
  delta_u_am <- -cu_amb + cu_ambm
  s_eta1 <- (delta_u_a * da_dmu1 * mu1 + delta_u_am * dam_dmu1 * mu1) / p_obs

  db_dmu2  <- if (pois2) -(y2 + 1L) * stats::dpois(y2 + 1L, mu2) / mu2
              else       -(y2 + 1L) * dnbinom(y2 + 1L, size = r2, mu = mu2) / mu2
  dbm_dmu2 <- if (pois2) ifelse(y2 > 0L, -y2 * stats::dpois(y2, mu2) / mu2, 0)
              else       ifelse(y2 > 0L, -y2 * dnbinom(y2, size = r2, mu = mu2) / mu2, 0)
  delta_v_b  <- cv_ab - cv_amb
  delta_v_bm <- -cv_abm + cv_ambm
  s_eta2 <- (delta_v_b * db_dmu2 * mu2 + delta_v_bm * dbm_dmu2 * mu2) / p_obs

  # Dispersion score: a Poisson margin's log_m is pinned -> zero score (the NB2
  # form (-r)*(...) would be (-Inf)*0 = NaN at r = Inf).
  if (pois1) {
    s_logm1 <- rep(0, length(y1))
  } else {
    da_dr1  <- mapply(.dnb_cdf_dr, y1,      mu1, r1)
    dam_dr1 <- ifelse(y1 > 0L, mapply(.dnb_cdf_dr, y1 - 1L, mu1, r1), 0)
    s_logm1 <- (-r1) * (delta_u_a * da_dr1 + delta_u_am * dam_dr1) / p_obs
  }
  if (pois2) {
    s_logm2 <- rep(0, length(y2))
  } else {
    db_dr2  <- mapply(.dnb_cdf_dr, y2,      mu2, r2)
    dbm_dr2 <- ifelse(y2 > 0L, mapply(.dnb_cdf_dr, y2 - 1L, mu2, r2), 0)
    s_logm2 <- (-r2) * (delta_v_b * db_dr2 + delta_v_bm * dbm_dr2) / p_obs
  }

  s_ztheta <- ct_rect * dth_dz / p_obs

  bad <- !ok | pm$underflow | !is.finite(s_eta1) | !is.finite(s_eta2) |
         !is.finite(s_logm1) | !is.finite(s_logm2) | !is.finite(s_ztheta)
  s_eta1[bad] <- 0; s_eta2[bad] <- 0
  s_logm1[bad] <- 0; s_logm2[bad] <- 0; s_ztheta[bad] <- 0

  list(p_obs = p_obs, s_eta1 = s_eta1, s_eta2 = s_eta2,
       s_logm1 = s_logm1, s_logm2 = s_logm2, s_ztheta = s_ztheta, ok = ok)
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

  r1  <- exp(-log_m1); r2 <- exp(-log_m2)
  mu1 <- .bound_mu(X1, beta1)
  mu2 <- .bound_mu(X2, beta2)
  theta  <- z_to_native(family, z_theta)
  dth_dz <- dnative_dz(family, z_theta)

  sc <- .copula_score_scalars(y1, y2, mu1, mu2, r1, r2, theta, dth_dz, family)

  g_beta1 <- as.vector(t(X1) %*% sc$s_eta1)
  g_beta2 <- as.vector(t(X2) %*% sc$s_eta2)
  g_logm1 <- sum(sc$s_logm1)
  g_logm2 <- sum(sc$s_logm2)
  g_z     <- sum(sc$s_ztheta)
  c(g_beta1, g_beta2, g_logm1, g_logm2, g_z)
}
