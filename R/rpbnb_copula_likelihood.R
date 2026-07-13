# Random-parameter BNB simulated log-likelihood with COPULA dependence.
# The per-draw dependence factor is the discrete-copula joint pmf
# (finite difference of the copula CDF over the two NB CDFs). Marginal means are
# random across individuals exactly as in the Famoye RP path. Internal.

#' Simulated log-likelihood for the copula RP-BNB model
#'
#' Parameter order: beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2),
#' log_m1, log_m2, z_theta. family is one of "frank", "normal", "kimeldorf".
#' @keywords internal
#' @noRd
bnbr_rp_copula_ll <- function(par, y1, y2, X1, X2, XR1, XR2,
                              rand_idx1, rand_idx2, Z1, Z2, family,
                              dist1 = NULL, dist2 = NULL,
                              sign1 = NULL, sign2 = NULL) {
  n  <- length(y1)
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  R  <- if (q1 + q2 > 0) nrow(Z1) else 1L

  beta1 <- par[1:k1]; beta2 <- par[(k1 + 1):(k1 + k2)]
  lg1 <- if (q1 > 0) (k1 + k2 + 1):(k1 + k2 + q1) else integer(0)
  lg2 <- if (q2 > 0) (k1 + k2 + q1 + 1):(k1 + k2 + q1 + q2) else integer(0)
  sd1 <- if (q1 > 0) exp(par[lg1]) else numeric(0)
  sd2 <- if (q2 > 0) exp(par[lg2]) else numeric(0)
  idx_end <- k1 + k2 + q1 + q2
  log_m1 <- par[idx_end + 1]; log_m2 <- par[idx_end + 2]; z_theta <- par[idx_end + 3]
  r1 <- exp(-log_m1); r2 <- exp(-log_m2)
  theta <- z_to_native(family, z_theta)

  if (is.null(dist1) && q1 > 0) dist1 <- rep("normal", q1)
  if (is.null(dist2) && q2 > 0) dist2 <- rep("normal", q2)
  if (is.null(sign1) && q1 > 0) sign1 <- rep(1, q1)
  if (is.null(sign2) && q2 > 0) sign2 <- rep(1, q2)

  xb1 <- as.vector(X1 %*% beta1); xb2 <- as.vector(X2 %*% beta2)
  real1 <- if (q1 > 0) rand_realize(Z1, dist1, sign1, beta1[rand_idx1], sd1) else NULL
  real2 <- if (q2 > 0) rand_realize(Z2, dist2, sign2, beta2[rand_idx2], sd2) else NULL
  XR1m <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2m <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else NULL

  cop_cdf <- switch(family, frank = frank_cdf, normal = normal_cdf,
                    kimeldorf = kimeldorf_cdf)

  LL <- matrix(0, n, R)
  for (r in seq_len(R)) {
    eta1 <- xb1 + if (q1 > 0) as.vector(XR1m %*% real1$dev[r, ]) else 0
    eta2 <- xb2 + if (q2 > 0) as.vector(XR2m %*% real2$dev[r, ]) else 0
    mu1 <- pmin(pmax(exp(eta1), 1e-300), 1e15)
    mu2 <- pmin(pmax(exp(eta2), 1e-300), 1e15)
    a  <- pnbinom(y1, size = r1, mu = mu1)
    am <- ifelse(y1 > 0, pnbinom(y1 - 1, size = r1, mu = mu1), 0)
    b  <- pnbinom(y2, size = r2, mu = mu2)
    bm <- ifelse(y2 > 0, pnbinom(y2 - 1, size = r2, mu = mu2), 0)
    p_obs <- cop_cdf(a, b, theta) - cop_cdf(am, b, theta) -
             cop_cdf(a, bm, theta) + cop_cdf(am, bm, theta)
    ok <- is.finite(a) & is.finite(am) & is.finite(b) & is.finite(bm) &
          is.finite(p_obs)
    col <- log(pmax(p_obs, 1e-300))
    col[!ok] <- -Inf
    LL[, r] <- col
  }
  lse <- row_log_sum_exp(LL)
  sum(lse - log(R))
}
