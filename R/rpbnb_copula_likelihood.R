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
                              sign1 = NULL, sign2 = NULL,
                              pois1 = FALSE, pois2 = FALSE,
                              off1 = NULL, off2 = NULL) {
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
  r1 <- if (pois1) Inf else exp(-log_m1)
  r2 <- if (pois2) Inf else exp(-log_m2)
  theta <- z_to_native(family, z_theta)

  if (is.null(dist1) && q1 > 0) dist1 <- rep("normal", q1)
  if (is.null(dist2) && q2 > 0) dist2 <- rep("normal", q2)
  if (is.null(sign1) && q1 > 0) sign1 <- rep(1, q1)
  if (is.null(sign2) && q2 > 0) sign2 <- rep(1, q2)

  xb1 <- as.vector(X1 %*% beta1) + .as_offset(off1, nrow(X1))
  xb2 <- as.vector(X2 %*% beta2) + .as_offset(off2, nrow(X2))
  real1 <- if (q1 > 0) rand_realize(Z1, dist1, sign1, beta1[rand_idx1], sd1) else NULL
  real2 <- if (q2 > 0) rand_realize(Z2, dist2, sign2, beta2[rand_idx2], sd2) else NULL
  XR1m <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2m <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else NULL

  LL <- matrix(0, n, R)
  for (r in seq_len(R)) {
    eta1 <- xb1 + if (q1 > 0) as.vector(XR1m %*% real1$dev[r, ]) else 0
    eta2 <- xb2 + if (q2 > 0) as.vector(XR2m %*% real2$dev[r, ]) else 0
    mu1 <- pmin(pmax(exp(eta1), 1e-300), 1e15)
    mu2 <- pmin(pmax(exp(eta2), 1e-300), 1e15)
    pm  <- .copula_pmf(y1, y2, mu1, mu2, r1, r2, theta, family)  # single pmf source
    col <- log(pm$p_obs); col[!pm$ok] <- -Inf
    LL[, r] <- col
  }
  lse <- row_log_sum_exp(LL)
  sum(lse - log(R))
}

#' Copula RP-BNB simulated LL with analytic gradient (and optional per-obs scores)
#'
#' Same value as bnbr_rp_copula_ll; additionally the analytic gradient via the
#' Louis mixture of per-draw copula scores. Parameter order matches
#' bnbr_rp_copula_ll. `want_scores` attaches the n x npar per-observation score
#' matrix (rows s_i = sum_r w_ir g_ir) for OPG covariance.
#' @keywords internal
#' @noRd
bnbr_rp_copula_ll_grad <- function(par, y1, y2, X1, X2, XR1, XR2,
                                   rand_idx1, rand_idx2, Z1, Z2, family,
                                   dist1 = NULL, dist2 = NULL,
                                   sign1 = NULL, sign2 = NULL,
                                   want_scores = FALSE, pois1 = FALSE, pois2 = FALSE,
                                   off1 = NULL, off2 = NULL) {
  n  <- length(y1)
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  R  <- if (q1 + q2 > 0) nrow(Z1) else 1L
  npar <- k1 + k2 + q1 + q2 + 3L

  beta1 <- par[1:k1]; beta2 <- par[(k1 + 1):(k1 + k2)]
  lg1 <- if (q1 > 0) (k1 + k2 + 1):(k1 + k2 + q1) else integer(0)
  lg2 <- if (q2 > 0) (k1 + k2 + q1 + 1):(k1 + k2 + q1 + q2) else integer(0)
  sd1 <- if (q1 > 0) exp(par[lg1]) else numeric(0)
  sd2 <- if (q2 > 0) exp(par[lg2]) else numeric(0)
  idx_end <- k1 + k2 + q1 + q2
  log_m1 <- par[idx_end + 1]; log_m2 <- par[idx_end + 2]; z_theta <- par[idx_end + 3]
  r1 <- if (pois1) Inf else exp(-log_m1)
  r2 <- if (pois2) Inf else exp(-log_m2)
  theta <- z_to_native(family, z_theta); dth_dz <- dnative_dz(family, z_theta)

  if (is.null(dist1) && q1 > 0) dist1 <- rep("normal", q1)
  if (is.null(dist2) && q2 > 0) dist2 <- rep("normal", q2)
  if (is.null(sign1) && q1 > 0) sign1 <- rep(1, q1)
  if (is.null(sign2) && q2 > 0) sign2 <- rep(1, q2)

  xb1 <- as.vector(X1 %*% beta1) + .as_offset(off1, nrow(X1))
  xb2 <- as.vector(X2 %*% beta2) + .as_offset(off2, nrow(X2))
  real1 <- if (q1 > 0) rand_realize(Z1, dist1, sign1, beta1[rand_idx1], sd1) else NULL
  real2 <- if (q2 > 0) rand_realize(Z2, dist2, sign2, beta2[rand_idx2], sd2) else NULL
  XR1m <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else matrix(0, n, 0)
  XR2m <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else matrix(0, n, 0)

  iB1 <- 1:k1; iB2 <- (k1 + 1):(k1 + k2)
  iL1 <- lg1; iL2 <- lg2
  im1 <- idx_end + 1; im2 <- idx_end + 2; iz <- idx_end + 3

  # Pass 1: per-draw LL matrix and stored means.
  mu1M <- matrix(0, n, R); mu2M <- matrix(0, n, R); LL <- matrix(0, n, R)
  for (r in seq_len(R)) {
    eta1 <- xb1 + if (q1 > 0) as.vector(XR1m %*% real1$dev[r, ]) else 0
    eta2 <- xb2 + if (q2 > 0) as.vector(XR2m %*% real2$dev[r, ]) else 0
    mu1 <- pmin(pmax(exp(eta1), 1e-300), 1e15); mu2 <- pmin(pmax(exp(eta2), 1e-300), 1e15)
    mu1M[, r] <- mu1; mu2M[, r] <- mu2
    sc <- .copula_score_scalars(y1, y2, mu1, mu2, r1, r2, theta, dth_dz, family)
    col <- log(sc$p_obs)   # p_obs already floored at 1e-300 in the helper
    col[!sc$ok] <- -Inf
    LL[, r] <- col
  }
  lse <- row_log_sum_exp(LL)
  value <- sum(lse - log(R))
  W <- exp(LL - lse)   # n x R softmax weights
  W[!is.finite(W)] <- 0

  # Pass 2: accumulate weighted per-draw scores into the total gradient (+ scores).
  grad <- numeric(npar)
  S <- if (want_scores) matrix(0, n, npar) else NULL
  for (r in seq_len(R)) {
    mu1 <- mu1M[, r]; mu2 <- mu2M[, r]; wv <- W[, r]
    sc <- .copula_score_scalars(y1, y2, mu1, mu2, r1, r2, theta, dth_dz, family)
    # design for this draw
    D1 <- X1; if (q1 > 0) for (j in seq_len(q1)) D1[, rand_idx1[j]] <- X1[, rand_idx1[j]] * real1$dloc[r, j]
    D2 <- X2; if (q2 > 0) for (j in seq_len(q2)) D2[, rand_idx2[j]] <- X2[, rand_idx2[j]] * real2$dloc[r, j]
    M1 <- if (q1 > 0) sweep(XR1m, 2, real1$dscale[r, ], `*`) else matrix(0, n, 0)
    M2 <- if (q2 > 0) sweep(XR2m, 2, real2$dscale[r, ], `*`) else matrix(0, n, 0)
    # per-draw score matrix G (n x npar)
    G <- matrix(0, n, npar)
    G[, iB1] <- D1 * sc$s_eta1
    G[, iB2] <- D2 * sc$s_eta2
    if (q1 > 0) G[, iL1] <- M1 * sc$s_eta1
    if (q2 > 0) G[, iL2] <- M2 * sc$s_eta2
    G[, im1] <- sc$s_logm1
    G[, im2] <- sc$s_logm2
    G[, iz]  <- sc$s_ztheta
    grad <- grad + colSums(G * wv)
    if (want_scores) S <- S + G * wv
  }

  if (pois1) { grad[im1] <- 0; if (want_scores) S[, im1] <- 0 }
  if (pois2) { grad[im2] <- 0; if (want_scores) S[, im2] <- 0 }

  out <- value
  attr(out, "gradient") <- grad
  if (want_scores) attr(out, "scores") <- S
  out
}

#' @keywords internal
#' @noRd
rpbnb_copula_cpp_available <- function() exists("rpbnb_copula_ll_grad_cpp", mode = "function")

#' Multithreaded C++ copula RP-BNB LL + gradient (+ scores). Same interface as
#' bnbr_rp_copula_ll_grad; used by the fit when the DLL is compiled.
#' @keywords internal
#' @noRd
bnbr_rp_copula_ll_grad_cpp <- function(par, y1, y2, X1, X2, XR1, XR2,
                                       rand_idx1, rand_idx2, Z1, Z2, family,
                                       dist1 = NULL, dist2 = NULL,
                                       sign1 = NULL, sign2 = NULL,
                                       want_scores = FALSE, n_threads = 0L, pois1 = FALSE, pois2 = FALSE,
                                       off1 = NULL, off2 = NULL) {
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  R  <- if (q1 + q2 > 0) nrow(Z1) else 1L
  beta1 <- par[1:k1]; beta2 <- par[(k1+1):(k1+k2)]
  lg1 <- if (q1>0) (k1+k2+1):(k1+k2+q1) else integer(0)
  lg2 <- if (q2>0) (k1+k2+q1+1):(k1+k2+q1+q2) else integer(0)
  sd1 <- if (q1>0) exp(par[lg1]) else numeric(0)
  sd2 <- if (q2>0) exp(par[lg2]) else numeric(0)
  idx_end <- k1+k2+q1+q2
  log_m1 <- par[idx_end+1]; log_m2 <- par[idx_end+2]; z_theta <- par[idx_end+3]
  r1 <- if (pois1) Inf else exp(-log_m1)
  r2 <- if (pois2) Inf else exp(-log_m2)
  theta <- z_to_native(family, z_theta); dth_dz <- dnative_dz(family, z_theta)
  if (is.null(dist1) && q1>0) dist1 <- rep("normal", q1)
  if (is.null(dist2) && q2>0) dist2 <- rep("normal", q2)
  if (is.null(sign1) && q1>0) sign1 <- rep(1, q1)
  if (is.null(sign2) && q2>0) sign2 <- rep(1, q2)
  real1 <- if (q1>0) rand_realize(Z1, dist1, sign1, beta1[rand_idx1], sd1)
           else list(dev=matrix(0,R,0), dloc=matrix(0,R,0), dscale=matrix(0,R,0))
  real2 <- if (q2>0) rand_realize(Z2, dist2, sign2, beta2[rand_idx2], sd2)
           else list(dev=matrix(0,R,0), dloc=matrix(0,R,0), dscale=matrix(0,R,0))
  xr1 <- if (q1>0) X1[, rand_idx1, drop=FALSE] else matrix(0, nrow(X1), 0)
  xr2 <- if (q2>0) X2[, rand_idx2, drop=FALSE] else matrix(0, nrow(X2), 0)
  fam_code <- match(family, c("frank","normal","kimeldorf")) - 1L
  res <- rpbnb_copula_ll_grad_cpp(
    y1, y2, X1, X2, xr1, xr2, as.integer(rand_idx1-1L), as.integer(rand_idx2-1L),
    real1$dev, real2$dev, real1$dloc, real2$dloc, real1$dscale, real2$dscale,
    as.vector(X1 %*% beta1) + .as_offset(off1, nrow(X1)),
    as.vector(X2 %*% beta2) + .as_offset(off2, nrow(X2)),
    r1, r2, theta, dth_dz, as.integer(fam_code),
    as.integer(isTRUE(want_scores)), as.integer(n_threads))
  val <- res$value
  attr(val, "gradient") <- res$gradient
  if (isTRUE(want_scores)) attr(val, "scores") <- res$scores
  val
}
