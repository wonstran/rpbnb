# Multithreaded (OpenMP) drop-in replacements for the RP-BNB simulated
# likelihood. These prepare the per-draw distribution transforms in R (reusing
# the tested rand_realize) and hand the numerically hot triple loop to the C++
# core in src/halton_parallel.cpp. Math-identical to bnbr_rp_ll_and_grad /
# bnbr_rp_ll_fixed_bounds; verified in tests/testthat/test-cpp-likelihood.R.

#' Is the multithreaded C++ likelihood available?
#' @keywords internal
#' @noRd
rpbnb_cpp_available <- function() {
  exists("rpbnb_ll_grad_cpp", mode = "function")
}

# Build the dev / dloc / dscale (R x q) matrices for one equation, matching the
# two code paths in bnbr_rp_ll_and_grad exactly.
.rp_realize_parts <- function(Z, dist, sign, b, s, q, R) {
  if (q == 0) {
    z <- matrix(0, R, 0)
    return(list(dev = z, dloc = z, dscale = z))
  }
  if (!is.null(dist)) {
    real <- rand_realize(Z, dist, sign, b = b, s = s)
    list(dev = real$dev, dloc = real$dloc, dscale = real$dscale)
  } else {
    Zsd <- sweep(Z, 2, s, `*`)          # legacy: pre-scaled normal deviations
    list(dev = Zsd, dloc = matrix(1, R, q), dscale = Zsd)
  }
}

# Shared unpacking for both C++ entry points. Returns everything the C++ core
# needs plus the parameter block sizes.
.rp_prepare <- function(par, y1, y2, X1, X2, XR1, XR2,
                        rand_idx1, rand_idx2, Z1, Z2,
                        dist1, dist2, sign1, sign2,
                        pois1 = FALSE, pois2 = FALSE) {
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  R  <- if (q1 + q2 > 0) nrow(Z1) else 1L

  beta1 <- par[1:k1]; beta2 <- par[(k1 + 1):(k1 + k2)]
  log_sd1 <- if (q1 > 0) par[(k1 + k2 + 1):(k1 + k2 + q1)] else numeric(0)
  log_sd2 <- if (q2 > 0) par[(k1 + k2 + q1 + 1):(k1 + k2 + q1 + q2)] else numeric(0)
  idx_end <- k1 + k2 + q1 + q2
  # m == 0 is the in-band Poisson flag for the C++ core (c_val -> exp(-d*mu),
  # nb_logpmf -> dpois, log_m gradient/score forced to 0). The pinned log_m in
  # `par` is ignored for a Poisson margin.
  m1 <- if (pois1) 0 else exp(par[idx_end + 1])
  m2 <- if (pois2) 0 else exp(par[idx_end + 2])
  zlam <- par[idx_end + 3]
  sd1 <- if (q1 > 0) exp(log_sd1) else numeric(0)
  sd2 <- if (q2 > 0) exp(log_sd2) else numeric(0)

  p1 <- .rp_realize_parts(Z1, dist1, sign1, beta1[rand_idx1], sd1, q1, R)
  p2 <- .rp_realize_parts(Z2, dist2, sign2, beta2[rand_idx2], sd2, q2, R)

  xr1 <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else matrix(0, nrow(X1), 0)
  xr2 <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else matrix(0, nrow(X2), 0)

  # S1/S2 feed only the (guarded) log_m gradient; a Poisson margin's r = Inf would
  # make digamma(Inf) a 0/0, so pass 0 there -- the C++ core skips its log_m term.
  r1v <- 1 / m1; r2v <- 1 / m2
  S1 <- if (pois1) numeric(length(y1)) else digamma(r1v + y1) - digamma(r1v)
  S2 <- if (pois2) numeric(length(y2)) else digamma(r2v + y2) - digamma(r2v)

  list(
    y1 = y1, y2 = y2, X1 = X1, X2 = X2, XR1 = xr1, XR2 = xr2,
    ri1 = as.integer(rand_idx1 - 1L), ri2 = as.integer(rand_idx2 - 1L),
    dev1 = p1$dev, dev2 = p2$dev, dloc1 = p1$dloc, dloc2 = p2$dloc,
    dscale1 = p1$dscale, dscale2 = p2$dscale,
    xb1 = as.vector(X1 %*% beta1), xb2 = as.vector(X2 %*% beta2),
    S1 = S1, S2 = S2, m1 = m1, m2 = m2, zlam = zlam
  )
}

#' Multithreaded RP-BNB simulated LL + gradient (C++ core)
#' @keywords internal
#' @noRd
bnbr_rp_ll_and_grad_cpp <- function(par, y1, y2, X1, X2, XR1, XR2,
                                    rand_idx1, rand_idx2, Z1, Z2,
                                    dist1 = NULL, dist2 = NULL,
                                    sign1 = NULL, sign2 = NULL,
                                    n_threads = 0L,
                                    pois1 = FALSE, pois2 = FALSE) {
  d <- .rp_prepare(par, y1, y2, X1, X2, XR1, XR2,
                   rand_idx1, rand_idx2, Z1, Z2, dist1, dist2, sign1, sign2,
                   pois1 = pois1, pois2 = pois2)
  res <- rpbnb_ll_grad_cpp(
    d$y1, d$y2, d$X1, d$X2, d$XR1, d$XR2, d$ri1, d$ri2,
    d$dev1, d$dev2, d$dloc1, d$dloc2, d$dscale1, d$dscale2,
    d$xb1, d$xb2, d$S1, d$S2, d$m1, d$m2, d$zlam,
    want_grad = 1L, use_fixed = 0L, lamLo_in = 0, lamHi_in = 0,
    want_scores = 0L, num_threads = as.integer(n_threads)
  )
  val <- res$value
  attr(val, "gradient") <- res$gradient
  val
}

#' Per-observation score matrix (n x npar) at a parameter vector, for OPG/BHHH
#'
#' Row i is s_i = sum_r W_ir * dlogP_ir/dtheta, so crossprod() of the returned
#' matrix is the outer-product-of-gradients (BHHH) information matrix. Uses the
#' same simulation draws as the objective. Parameter/column order matches the
#' gradient: beta1, beta2, log_sd1, log_sd2, log_m1, log_m2, z_lambda.
#' @keywords internal
#' @noRd
bnbr_rp_scores_cpp <- function(par, y1, y2, X1, X2, XR1, XR2,
                               rand_idx1, rand_idx2, Z1, Z2,
                               dist1 = NULL, dist2 = NULL,
                               sign1 = NULL, sign2 = NULL,
                               n_threads = 0L,
                               pois1 = FALSE, pois2 = FALSE) {
  d <- .rp_prepare(par, y1, y2, X1, X2, XR1, XR2,
                   rand_idx1, rand_idx2, Z1, Z2, dist1, dist2, sign1, sign2,
                   pois1 = pois1, pois2 = pois2)
  res <- rpbnb_ll_grad_cpp(
    d$y1, d$y2, d$X1, d$X2, d$XR1, d$XR2, d$ri1, d$ri2,
    d$dev1, d$dev2, d$dloc1, d$dloc2, d$dscale1, d$dscale2,
    d$xb1, d$xb2, d$S1, d$S2, d$m1, d$m2, d$zlam,
    want_grad = 0L, use_fixed = 0L, lamLo_in = 0, lamHi_in = 0,
    want_scores = 1L, num_threads = as.integer(n_threads)
  )
  res$scores
}

#' OPG (BHHH) covariance from per-observation scores, with curvature diagnostics
#'
#' V = (S'S)^{-1}. Delegates to `.observed_info_vcov()` so a singular S'S (e.g. a
#' random-coefficient SD collapsed to ~0, contributing almost no score variation
#' and hence weakly identified) is repaired non-silently and recorded. Returns
#' the full list(vcov, se, diag).
#' @keywords internal
#' @noRd
opg_vcov <- function(scores, par_names) {
  .observed_info_vcov(crossprod(scores), par_names, label = "OPG (BHHH)")
}

#' Fixed-bounds RP-BNB simulated LL (C++ core, value only)
#' @keywords internal
#' @noRd
bnbr_rp_ll_fixed_bounds_cpp <- function(par, y1, y2, X1, X2, XR1, XR2,
                                        rand_idx1, rand_idx2, Z1, Z2,
                                        lamLo, lamHi,
                                        dist1 = NULL, dist2 = NULL,
                                        sign1 = NULL, sign2 = NULL,
                                        n_threads = 0L,
                                        pois1 = FALSE, pois2 = FALSE) {
  d <- .rp_prepare(par, y1, y2, X1, X2, XR1, XR2,
                   rand_idx1, rand_idx2, Z1, Z2, dist1, dist2, sign1, sign2,
                   pois1 = pois1, pois2 = pois2)
  res <- rpbnb_ll_grad_cpp(
    d$y1, d$y2, d$X1, d$X2, d$XR1, d$XR2, d$ri1, d$ri2,
    d$dev1, d$dev2, d$dloc1, d$dloc2, d$dscale1, d$dscale2,
    d$xb1, d$xb2, d$S1, d$S2, d$m1, d$m2, d$zlam,
    want_grad = 0L, use_fixed = 1L, lamLo_in = lamLo, lamHi_in = lamHi,
    want_scores = 0L, num_threads = as.integer(n_threads)
  )
  res$value
}

#' Number of CPU threads available for the multithreaded likelihood
#'
#' @return Integer thread count reported by OpenMP (1 if OpenMP is unavailable).
#' @export
rpbnb_threads <- function() {
  if (exists("get_num_threads", mode = "function")) get_num_threads() else 1L
}
