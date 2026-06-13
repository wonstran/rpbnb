# Random-parameter BNB (Famoye) simulated-likelihood internals.
# Ported math-identical from Rcodes/rpbnbr_faymore.R. Core math comes from
# famoye_core.R; row_log_sum_exp from utilities.R.

#' Simulated log-likelihood + analytic gradient for the RP-BNB model
#'
#' Parameter vector order: beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2),
#' log_m1, log_m2, z_lambda.
#' @keywords internal
#' @noRd
bnbr_rp_ll_and_grad <- compiler::cmpfun(function(par, y1, y2, X1, X2, XR1, XR2,
                                                 rand_idx1, rand_idx2, Z1, Z2, cl = NULL) {
  n   <- length(y1)
  k1  <- ncol(X1); k2 <- ncol(X2)
  q1  <- length(rand_idx1); q2 <- length(rand_idx2)
  R   <- if (q1 + q2 > 0) nrow(Z1) else 1L

  # unpack
  i1 <- 1:k1
  i2 <- (k1+1):(k1+k2)
  beta1   <- par[i1]
  beta2   <- par[i2]
  lg1     <- if (q1>0) (k1+k2+1):(k1+k2+q1) else integer(0)
  lg2     <- if (q2>0) (k1+k2+q1+1):(k1+k2+q1+q2) else integer(0)
  log_sd1 <- if (q1>0) par[lg1] else numeric(0)
  log_sd2 <- if (q2>0) par[lg2] else numeric(0)
  idx_end <- k1 + k2 + q1 + q2
  log_m1  <- par[idx_end + 1]
  log_m2  <- par[idx_end + 2]
  zlam    <- par[idx_end + 3]

  m1 <- exp(log_m1); r1 <- 1/m1
  m2 <- exp(log_m2); r2 <- 1/m2
  sd1 <- if (q1 > 0) exp(log_sd1) else numeric(0)
  sd2 <- if (q2 > 0) exp(log_sd2) else numeric(0)

  xb1 <- as.vector(X1 %*% beta1)
  xb2 <- as.vector(X2 %*% beta2)

  # pre-scale Halton by sd (so per-draw we only multiply XR*row)
  Z1sd <- if (q1>0) sweep(Z1, 2, sd1, `*`) else matrix(0, nrow = max(1L, R), ncol = 0)
  Z2sd <- if (q2>0) sweep(Z2, 2, sd2, `*`) else matrix(0, nrow = max(1L, R), ncol = 0)

  # ---- Pass 1: mu,c + bounds per draw ----
  pass1_fun <- function(r) {
    mu1_r <- if (q1 > 0) exp(xb1 + as.vector(XR1 %*% Z1sd[r,])) else exp(xb1)
    mu2_r <- if (q2 > 0) exp(xb2 + as.vector(XR2 %*% Z2sd[r,])) else exp(xb2)
    c1_r <- c_val(mu1_r, m1)
    c2_r <- c_val(mu2_r, m2)
    b <- lambda_bounds_vec(c1_r, c2_r)
    list(mu1 = mu1_r, mu2 = mu2_r, c1 = c1_r, c2 = c2_r, lamLo_r = b[1], lamHi_r = b[2])
  }
  pass1 <- if (!is.null(cl)) parallel::parLapply(cl, seq_len(R), pass1_fun) else lapply(seq_len(R), pass1_fun)

  lamLo <- max(vapply(pass1, `[[`, numeric(1), "lamLo_r"))
  lamHi <- min(vapply(pass1, `[[`, numeric(1), "lamHi_r"))
  if (!(lamLo < lamHi && is.finite(lamLo) && is.finite(lamHi))) {
    val <- -1e50; attr(val, "gradient") <- rep(0, length(par)); return(val)
  }
  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  dlam_dz <- (lamHi - lamLo) * (1 - 2*eps) * sig * (1 - sig)

  # ---- Pass 2: LL matrix (n x R) ----
  pass2_fun <- function(r) {
    mu1_r <- pass1[[r]]$mu1; mu2_r <- pass1[[r]]$mu2; c1_r <- pass1[[r]]$c1; c2_r <- pass1[[r]]$c2
    logNB1 <- nb_logpmf_y_mu_r(y1, mu1_r, r1)
    logNB2 <- nb_logpmf_y_mu_r(y2, mu2_r, r2)
    dep <- 1 + lam * (exp(-y1) - c1_r) * (exp(-y2) - c2_r)
    dep <- pmax(dep, 1e-300)
    logNB1 + logNB2 + log(dep)
  }
  cols <- if (!is.null(cl)) parallel::parLapply(cl, seq_len(R), pass2_fun) else lapply(seq_len(R), pass2_fun)
  LL <- do.call(cbind, cols)

  lse <- row_log_sum_exp(LL); val <- sum(lse - log(R))
  W <- exp(LL - lse)

  # ---- Analytic gradient (weighted avg over draws) ----
  g_beta1 <- numeric(k1); g_beta2 <- numeric(k2)
  g_logsd1 <- if (q1>0) numeric(q1) else numeric(0)
  g_logsd2 <- if (q2>0) numeric(q2) else numeric(0)
  g_logm1 <- 0; g_logm2 <- 0; g_z <- 0

  for (r in 1:R) {
    mu1_r <- pass1[[r]]$mu1; mu2_r <- pass1[[r]]$mu2
    c1_r  <- pass1[[r]]$c1;  c2_r  <- pass1[[r]]$c2
    w_ir  <- W[, r]

    k1v <- exp(-y1) - c1_r
    k2v <- exp(-y2) - c2_r
    dep <- 1 + lam * (k1v * k2v); inv_dep <- 1 / pmax(dep, 1e-300)

    dc1_dm1 <- dct_dm(mu1_r, m1, c1_r); dc2_dm2 <- dct_dm(mu2_r, m2, c2_r)
    dc1_dbetas <- dc_dbeta_mat(mu1_r, m1, c1_r, X1)
    dc2_dbetas <- dc_dbeta_mat(mu2_r, m2, c2_r, X2)

    w1 <- (y1 - mu1_r) / (1 + m1 * mu1_r)
    w2 <- (y2 - mu2_r) / (1 + m2 * mu2_r)

    pen1 <- lam * k2v * inv_dep; pen2 <- lam * k1v * inv_dep

    score_b1 <- sweep(X1, 1, w1, `*`) - sweep(dc1_dbetas, 1, pen1, `*`)
    score_b2 <- sweep(X2, 1, w2, `*`) - sweep(dc2_dbetas, 1, pen2, `*`)
    g_beta1 <- g_beta1 + colSums(sweep(score_b1, 1, w_ir, `*`))
    g_beta2 <- g_beta2 + colSums(sweep(score_b2, 1, w_ir, `*`))

    r1v <- 1/m1; r2v <- 1/m2
    S1 <- digamma(r1v + y1) - digamma(r1v)
    S2 <- digamma(r2v + y2) - digamma(r2v)
    term_m1 <- r1v^2 * log(m1) + r1v^2 * (log(mu1_r + r1v) - 1) +
      r1v^2 * (y1 + r1v)/(mu1_r + r1v) - r1v^2 * S1 - (lam * k2v * inv_dep) * dc1_dm1
    term_m2 <- r2v^2 * log(m2) + r2v^2 * (log(mu2_r + r2v) - 1) +
      r2v^2 * (y2 + r2v)/(mu2_r + r2v) - r2v^2 * S2 - (lam * k1v * inv_dep) * dc2_dm2
    g_logm1 <- g_logm1 + sum(w_ir * (m1 * term_m1))
    g_logm2 <- g_logm2 + sum(w_ir * (m2 * term_m2))

    g_z <- g_z + sum(w_ir * ((k1v * k2v) * inv_dep * dlam_dz))

    if (q1 > 0) {
      M1 <- sweep(XR1, 2, Z1sd[r,], `*`)     # dη/d log_sd = XR1 * (sd*z)
      part_nb <- sweep(M1, 1, w1, `*`)
      row_factor1 <- -(d_const() * c1_r * mu1_r) / (1 + d_const() * m1 * mu1_r)
      part_c  <- sweep(M1, 1, row_factor1, `*`)
      score_logsd1 <- part_nb - sweep(part_c, 1, pen1, `*`)
      g_logsd1 <- g_logsd1 + colSums(sweep(score_logsd1, 1, w_ir, `*`))
    }
    if (q2 > 0) {
      M2 <- sweep(XR2, 2, Z2sd[r,], `*`)
      part_nb2 <- sweep(M2, 1, w2, `*`)
      row_factor2 <- -(d_const() * c2_r * mu2_r) / (1 + d_const() * m2 * mu2_r)
      part_c2  <- sweep(M2, 1, row_factor2, `*`)
      score_logsd2 <- part_nb2 - sweep(part_c2, 1, pen2, `*`)
      g_logsd2 <- g_logsd2 + colSums(sweep(score_logsd2, 1, w_ir, `*`))
    }
  }

  grad <- c(g_beta1, g_beta2, g_logsd1, g_logsd2, g_logm1, g_logm2, g_z)
  attr(val, "gradient") <- grad
  val
})

#' Fixed-bounds simulated log-likelihood for the RP-BNB Hessian
#' @keywords internal
#' @noRd
bnbr_rp_ll_fixed_bounds <- function(par, y1, y2, X1, X2, XR1, XR2,
                                    rand_idx1, rand_idx2, Z1, Z2,
                                    lamLo, lamHi, cl = NULL) {
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  R  <- if (q1 + q2 > 0) nrow(Z1) else 1L

  # unpack
  i1 <- 1:k1; i2 <- (k1+1):(k1+k2)
  beta1 <- par[i1]; beta2 <- par[i2]
  lg1   <- if (q1>0) (k1+k2+1):(k1+k2+q1) else integer(0)
  lg2   <- if (q2>0) (k1+k2+q1+1):(k1+k2+q1+q2) else integer(0)
  log_sd1 <- if (q1>0) par[lg1] else numeric(0)
  log_sd2 <- if (q2>0) par[lg2] else numeric(0)
  idx_end <- k1+k2+q1+q2
  log_m1  <- par[idx_end+1]
  log_m2  <- par[idx_end+2]
  zlam    <- par[idx_end+3]

  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1/m1;       r2 <- 1/m2
  sd1 <- if (q1>0) exp(log_sd1) else numeric(0)
  sd2 <- if (q2>0) exp(log_sd2) else numeric(0)

  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)

  xb1 <- as.vector(X1 %*% beta1)
  xb2 <- as.vector(X2 %*% beta2)
  Z1sd <- if (q1>0) sweep(Z1, 2, sd1, `*`) else matrix(0, nrow = max(1L, R), ncol = 0)
  Z2sd <- if (q2>0) sweep(Z2, 2, sd2, `*`) else matrix(0, nrow = max(1L, R), ncol = 0)

  pass_fun <- function(r) {
    mu1_r <- if (q1>0) exp(xb1 + as.vector(XR1 %*% Z1sd[r,])) else exp(xb1)
    mu2_r <- if (q2>0) exp(xb2 + as.vector(XR2 %*% Z2sd[r,])) else exp(xb2)
    c1_r  <- c_val(mu1_r, m1)
    c2_r  <- c_val(mu2_r, m2)
    logNB1 <- nb_logpmf_y_mu_r(y1, mu1_r, r1)
    logNB2 <- nb_logpmf_y_mu_r(y2, mu2_r, r2)
    dep    <- 1 + lam * (exp(-y1) - c1_r) * (exp(-y2) - c2_r)
    dep    <- pmax(dep, 1e-300)
    logNB1 + logNB2 + log(dep)
  }

  cols <- if (!is.null(cl)) parallel::parLapply(cl, seq_len(R), pass_fun) else lapply(seq_len(R), pass_fun)
  LL   <- do.call(cbind, cols)

  m <- apply(LL, 1, max)
  sum(m + log(rowSums(exp(LL - m))) - log(R))
}
