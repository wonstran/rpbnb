# Analytic observed-information Hessian for the random-parameter BNB simulated
# likelihood. Implements the Louis mixture-Hessian using Famoye (2010) Appendix
# per-draw second derivatives, chain-ruled to the (beta, log_sd, log_m, z_lambda)
# parameterization. See docs/rpbnb_analytic_hessian.md. Validated against
# numDeriv::hessian in tests/testthat/test-rpbnb-hessian.R.
#
# Returns the npar x npar Hessian of the simulated log-likelihood (NOT the
# information); the caller negates and inverts for the covariance.

#' @keywords internal
#' @noRd
bnbr_rp_hessian <- function(par, y1, y2, X1, X2, XR1, XR2,
                            rand_idx1, rand_idx2, Z1, Z2,
                            dist1 = NULL, dist2 = NULL,
                            sign1 = NULL, sign2 = NULL,
                            lamLo = NULL, lamHi = NULL,
                            pois1 = FALSE, pois2 = FALSE) {
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
  # A Poisson margin uses the exact m = 0 limit: the beta/log_sd/lambda blocks
  # reduce to their Poisson values at m = 0, and its log_m row/column (a fixed
  # parameter, a 0/0 in the NB2 dispersion curvature) is zeroed at the end.
  m1 <- if (pois1) 0 else exp(par[idx_end + 1])
  m2 <- if (pois2) 0 else exp(par[idx_end + 2])
  zlam <- par[idx_end + 3]
  r1 <- if (pois1) Inf else 1 / m1; r2 <- if (pois2) Inf else 1 / m2
  d <- d_const()

  if (is.null(dist1) && q1 > 0) dist1 <- rep("normal", q1)
  if (is.null(dist2) && q2 > 0) dist2 <- rep("normal", q2)
  if (is.null(sign1) && q1 > 0) sign1 <- rep(1, q1)
  if (is.null(sign2) && q2 > 0) sign2 <- rep(1, q2)

  xb1 <- as.vector(X1 %*% beta1); xb2 <- as.vector(X2 %*% beta2)

  # Per-draw realizations (dev/dloc/dscale/base) via the tested registry path.
  real1 <- if (q1 > 0) rand_realize(Z1, dist1, sign1, beta1[rand_idx1], sd1)
           else list(dev = matrix(0, R, 0), dloc = matrix(0, R, 0),
                     dscale = matrix(0, R, 0), base = matrix(0, R, 0))
  real2 <- if (q2 > 0) rand_realize(Z2, dist2, sign2, beta2[rand_idx2], sd2)
           else list(dev = matrix(0, R, 0), dloc = matrix(0, R, 0),
                     dscale = matrix(0, R, 0), base = matrix(0, R, 0))

  # Second-order design factors d2_ss / d2_bb / d2_bs per (draw, random col).
  d2_factors <- function(real, dist, s, q) {
    if (q == 0) return(list(bb = matrix(0, R, 0), bs = matrix(0, R, 0),
                            ss = matrix(0, R, 0)))
    bb <- matrix(0, R, q); bs <- matrix(0, R, q); ss <- matrix(0, R, q)
    for (j in seq_len(q)) {
      if (identical(dist[j], "lognormal")) {
        coefj <- real$coef[, j]; dscj <- real$dscale[, j]
        sbase <- s[j] * real$base[, j]
        bb[, j] <- coefj                 # d2 coef / db2      = coef
        bs[, j] <- dscj                  # d2 coef / db dlogs = dscale
        ss[, j] <- dscj * (1 + sbase)    # d2 coef / dlogs2
      } else {
        ss[, j] <- real$dscale[, j]      # linear dists: only d2/dlogs2 = dscale
      }
    }
    list(bb = bb, bs = bs, ss = ss)
  }
  f1 <- d2_factors(real1, dist1, sd1, q1)
  f2 <- d2_factors(real2, dist2, sd2, q2)

  XR1m <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else matrix(0, n, 0)
  XR2m <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else matrix(0, n, 0)

  # ---- Pass 1: per-draw mu, c, LL; reduce lambda bounds -----------------------
  mu1M <- matrix(0, n, R); mu2M <- matrix(0, n, R)
  c1M  <- matrix(0, n, R); c2M  <- matrix(0, n, R)
  LL   <- matrix(0, n, R)
  lamLo_r <- numeric(R); lamHi_r <- numeric(R)
  ey1 <- exp(-y1); ey2 <- exp(-y2)
  for (r in seq_len(R)) {
    eta1 <- xb1 + if (q1 > 0) as.vector(XR1m %*% real1$dev[r, ]) else 0
    eta2 <- xb2 + if (q2 > 0) as.vector(XR2m %*% real2$dev[r, ]) else 0
    mu1 <- pmin(exp(eta1), 1e15); mu2 <- pmin(exp(eta2), 1e15)
    cc1 <- c_val(mu1, m1); cc2 <- c_val(mu2, m2)
    mu1M[, r] <- mu1; mu2M[, r] <- mu2; c1M[, r] <- cc1; c2M[, r] <- cc2
    b <- lambda_bounds_vec(cc1, cc2)
    lamLo_r[r] <- b[1]; lamHi_r[r] <- b[2]
  }
  if (is.null(lamLo) || is.null(lamHi)) {
    lamLo <- max(lamLo_r); lamHi <- min(lamHi_r)
  }
  eps <- 1e-6; sig <- plogis(zlam)
  lam       <- lamLo + (lamHi - lamLo) * (eps + (1 - 2 * eps) * sig)
  dlam_dz   <- (lamHi - lamLo) * (1 - 2 * eps) * sig * (1 - sig)
  d2lam_dz2 <- (lamHi - lamLo) * (1 - 2 * eps) * sig * (1 - sig) * (1 - 2 * sig)

  for (r in seq_len(R)) {
    dep <- pmax(1 + lam * (ey1 - c1M[, r]) * (ey2 - c2M[, r]), 1e-300)
    LL[, r] <- nb_logpmf_y_mu_r(y1, mu1M[, r], r1) +
               nb_logpmf_y_mu_r(y2, mu2M[, r], r2) + log(dep)
  }
  lse <- row_log_sum_exp(LL)
  W   <- exp(LL - lse)   # n x R softmax weights

  # ---- accumulate the three Hessian pieces -----------------------------------
  H_H  <- matrix(0, npar, npar)   # Sum_r Sum_i w H_ir  (design-contracted)
  H_gg <- matrix(0, npar, npar)   # Sum_r G_r' diag(w) G_r
  S    <- matrix(0, n, npar)      # Sum_r diag(W[,r]) G_r  (per-obs scores)

  iB1 <- 1:k1; iB2 <- (k1 + 1):(k1 + k2)
  iL1 <- if (q1 > 0) lg1 else integer(0)
  iL2 <- if (q2 > 0) lg2 else integer(0)
  im1 <- idx_end + 1; im2 <- idx_end + 2; iz <- idx_end + 3

  for (r in seq_len(R)) {
    mu1 <- mu1M[, r]; mu2 <- mu2M[, r]; cc1 <- c1M[, r]; cc2 <- c2M[, r]
    wv  <- W[, r]
    k1v <- ey1 - cc1; k2v <- ey2 - cc2
    dep <- pmax(1 + lam * (k1v * k2v), 1e-300); invD <- 1 / dep; invD2 <- invD^2
    pen1 <- lam * k2v * invD; pen2 <- lam * k1v * invD
    G1 <- 1 + d * m1 * mu1; G2 <- 1 + d * m2 * mu2
    w1 <- (y1 - mu1) / (1 + m1 * mu1); w2 <- (y2 - mu2) / (1 + m2 * mu2)

    rf1 <- -(d * cc1 * mu1) / G1;  rf2 <- -(d * cc2 * mu2) / G2
    dcm1 <- dct_dm(mu1, m1, cc1);  dcm2 <- dct_dm(mu2, m2, cc2)
    b1 <- d2ct_dbeta2_factor(mu1, m1, cc1); b2 <- d2ct_dbeta2_factor(mu2, m2, cc2)
    e1 <- d2ct_dmdbeta_factor(mu1, m1, cc1); e2 <- d2ct_dmdbeta_factor(mu2, m2, cc2)
    d2cm1 <- d2ct_dm2(mu1, m1, cc1); d2cm2 <- d2ct_dm2(mu2, m2, cc2)

    # NB dispersion first/second derivatives (natural m) via digamma/trigamma.
    A1 <- (digamma(y1 + r1) - digamma(r1)) + log(r1) + 1 - log(r1 + mu1) - (r1 + y1) / (r1 + mu1)
    B1 <- (trigamma(y1 + r1) - trigamma(r1)) + 1 / r1 - 1 / (r1 + mu1) + (y1 - mu1) / (r1 + mu1)^2
    A2 <- (digamma(y2 + r2) - digamma(r2)) + log(r2) + 1 - log(r2 + mu2) - (r2 + y2) / (r2 + mu2)
    B2 <- (trigamma(y2 + r2) - trigamma(r2)) + 1 / r2 - 1 / (r2 + mu2) + (y2 - mu2) / (r2 + mu2)^2
    nb_dm1 <- -r1^2 * A1; nb_d2m1 <- r1^4 * B1 + 2 * r1^3 * A1
    nb_dm2 <- -r2^2 * A2; nb_d2m2 <- r2^4 * B2 + 2 * r2^3 * A2

    # First derivatives (natural).
    deta1 <- w1 - pen1 * rf1;  deta2 <- w2 - pen2 * rf2
    dm1n  <- nb_dm1 - pen1 * dcm1;  dm2n <- nb_dm2 - pen2 * dcm2
    dln   <- k1v * k2v * invD

    # Reduced second derivatives (natural m, natural lambda).
    Heta1 <- -mu1 * (1 + m1 * y1) / (1 + m1 * mu1)^2 - lam^2 * k2v^2 * invD2 * rf1^2 - pen1 * b1
    Heta2 <- -mu2 * (1 + m2 * y2) / (1 + m2 * mu2)^2 - lam^2 * k1v^2 * invD2 * rf2^2 - pen2 * b2
    He1e2 <- lam * invD2 * rf1 * rf2
    He1m1 <- mu1 * (mu1 - y1) / (1 + m1 * mu1)^2 - lam^2 * k2v^2 * invD2 * rf1 * dcm1 - pen1 * e1
    He2m2 <- mu2 * (mu2 - y2) / (1 + m2 * mu2)^2 - lam^2 * k1v^2 * invD2 * rf2 * dcm2 - pen2 * e2
    He1m2 <- lam * invD2 * rf1 * dcm2
    He2m1 <- lam * invD2 * rf2 * dcm1
    Hm1m1 <- nb_d2m1 - lam^2 * k2v^2 * invD2 * dcm1^2 - pen1 * d2cm1
    Hm2m2 <- nb_d2m2 - lam^2 * k1v^2 * invD2 * dcm2^2 - pen2 * d2cm2
    Hm1m2 <- lam * invD2 * dcm1 * dcm2
    He1l <- -k2v * invD2 * rf1;  He2l <- -k1v * invD2 * rf2
    Hm1l <- -k2v * invD2 * dcm1; Hm2l <- -k1v * invD2 * dcm2
    Hll  <- -(k1v * k2v)^2 * invD2

    # Design matrices for this draw.
    D1 <- X1; if (q1 > 0) for (j in seq_len(q1)) D1[, rand_idx1[j]] <- X1[, rand_idx1[j]] * real1$dloc[r, j]
    D2 <- X2; if (q2 > 0) for (j in seq_len(q2)) D2[, rand_idx2[j]] <- X2[, rand_idx2[j]] * real2$dloc[r, j]
    M1 <- if (q1 > 0) sweep(XR1m, 2, real1$dscale[r, ], `*`) else matrix(0, n, 0)
    M2 <- if (q2 > 0) sweep(XR2m, 2, real2$dscale[r, ], `*`) else matrix(0, n, 0)

    # Per-draw score matrix G_r (n x npar).
    G <- matrix(0, n, npar)
    G[, iB1] <- D1 * deta1
    G[, iB2] <- D2 * deta2
    if (q1 > 0) G[, iL1] <- M1 * deta1
    if (q2 > 0) G[, iL2] <- M2 * deta2
    G[, im1] <- m1 * dm1n
    G[, im2] <- m2 * dm2n
    G[, iz]  <- dln * dlam_dz

    H_gg <- H_gg + crossprod(G, G * wv)   # G' diag(wv) G
    S    <- S + G * wv

    # ---- term_H: contract the per-draw density Hessian, weighted by wv --------
    Hb <- matrix(0, npar, npar)
    D1w <- D1 * wv; D2w <- D2 * wv
    # eta-eta blocks
    Hb[iB1, iB1] <- crossprod(D1w * Heta1, D1)
    Hb[iB2, iB2] <- crossprod(D2w * Heta2, D2)
    Hb[iB1, iB2] <- crossprod(D1w * He1e2, D2); Hb[iB2, iB1] <- t(Hb[iB1, iB2])
    if (q1 > 0) {
      Hb[iL1, iL1] <- crossprod(M1 * (wv * Heta1), M1)
      Hb[iB1, iL1] <- crossprod(D1w * Heta1, M1); Hb[iL1, iB1] <- t(Hb[iB1, iL1])
    }
    if (q2 > 0) {
      Hb[iL2, iL2] <- crossprod(M2 * (wv * Heta2), M2)
      Hb[iB2, iL2] <- crossprod(D2w * Heta2, M2); Hb[iL2, iB2] <- t(Hb[iB2, iL2])
    }
    if (q1 > 0 && q2 > 0) {
      Hb[iL1, iL2] <- crossprod(M1 * (wv * He1e2), M2); Hb[iL2, iL1] <- t(Hb[iL1, iL2])
    }
    if (q1 > 0) { Hb[iB2, iL1] <- crossprod(D2w * He1e2, M1); Hb[iL1, iB2] <- t(Hb[iB2, iL1]) }
    if (q2 > 0) { Hb[iB1, iL2] <- crossprod(D1w * He1e2, M2); Hb[iL2, iB1] <- t(Hb[iB1, iL2]) }

    # eta-m blocks (log_m chain: dm/dlogm = m)
    Hb[iB1, im1] <- crossprod(D1, wv * He1m1 * m1); Hb[im1, iB1] <- t(Hb[iB1, im1])
    Hb[iB1, im2] <- crossprod(D1, wv * He1m2 * m2); Hb[im2, iB1] <- t(Hb[iB1, im2])
    Hb[iB2, im1] <- crossprod(D2, wv * He2m1 * m1); Hb[im1, iB2] <- t(Hb[iB2, im1])
    Hb[iB2, im2] <- crossprod(D2, wv * He2m2 * m2); Hb[im2, iB2] <- t(Hb[iB2, im2])
    if (q1 > 0) {
      Hb[iL1, im1] <- crossprod(M1, wv * He1m1 * m1); Hb[im1, iL1] <- t(Hb[iL1, im1])
      Hb[iL1, im2] <- crossprod(M1, wv * He1m2 * m2); Hb[im2, iL1] <- t(Hb[iL1, im2])
    }
    if (q2 > 0) {
      Hb[iL2, im1] <- crossprod(M2, wv * He2m1 * m1); Hb[im1, iL2] <- t(Hb[iL2, im1])
      Hb[iL2, im2] <- crossprod(M2, wv * He2m2 * m2); Hb[im2, iL2] <- t(Hb[iL2, im2])
    }
    # eta-z blocks (lambda chain: dlam/dz = dlam_dz)
    Hb[iB1, iz] <- crossprod(D1, wv * He1l * dlam_dz); Hb[iz, iB1] <- t(Hb[iB1, iz])
    Hb[iB2, iz] <- crossprod(D2, wv * He2l * dlam_dz); Hb[iz, iB2] <- t(Hb[iB2, iz])
    if (q1 > 0) { Hb[iL1, iz] <- crossprod(M1, wv * He1l * dlam_dz); Hb[iz, iL1] <- t(Hb[iL1, iz]) }
    if (q2 > 0) { Hb[iL2, iz] <- crossprod(M2, wv * He2l * dlam_dz); Hb[iz, iL2] <- t(Hb[iL2, iz]) }

    # scalar blocks (m-m, m-z, z-z) with log/logistic chain corrections
    Hb[im1, im1] <- sum(wv * (Hm1m1 * m1^2 + dm1n * m1))
    Hb[im2, im2] <- sum(wv * (Hm2m2 * m2^2 + dm2n * m2))
    Hb[im1, im2] <- sum(wv * (Hm1m2 * m1 * m2)); Hb[im2, im1] <- Hb[im1, im2]
    Hb[im1, iz]  <- sum(wv * (Hm1l * m1 * dlam_dz)); Hb[iz, im1] <- Hb[im1, iz]
    Hb[im2, iz]  <- sum(wv * (Hm2l * m2 * dlam_dz)); Hb[iz, im2] <- Hb[im2, iz]
    Hb[iz, iz]   <- sum(wv * (Hll * dlam_dz^2 + dln * d2lam_dz2))

    # second-order design corrections (random columns only)
    if (q1 > 0) for (j in seq_len(q1)) {
      col <- rand_idx1[j]; xr <- XR1m[, j]; wd <- wv * deta1
      Hb[col, col]    <- Hb[col, col]    + sum(wd * xr * f1$bb[r, j])
      Hb[col, iL1[j]] <- Hb[col, iL1[j]] + sum(wd * xr * f1$bs[r, j]); Hb[iL1[j], col] <- Hb[col, iL1[j]]
      Hb[iL1[j], iL1[j]] <- Hb[iL1[j], iL1[j]] + sum(wd * xr * f1$ss[r, j])
    }
    if (q2 > 0) for (j in seq_len(q2)) {
      col <- rand_idx2[j]; xr <- XR2m[, j]; wd <- wv * deta2; bc <- k1 + col
      Hb[bc, bc]      <- Hb[bc, bc]      + sum(wd * xr * f2$bb[r, j])
      Hb[bc, iL2[j]]  <- Hb[bc, iL2[j]]  + sum(wd * xr * f2$bs[r, j]); Hb[iL2[j], bc] <- Hb[bc, iL2[j]]
      Hb[iL2[j], iL2[j]] <- Hb[iL2[j], iL2[j]] + sum(wd * xr * f2$ss[r, j])
    }

    H_H <- H_H + Hb
  }

  H_ss <- crossprod(S)
  H <- H_H + H_gg - H_ss
  # Zero each Poisson margin's log_m row/column: its NB2 dispersion curvature is a
  # 0/0 at m = 0 and enters only that row/column, which .free_index_vcov() drops
  # from the free-parameter information. (The beta/log_sd/lambda blocks already
  # used the exact m = 0 limits.)
  if (pois1) { H[im1, ] <- 0; H[, im1] <- 0 }
  if (pois2) { H[im2, ] <- 0; H[, im2] <- 0 }
  (H + t(H)) / 2
}
