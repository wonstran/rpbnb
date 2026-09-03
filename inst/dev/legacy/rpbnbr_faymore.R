# =========================================================
# Random-Parameters BNBR (Famoye) -- BFGS + Analytic Gradient
# - Halton draws (randtoolbox), optional parallel over draws
# - Numeric Hessian via numDeriv with smaller draw count
# - Displays SDs (not log SDs) for random coefficients
# - Convergence plots
# =========================================================

# ---- Packages ----
need <- c("maxLik", "numDeriv", "randtoolbox", "parallel")
to_install <- need[!need %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
suppressPackageStartupMessages({
  library(maxLik); library(numDeriv); library(randtoolbox); library(parallel)
})
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")
if (!requireNamespace("compiler", quietly = TRUE)) install.packages("compiler")
compiler::enableJIT(3)

# ---------- Core math ----------
d_const <- 1 - exp(-1)
c_val   <- function(mu, m) (1 + d_const * m * mu)^(-1/m)

nb_logpmf_y_mu_r <- function(y, mu, r) {
  p <- r / (r + mu)
  lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r*log(p) + y*log1p(-p)
}

lambda_bounds_vec <- function(c1, c2) {
  lam_min <- -1 / ((1 - c1) * (1 - c2))
  lam_max <-  1 / pmax(c1 * (1 - c2), c2 * (1 - c1))
  c(max(lam_min), min(lam_max))
}

# Derivatives of c(mu,m)
dct_dm <- function(mu, m, c) {
  m_inv <- 1/m
  denom <- 1 + d_const * m * mu
  term  <- m_inv * ( m_inv * log(denom) - (d_const * mu) / denom )
  term * c
}
dc_dbeta_mat <- function(mu, m, c, X) {
  denom <- 1 + d_const * m * mu
  row_factor <- -(d_const * c * mu) / denom
  sweep(X, 1, row_factor, `*`)
}

# ---------- Utilities ----------
rowLogSumExp <- function(M) {
  m <- apply(M, 1, max)
  m + log(rowSums(exp(M - m)))
}
make_halton_norm <- function(n_draws, d, skip = 100, burn = 200) {
  n <- burn + n_draws
  U <- randtoolbox::halton(n = n, dim = d, normal = FALSE, usetime = TRUE, init = TRUE)
  qnorm(U[(burn+1):n, , drop = FALSE])
}

# ---------- Simulated LL + analytic gradient ----------
# par = [ beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2), log_m1, log_m2, z_lambda ]
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
  pass1 <- if (!is.null(cl)) parLapply(cl, seq_len(R), pass1_fun) else lapply(seq_len(R), pass1_fun)
  
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
  cols <- if (!is.null(cl)) parLapply(cl, seq_len(R), pass2_fun) else lapply(seq_len(R), pass2_fun)
  LL <- do.call(cbind, cols)
  
  lse <- rowLogSumExp(LL); val <- sum(lse - log(R))
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
      row_factor1 <- -(d_const * c1_r * mu1_r) / (1 + d_const * m1 * mu1_r)
      part_c  <- sweep(M1, 1, row_factor1, `*`)
      score_logsd1 <- part_nb - sweep(part_c, 1, pen1, `*`)
      g_logsd1 <- g_logsd1 + colSums(sweep(score_logsd1, 1, w_ir, `*`))
    }
    if (q2 > 0) {
      M2 <- sweep(XR2, 2, Z2sd[r,], `*`)
      part_nb2 <- sweep(M2, 1, w2, `*`)
      row_factor2 <- -(d_const * c2_r * mu2_r) / (1 + d_const * m2 * mu2_r)
      part_c2  <- sweep(M2, 1, row_factor2, `*`)
      score_logsd2 <- part_nb2 - sweep(part_c2, 1, pen2, `*`)
      g_logsd2 <- g_logsd2 + colSums(sweep(score_logsd2, 1, w_ir, `*`))
    }
  }
  
  grad <- c(g_beta1, g_beta2, g_logsd1, g_logsd2, g_logm1, g_logm2, g_z)
  attr(val, "gradient") <- grad
  val
})

# ---------- Fixed-bounds LL (for Hessian) ----------
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

# ---------- Fitter wrapper (BFGS) ----------
rpbnbr_bfgs <- function(data, f1, f2,
                        rand_names1 = character(0),
                        rand_names2 = character(0),
                        n_draws = 400,                 # optimize with this many draws
                        n_draws_hess = 100,            # Hessian draws (smaller -> faster)
                        halton_burn = 300, halton_skip = 100,
                        start = NULL,
                        control = list(iterlim = 300, reltol = 1e-8, printLevel = 2),
                        method = "BFGS",
                        n_cores = 1,                   # parallel over draws
                        compute_se = TRUE,             # FALSE skips Hessian/SEs
                        print_output = TRUE) {
  
  stopifnot(is.data.frame(data))
  mf1 <- model.frame(f1, data = data)
  mf2 <- model.frame(f2, data = data)
  Y1  <- as.integer(model.response(mf1))
  Y2  <- as.integer(model.response(mf2))
  if (any(Y1 < 0 | Y2 < 0)) stop("Responses must be non-negative counts.")
  X1  <- model.matrix(f1, mf1)
  X2  <- model.matrix(f2, mf2)
  
  # random names -> indices
  idx_from_names <- function(who, X) {
    if (!length(who)) return(integer(0))
    m <- match(who, colnames(X)); m <- m[!is.na(m)]
    if (!length(m)) integer(0) else as.integer(m)
  }
  rand_idx1 <- idx_from_names(rand_names1, X1)
  rand_idx2 <- idx_from_names(rand_names2, X2)
  
  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  XR1 <- if (q1>0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2 <- if (q2>0) X2[, rand_idx2, drop = FALSE] else NULL
  
  # Halton normals for optimization phase
  if ((q1+q2) > 0) {
    Z_opt <- make_halton_norm(n_draws, q1 + q2, burn = halton_burn, skip = halton_skip)
    Z1_opt <- if (q1>0) Z_opt[, 1:q1, drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
    Z2_opt <- if (q2>0) Z_opt[, (q1+1):(q1+q2), drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
  } else {
    Z1_opt <- matrix(0, nrow = n_draws, ncol = 0)
    Z2_opt <- matrix(0, nrow = n_draws, ncol = 0)
  }
  
  # Start vector
  if (is.null(start)) {
    start <- c(rep(0, k1 + k2),
               if (q1>0) rep(log(0.2), q1) else NULL,
               if (q2>0) rep(log(0.2), q2) else NULL,
               log(0.5), log(0.5), 0)
  }
  par_names <- c(paste0("b1:", colnames(X1)),
                 paste0("b2:", colnames(X2)),
                 if (q1>0) paste0("log_sd1:", colnames(X1)[rand_idx1]) else NULL,
                 if (q2>0) paste0("log_sd2:", colnames(X2)[rand_idx2]) else NULL,
                 "log_m1","log_m2","z_lambda")
  names(start) <- par_names
  
  # Optional cluster (for optimization draws)
  cl <- NULL
  if (n_cores > 1) {
    cl <- parallel::makeCluster(max(1L, as.integer(n_cores)))
    on.exit({ try(parallel::stopCluster(cl), silent = TRUE) }, add = TRUE)
    y1 <- Y1; y2 <- Y2
    parallel::clusterExport(cl,
                            c("X1","X2","XR1","XR2","y1","y2","Z1_opt","Z2_opt",
                              "c_val","lambda_bounds_vec","nb_logpmf_y_mu_r","dct_dm","dc_dbeta_mat","d_const"),
                            envir = environment()
    )
  }
  
  # LL trace for plotting
  ll_trace <- numeric(0)
  ll_fun <- function(p) {
    v <- bnbr_rp_ll_and_grad(p, Y1, Y2, X1, X2, XR1, XR2,
                             rand_idx1, rand_idx2, Z1_opt, Z2_opt, cl = cl)
    ll_trace <<- c(ll_trace, as.numeric(v))
    v
  }
  
  # --------- BFGS optimization (analytic gradient) ----------
  fit <- maxLik::maxLik(logLik = ll_fun, start = start, method = method, control = control)
  par_hat <- coef(fit); names(par_hat) <- par_names
  
  # --- Build frozen λ-bounds at optimum using optimization draws ---
  rebuild_bounds <- function(p) {
    i1 <- 1:k1; i2 <- (k1+1):(k1+k2)
    beta1 <- p[i1]; beta2 <- p[i2]
    lg1 <- if (q1>0) (k1+k2+1):(k1+k2+q1) else integer(0)
    lg2 <- if (q2>0) (k1+k2+q1+1):(k1+k2+q1+q2) else integer(0)
    log_sd1 <- if (q1>0) p[lg1] else numeric(0)
    log_sd2 <- if (q2>0) p[lg2] else numeric(0)
    idx_end <- k1+k2+q1+q2
    m1 <- exp(p[idx_end+1]); m2 <- exp(p[idx_end+2])
    sd1 <- if (q1>0) exp(log_sd1) else numeric(0)
    sd2 <- if (q2>0) exp(log_sd2) else numeric(0)
    
    xb1 <- as.vector(X1 %*% beta1); xb2 <- as.vector(X2 %*% beta2)
    Z1sd <- if (q1>0) sweep(Z1_opt, 2, sd1, `*`) else matrix(0, nrow = n_draws, ncol = 0)
    Z2sd <- if (q2>0) sweep(Z2_opt, 2, sd2, `*`) else matrix(0, nrow = n_draws, ncol = 0)
    
    lamLo <- -Inf; lamHi <- Inf
    Rloc <- ifelse(q1+q2>0, nrow(Z1sd), 1)
    for (r in 1:Rloc) {
      mu1_r <- if (q1>0) exp(xb1 + as.vector(XR1 %*% Z1sd[r,])) else exp(xb1)
      mu2_r <- if (q2>0) exp(xb2 + as.vector(XR2 %*% Z2sd[r,])) else exp(xb2)
      b <- lambda_bounds_vec(c_val(mu1_r, m1), c_val(mu2_r, m2))
      lamLo <- max(lamLo, b[1]); lamHi <- min(lamHi, b[2])
    }
    c(lamLo, lamHi)
  }
  lam_b  <- rebuild_bounds(par_hat)
  lamLo_h <- lam_b[1]; lamHi_h <- lam_b[2]
  if (!(lamLo_h < lamHi_h)) warning("Frozen bounds invalid at optimum; SEs may be unstable.")
  
  # --------- SEs via small Hessian draws (fast) ----------
  vc <- NULL
  se <- rep(NA_real_, length(par_hat)); names(se) <- par_names
  if (isTRUE(compute_se)) {
    # fresh small Halton for Hessian
    if ((q1+q2) > 0) {
      Z_h <- make_halton_norm(n_draws_hess, q1 + q2, burn = max(50, floor(halton_burn/3)), skip = halton_skip)
      Z1_h <- if (q1>0) Z_h[, 1:q1, drop = FALSE] else matrix(0, nrow = n_draws_hess, ncol = 0)
      Z2_h <- if (q2>0) Z_h[, (q1+1):(q1+q2), drop = FALSE] else matrix(0, nrow = n_draws_hess, ncol = 0)
    } else {
      Z1_h <- matrix(0, nrow = n_draws_hess, ncol = 0)
      Z2_h <- matrix(0, nrow = n_draws_hess, ncol = 0)
    }
    
    cl_h <- NULL
    if (n_cores > 1) {
      cl_h <- parallel::makeCluster(max(1L, as.integer(min(n_cores, 4))))
      on.exit({ try(parallel::stopCluster(cl_h), silent = TRUE) }, add = TRUE)
      y1 <- Y1; y2 <- Y2
      parallel::clusterExport(cl_h,
                              c("X1","X2","XR1","XR2","y1","y2","Z1_h","Z2_h",
                                "c_val","lambda_bounds_vec","nb_logpmf_y_mu_r","dct_dm","dc_dbeta_mat","d_const"),
                              envir = environment()
      )
    }
    
    ll_fb <- function(p) bnbr_rp_ll_fixed_bounds(p, Y1, Y2, X1, X2, XR1, XR2,
                                                 rand_idx1, rand_idx2, Z1_h, Z2_h,
                                                 lamLo_h, lamHi_h, cl = cl_h)
    H  <- numDeriv::hessian(ll_fb, par_hat, method.args = list(r = 4, eps = 1e-5))
    info <- -H; info <- (info + t(info)) / 2
    if (any(!is.finite(info))) {
      H <- numDeriv::hessian(ll_fb, par_hat, method.args = list(r = 6, eps = 1e-4))
      info <- -H; info <- (info + t(info)) / 2
    }
    ok <- try(eigen(info, symmetric = TRUE, only.values = TRUE), silent = TRUE)
    if (inherits(ok, "try-error") || any(!is.finite(ok$values)) || min(ok$values) <= 0) {
      ridge <- if (inherits(ok, "try-error") || any(!is.finite(ok$values))) 1e-2 else (1e-8 - min(ok$values))
      info <- info + diag(ridge, nrow(info))
    }
    vc  <- try(solve(info), silent = TRUE)
    if (inherits(vc, "try-error")) vc <- MASS::ginv(info)
    se  <- sqrt(pmax(diag(vc), 0)); names(se) <- par_names
  }
  
  # ---- Display table (show sd, not log_sd) ----
  idx_logsd1 <- if (q1 > 0) which(startsWith(par_names, "log_sd1:")) else integer(0)
  idx_logsd2 <- if (q2 > 0) which(startsWith(par_names, "log_sd2:")) else integer(0)
  disp_est   <- par_hat
  disp_se    <- se
  disp_names <- par_names
  if (length(idx_logsd1)) {
    sd1_hat <- exp(par_hat[idx_logsd1])
    disp_est[idx_logsd1] <- sd1_hat
    if (all(is.finite(se[idx_logsd1]))) disp_se[idx_logsd1] <- abs(sd1_hat * se[idx_logsd1])
    disp_names[idx_logsd1] <- sub("^log_sd1:", "sd1:", disp_names[idx_logsd1])
  }
  if (length(idx_logsd2)) {
    sd2_hat <- exp(par_hat[idx_logsd2])
    disp_est[idx_logsd2] <- sd2_hat
    if (all(is.finite(se[idx_logsd2]))) disp_se[idx_logsd2] <- abs(sd2_hat * se[idx_logsd2])
    disp_names[idx_logsd2] <- sub("^log_sd2:", "sd2:", disp_names[idx_logsd2])
  }
  zval <- disp_est / disp_se
  pval <- 2 * pnorm(-abs(zval))
  stars <- symnum(pval, corr = FALSE, na = FALSE,
                  cutpoints = c(0, .001, .01, .05, .1, 1),
                  symbols   = c("***","**","*","."," "))
  coef_table_display <- data.frame(
    Parameter = disp_names,
    Estimate  = as.numeric(disp_est),
    StdErr    = as.numeric(disp_se),
    z         = as.numeric(zval),
    p         = as.numeric(pval),
    Signif    = as.character(stars),
    check.names = FALSE, row.names = NULL
  )
  
  # readable transforms
  log_m1_hat <- par_hat["log_m1"]; log_m2_hat <- par_hat["log_m2"]; z_hat <- par_hat["z_lambda"]
  m1_hat <- exp(log_m1_hat); m2_hat <- exp(log_m2_hat)
  sig <- plogis(z_hat); eps <- 1e-6
  lamLo_h <- as.numeric(lamLo_h); lamHi_h <- as.numeric(lamHi_h)
  lam_hat <- lamLo_h + (lamHi_h - lamLo_h) * (eps + (1 - 2*eps) * sig)
  se_m1 <- if (is.finite(se["log_m1"])) m1_hat * se["log_m1"] else NA_real_
  se_m2 <- if (is.finite(se["log_m2"])) m2_hat * se["log_m2"] else NA_real_
  dlam_dz <- (lamHi_h - lamLo_h) * (1 - 2*eps) * sig * (1 - sig)
  se_lam <- if (is.finite(se["z_lambda"])) abs(dlam_dz) * se["z_lambda"] else NA_real_
  
  if (print_output) {
    cat("\n--- Random-Parameters BNBR (BFGS) ---\n")
    cat("Eq1 random: ", if (q1>0) paste(colnames(X1)[rand_idx1], collapse=", ") else "(none)", "\n", sep="")
    cat("Eq2 random: ", if (q2>0) paste(colnames(X2)[rand_idx2], collapse=", ") else "(none)", "\n", sep="")
    cat(sprintf("Draws: optimize=%d, hessian=%d, Cores=%d\n",
                ifelse(q1+q2>0, nrow(Z1_opt), 1L),
                ifelse(q1+q2>0, n_draws_hess, 1L),
                ifelse(is.null(cl), 1L, length(cl))))
    cat("\n--- Coefficients (sd shown, not log_sd) ---\n")
    print(coef_table_display, row.names = FALSE, digits = 4, right = TRUE)
    cat("\n--- Transformed ---\n")
    cat(sprintf("m1 = %.4f (SE %s)\n", m1_hat, ifelse(is.na(se_m1), "NA", sprintf("%.4f", se_m1))))
    cat(sprintf("m2 = %.4f (SE %s)\n", m2_hat, ifelse(is.na(se_m2), "NA", sprintf("%.4f", se_m2))))
    cat(sprintf("lambda = %.6f (SE %s)  [bounds %.6f, %.6f]\n",
                lam_hat, ifelse(is.na(se_lam), "NA", sprintf("%.6f", se_lam)), lamLo_h, lamHi_h))
    cat("\nlogLik =", as.numeric(logLik(fit)),
        "  AIC =", AIC(fit), "  BIC =", BIC(fit), "\n")
    if (!compute_se) cat("(SEs skipped — set compute_se=TRUE to compute with n_draws_hess)\n")
  }
  
  invisible(list(
    maxLik      = fit,
    par_opt     = par_hat,
    se_opt      = if (compute_se) se else NULL,
    vcov_opt    = if (compute_se) vc else NULL,
    coef_table  = coef_table_display,
    readable    = list(m1 = m1_hat, se_m1 = se_m1,
                       m2 = m2_hat, se_m2 = se_m2,
                       lambda = lam_hat, se_lambda = se_lam,
                       bounds = c(lamLo_h, lamHi_h)),
    ll_trace    = ll_trace,
    X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
    rand_idx1 = rand_idx1, rand_idx2 = rand_idx2
  ))
}

# ---------- Plot helpers ----------
get_iteration_trace <- function(ll, tol = 1e-8) {
  ll <- ll[is.finite(ll)]; if (!length(ll)) return(data.frame(iter = integer(0), logLik = numeric(0)))
  best <- -Inf; iter <- integer(0); val <- numeric(0)
  for (i in seq_along(ll)) if (ll[i] > best + tol) { best <- ll[i]; iter <- c(iter, length(iter)+1L); val <- c(val, ll[i]) }
  data.frame(iter = iter, logLik = val)
}
plot_ll_evals <- function(fit_obj) {
  ll <- fit_obj$ll_trace; keep <- is.finite(ll)
  eval_id <- which(keep); ll <- ll[keep]
  plot(eval_id, ll, type = "l", lwd = 1,
       xlab = "Function evaluation", ylab = "Log-likelihood",
       main = "RP-BNBR — Convergence (function evaluations)")
  lines(eval_id, cummax(ll), lwd = 2)
  abline(h = as.numeric(logLik(fit_obj$maxLik)), col = "red", lty = 2, lwd = 2)
  legend("bottomright", c("Evaluations","Best-so-far","Final logLik"),
         lty = c(1,1,2), lwd = c(1,2,2), col = c("black","black","red"), bty="n")
}
plot_ll_iterations <- function(fit_obj, tol = 1e-8) {
  tr <- get_iteration_trace(fit_obj$ll_trace, tol)
  if (!nrow(tr)) return(plot_ll_evals(fit_obj))
  plot(tr$iter, tr$logLik, type = "o",
       xlab = "Iteration (accepted BFGS steps)", ylab = "Log-likelihood",
       main = "RP-BNBR — Log-likelihood by accepted iteration")
  abline(h = as.numeric(logLik(fit_obj$maxLik)), lty = 2, lwd = 2, col = "red")
  legend("bottomright", c("Accepted steps","Final logLik"),
         lty = c(1,2), pch = c(1,NA), lwd = c(1,2), col = c("black","red"), bty="n")
}

# =========================================================
# EXAMPLE USAGE
# =========================================================
# 1) Simulated quick demo (uncomment to test):
# set.seed(123)
# n  <- 1000
# df <- data.frame(x1 = rnorm(n), x2 = runif(n,-1,1),
#                  z1 = rnorm(n), z2 = rbinom(n,1,0.4))
# # simulate simple counts (placeholder; plug your own simulator if needed)
# Y1 <- rpois(n, exp(0.5 + 0.3*df$x1 - 0.2*df$x2))
# Y2 <- rpois(n, exp(1.0 - 0.1*df$z1 + 0.5*df$z2))
# df$Y1 <- Y1; df$Y2 <- Y2
# f1 <- Y1 ~ x1 + x2
# f2 <- Y2 ~ z1 + z2
# fit <- fit_bnbr_rp(df, f1, f2,
#                    rand_names1 = "x1", rand_names2 = "z2",
#                    n_draws = 400, n_draws_hess = 80,
#                    n_cores = max(1L, parallel::detectCores()-1L))
# plot_ll_evals(fit); plot_ll_iterations(fit)

# 2) CSV example (edit paths/formulas/names):
# csv_path <- "C:/Users/wonst/OneDrive/Project/MVNB/data/rwm1984_clean.csv"
# raw <- read.csv(csv_path)
# f1 <- docvis ~ MarM + SinM + SinF + kids + outwork + postHS
# f2 <- hospvis ~ MarM + SinM + SinF + kids + outwork + postHS
# fit_csv <- fit_bnbr_rp(
#   data = raw, f1 = f1, f2 = f2,
#   # choose which coefficients are random BY NAME (must match model.matrix columns):
#   rand_names1 = c("MarM","kids"),          # example
#   rand_names2 = c("outwork","postHS"),     # example
#   n_draws = 500,          # for BFGS
#   n_draws_hess = 80,      # for SEs (fast)
#   n_cores = max(1L, parallel::detectCores()-1L),
#   compute_se = TRUE,      # set FALSE to skip SEs entirely
#   print_output = TRUE
# )
# plot_ll_evals(fit_csv); plot_ll_iterations(fit_csv)
