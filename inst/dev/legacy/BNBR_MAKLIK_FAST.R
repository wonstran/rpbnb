# =========================================================
# BNBR (Famoye) via maxLik (BFGS)
# * Analytic gradient (Appendix-style) + numeric Hessian (numDeriv)
# * Safe Hessian (freeze lambda bounds, finite guard, ridge if needed)
# * Iteration plot: log-likelihood vs accepted steps
# =========================================================
rm(list = ls())

# ---- Packages ----
need <- c("maxLik", "numDeriv")
to_install <- need[!need %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
library(maxLik)
library(numDeriv)

# (optional) for fallback pseudo-inverse when information is near-singular
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")

# ---------- Core math ----------
d_const <- 1 - exp(-1)
c_val   <- function(mu, m) (1 + d_const * m * mu)^(-1/m)

.safe_prob <- function(mu, m) { # NB2 prob parameter, clamped for safety
  pr <- 1 / (1 + m * mu)
  eps <- 1e-12
  pmin(pmax(pr, eps), 1 - eps)
}

nb_logpmf_y_mu_r <- function(y, mu, r) {
  p <- r / (r + mu)
  lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r*log(p) + y*log1p(-p)
}

# Sufficient (global) λ-bounds given vectors c1, c2 (conservative global box)
lambda_bounds_vec <- function(c1, c2) {
  lam_min <- -1 / ((1 - c1) * (1 - c2))                 # elementwise
  lam_max <-  1 / pmax(c1 * (1 - c2), c2 * (1 - c1))
  c(max(lam_min), min(lam_max))
}

# ---------- Simulator (for demo/testing) ----------
rnb_mu_r <- function(n, mu, r) rpois(n, rgamma(n, shape = r, scale = mu/r))

rY2_given_Y1 <- function(y1, mu2, m2, lambda, c1) {
  r2 <- 1/m2; c2 <- c_val(mu2, m2); a <- lambda * (exp(-y1) - c1)
  M  <- if (a >= 0) 1 + a*(1 - c2) else 1 - a*c2
  repeat {
    y2 <- rnb_mu_r(1, mu2, r2)
    w  <- 1 + a * (exp(-y2) - c2); if (w < 0 && w > -1e-15) w <- 0
    if (runif(1) <= max(0, w)/M) return(y2)
  }
}

rBNBR <- function(n, X1, X2, beta1, beta2, m1, m2, lambda, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  mu1 <- as.vector(exp(X1 %*% beta1)); mu2 <- as.vector(exp(X2 %*% beta2))
  r1  <- 1/m1; c1 <- c_val(mu1, m1)
  y1  <- rnb_mu_r(n, mu1, r1)
  y2  <- integer(n)
  for (i in seq_len(n)) y2[i] <- rY2_given_Y1(y1[i], mu2[i], m2, lambda, c1[i])
  data.frame(Y1 = y1, Y2 = y2)
}

# ---------- Derivatives for c(mu,m) ----------
# c(mu,m) = (1 + d*m*mu)^(-1/m)
dct_dm <- function(mu, m, c) {
  m_inv <- 1/m
  denom <- 1 + d_const * m * mu
  term  <- m_inv * ( m_inv * log(denom) - (d_const * mu) / denom )
  term * c
}

# ∂c/∂β_j = - d * c * μ * x_j / (1 + d m μ)  ==> row factor * X column
dc_dbeta_mat <- function(mu, m, c, X) {
  denom <- 1 + d_const * m * mu
  row_factor <- -(d_const * c * mu) / denom   # length n
  sweep(X, 1, row_factor, `*`)                # n x p
}

# ---------- Log-likelihood (per observation) + analytic score ----------
# Parameter vector: [beta1 (k1), beta2 (k2), log_m1, log_m2, z_lambda]
bnb_loglik_vec <- function(par, y1, y2, X1, X2) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  r1  <- 1/m1; r2 <- 1/m2
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)
  
  # Global lambda bounds + logistic map (kept strictly interior)
  bnds  <- lambda_bounds_vec(c1, c2); lamLo <- bnds[1]; lamHi <- bnds[2]
  eps <- 1e-6
  sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  
  # Marginal NB parts
  lnb1 <- nb_logpmf_y_mu_r(y1, mu1, r1)
  lnb2 <- nb_logpmf_y_mu_r(y2, mu2, r2)
  
  # Dependence factor (finite guard)
  dep <- 1 + lam * (exp(-y1) - c1) * (exp(-y2) - c2)
  dep <- pmax(dep, 1e-300)
  
  lnb1 + lnb2 + log(dep)
}

# Per-observation analytic SCORE matrix (n x k)
bnb_score_mat <- function(par, y1, y2, X1, X2) {
  n  <- length(y1); p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)
  
  # lambda map (same as in ll)
  bnds  <- lambda_bounds_vec(c1, c2); lamLo <- bnds[1]; lamHi <- bnds[2]
  eps <- 1e-6
  sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  
  # pieces
  k1 <- exp(-y1) - c1
  k2 <- exp(-y2) - c2
  dep <- 1 + lam * (k1 * k2)
  inv_dep <- 1 / pmax(dep, 1e-300)
  
  # d c / d m and d c / d beta (matrix form; no loops)
  dc1_dm1 <- dct_dm(mu1, m1, c1)
  dc2_dm2 <- dct_dm(mu2, m2, c2)
  dc1_dbetas <- dc_dbeta_mat(mu1, m1, c1, X1)  # n x p1
  dc2_dbetas <- dc_dbeta_mat(mu2, m2, c2, X2)  # n x p2
  
  # Marginal NB score for betas: dℓ/dη = (y - μ)/(1 + m μ), and dη/dβ = X
  w1 <- (y1 - mu1) / (1 + m1 * mu1)   # length n
  w2 <- (y2 - mu2) / (1 + m2 * mu2)
  
  # dependence penalty multipliers
  pen1 <- lam * k2 * inv_dep     # length n
  pen2 <- lam * k1 * inv_dep
  
  # ∂ℓ/∂β blocks
  score_beta1_mat <- sweep(X1, 1, w1, `*`) - sweep(dc1_dbetas, 1, pen1, `*`)
  score_beta2_mat <- sweep(X2, 1, w2, `*`) - sweep(dc2_dbetas, 1, pen2, `*`)
  
  # NB2 dispersion part via r = 1/m and digamma identity
  r1 <- 1/m1; r2 <- 1/m2
  S1 <- digamma(r1 + y1) - digamma(r1)
  S2 <- digamma(r2 + y2) - digamma(r2)
  
  # dℓ/dm (see paper's appendix; implemented in closed form) + dependence c-part
  term_m1_i <- r1^2 * log(m1) + r1^2 * (log(mu1 + r1) - 1) +
    r1^2 * (y1 + r1) / (mu1 + r1) - r1^2 * S1 -
    (lam * k2 * inv_dep) * dc1_dm1
  term_m2_i <- r2^2 * log(m2) + r2^2 * (log(mu2 + r2) - 1) +
    r2^2 * (y2 + r2) / (mu2 + r2) - r2^2 * S2 -
    (lam * k1 * inv_dep) * dc2_dm2
  
  # d/d log m = m * d/d m
  score_logm1 <- m1 * term_m1_i
  score_logm2 <- m2 * term_m2_i
  
  # λ part: dℓ_i/dλ = (k1*k2)/dep ; chain z->λ
  dlam_dz <- (lamHi - lamLo) * (1 - 2*eps) * sig * (1 - sig)
  score_z <- (k1 * k2) * inv_dep * dlam_dz
  
  cbind(score_beta1_mat, score_beta2_mat, score_logm1, score_logm2, score_z)
}

bnb_grad_vec <- function(par, y1, y2, X1, X2) {
  colSums(bnb_score_mat(par, y1, y2, X1, X2))
}

# ---------- LL with fixed λ-bounds (for SAFE numeric Hessian) ----------
bnbr_loglik_fixed_bounds <- function(par, Y1, Y2, X1, X2, lamLo, lamHi, tiny = 1e-10) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  r1  <- 1/m1; r2 <- 1/m2
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)
  
  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  
  logNB1 <- nb_logpmf_y_mu_r(Y1, mu1, r1)
  logNB2 <- nb_logpmf_y_mu_r(Y2, mu2, r2)
  
  fac <- 1 + lam * (exp(-Y1) - c1) * (exp(-Y2) - c2)
  fac <- pmax(fac, tiny)
  
  sum(logNB1 + logNB2 + log(fac))
}

# ---------- Iteration trace & plot ----------
get_iteration_trace <- function(ll, tol = 1e-8) {
  ll <- ll[is.finite(ll)]
  if (!length(ll)) return(data.frame(iter = integer(0), logLik = numeric(0)))
  best <- -Inf
  iter <- integer(0); val <- numeric(0)
  for (i in seq_along(ll)) {
    if (ll[i] > best + tol) {
      best <- ll[i]
      iter <- c(iter, length(iter) + 1L)
      val  <- c(val,  ll[i])
    }
  }
  data.frame(iter = iter, logLik = val)
}

plot_ll_by_iteration <- function(fit_list, tol = 1e-8) {
  tr <- get_iteration_trace(fit_list$ll_trace, tol)
  if (!nrow(tr)) {
    message("No improvements recorded; cannot build iteration trace.")
    return(invisible(NULL))
  }
  plot(tr$iter, tr$logLik, type = "o",
       xlab = "Iteration (accepted steps)", ylab = "Log-likelihood",
       main = "BNBR MLE (BFGS) — Iteration vs Log-likelihood")
  abline(h = as.numeric(logLik(fit_list$maxLik)), lty = 2, lwd = 2)
  legend("bottomright", c("Accepted steps", "Final logLik"),
         lty = c(1, 2), pch = c(1, NA), lwd = c(1, 2), bty = "n")
}

# ---------- Estimation wrapper (data.frame + formulas) ----------
fit_bnbr_maxlik_df <- function(data, f1, f2,
                               start = NULL, method = "BFGS",
                               control = list(iterlim = 200, reltol = 1e-8, printLevel = 2)) {
  stopifnot(is.data.frame(data))
  mf1 <- model.frame(f1, data = data)
  mf2 <- model.frame(f2, data = data)
  Y1  <- as.integer(model.response(mf1))
  Y2  <- as.integer(model.response(mf2))
  X1  <- model.matrix(f1, mf1)   # includes intercept by default
  X2  <- model.matrix(f2, mf2)
  
  k1 <- NCOL(X1); k2 <- NCOL(X2)
  cn1 <- colnames(X1); cn2 <- colnames(X2)
  
  if (is.null(start)) start <- c(rep(0, k1 + k2), log(0.5), log(0.5), 0)
  names(start) <- c(paste0("b1:", cn1), paste0("b2:", cn2), "log_m1", "log_m2", "z_lambda")
  
  # capture LL values at each evaluation
  ll_trace <- numeric(0)
  ll_fun <- function(p) {
    v <- bnb_loglik_vec(p, Y1, Y2, X1, X2)
    s <- sum(v)
    ll_trace <<- c(ll_trace, s)
    v
  }
  
  # BFGS with analytic gradient (vector)
  fit <- maxLik(
    logLik = ll_fun,
    grad   = function(p) bnb_grad_vec(p, Y1, Y2, X1, X2),
    start  = start,
    method = method,
    control = control
  )
  
  par_hat <- coef(fit)
  
  # Transform back to interpretable values:
  beta1_hat <- par_hat[ paste0("b1:", cn1) ]
  beta2_hat <- par_hat[ paste0("b2:", cn2) ]
  m1_hat    <- exp(par_hat["log_m1"])
  m2_hat    <- exp(par_hat["log_m2"])
  z_hat     <- par_hat["z_lambda"]
  
  mu1_hat <- as.vector(exp(X1 %*% beta1_hat)); mu2_hat <- as.vector(exp(X2 %*% beta2_hat))
  c1_hat  <- c_val(mu1_hat, m1_hat);           c2_hat  <- c_val(mu2_hat, m2_hat)
  bnds_hat <- lambda_bounds_vec(c1_hat, c2_hat); eps <- 1e-6
  lambda_hat <- bnds_hat[1] + (bnds_hat[2] - bnds_hat[1]) * (eps + (1 - 2*eps) * plogis(z_hat))
  
  # Pretty prints
  cat("\nPredictors (Y1): ", paste(cn1, collapse = ", "), "\n", sep = "")
  cat("Predictors (Y2): ", paste(cn2, collapse = ", "), "\n", sep = "")
  
  # ---- Safe numeric Hessian / SEs: freeze lambda bounds at optimum, clamp fac ----
  lamLo_h <- bnds_hat[1]; lamHi_h <- bnds_hat[2]
  ll_fb <- function(p) bnbr_loglik_fixed_bounds(p, Y1, Y2, X1, X2, lamLo_h, lamHi_h)
  
  H <- numDeriv::hessian(ll_fb, par_hat, method.args = list(r = 4, eps = 1e-5))
  info <- -H; info <- (info + t(info)) / 2
  
  # If non-finite, retry with gentler steps
  if (any(!is.finite(info))) {
    warning("Non-finite information matrix; retrying Hessian with larger step.")
    H <- numDeriv::hessian(ll_fb, par_hat, method.args = list(r = 6, eps = 1e-4))
    info <- -H; info <- (info + t(info)) / 2
  }
  
  # Ridge if not PD
  ok_eig <- try(eigen(info, symmetric = TRUE, only.values = TRUE), silent = TRUE)
  if (inherits(ok_eig, "try-error") || any(!is.finite(ok_eig$values)) || min(ok_eig$values) <= 0) {
    ridge <- if (inherits(ok_eig, "try-error") || any(!is.finite(ok_eig$values))) 1e-2 else (1e-8 - min(ok_eig$values))
    info <- info + diag(ridge, nrow(info))
  }
  
  vc  <- try(solve(info), silent = TRUE)
  if (inherits(vc, "try-error")) vc <- MASS::ginv(info)
  se  <- sqrt(pmax(diag(vc), 0))
  
  zval <- par_hat / se
  pval <- 2 * pnorm(-abs(zval))
  coef_table <- data.frame(
    Parameter = names(par_hat),
    Estimate  = as.numeric(par_hat),
    StdErr    = as.numeric(se),
    z         = as.numeric(zval),
    p         = as.numeric(pval),
    row.names = NULL
  )
  
  cat("\n--- Coefficients (optimization scale; numeric Hessian SEs) ---\n")
  print(coef_table, row.names = FALSE, digits = 4)
  
  names(beta1_hat) <- cn1; names(beta2_hat) <- cn2
  cat("\n--- Transformed parameters (readable scale) ---\n")
  print(list(beta1 = beta1_hat, beta2 = beta2_hat, m1 = m1_hat, m2 = m2_hat, lambda = lambda_hat))
  
  invisible(list(
    maxLik      = fit,
    coef        = list(beta1 = beta1_hat, beta2 = beta2_hat, m1 = m1_hat, m2 = m2_hat, lambda = lambda_hat),
    bounds      = list(lambda_lo = bnds_hat[1], lambda_hi = bnds_hat[2]),
    coef_table  = coef_table,
    X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
    ll_trace    = ll_trace
  ))
}

# =========================================================
# DEMO using a data.frame + formulas (RHS-only for simulation)
# =========================================================
set.seed(123)
n  <- 5000
df <- data.frame(
  x1_norm = rnorm(n),
  x1_uni  = runif(n, -1, 1),
  x2_norm = rnorm(n),
  x2_bin  = rbinom(n, 1, 0.4)
)

# Build design matrices for simulation (RHS-only, automatic intercept)
f1_rhs <- ~ x1_norm + x1_uni
f2_rhs <- ~ x2_norm + x2_bin
X1_sim <- model.matrix(f1_rhs, df)
X2_sim <- model.matrix(f2_rhs, df)

# True parameters (match columns of X*_sim)
beta1_true <- c(log(2.0), 0.30, -0.20)  # (Intercept), x1_norm, x1_uni
beta2_true <- c(log(4.0), -0.15, 0.50)  # (Intercept), x2_norm, x2_bin
m1_true    <- 0.60
m2_true    <- 0.40

mu1_true <- as.vector(exp(X1_sim %*% beta1_true))
mu2_true <- as.vector(exp(X2_sim %*% beta2_true))
c1_true  <- c_val(mu1_true, m1_true)
c2_true  <- c_val(mu2_true, m2_true)
b_sim    <- lambda_bounds_vec(c1_true, c2_true)
lambda_true <- 0.5 * b_sim[2]

cat(sprintf("Simulation λ bounds: (%.3f, %.3f); using λ=%.3f\n",
            b_sim[1], b_sim[2], lambda_true))

sim <- rBNBR(n, X1_sim, X2_sim, beta1_true, beta2_true, m1_true, m2_true, lambda_true, seed = 99)
df$Y1 <- as.integer(sim$Y1)
df$Y2 <- as.integer(sim$Y2)

# Fit with formulas (responses now present in df)
f1_fit <- Y1 ~ x1_norm + x1_uni
f2_fit <- Y2 ~ x2_norm + x2_bin

fit <- fit_bnbr_maxlik_df(
  data = df,
  f1   = f1_fit,   # NB for Y1
  f2   = f2_fit,   # NB for Y2
  start = NULL, method = "BFGS"
)

cat("\n--- Final logLik ---\n")
print(logLik(fit$maxLik))

cat("\n--- True parameters ---\n")
print(list(beta1 = beta1_true, beta2 = beta2_true,
           m1 = m1_true, m2 = m2_true, lambda = lambda_true))

# ----------------------------
# Plot: iteration vs log-likelihood
# ----------------------------
plot_ll_by_iteration(fit)

                                     