# =========================================================
# Famoye BNBR with maxLik (BFGS) + analytic gradient
# Vectorized, NA-safe, named outputs, tidy summary + significance stars
# =========================================================

# ---- Shared helpers (guards kept) ----
d_const <- 1 - exp(-1)

.safe_prob <- function(mu, m) {
  pr <- 1 / (1 + m * mu)
  eps <- 1e-12
  pmin(pmax(pr, eps), 1 - eps)
}

c_vec <- function(mu, m) {
  base <- 1 + d_const * m * mu
  base <- pmax(base, 1e-300)               # avoid <=0 from underflow
  base^(-1 / m)
}

lambda_bounds_global <- function(mu1, mu2, m1, m2) {
  c1 <- c_vec(mu1, m1)
  c2 <- c_vec(mu2, m2)
  Ppos <- pmax( (1 - c1) * (1 - c2),  c1 * c2, 1e-15 )
  Pneg <- pmax( (1 - c1) * c2,        c1 * (1 - c2), 1e-15 )
  lower_i <- -1 / Ppos
  upper_i <-  1 / Pneg
  c(lower = max(lower_i), upper = min(upper_i))
}

map_lambda <- function(z_lambda, mu1, mu2, m1, m2) {
  b <- lambda_bounds_global(mu1, mu2, m1, m2)
  L <- b["lower"]; U <- b["upper"]
  if (!is.finite(L) || !is.finite(U) || L >= U) { L <- -0.99; U <- 0.99 }
  mid <- 0.5 * (L + U); rad <- 0.5 * (U - L)
  mid + rad * tanh(z_lambda)
}

# ---- Derivatives of c(mu,m) = (1 + d*m*mu)^(-1/m) ----
dct_dm <- function(mu, m, c) {
  m_inv <- 1/m
  term <- m_inv * ( m_inv * log(1 + d_const * m * mu) - (d_const * mu) / (1 + d_const * m * mu) )
  term * c
}
dct_dbeta <- function(mu, m, c, x_col) {
  -(d_const * c * mu * x_col) / (1 + d_const * m * mu)
}

# ---- Log-likelihood contributions (n-vector) ----
# par = c(beta1 (p1), beta2 (p2), log_m1, log_m2, z_lambda)
bnb_loglik_vec <- function(par, y1, y2, X1, X2) {
  n <- length(y1)
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  if (!is.finite(m1) || !is.finite(m2) || m1 <= 0 || m2 <= 0) return(rep(-1e6, n))
  
  eta1 <- as.vector(X1 %*% beta1)
  eta2 <- as.vector(X2 %*% beta2)
  if (any(!is.finite(eta1)) || any(!is.finite(eta2))) return(rep(-1e6, n))
  
  mu1 <- exp(eta1); mu2 <- exp(eta2)
  if (any(!is.finite(mu1)) || any(!is.finite(mu2))) return(rep(-1e6, n))
  
  size1 <- 1/m1; size2 <- 1/m2
  prob1 <- .safe_prob(mu1, m1)
  prob2 <- .safe_prob(mu2, m2)
  
  lnb1 <- dnbinom(y1, size = size1, prob = prob1, log = TRUE)
  lnb2 <- dnbinom(y2, size = size2, prob = prob2, log = TRUE)
  if (any(!is.finite(lnb1)) || any(!is.finite(lnb2))) return(rep(-1e6, n))
  
  c1 <- c_vec(mu1, m1)
  c2 <- c_vec(mu2, m2)
  lam <- map_lambda(zlam, mu1, mu2, m1, m2)
  if (!is.finite(lam)) return(rep(-1e6, n))
  
  k1  <- exp(-y1) - c1
  k2  <- exp(-y2) - c2
  dep <- 1 + lam * (k1 * k2)
  
  # Penalize infeasible dep, but avoid -Inf to keep numeric derivatives sane
  ll <- lnb1 + lnb2 + log(pmax(dep, 1e-300))
  bad <- (!is.finite(dep)) | (dep <= 0)
  ll[bad] <- -1e6
  ll
}

# ---- Per-observation analytic SCORE matrix (n x k) ----
bnb_score_mat <- function(par, y1, y2, X1, X2) {
  n <- length(y1); p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  eta1 <- as.vector(X1 %*% beta1)
  eta2 <- as.vector(X2 %*% beta2)
  mu1 <- exp(eta1); mu2 <- exp(eta2)
  
  c1 <- c_vec(mu1, m1)
  c2 <- c_vec(mu2, m2)
  lam <- map_lambda(zlam, mu1, mu2, m1, m2)
  
  k1 <- exp(-y1) - c1
  k2 <- exp(-y2) - c2
  dep <- 1 + lam * (k1 * k2)
  inv_dep <- 1 / pmax(dep, 1e-300)
  
  # d c / d m and d c / d beta
  dc1_dm1 <- dct_dm(mu1, m1, c1)
  dc2_dm2 <- dct_dm(mu2, m2, c2)
  
  dci_dbeta <- function(ct, mu, m, X) {
    out <- matrix(0.0, nrow = n, ncol = ncol(X))
    for (j in seq_len(ncol(X))) out[, j] <- dct_dbeta(mu, m, ct, X[, j])
    out
  }
  dc1_dbetas <- dci_dbeta(c1, mu1, m1, X1)  # n x p1
  dc2_dbetas <- dci_dbeta(c2, mu2, m2, X2)  # n x p2
  
  # Marginal NB pieces for betas:
  w1 <- (y1 - mu1) / (1 + m1 * mu1)
  w2 <- (y2 - mu2) / (1 + m2 * mu2)
  
  pen1 <- lam * k2 * inv_dep
  pen2 <- lam * k1 * inv_dep
  
  score_beta1_mat <- sweep(X1, 1, w1, `*`)
  for (j in seq_len(p1)) score_beta1_mat[, j] <- score_beta1_mat[, j] - pen1 * dc1_dbetas[, j]
  
  score_beta2_mat <- sweep(X2, 1, w2, `*`)
  for (j in seq_len(p2)) score_beta2_mat[, j] <- score_beta2_mat[, j] - pen2 * dc2_dbetas[, j]
  
  # helpers for m1,m2: sum_{j=0}^{y-1} 1/(r + j), r = 1/m
  sum_inv_rpj <- function(y, r) {
    sapply(y, function(yy) if (yy <= 0) 0 else sum(1 / (r + 0:(yy - 1))))
  }
  r1 <- 1/m1; r2 <- 1/m2
  S1 <- sum_inv_rpj(y1, r1)
  S2 <- sum_inv_rpj(y2, r2)
  
  term_m1_i <- r1^2 * log(m1) + r1^2 * (log(mu1 + r1) - 1) +
    r1^2 * (y1 + r1) / (mu1 + r1) - r1^2 * S1 -
    (lam * k2 * inv_dep) * dc1_dm1
  term_m2_i <- r2^2 * log(m2) + r2^2 * (log(mu2 + r2) - 1) +
    r2^2 * (y2 + r2) / (mu2 + r2) - r2^2 * S2 -
    (lam * k1 * inv_dep) * dc2_dm2
  
  score_logm1 <- m1 * term_m1_i      # d/d log m = m * d/d m
  score_logm2 <- m2 * term_m2_i
  
  # λ part per obs: dℓ_i/dλ = (k1*k2)/dep ; z->λ: dλ/dz = rad * sech^2(z)
  b <- lambda_bounds_global(mu1, mu2, m1, m2)
  rad <- as.numeric(0.5 * (b["upper"] - b["lower"]))
  dlam_dz <- rad * (1 - tanh(zlam)^2)
  score_z <- (k1 * k2) * inv_dep * dlam_dz
  
  cbind(
    score_beta1_mat,          # n x p1
    score_beta2_mat,          # n x p2
    score_logm1,              # n x 1
    score_logm2,              # n x 1
    score_z                   # n x 1
  )
}

# ---- Total gradient vector (k-vector) for BFGS ----
bnb_grad_vec <- function(par, y1, y2, X1, X2) {
  colSums(bnb_score_mat(par, y1, y2, X1, X2))
}

# ---- Numeric Hessian (of SUM logLik) for SEs ----
bnb_hess <- function(par, y1, y2, X1, X2) {
  if (!requireNamespace("numDeriv", quietly = TRUE))
    stop("Please install.packages('numDeriv') for numeric Hessian.")
  numDeriv::hessian(function(p) sum(bnb_loglik_vec(p, y1, y2, X1, X2)), par)
}

# ---- Lambda on feasible scale at a parameter vector ----
bnb_lambda_at <- function(par, X1, X2) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  mu1 <- exp(as.vector(X1 %*% beta1))
  mu2 <- exp(as.vector(X2 %*% beta2))
  map_lambda(zlam, mu1, mu2, m1, m2)
}

# =========================================================
# Demo / Example (replace with your data)
# =========================================================
set.seed(1)
n  <- 200
X1 <- cbind(1, rnorm(n))
X2 <- cbind(1, rnorm(n))

# Column names for pretty output
colnames(X1) <- c("(Intercept)", "x1")
colnames(X2) <- c("(Intercept)", "x2")

p1 <- ncol(X1); p2 <- ncol(X2)

# True params for simulation
beta1_true <- c(log(6),  0.20)
beta2_true <- c(log(10), -0.15)
m1_true <- 0.4;  m2_true <- 0.6
mu1_true <- exp(drop(X1 %*% beta1_true))
mu2_true <- exp(drop(X2 %*% beta2_true))

# Feasible lambda and sample (toy AR)
bnds <- lambda_bounds_global(mu1_true, mu2_true, m1_true, m2_true)
lambda_true <- 0.5 * (bnds["lower"] + bnds["upper"])

rBNBR <- function(n, mu1, mu2, m1, m2, lambda) {
  y1 <- y2 <- integer(n)
  for (i in seq_len(n)) {
    repeat {
      y1i <- rnbinom(1, size = 1/m1, prob = .safe_prob(mu1[i], m1))
      y2i <- rnbinom(1, size = 1/m2, prob = .safe_prob(mu2[i], m2))
      c1i <- (1 + d_const * m1 * mu1[i])^(-1/m1)
      c2i <- (1 + d_const * m2 * mu2[i])^(-1/m2)
      acc <- 1 + lambda * (exp(-y1i) - c1i) * (exp(-y2i) - c2i)
      if (!is.finite(acc) || acc <= 0) next
      if (runif(1) <= min(1, acc)) { y1[i] <- y1i; y2[i] <- y2i; break }
    }
  }
  list(y1 = y1, y2 = y2)
}
samp <- rBNBR(n, mu1_true, mu2_true, m1_true, m2_true, lambda_true)
y1 <- samp$y1; y2 <- samp$y2

# =========================================================
# maxLik fit (BFGS) with NAMED PARAMETERS
# =========================================================
param_names <- c(
  paste0("eq1_", colnames(X1)),  # p1 names
  paste0("eq2_", colnames(X2)),  # p2 names
  "log_m1", "log_m2", "z_lambda"
)
par0 <- c(rep(0, p1 + p2), log(0.5), log(0.5), 0)
names(par0) <- param_names

library(maxLik)

fit <- maxLik(
  logLik  = function(p) bnb_loglik_vec(p, y1, y2, X1, X2),   # n-vector
  grad    = function(p) bnb_grad_vec(p, y1, y2, X1, X2),     # k-vector
  hess    = function(p) bnb_hess(p, y1, y2, X1, X2),         # numeric Hessian for SEs
  start   = par0,
  method  = "BFGS",
  control = list(printLevel = 2, tol = 1e-8)
)

# =========================================================
# Robust, named tidy summary + significance stars
# =========================================================
cat("\nConverged:", fit$code == 0, "  logLik:", as.numeric(logLik(fit)), "\n\n")

summ <- summary(fit)
print(summ)  # may omit z/p cols depending on Hessian quality

# Build robust tidy table
est_vec <- coef(fit)
V <- tryCatch(vcov(fit), error = function(e) NULL)
se_vec <- if (!is.null(V)) sqrt(pmax(diag(V), 0)) else rep(NA_real_, length(est_vec))

est_mat <- tryCatch(as.matrix(summ$estimate), error = function(e) NULL)
est_from_summ <- if (!is.null(est_mat)) est_mat[, 1] else est_vec

# Try to find a "std" column in summary; else fallback to vcov-derived SEs
se_from_summ <- NULL
if (!is.null(est_mat)) {
  cnl <- tolower(colnames(est_mat))
  j <- which(grepl("std", cnl))
  if (length(j) == 1) se_from_summ <- est_mat[, j]
}
se_use <- if (!is.null(se_from_summ)) se_from_summ else se_vec
z_use  <- ifelse(is.finite(se_use) & se_use > 0, est_from_summ / se_use, NA_real_)
p_use  <- ifelse(is.finite(z_use), 2 * pnorm(-abs(z_use)), NA_real_)

# Significance stars
sig_stars <- function(p) {
  sapply(p, function(pi) {
    if (!is.finite(pi)) return("")
    if (pi < 0.001) return("***")
    if (pi < 0.01)  return("**")
    if (pi < 0.05)  return("*")
    if (pi < 0.10)  return(".")
    ""
  })
}
stars <- sig_stars(p_use)

coef_tab <- data.frame(
  param     = names(est_vec),
  estimate  = as.numeric(est_from_summ),
  std_error = as.numeric(se_use),
  z_value   = as.numeric(z_use),
  p_value   = as.numeric(p_use),
  signif    = stars,
  row.names = NULL,
  check.names = FALSE
)

# Optional: pretty-printed estimate with stars appended
coef_tab$estimate_fmt <- sprintf("%.6f %s", coef_tab$estimate, coef_tab$signif)
coef_tab_out <- coef_tab[, c("param","estimate_fmt","std_error","z_value","p_value","signif")]
names(coef_tab_out)[2] <- "estimate"

print(coef_tab_out, row.names = FALSE)
cat("\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n")

# Derived parameters with labels
est <- fit$estimate
names(est) <- names(est_vec)
m1_hat <- exp(est["log_m1"])
m2_hat <- exp(est["log_m2"])
lambda_hat <- bnb_lambda_at(est, X1, X2)

cat("\nDerived parameters:\n")
cat(sprintf("  m1 = %.6f\n  m2 = %.6f\n  lambda (feasible scale) = %.6f\n",
            m1_hat, m2_hat, lambda_hat))
