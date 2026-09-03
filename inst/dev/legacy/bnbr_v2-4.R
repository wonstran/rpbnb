# =========================================================
# BNBR (Famoye/Sarmanov) — maxLik + BFGS + BHHH
#  * Analytic gradient (Appendix A1–A5 + chain rules)
#  * Numeric Hessian (frozen λ-bounds) for SEs
#  * Log-likelihood trace over evaluations + accepted steps
#  * Pretty printing (no scientific notation)
# =========================================================

options(scipen = 999, digits = 10)

# ---- Packages ----
need <- c("maxLik", "numDeriv")
to_install <- need[!need %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
suppressPackageStartupMessages({
  library(maxLik)
  library(numDeriv)
})
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")

# ---- Pretty printing helpers ----
STAR_SYMS <- c("***","**","*","."," ")
fmt_fix <- function(x, digits = 4) if (is.numeric(x)) formatC(x, format = "f", digits = digits) else x
format_df_fixed <- function(df, digits = 4) {
  num <- vapply(df, is.numeric, TRUE)
  df[num] <- lapply(df[num], function(col) formatC(col, format = "f", digits = digits))
  df
}
print_fixed <- function(df, digits = 4, row.names = FALSE, ...) {
  print(format_df_fixed(df, digits), row.names = row.names, right = TRUE, ...)
}

# ---- Log-likelihood traces & plotting ----
get_iteration_trace <- function(ll_vals, tol = 1e-8) {
  ll_vals <- ll_vals[is.finite(ll_vals)]
  if (!length(ll_vals)) return(data.frame(iter = integer(0), logLik = numeric(0)))
  best <- -Inf; iter <- integer(0); val <- numeric(0)
  for (i in seq_along(ll_vals)) {
    if (ll_vals[i] > best + tol) { best <- ll_vals[i]; iter <- c(iter, length(iter)+1L); val <- c(val, ll_vals[i]) }
  }
  data.frame(iter = iter, logLik = val)
}

plot_ll_all <- function(fit, col = "gray40", pch = 1, lwd = 2, ...) {
  df <- fit$ll_eval_df
  if (is.null(df) || !nrow(df)) { message("No evaluations recorded."); return(invisible(NULL)) }
  plot(df$eval, df$logLik, type = "o",
       xlab = "Function evaluations (BFGS calls)",
       ylab = "Log-likelihood",
       main = "BNBR MLE — Log-likelihood (all evaluations)",
       col = col, pch = pch, lwd = lwd, ...)
}

plot_ll_accepted <- function(fit, col = "gray40", pch = 1, lwd = 2,
                             ref_col = "red", ...) {
  df <- fit$ll_accepted_df
  if (is.null(df) || !nrow(df)) { message("No accepted-improvement points."); return(invisible(NULL)) }
  plot(df$iter, df$logLik, type = "o",
       xlab = "Accepted steps",
       ylab = "Log-likelihood",
       main = "BNBR MLE — Log-likelihood (accepted improvements only)",
       col = col, pch = pch, lwd = lwd, ...)
  abline(h = as.numeric(logLik(fit$maxLik)), lty = 2, lwd = 2, col = ref_col)
  legend("bottomright",
         c("Accepted steps", "Final logLik"),
         lty = c(1, 2), pch = c(pch, NA), lwd = c(lwd, 2),
         col = c(col, ref_col), bty = "n")
}

# =========================================================
# Goodness-of-fit: LL, AIC, BIC, Pseudo R^2
# =========================================================
bnbr_gof <- function(fit,
                     data = NULL,
                     f1_null = NULL,
                     f2_null = NULL,
                     digits = 4,
                     quiet_null = TRUE) {
  
  stopifnot(is.list(fit), !is.null(fit$maxLik))
  
  # Full model metrics
  ll_full <- as.numeric(logLik(fit$maxLik))
  k       <- length(coef(fit$maxLik))
  n       <- NROW(fit$X1)
  AIC_val <- -2*ll_full +  2*k
  BIC_val <- -2*ll_full + log(n)*k
  
  # ---------- Null model (intercept-only) ----------
  if (is.null(f1_null) && is.null(f2_null)) {
    # Build a minimal data frame with only responses
    df_null <- data.frame(Y1 = fit$Y1, Y2 = fit$Y2)
    f1n <- Y1 ~ 1
    f2n <- Y2 ~ 1
  } else {
    if (is.null(data))
      stop("Please provide `data` when supplying custom f1_null/f2_null formulas.")
    df_null <- data
    f1n <- if (is.null(f1_null)) stop("Missing f1_null.") else f1_null
    f2n <- if (is.null(f2_null)) stop("Missing f2_null.") else f2_null
  }
  
  ctrl_null <- list(iterlim = 200, reltol = 1e-8, printLevel = if (quiet_null) 0 else 2)
  fit_null  <- bnbr_famoye_bfgs(df_null, f1n, f2n,
                                control = ctrl_null,
                                print_output = !quiet_null)
  ll_null <- as.numeric(logLik(fit_null$maxLik))
  
  # ---------- Pseudo R^2 ----------
  R2_MF   <- 1 - (ll_full/ll_null)
  R2_MFa  <- 1 - ((ll_full - k)/ll_null)
  R2_CS   <- 1 - exp((2/n) * (ll_null - ll_full))
  denom   <- 1 - exp((2/n) * ll_null)
  R2_NK   <- if (abs(denom) < .Machine$double.eps) NA_real_ else R2_CS/denom
  
  # Clamp to [0,1] when numeric noise pushes slightly out
  clamp01 <- function(x) ifelse(is.na(x), NA_real_, pmin(pmax(x, 0), 1))
  R2_MF  <- clamp01(R2_MF)
  R2_MFa <- clamp01(R2_MFa)
  R2_CS  <- clamp01(R2_CS)
  R2_NK  <- clamp01(R2_NK)
  
  # ---------- Print nicely ----------
  cat("\n--- Goodness of Fit ---\n")
  cat(sprintf("n = %d, k = %d\n", n, k))
  cat(sprintf("logLik(full) = %s\n", formatC(ll_full, format = "f", digits = digits)))
  cat(sprintf("AIC = %s    BIC = %s\n",
              formatC(AIC_val, format = "f", digits = digits),
              formatC(BIC_val, format = "f", digits = digits)))
  cat("\nPseudo R-squared:\n")
  gof_tab <- data.frame(
    Metric = c("McFadden", "McFadden_adj", "CoxSnell", "Nagelkerke"),
    Value  = c(R2_MF, R2_MFa, R2_CS, R2_NK),
    check.names = FALSE
  )
  # fixed decimals
  gof_tab$Value <- formatC(gof_tab$Value, format = "f", digits = digits)
  print(gof_tab, row.names = FALSE, right = TRUE)
  
  invisible(list(
    n = n, k = k,
    logLik_full = ll_full,
    logLik_null = ll_null,
    AIC = AIC_val, BIC = BIC_val,
    pseudoR2 = c(McFadden = as.numeric(gof_tab$Value[1]),
                 McFadden_adj = as.numeric(gof_tab$Value[2]),
                 CoxSnell = as.numeric(gof_tab$Value[3]),
                 Nagelkerke = as.numeric(gof_tab$Value[4])),
    null_fit = fit_null
  ))
}


# ---------- Core constants & helpers ----------
d_const <- 1 - exp(-1)                                   # Famoye's d
c_val   <- function(mu, m) (1 + d_const * m * mu)^(-1/m) # E[e^{-Y}] under NB2(mu,m)

# NB log pmf with (mu, r=size=1/m)
nb_logpmf_y_mu_r <- function(y, mu, r) {
  p <- r / (r + mu)
  lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r*log(p) + y*log1p(-p)
}

# Conservative global λ-bounds from c1, c2 (vectorized over obs)
lambda_bounds_vec <- function(c1, c2) {
  lam_min <- -1 / ((1 - c1) * (1 - c2))
  lam_max <-  1 / pmax(c1 * (1 - c2), c2 * (1 - c1))
  c(max(lam_min), min(lam_max))
}

# ---------- Derivatives of c(mu,m) ----------
# ∂c/∂m
dct_dm <- function(mu, m, c) {
  m_inv <- 1/m
  denom <- 1 + d_const * m * mu
  term  <- m_inv * ( m_inv * log(denom) - (d_const * mu) / denom )
  term * c
}
# ∂c/∂β_j = − d * c * μ * x_j / (1 + d m μ)  (returns n×p)
dc_dbeta_mat <- function(mu, m, c, X) {
  denom <- 1 + d_const * m * mu
  row_factor <- -(d_const * c * mu) / denom
  sweep(X, 1, row_factor, `*`)
}

# ---------- Per-observation log-likelihood (vector) ----------
# Parameter vector: [beta1 (p1), beta2 (p2), log_m1, log_m2, z_lambda]
bnb_loglik_vec <- function(par, y1, y2, X1, X2) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1/m1;        r2 <- 1/m2
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)
  
  # Data-adaptive bounds and interior logistic map (strictly inside)
  bnds <- lambda_bounds_vec(c1, c2); lamLo <- bnds[1]; lamHi <- bnds[2]
  eps  <- 1e-6
  sig  <- plogis(zlam)
  lam  <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  
  lnb1 <- nb_logpmf_y_mu_r(y1, mu1, r1)
  lnb2 <- nb_logpmf_y_mu_r(y2, mu2, r2)
  dep  <- 1 + lam * (exp(-y1) - c1) * (exp(-y2) - c2)
  dep  <- pmax(dep, 1e-300)
  
  lnb1 + lnb2 + log(dep)
}

# ---------- Analytic score (per observation) ----------
# Matches paper Appendix A1–A5 + chain rules (∂/∂log m = m∂/∂m; z→λ via logistic)
bnb_score_mat <- function(par, y1, y2, X1, X2) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1/m1;        r2 <- 1/m2
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)
  
  bnds <- lambda_bounds_vec(c1, c2); lamLo <- bnds[1]; lamHi <- bnds[2]
  eps  <- 1e-6; sig <- plogis(zlam)
  lam  <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  dlam_dz <- (lamHi - lamLo) * (1 - 2*eps) * sig * (1 - sig)
  
  k1 <- exp(-y1) - c1
  k2 <- exp(-y2) - c2
  dep <- pmax(1 + lam * (k1 * k2), 1e-300)
  inv_dep <- 1/dep
  
  # β blocks
  w1 <- (y1 - mu1) / (1 + m1 * mu1)
  w2 <- (y2 - mu2) / (1 + m2 * mu2)
  dc1_dbetas <- dc_dbeta_mat(mu1, m1, c1, X1)     # n×p1
  dc2_dbetas <- dc_dbeta_mat(mu2, m2, c2, X2)     # n×p2
  pen1 <- lam * k2 * inv_dep
  pen2 <- lam * k1 * inv_dep
  score_beta1_mat <- sweep(X1, 1, w1, `*`) - sweep(dc1_dbetas, 1, pen1, `*`)
  score_beta2_mat <- sweep(X2, 1, w2, `*`) - sweep(dc2_dbetas, 1, pen2, `*`)
  
  # m blocks (A2–A3) + chain to log m
  S1 <- digamma(r1 + y1) - digamma(r1)
  S2 <- digamma(r2 + y2) - digamma(r2)
  dc1_dm1 <- dct_dm(mu1, m1, c1)
  dc2_dm2 <- dct_dm(mu2, m2, c2)
  
  s_m1 <-  (m1^(-2)) * ( -S1 + log(m1) + log(mu1 + r1) - 1 + (y1 + r1)/(mu1 + r1) ) -
    (lam * k2 * inv_dep) * dc1_dm1
  s_m2 <-  (m2^(-2)) * ( -S2 + log(m2) + log(mu2 + r2) - 1 + (y2 + r2)/(mu2 + r2) ) -
    (lam * k1 * inv_dep) * dc2_dm2
  
  score_logm1 <- m1 * s_m1
  score_logm2 <- m2 * s_m2
  
  # λ part (A1) + chain rule z -> λ
  dL_dlambda <- (k1 * k2) * inv_dep
  score_z <- dL_dlambda * dlam_dz
  
  cbind(score_beta1_mat, score_beta2_mat, score_logm1, score_logm2, score_z)
}

# Summed gradient for BFGS
bnb_grad_vec <- function(par, y1, y2, X1, X2) colSums(bnb_score_mat(par, y1, y2, X1, X2))

# ---------- Fixed-bounds logLik for numeric Hessian (freeze at optimum) ----------
bnbr_loglik_fixed_bounds <- function(par, Y1, Y2, X1, X2, lamLo, lamHi, tiny = 1e-10) {
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1  <- par[seq_len(p1)]
  beta2  <- par[p1 + seq_len(p2)]
  log_m1 <- par[p1 + p2 + 1]
  log_m2 <- par[p1 + p2 + 2]
  zlam   <- par[p1 + p2 + 3]
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1/m1;        r2 <- 1/m2
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  c1  <- c_val(mu1, m1); c2 <- c_val(mu2, m2)
  
  eps <- 1e-6; sig <- plogis(zlam)
  lam <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * sig)
  
  logNB1 <- nb_logpmf_y_mu_r(Y1, mu1, r1)
  logNB2 <- nb_logpmf_y_mu_r(Y2, mu2, r2)
  fac <- 1 + lam * (exp(-Y1) - c1) * (exp(-Y2) - c2)
  fac <- pmax(fac, tiny)
  
  sum(logNB1 + logNB2 + log(fac))
}

# ---------- Estimator: BFGS + analytic grad + numeric Hessian + LL trace ----------
bnbr_famoye_bfgs <- function(data, f1, f2,
                             start = NULL,
                             control = list(iterlim = 200, reltol = 1e-8, printLevel = 2),
                             print_output = TRUE,
                             hess_eps = 1e-5, hess_r = 4) {
  stopifnot(is.data.frame(data))
  
  mf1 <- model.frame(f1, data = data)
  mf2 <- model.frame(f2, data = data)
  Y1  <- as.integer(model.response(mf1))
  Y2  <- as.integer(model.response(mf2))
  if (any(Y1 < 0 | Y2 < 0)) stop("Counts must be non-negative")
  
  X1  <- model.matrix(f1, mf1)
  X2  <- model.matrix(f2, mf2)
  
  p1 <- NCOL(X1); p2 <- NCOL(X2)
  cn1 <- colnames(X1); cn2 <- colnames(X2)
  
  if (is.null(start)) start <- c(rep(0, p1 + p2), log(0.5), log(0.5), 0)
  names(start) <- c(paste0("b1:", cn1), paste0("b2:", cn2), "log_m1", "log_m2", "z_lambda")
  
  # --- capture logLik at every evaluation ---
  .ll_eval <- numeric(0)
  ll_fun <- function(p) {
    v <- bnb_loglik_vec(p, Y1, Y2, X1, X2)   # per-obs vector
    s <- sum(v)
    .ll_eval <<- c(.ll_eval, s)              # record total logLik
    v
  }
  grad_fun <- function(p) bnb_grad_vec(p, Y1, Y2, X1, X2)
  
  fit <- maxLik::maxLik(
    logLik  = ll_fun,
    grad    = grad_fun,      # analytic gradient
    start   = start,
    method  = "BFGS",
    control = control
  )
  par_hat <- coef(fit)
  
  # --- build traces (all evals + accepted improvements) ---
  ll_eval_df      <- data.frame(eval = seq_along(.ll_eval), logLik = .ll_eval)
  ll_accepted_df  <- get_iteration_trace(.ll_eval, tol = 1e-8)
  
  # Back-transform and λ at fitted bounds
  beta1_hat <- par_hat[ paste0("b1:", cn1) ]
  beta2_hat <- par_hat[ paste0("b2:", cn2) ]
  m1_hat <- unname(exp(par_hat["log_m1"]))
  m2_hat <- unname(exp(par_hat["log_m2"]))
  z_hat  <- unname(par_hat["z_lambda"])
  
  mu1_hat <- as.vector(exp(X1 %*% beta1_hat))
  mu2_hat <- as.vector(exp(X2 %*% beta2_hat))
  c1_hat  <- c_val(mu1_hat, m1_hat)
  c2_hat  <- c_val(mu2_hat, m2_hat)
  bnds_hat <- lambda_bounds_vec(c1_hat, c2_hat)
  eps <- 1e-6
  lambda_hat <- unname(bnds_hat[1] + (bnds_hat[2] - bnds_hat[1]) *
                         (eps + (1 - 2*eps) * plogis(z_hat)))
  
  # Numeric Hessian with λ-bounds frozen at optimum
  lamLo_h <- bnds_hat[1]; lamHi_h <- bnds_hat[2]
  ll_fb <- function(p) bnbr_loglik_fixed_bounds(p, Y1, Y2, X1, X2, lamLo_h, lamHi_h)
  
  H <- numDeriv::hessian(ll_fb, par_hat, method.args = list(r = hess_r, eps = hess_eps))
  info <- -H; info <- (info + t(info)) / 2
  if (any(!is.finite(info))) {
    warning("Non-finite information; retrying Hessian with larger step.")
    H <- numDeriv::hessian(ll_fb, par_hat, method.args = list(r = max(6, hess_r + 2), eps = max(1e-4, 5*hess_eps)))
    info <- -H; info <- (info + t(info)) / 2
  }
  ok_eig <- try(eigen(info, symmetric = TRUE, only.values = TRUE), silent = TRUE)
  if (inherits(ok_eig, "try-error") || any(!is.finite(ok_eig$values)) || min(ok_eig$values) <= 0) {
    ridge <- if (inherits(ok_eig, "try-error") || any(!is.finite(ok_eig$values))) 1e-2 else (1e-8 - min(ok_eig$values))
    info <- info + diag(ridge, nrow(info))
  }
  vc  <- try(solve(info), silent = TRUE)
  if (inherits(vc, "try-error")) vc <- MASS::ginv(info)
  se  <- sqrt(pmax(diag(vc), 0))
  names(se) <- names(par_hat)
  
  # ----- Tables -----
  zval <- par_hat / se; pval <- 2*pnorm(-abs(zval))
  coef_table_opt <- data.frame(
    Parameter = names(par_hat),
    Estimate  = as.numeric(par_hat),
    StdErr    = as.numeric(se),
    z         = as.numeric(zval),
    p         = as.numeric(pval),
    Signif    = as.character(symnum(pval, corr=FALSE, na=FALSE,
                                    cutpoints=c(0,.001,.01,.05,.1,1),
                                    symbols=STAR_SYMS)),
    row.names = NULL, check.names = FALSE
  )
  
  idx_b1 <- match(paste0("b1:", cn1), names(par_hat))
  idx_b2 <- match(paste0("b2:", cn2), names(par_hat))
  idx_m1 <- match("log_m1", names(par_hat))
  idx_m2 <- match("log_m2", names(par_hat))
  idx_z  <- match("z_lambda", names(par_hat))
  
  se_b1 <- se[idx_b1]; se_b2 <- se[idx_b2]
  se_m1 <- abs(m1_hat) * se[idx_m1]    # delta for exp
  se_m2 <- abs(m2_hat) * se[idx_m2]
  sig_hat <- plogis(z_hat)
  dlam_dz_hat <- (bnds_hat[2] - bnds_hat[1]) * (1 - 2*eps) * sig_hat * (1 - sig_hat)
  se_lam <- abs(dlam_dz_hat) * se[idx_z]
  
  tb1 <- data.frame(Block="beta1", Name=cn1,
                    Estimate=unname(beta1_hat), StdErr=unname(se_b1))
  tb1$z <- with(tb1, as.numeric(Estimate)/as.numeric(StdErr))
  tb1$p <- 2*pnorm(-abs(tb1$z))
  tb1$Signif <- as.character(symnum(tb1$p, corr=FALSE, na=FALSE,
                                    cutpoints=c(0,.001,.01,.05,.1,1),
                                    symbols=STAR_SYMS))
  
  tb2 <- data.frame(Block="beta2", Name=cn2,
                    Estimate=unname(beta2_hat), StdErr=unname(se_b2))
  tb2$z <- with(tb2, as.numeric(Estimate)/as.numeric(StdErr))
  tb2$p <- 2*pnorm(-abs(tb2$z))
  tb2$Signif <- as.character(symnum(tb2$p, corr=FALSE, na=FALSE,
                                    cutpoints=c(0,.001,.01,.05,.1,1),
                                    symbols=STAR_SYMS))
  
  td <- rbind(
    tb1, tb2,
    data.frame(Block="dispersion", Name="m1", Estimate=m1_hat, StdErr=se_m1,
               z=as.numeric(m1_hat)/as.numeric(se_m1),
               p=2*pnorm(-abs(as.numeric(m1_hat)/as.numeric(se_m1))),
               Signif=as.character(symnum(2*pnorm(-abs(as.numeric(m1_hat)/as.numeric(se_m1))),
                                          corr=FALSE, na=FALSE,
                                          cutpoints=c(0,.001,.01,.05,.1,1),
                                          symbols=STAR_SYMS))),
    data.frame(Block="dispersion", Name="m2", Estimate=m2_hat, StdErr=se_m2,
               z=as.numeric(m2_hat)/as.numeric(se_m2),
               p=2*pnorm(-abs(as.numeric(m2_hat)/as.numeric(se_m2))),
               Signif=as.character(symnum(2*pnorm(-abs(as.numeric(m2_hat)/as.numeric(se_m2))),
                                          corr=FALSE, na=FALSE,
                                          cutpoints=c(0,.001,.01,.05,.1,1),
                                          symbols=STAR_SYMS))),
    data.frame(Block="dependence", Name="lambda", Estimate=lambda_hat, StdErr=se_lam,
               z=as.numeric(lambda_hat)/as.numeric(se_lam),
               p=2*pnorm(-abs(as.numeric(lambda_hat)/as.numeric(se_lam))),
               Signif=as.character(symnum(2*pnorm(-abs(as.numeric(lambda_hat)/as.numeric(se_lam))),
                                          corr=FALSE, na=FALSE,
                                          cutpoints=c(0,.001,.01,.05,.1,1),
                                          symbols=STAR_SYMS)))
  )
  rownames(td) <- NULL
  
  if (print_output) {
    cat("\n--- BNBR (Famoye/Sarmanov) — BFGS (analytic grad) + numeric Hessian ---\n")
    cat("Predictors (Y1): ", paste(cn1, collapse = ", "), "\n", sep = "")
    cat("Predictors (Y2): ", paste(cn2, collapse = ", "), "\n", sep = "")
    print_fixed(td, digits = 4)
    cat("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  }
  
  invisible(list(
    maxLik      = fit,
    coef        = list(beta1 = `names<-`(beta1_hat, cn1),
                       beta2 = `names<-`(beta2_hat, cn2),
                       m1 = m1_hat, m2 = m2_hat, lambda = lambda_hat),
    bounds      = list(lambda_lo = bnds_hat[1], lambda_hi = bnds_hat[2]),
    vcov        = vc,
    se          = se,
    coef_table_opt = coef_table_opt,
    coef_table_readable = td,
    X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
    # NEW:
    ll_eval        = .ll_eval,
    ll_eval_df     = ll_eval_df,
    ll_accepted    = ll_accepted_df$logLik,
    ll_accepted_df = ll_accepted_df
  ))
}

# =========================================================
# Marginal effects for BNBR (per-margin)
# - Continuous vars: AME & MEM with delta-method SEs (analytic)
# - Binary 0->1 discrete change: AME & MEM with delta-method SEs (numeric Jacobian)
# =========================================================

# ---- internal: tidy table ----
.bnbr_me_tidy <- function(names, est, se, digits = 4) {
  z <- est / se
  p <- 2 * pnorm(-abs(z))
  STAR_SYMS <- c("***","**","*","."," ")
  Signif <- as.character(symnum(p, corr=FALSE, na=FALSE,
                                cutpoints=c(0,.001,.01,.05,.1,1),
                                symbols=STAR_SYMS))
  out <- data.frame(Name = names, Estimate = est, StdErr = se, z = z, p = p, Signif = Signif,
                    check.names = FALSE)
  out
}

bnbr_me <- function(fit,
                    which = c("y1","y2","both","all"),
                    type  = c("AME","MEM"),
                    vars  = NULL,             # <- new
                    include_intercept = FALSE,
                    digits = 4,
                    print_output = TRUE) {
  
  type  <- match.arg(type, c("AME","MEM"))
  which <- match.arg(which, c("y1","y2","both","all"))
  
  # worker for ONE response -----------------------------------------
  .one_resp <- function(X, beta, Vb, resp_name) {
    cn <- colnames(X)
    p  <- ncol(X)
    
    # ---- select variables first ----
    if (is.null(vars)) {
      idx <- seq_len(p)
    } else if (is.numeric(vars)) {
      if (any(vars < 1 | vars > p)) stop("Index out of range in `vars`.")
      idx <- vars
    } else {  # character
      idx <- match(vars, cn)
      if (anyNA(idx)) {
        stop("Unknown vars in ", resp_name, ": ",
             paste(vars[is.na(idx)], collapse = ", "),
             ". Available: ", paste(cn, collapse = ", "))
      }
    }
    
    # drop intercept if asked
    if (!include_intercept) {
      idx <- idx[cn[idx] != "(Intercept)"]
    }
    if (!length(idx)) stop("No variables selected after filtering the intercept.")
    
    # ---- now auto-judge *within* the selected vars ----
    is_bin <- vapply(
      idx,
      function(j) {
        x <- X[, j]; x <- x[!is.na(x)]
        length(x) > 0 && all(x %in% c(0,1))
      },
      logical(1)
    )
    
    pos_bin  <- idx[is_bin]
    pos_cont <- idx[!is_bin]
    
    mu <- as.vector(exp(X %*% beta))
    n  <- NROW(X)
    out_list <- list()
    
    # ========== continuous part ==========
    if (length(pos_cont)) {
      est_c <- se_c <- numeric(length(pos_cont))
      nm_c  <- cn[pos_cont]
      
      if (type == "AME") {
        mu_bar  <- mean(mu)
        Xmubar  <- as.numeric(t(X) %*% (mu / n))   # E[mu * x_k]
        for (h in seq_along(pos_cont)) {
          j <- pos_cont[h]
          est_c[h] <- mean(beta[j] * mu)
          g <- beta[j] * Xmubar
          g[j] <- g[j] + mu_bar
          se_c[h] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        }
      } else {  # MEM
        Xbar   <- colMeans(X)
        mu_bar <- exp(sum(Xbar * beta))
        for (h in seq_along(pos_cont)) {
          j <- pos_cont[h]
          est_c[h] <- beta[j] * mu_bar
          g <- beta[j] * (mu_bar * Xbar)
          g[j] <- g[j] + mu_bar
          se_c[h] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        }
      }
      
      tab_c <- .bnbr_me_tidy(nm_c, est_c, se_c, digits)
      tab_c$var_type <- "continuous"
      out_list[[length(out_list)+1]] <- tab_c
    }
    
    # ========== binary 0→1 part ==========
    if (length(pos_bin)) {
      est_b <- se_b <- numeric(length(pos_bin))
      nm_b  <- cn[pos_bin]
      
      for (k in seq_along(pos_bin)) {
        j <- pos_bin[k]
        
        if (type == "AME") {
          X0 <- X; X0[, j] <- 0
          X1 <- X; X1[, j] <- 1
          
          fval <- function(b) mean(exp(X1 %*% b) - exp(X0 %*% b))
          est_b[k] <- fval(beta)
          
          g <- as.numeric(numDeriv::jacobian(fval, beta))
          se_b[k] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        } else {  # MEM
          Xbar <- matrix(colMeans(X), nrow = 1)
          X0   <- Xbar; X0[, j] <- 0
          X1   <- Xbar; X1[, j] <- 1
          
          mu0  <- as.numeric(exp(X0 %*% beta))
          mu1  <- as.numeric(exp(X1 %*% beta))
          
          est_b[k] <- mu1 - mu0
          
          g <- as.numeric(mu1 * X1 - mu0 * X0)
          se_b[k] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        }
      }
      
      tab_b <- .bnbr_me_tidy(nm_b, est_b, se_b, digits)
      tab_b$var_type <- "binary(0→1)"
      out_list[[length(out_list)+1]] <- tab_b
    }
    
    out <- do.call(rbind, out_list)
    rownames(out) <- NULL
    
    if (print_output) {
      cat(sprintf("\n--- Marginal effects (auto) for %s (%s) ---\n",
                  resp_name, type))
      print_fixed(out, digits = digits)
      cat("continuous: dE[Y]/dx_j = β_j * μ\n")
      cat("binary: E[Y|x_j=1] − E[Y|x_j=0]\n")
    }
    
    out
  }
  
  # ------------------------------------------------------------
  # y1 + y2
  # ------------------------------------------------------------
  if (which %in% c("both","all")) {
    res <- list()
    
    # y1
    X1    <- fit$X1
    beta1 <- fit$coef$beta1
    p1    <- ncol(X1)
    Vb1   <- fit$vcov[1:p1, 1:p1, drop = FALSE]
    res$y1 <- .one_resp(X1, beta1, Vb1, "y1")
    
    # y2
    X2    <- fit$X2
    beta2 <- fit$coef$beta2
    p2    <- ncol(X2)
    Vb2   <- fit$vcov[(p1+1):(p1+p2), (p1+1):(p1+p2), drop = FALSE]
    res$y2 <- .one_resp(X2, beta2, Vb2, "y2")
    
    return(invisible(res))
  }
  
  # single response
  if (which == "y1") {
    X    <- fit$X1
    beta <- fit$coef$beta1
    p    <- ncol(X)
    Vb   <- fit$vcov[1:p, 1:p, drop = FALSE]
  } else {
    p1   <- ncol(fit$X1)
    X    <- fit$X2
    beta <- fit$coef$beta2
    p    <- ncol(X)
    Vb   <- fit$vcov[(p1+1):(p1+p), (p1+1):(p1+p), drop = FALSE]
  }
  
  out <- .one_resp(X, beta, Vb, which)
  invisible(out)
}

bnbr_elasticities <- function(fit,
                              which = c("y1", "y2", "both"),
                              type  = c("AME", "MEM"),
                              vars  = NULL,
                              include_intercept = FALSE,   # <- new
                              digits = 4,
                              print_output = TRUE) {
  
  type  <- match.arg(type)
  which <- match.arg(which)
  
  # ------------------------------------------------------------
  # worker for ONE response
  # ------------------------------------------------------------
  .one_resp <- function(X, beta, Vb, resp_name) {
    cn <- colnames(X)
    
    # restrict to user vars if given
    if (!is.null(vars)) {
      pos <- match(vars, cn)
      if (anyNA(pos)) {
        stop("These vars not found in design matrix for '", resp_name, "': ",
             paste(vars[is.na(pos)], collapse = ", "))
      }
      idxs      <- pos
      out_names <- vars
    } else {
      idxs      <- seq_along(cn)
      out_names <- cn
    }
    
    # drop intercept if requested
    if (!include_intercept) {
      keep <- out_names != "(Intercept)"
      idxs      <- idxs[keep]
      out_names <- out_names[keep]
    }
    if (!length(idxs)) {
      stop("No variables selected after intercept filtering for '", resp_name, "'.")
    }
    
    xbar <- colMeans(X)
    est  <- se <- numeric(length(idxs))
    vtype <- character(length(idxs))
    
    for (m in seq_along(idxs)) {
      j    <- idxs[m]
      xj   <- X[, j]
      bj   <- beta[j]
      varb <- Vb[j, j]
      is_bin <- all(xj %in% c(0, 1, NA))
      
      if (is_bin) {
        # 0/1 var -> semi-elasticity for dummy: exp(beta) - 1
        eff     <- exp(bj) - 1
        se_j    <- exp(bj) * sqrt(max(varb, 0))
        est[m]  <- eff
        se[m]   <- se_j
        vtype[m] <- "binary (0→1): exp(beta)-1"
      } else {
        # continuous var -> elasticity = beta * E[x_j]
        if (type == "AME") {
          m_xj    <- mean(xj, na.rm = TRUE)
          est[m]  <- bj * m_xj
          se[m]   <- sqrt(max(varb, 0)) * abs(m_xj)
        } else {
          est[m]  <- bj * xbar[j]
          se[m]   <- sqrt(max(varb, 0)) * abs(xbar[j])
        }
        vtype[m] <- "continuous: beta * E[x]"
      }
    }
    
    tab <- .bnbr_me_tidy(out_names, est, se, digits)
    tab$var_type <- vtype
    
    if (print_output) {
      cat(sprintf("\n--- Elasticities / semi-elasticities (%s, %s) ---\n",
                  resp_name, type))
      print_fixed(tab, digits = digits)
      cat("Continuous vars: elasticity = beta * E[x].\n")
      cat("0/1 vars: semi-elasticity = exp(beta) - 1 (percent change from 0→1).\n")
    }
    
    tab
  }
  
  # ------------------------------------------------------------
  # y1, y2, or both
  # ------------------------------------------------------------
  if (which == "both") {
    # y1
    X1    <- fit$X1
    beta1 <- fit$coef$beta1
    p1    <- ncol(X1)
    Vb1   <- fit$vcov[1:p1, 1:p1, drop = FALSE]
    tab1  <- .one_resp(X1, beta1, Vb1, "y1")
    
    # y2
    X2    <- fit$X2
    beta2 <- fit$coef$beta2
    p2    <- ncol(X2)
    Vb2   <- fit$vcov[(p1 + 1):(p1 + p2), (p1 + 1):(p1 + p2), drop = FALSE]
    tab2  <- .one_resp(X2, beta2, Vb2, "y2")
    
    return(invisible(list(y1 = tab1, y2 = tab2)))
  }
  
  # single response
  if (which == "y1") {
    X    <- fit$X1
    beta <- fit$coef$beta1
    p    <- ncol(X)
    Vb   <- fit$vcov[1:p, 1:p, drop = FALSE]
    tab  <- .one_resp(X, beta, Vb, "y1")
  } else {
    p1   <- ncol(fit$X1)
    X    <- fit$X2
    beta <- fit$coef$beta2
    p    <- ncol(X)
    Vb   <- fit$vcov[(p1 + 1):(p1 + p), (p1 + 1):(p1 + p), drop = FALSE]
    tab  <- .one_resp(X, beta, Vb, "y2")
  }
  
  invisible(tab)
}
