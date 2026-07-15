#!/usr/bin/env Rscript
# =============================================================================
# Fit the copula random-parameter bivariate NB (RP-BNB) model to the complex
# copula sample data produced by inst/simulate_rpbnb_copula_complex.R, then
# compare the estimates against the known ground-truth parameters.
#
#   * 8 independent variables (3 continuous + 5 dummy)
#   * random coefficients on x_age (eq 1) and x_income (eq 2), continuous
#   * GENUINE copula dependence (Gaussian copula, rho) between y1 and y2
#   * multithreaded (OpenMP) simulated-likelihood estimation, OPG standard errors
#
# RUNTIME: the copula RP path is numerically heavy (the discrete-copula pmf +
# per-draw NB CDF corners), so a full fit at n = 5000 with draws = 500 takes on
# the order of 10-20 minutes even multithreaded -- noticeably slower than the
# Famoye path. Lower `draws`, subset the data, or set compute_se = FALSE for a
# quicker look. The mean coefficients and the copula dependence are well
# identified and recover first; the NB-dispersion-vs-random-SD variance split is
# the slow, fuzzy part of the surface (see the copula-recovery memo).
# =============================================================================

devtools::load_all(quiet = TRUE)

# ---- 1. Load data + ground truth -------------------------------------------
data  <- read.csv(file.path("data", "simulated_rpbnb_copula_complex.csv"))
truth <- readRDS(file.path("data", "simulated_rpbnb_copula_complex_truth.rds"))

cat("=== Fitting copula RP-BNB to complex sample data ===\n")
cat("Observations :", nrow(data), "\n")
cat("Threads      :", rpbnb_threads(), "\n")
cat(sprintf("True copula  : %s  (rho = %.2f, Kendall tau = %.3f)\n\n",
            truth$copula, truth$theta, truth$tau))

# ---- 2. Model specification -------------------------------------------------
rhs <- ~ x_age + x_income + x_score + d_female + d_urban + d_married + d_college + d_smoker
f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

t_fit <- system.time(
  fit <- fit_rpbnb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    random_1  = truth$random_names_1,   # c("x_age")
    random_2  = truth$random_names_2,   # c("x_income")
    dependence = copula(truth$copula),  # Gaussian copula; rho estimated
    draws     = 400,
    seed      = 20240712,
    control   = rpbnb_control(
      print_level = 2,
      n_cores     = rpbnb_threads(),  # OpenMP thread count
      se_method   = "opg"             # fast analytic BHHH/OPG SEs (one score
      # pass). The copula-path default is se_method = "numeric" (observed-info
      # Hessian, slower). Use compute_se = FALSE to skip SEs entirely.
    )
  )
)[["elapsed"]]

cat(sprintf("\nEstimation finished in %.1f s\n\n", t_fit))

# ---- 3. Model summary -------------------------------------------------------
print(fit)

# ---- 4. Compare estimated MEAN coefficients vs truth ------------------------
compare_means <- function(prefix, true_beta) {
  est <- fit$coef[paste0(prefix, ":", names(true_beta))]
  se  <- fit$se[paste0(prefix, ":", names(true_beta))]
  data.frame(
    Variable  = names(true_beta),
    True      = as.numeric(true_beta),
    Estimate  = as.numeric(est),
    StdErr    = as.numeric(se),
    z_vs_true = (as.numeric(est) - as.numeric(true_beta)) / as.numeric(se),
    row.names = NULL
  )
}

cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("EQUATION 1 (y1) - mean coefficients: estimate vs true\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
print(compare_means("b1", truth$beta1), digits = 4)

cat("\nEQUATION 2 (y2) - mean coefficients: estimate vs true\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
print(compare_means("b2", truth$beta2), digits = 4)

# ---- 5. Compare random-coefficient SDs vs truth -----------------------------
compare_sd <- function(prefix, random_spec) {
  nm  <- names(random_spec)
  est <- exp(fit$coef[paste0("log_", prefix, ":", nm)])   # natural-scale SD
  true_sd <- vapply(random_spec, function(z) z$sd, numeric(1))
  data.frame(
    Variable = nm,
    True_SD  = as.numeric(true_sd),
    Est_SD   = as.numeric(est),
    row.names = NULL
  )
}

cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("RANDOM-COEFFICIENT STANDARD DEVIATIONS: estimate vs true\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
cat("Equation 1:\n"); print(compare_sd("sd1", truth$random_1), digits = 4)
cat("\nEquation 2:\n"); print(compare_sd("sd2", truth$random_2), digits = 4)

# ---- 6. Dispersion & copula dependence vs truth -----------------------------
rho_hat <- tanh(fit$coef[["z_theta"]])
tau_hat <- rpbnb:::normal_tau(rho_hat)
cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("DISPERSION & COPULA DEPENDENCE: estimate vs true\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
disp_tbl <- data.frame(
  Parameter = c("m1", "m2", "rho", "Kendall tau"),
  True      = c(truth$dispersion[["m1"]], truth$dispersion[["m2"]], truth$theta, truth$tau),
  Estimate  = c(fit$m1, fit$m2, rho_hat, tau_hat),
  row.names = NULL
)
print(disp_tbl, digits = 4)

cat("\nlogLik =", round(fit$logLik, 2),
    "  AIC =", round(fit$AIC, 2),
    "  BIC =", round(fit$BIC, 2),
    "  npar =", fit$npar, "\n")
cat("\nNote: random coefficients sit on CONTINUOUS regressors -- the copula RP\n")
cat("      model is weakly identified when a random coefficient sits on a 0/1\n")
cat("      dummy (NB dispersion vs random-coefficient scale). The five dummies\n")
cat("      here are ordinary fixed effects and recover normally.\n")
