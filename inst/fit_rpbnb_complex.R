#!/usr/bin/env Rscript
# =============================================================================
# Fit a random-parameter bivariate NB (RP-BNB) model to the complex sample data
# produced by inst/simulate_rpbnb_complex.R, then compare the estimates against
# the known ground-truth parameters.
#
#   * 8 independent variables (3 continuous + 5 dummy)
#   * random coefficients on x_age (eq 1) and x_income (eq 2)
#   * GENUINE Famoye/Sarmanov dependence (non-zero lambda) between y1 and y2
#   * multithreaded (OpenMP) simulated-likelihood estimation
# =============================================================================

devtools::load_all(quiet = TRUE)

# ---- 1. Load data + ground truth -------------------------------------------
data  <- read.csv(file.path("data", "simulated_rpbnb_dependent.csv"))
truth <- readRDS(file.path("data", "simulated_rpbnb_dependent_truth.rds"))

cat("=== Fitting RP-BNB to complex sample data ===\n")
cat("Observations :", nrow(data), "\n")
cat("Threads      :", rpbnb_threads(), "\n\n")

# ---- 2. Model specification -------------------------------------------------
# Same 8 covariates in both equations; random coefficients matched to the DGP.
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
    draws     = 500,
    seed      = 20240712,
    control   = rpbnb_control(
      print_level = 2,
      n_cores     = rpbnb_threads(),  # OpenMP thread count
      se_method   = "opg"             # This demo opts into the fast analytic
      # BHHH/OPG standard errors (one score pass, seconds) to keep the 5000-obs
      # run quick. The PACKAGE DEFAULT is se_method = "numeric" (the
      # observed-information Hessian, minutes for a model this size but robust at
      # boundaries). Use compute_se = FALSE to skip SEs entirely.
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

# ---- 6. Dispersion & dependence vs truth ------------------------------------
cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("DISPERSION & DEPENDENCE: estimate vs true\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
disp_tbl <- data.frame(
  Parameter = c("m1", "m2", "lambda"),
  True      = c(truth$dispersion[["m1"]], truth$dispersion[["m2"]], truth$lambda),
  Estimate  = c(fit$m1, fit$m2, fit$lambda),
  row.names = NULL
)
print(disp_tbl, digits = 4)

cat("\nlogLik =", round(fit$logLik, 2),
    "  AIC =", round(fit$AIC, 2),
    "  BIC =", round(fit$BIC, 2),
    "  npar =", fit$npar, "\n")
cat(sprintf("\nNote: TRUE lambda = %.4f (genuine Famoye/Sarmanov dependence) -- the fit\n",
            truth$lambda))
cat("      should recover a significantly non-zero lambda here (contrast with the\n")
cat("      independent dataset, where lambda ~ 0, n.s.).\n")
cat("Note: the raw cor(y1, y2) is modest by construction -- the Famoye e^{-y} tilt\n")
cat("      concentrates dependence in the low-count region, so the structural\n")
cat("      lambda is strong even though the Pearson correlation looks small.\n")

# ---- 7. Marginal effects & elasticities (AME) -------------------------------
# Built on the Monte-Carlo integrated mean E[exp(x'beta)] (the same estimand
# predict() uses), so x_age (random in eq 1) and x_income (random in eq 2) use
# the draw-integrated formula while every other covariate reduces to the
# classic fixed-coefficient result. See ?rpbnb_marginal_effects.
cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("AVERAGE MARGINAL EFFECTS (AME)\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
me <- rpbnb_marginal_effects(fit, which = "both", type = "AME",
                             n_cores = rpbnb_threads())

cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
cat("ELASTICITIES / SEMI-ELASTICITIES (AME)\n")
cat(paste(rep("=", 72), collapse = ""), "\n")
el <- rpbnb_elasticities(fit, which = "both", type = "AME",
                         n_cores = rpbnb_threads())
