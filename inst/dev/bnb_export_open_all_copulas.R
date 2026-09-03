#!/usr/bin/env Rscript
# =============================================================================
# Fit BNB models (non-TMB, fixed-effects) to export_open_all.csv
# Compares different copula structures: Frank, Gaussian, Clayton
# =============================================================================

library(rpbnb)

# ---- Load data ---------------------------------------------------------------
data <- read.csv(system.file(
  "extdata", "export_open_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== BNB Model Comparison (non-TMB, fixed-effects) ===\n")
cat("Data: export_open_all.csv\n")
cat("Observations:", nrow(data), "\n\n")

# ---- Model specification ----------------------------------------------------
numeric_cols <- names(data)[sapply(data, is.numeric) & names(data) != "y1" & names(data) != "y2"]
rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))

f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

cat("Formula 1:", deparse(f1), "\n")
cat("Formula 2:", deparse(f2), "\n")
cat("Type: Fixed-effects bivariate NB (no random coefficients)\n\n")

# ---- Model 1: Famoye dependence -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("Model 1: Famoye/Sarmanov Dependence\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

t_famoye <- system.time(
  fit_famoye <- fit_bnb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = "famoye",
    control   = rpbnb_control(
      n_cores = parallel::detectCores() - 2,
      compute_se = TRUE,
      print_level = 1
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n",
            t_famoye, fit_famoye$logLik, fit_famoye$AIC, fit_famoye$BIC, fit_famoye$npar))

# ---- Model 2: Frank copula -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("Model 2: Frank Copula\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

t_frank <- system.time(
  fit_frank <- fit_bnb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = copula("frank"),
    control   = rpbnb_control(
      n_cores = parallel::detectCores() - 2,
      compute_se = TRUE,
      print_level = 1
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n",
            t_frank, fit_frank$logLik, fit_frank$AIC, fit_frank$BIC, fit_frank$npar))
cat("Copula parameter:", round(fit_frank$cop_par, 4), "| tau:", round(fit_frank$cop_tau, 4), "\n")

# ---- Model 3: Gaussian copula -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("Model 3: Gaussian (Normal) Copula\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

t_gaussian <- system.time(
  fit_gaussian <- fit_bnb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = copula("normal"),
    control   = rpbnb_control(
      n_cores = parallel::detectCores() - 2,
      compute_se = TRUE,
      print_level = 1
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n",
            t_gaussian, fit_gaussian$logLik, fit_gaussian$AIC, fit_gaussian$BIC, fit_gaussian$npar))
cat("Copula parameter:", round(fit_gaussian$cop_par, 4), "| tau:", round(fit_gaussian$cop_tau, 4), "\n")

# ---- Model 4: Clayton copula -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("Model 4: Clayton Copula\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

t_clayton <- system.time(
  fit_clayton <- fit_bnb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = copula("clayton"),
    control   = rpbnb_control(
      n_cores = parallel::detectCores() - 2,
      compute_se = TRUE,
      print_level = 1
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n",
            t_clayton, fit_clayton$logLik, fit_clayton$AIC, fit_clayton$BIC, fit_clayton$npar))
cat("Copula parameter:", round(fit_clayton$cop_par, 4), "| tau:", round(fit_clayton$cop_tau, 4), "\n")

# ---- Model comparison -----------------------------------------------
cat("\n\n", paste(rep("=", 100), collapse = ""), "\n", sep = "")
cat("MODEL COMPARISON\n")
cat(paste(rep("=", 100), collapse = ""), "\n")

comparison <- data.frame(
  Model = c("Famoye", "Frank", "Gaussian", "Clayton"),
  Time_sec = c(t_famoye, t_frank, t_gaussian, t_clayton),
  logLik = c(fit_famoye$logLik, fit_frank$logLik, fit_gaussian$logLik, fit_clayton$logLik),
  AIC = c(fit_famoye$AIC, fit_frank$AIC, fit_gaussian$AIC, fit_clayton$AIC),
  BIC = c(fit_famoye$BIC, fit_frank$BIC, fit_gaussian$BIC, fit_clayton$BIC),
  npar = c(fit_famoye$npar, fit_frank$npar, fit_gaussian$npar, fit_clayton$npar),
  row.names = NULL
)

print(comparison, digits = 3)

cat("\nBest AIC model:", comparison[which.min(comparison$AIC), "Model"], "\n")
cat("Best BIC model:", comparison[which.min(comparison$BIC), "Model"], "\n")
