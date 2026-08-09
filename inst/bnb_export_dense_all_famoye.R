#!/usr/bin/env Rscript
# =============================================================================
# Fit BNB model (non-TMB, fixed-effects) to export_dense_all.csv
# Famoye/Sarmanov dependence, no random coefficients
# =============================================================================

library(rpbnb)

# ---- Load data ---------------------------------------------------------------
data <- read.csv(system.file(
  "extdata", "export_dense_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== Fitting BNB (non-TMB, fixed-effects, Famoye) to export_dense_all.csv ===\n")
cat("Observations :", nrow(data), "\n")
cat("Columns     :", paste(names(data), collapse = ", "), "\n\n")

# ---- Model specification ----------------------------------------------------
numeric_cols <- names(data)[sapply(data, is.numeric) & names(data) != "y1" & names(data) != "y2"]
rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))

f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

cat("Formula 1:", deparse(f1), "\n")
cat("Formula 2:", deparse(f2), "\n")
cat("Type: Fixed-effects bivariate NB (no random coefficients)\n\n")

# ---- Fit model ---------------------------------------------------------------
t_fit <- system.time(
  fit <- fit_bnb(
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

cat(sprintf("\nEstimation finished in %.1f s\n\n", t_fit))

# ---- Model summary ----------------------------------------------------------
print(fit)

cat("\nlogLik =", round(fit$logLik, 2),
    "  AIC =", round(fit$AIC, 2),
    "  BIC =", round(fit$BIC, 2),
    "  npar =", fit$npar, "\n")

cat("\nNote: BNB models estimate marginal dispersion (m1, m2) and dependence (lambda)\n")
cat("      but no random coefficients. This is a simpler/faster alternative to RP-BNB.\n")
