#!/usr/bin/env Rscript
# =============================================================================
# Fit RP-BNB model (non-TMB) to export_open_all.csv with Famoye dependence
# Uses BFGS optimization with random coefficients on two variables
# =============================================================================

library(rpbnb)

# ---- Load data ---------------------------------------------------------------
data <- read.csv(system.file(
  "extdata", "export_open_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== Fitting RP-BNB (non-TMB, Famoye) to export_open_all.csv ===\n")
cat("Observations :", nrow(data), "\n")
cat("Columns     :", paste(names(data), collapse = ", "), "\n\n")

# ---- Model specification ----------------------------------------------------
numeric_cols <- names(data)[sapply(data, is.numeric) & names(data) != "y1" & names(data) != "y2"]
rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))

f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

random_vars <- numeric_cols[1:min(2, length(numeric_cols))]

cat("Formula 1:", deparse(f1), "\n")
cat("Formula 2:", deparse(f2), "\n")
cat("Random on:", paste(random_vars, collapse = ", "), "\n\n")

# ---- Fit model ---------------------------------------------------------------
t_fit <- system.time(
  fit <- fit_rpbnb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    random_1  = random_vars,
    random_2  = random_vars,
    dependence = "famoye",
    draws     = 300,
    seed      = 20260809,
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

cat("\nConvergence:", fit$convergence, "\n")
