#!/usr/bin/env Rscript
# =============================================================================
# Fit RP-BNB model to export_open_all.csv with Famoye dependence
# Uses Laplace approximation with random coefficients
# =============================================================================

devtools::load_all()

library(rpbnb)

# ---- Load data --------------------------------------------------------------
data <- read.csv(system.file(
  "extdata", "export_open_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== Fitting RP-BNB (Famoye, Laplace) to export_open_all.csv ===\n")
cat("Observations :", nrow(data), "\n")
cat("Columns     :", paste(names(data), collapse = ", "), "\n\n")

# ---- Model specification ----------------------------------------------------
numeric_cols <- names(data)[sapply(data, is.numeric) & names(data) != "y1" & names(data) != "y2"]
rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))

f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

# Use first and second numeric columns as random coefficients
random_vars <- numeric_cols[1:min(2, length(numeric_cols))]
cat("Formula 1:", deparse(f1), "\n")
cat("Formula 2:", deparse(f2), "\n")
cat("Random coefficients:", paste(random_vars, collapse = ", "), "\n\n")

# ---- Fit model ---------------------------------------------------------------
t_fit <- system.time(
  fit <- fit_rpbnb_tmb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    random_1  = random_vars,
    random_2  = random_vars,
    dependence = "famoye",
    draws     = 200,
    seed      = 20260809,
    method    = "laplace",
    control   = rpbnb_tmb_control(
      print_level = 2,
      n_cores     = parallel::detectCores() - 2
    )
  )
)[["elapsed"]]

cat(sprintf("\nEstimation finished in %.1f s\n\n", t_fit))

# ---- Model summary ----------------------------------------------------------
print(fit)

cat("\nlogLik =", round(fit$logLik, 2),
    "  AIC =", round(AIC(fit), 2),
    "  BIC =", round(BIC(fit), 2),
    "  npar =", fit$npar, "\n")
