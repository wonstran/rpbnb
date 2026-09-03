#!/usr/bin/env Rscript
# =============================================================================
# Fit RP-BNB model to export_open_all.csv with independence assumption
# Uses SML estimation
# =============================================================================

devtools::load_all()

library(rpbnb)

# ---- Load data --------------------------------------------------------------
data <- read.csv(system.file(
  "extdata", "export_open_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== Fitting RP-BNB (Independence, SML) to export_open_all.csv ===\n")
cat("Observations :", nrow(data), "\n")
cat("Columns     :", paste(names(data), collapse = ", "), "\n\n")

# ---- Model specification ----------------------------------------------------
numeric_cols <- names(data)[sapply(data, is.numeric) & names(data) != "y1" & names(data) != "y2"]
rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))

f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

cat("Formula 1:", deparse(f1), "\n")
cat("Formula 2:", deparse(f2), "\n")
cat("Dependence: independence (separate negative binomials)\n\n")

# ---- Fit model ---------------------------------------------------------------
t_fit <- system.time(
  fit <- fit_rpbnb_tmb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = "independence",
    random_1  = NULL,
    random_2  = NULL,
    draws     = 100,
    seed      = 20260809,
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

cat("\nNote: Independence model has no dependence parameter.\n")
cat("Use this as a baseline for comparing model fit against copula/Famoye variants.\n")
