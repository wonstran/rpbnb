#!/usr/bin/env Rscript
# =============================================================================
# Fit RP-BNB model to export_dense_all.csv with random coefficients
# Tests different random coefficient specifications
# =============================================================================

devtools::load_all()

library(rpbnb)

# ---- Load data --------------------------------------------------------------
data <- read.csv(system.file(
  "extdata", "export_dense_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== RP-BNB with Random Coefficients: export_dense_all.csv ===\n")
cat("Observations :", nrow(data), "\n\n")

# ---- Model specification ----------------------------------------------------
numeric_cols <- names(data)[sapply(data, is.numeric) & names(data) != "y1" & names(data) != "y2"]
rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))

f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

cat("Formula 1:", deparse(f1), "\n")
cat("Formula 2:", deparse(f2), "\n")
cat("Available numeric predictors:", paste(numeric_cols, collapse = ", "), "\n\n")

# ---- Model 1: One random coefficient in each equation -------------------------
cat("\n### Model 1: Single random coefficient (normal distribution) ###\n")
random_1_var <- numeric_cols[1]
random_2_var <- numeric_cols[2]

cat("Random on:", random_1_var, "and", random_2_var, "\n\n")

t_fit1 <- system.time(
  fit1 <- fit_rpbnb_tmb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    random_1  = random_1_var,
    random_2  = random_2_var,
    dependence = "famoye",
    draws     = 250,
    seed      = 20260809,
    method    = "sml",
    control   = rpbnb_tmb_control(
      print_level = 2,
      n_cores     = parallel::detectCores() - 2
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n\n",
            t_fit1, fit1$logLik, AIC(fit1), BIC(fit1), fit1$npar))

# ---- Model 2: Multiple random coefficients -------------------------
if (length(numeric_cols) >= 3) {
  cat("\n### Model 2: Multiple random coefficients ###\n")
  random_vars <- numeric_cols[1:min(3, length(numeric_cols))]
  cat("Random on:", paste(random_vars, collapse = ", "), "\n\n")

  t_fit2 <- system.time(
    fit2 <- fit_rpbnb_tmb(
      formula_1 = f1,
      formula_2 = f2,
      data      = data,
      random_1  = random_vars,
      random_2  = random_vars,
      dependence = "famoye",
      draws     = 200,
      seed      = 20260809,
      method    = "sml",
      control   = rpbnb_tmb_control(
        print_level = 2,
        n_cores     = parallel::detectCores() - 2
      )
    )
  )[["elapsed"]]

  cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n\n",
              t_fit2, fit2$logLik, AIC(fit2), BIC(fit2), fit2$npar))
}

# ---- Model 3: Random coefficients with Laplace approximation -------
cat("\n### Model 3: Random coefficients (Laplace method) ###\n")
random_vars <- numeric_cols[1:min(2, length(numeric_cols))]
cat("Random on:", paste(random_vars, collapse = ", "), "\n")
cat("Using Laplace approximation for memory efficiency\n\n")

t_fit3 <- system.time(
  fit3 <- fit_rpbnb_tmb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    random_1  = random_vars,
    random_2  = random_vars,
    dependence = "famoye",
    draws     = 150,
    seed      = 20260809,
    method    = "laplace",
    control   = rpbnb_tmb_control(
      print_level = 2,
      n_cores     = parallel::detectCores() - 2
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n\n",
            t_fit3, fit3$logLik, AIC(fit3), BIC(fit3), fit3$npar))

# ---- Model comparison -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("MODEL COMPARISON\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

comparison <- data.frame(
  Model = c("Single RC (SML)", "Multiple RC (SML)", "Multiple RC (Laplace)"),
  Time_sec = c(t_fit1, if (exists("t_fit2")) t_fit2 else NA_real_, t_fit3),
  logLik = c(fit1$logLik, if (exists("fit2")) fit2$logLik else NA_real_, fit3$logLik),
  AIC = c(AIC(fit1), if (exists("fit2")) AIC(fit2) else NA_real_, AIC(fit3)),
  BIC = c(BIC(fit1), if (exists("fit2")) BIC(fit2) else NA_real_, BIC(fit3)),
  npar = c(fit1$npar, if (exists("fit2")) fit2$npar else NA_integer_, fit3$npar),
  row.names = NULL
)

print(comparison, digits = 3)
