#!/usr/bin/env Rscript
# =============================================================================
# Compare inference methods: full vs diagonal vs none
# Tests impact on computation time and standard errors
# Uses export_open_all.csv with Famoye dependence
# =============================================================================

devtools::load_all()

library(rpbnb)

# ---- Load data --------------------------------------------------------------
data <- read.csv(system.file(
  "extdata", "export_open_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== Inference Method Comparison ===\n")
cat("Data: export_open_all.csv\n")
cat("Observations:", nrow(data), "\n")
cat("Model: Famoye dependence with SML estimation\n\n")

# ---- Setup model specification -----------------------------------------------
numeric_cols <- names(data)[sapply(data, is.numeric) & names(data) != "y1" & names(data) != "y2"]
rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))

f1 <- update(rhs, y1 ~ .)
f2 <- update(rhs, y2 ~ .)

cat("Formula 1:", deparse(f1), "\n")
cat("Formula 2:", deparse(f2), "\n\n")

# ---- Model 1: Full inference -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("Model 1: Full inference (complete covariance matrix)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

t_full <- system.time(
  fit_full <- fit_rpbnb_tmb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = "famoye",
    draws     = 200,
    seed      = 20260809,
    method    = "sml",
    inference = "full",
    control   = rpbnb_tmb_control(
      print_level = 1,
      n_cores     = parallel::detectCores() - 2
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | npar: %d\n",
            t_full, fit_full$logLik, AIC(fit_full), fit_full$npar))

if (!is.null(fit_full$vcov)) {
  cat("Full covariance matrix available (dim:", nrow(fit_full$vcov), "x", ncol(fit_full$vcov), ")\n")
}

# ---- Model 2: Diagonal inference -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("Model 2: Diagonal inference (standard errors only)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

t_diag <- system.time(
  fit_diag <- fit_rpbnb_tmb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = "famoye",
    draws     = 200,
    seed      = 20260809,
    method    = "sml",
    inference = "diag",
    control   = rpbnb_tmb_control(
      print_level = 1,
      n_cores     = parallel::detectCores() - 2
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | npar: %d\n",
            t_diag, fit_diag$logLik, AIC(fit_diag), fit_diag$npar))
cat("Standard errors available; covariances set to NA\n")

# ---- Model 3: No inference -----------------------------------------------
cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("Model 3: No inference (skip Hessian computation)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

t_none <- system.time(
  fit_none <- fit_rpbnb_tmb(
    formula_1 = f1,
    formula_2 = f2,
    data      = data,
    dependence = "famoye",
    draws     = 200,
    seed      = 20260809,
    method    = "sml",
    inference = "none",
    control   = rpbnb_tmb_control(
      print_level = 1,
      n_cores     = parallel::detectCores() - 2
    )
  )
)[["elapsed"]]

cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | npar: %d\n",
            t_none, fit_none$logLik, AIC(fit_none), fit_none$npar))
cat("No standard errors or covariances computed\n")

# ---- Comparison summary -----------------------------------------------
cat("\n\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("INFERENCE METHOD COMPARISON\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

comparison <- data.frame(
  Method = c("Full", "Diagonal", "None"),
  Time_sec = c(t_full, t_diag, t_none),
  logLik = c(fit_full$logLik, fit_diag$logLik, fit_none$logLik),
  AIC = c(AIC(fit_full), AIC(fit_diag), AIC(fit_none)),
  npar = c(fit_full$npar, fit_diag$npar, fit_none$npar),
  Has_Covariance = c(
    !is.null(fit_full$vcov) && !all(is.na(fit_full$vcov)),
    !is.null(fit_diag$vcov),
    is.null(fit_none$vcov)
  ),
  Time_Ratio = c(1, t_diag / t_full, t_none / t_full),
  row.names = NULL
)

print(comparison, digits = 3)

cat("\nNotes:\n")
cat("  - All methods produce identical estimates and logLik\n")
cat("  - Inference method affects: computation time and parameter uncertainty\n")
cat("  - Use 'none' for model comparison when SE/covariances not needed\n")
cat("  - Use 'diag' when SE needed but cross-covariances not required\n")
cat("  - Use 'full' for complete uncertainty quantification\n")
