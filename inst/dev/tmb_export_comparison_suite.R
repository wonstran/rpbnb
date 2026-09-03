#!/usr/bin/env Rscript
# =============================================================================
# Comprehensive RP-BNB model comparison suite
# Fits multiple models to both export_dense_all.csv and export_open_all.csv
# Compares: Famoye, Frank copula, Gaussian copula, Clayton copula, independence
# =============================================================================

devtools::load_all()

library(rpbnb)

# ---- Helper function to fit and report -----------------------------------------------
fit_and_report <- function(data_name, data, formula_1, formula_2,
                          random_1 = NULL, random_2 = NULL,
                          dependence_name, dependence_obj, method = "sml") {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
  cat(sprintf("Model: %s | Data: %s | Method: %s\n", dependence_name, data_name, method))
  cat(paste(rep("=", 80), collapse = ""), "\n")

  tryCatch({
    t_fit <- system.time(
      fit <- fit_rpbnb_tmb(
        formula_1 = formula_1,
        formula_2 = formula_2,
        data      = data,
        random_1  = random_1,
        random_2  = random_2,
        dependence = dependence_obj,
        draws     = if (method == "sml") 200 else 100,
        seed      = 20260809,
        method    = method,
        control   = rpbnb_tmb_control(
          print_level = 1,
          n_cores     = max(1, parallel::detectCores() - 2)
        )
      )
    )[["elapsed"]]

    result <- data.frame(
      data_name = data_name,
      model = dependence_name,
      method = method,
      n_obs = nrow(data),
      time_sec = t_fit,
      logLik = fit$logLik,
      AIC = AIC(fit),
      BIC = BIC(fit),
      npar = fit$npar,
      convergence = fit$optimizer$convergence,
      max_grad = fit$optimizer$max_abs_gradient,
      row.names = NULL
    )

    cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n",
                t_fit, fit$logLik, AIC(fit), BIC(fit), fit$npar))
    cat(sprintf("Convergence: %d | Max gradient: %.2e\n",
                fit$optimizer$convergence, fit$optimizer$max_abs_gradient))

    return(result)
  }, error = function(e) {
    cat("ERROR:", e$message, "\n")
    return(data.frame(
      data_name = data_name,
      model = dependence_name,
      method = method,
      n_obs = nrow(data),
      time_sec = NA_real_,
      logLik = NA_real_,
      AIC = NA_real_,
      BIC = NA_real_,
      npar = NA_integer_,
      convergence = NA_integer_,
      max_grad = NA_real_,
      row.names = NULL
    ))
  })
}

# ---- Load both datasets -----------------------------------------------------------
data_dense <- read.csv(system.file(
  "extdata", "export_dense_all.csv",
  package = "rpbnb", mustWork = TRUE
))

data_open <- read.csv(system.file(
  "extdata", "export_open_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== RP-BNB Comprehensive Model Comparison Suite ===\n")
cat("Dense dataset observations:", nrow(data_dense), "\n")
cat("Open dataset observations:", nrow(data_open), "\n\n")

# ---- Setup formulas (adapt to your actual data columns) --------------------------
# Using first 5 numeric columns as predictors
get_formula <- function(data) {
  numeric_cols <- names(data)[sapply(data, is.numeric) &
                              names(data) != "y1" & names(data) != "y2"]
  rhs <- as.formula(paste("~", paste(numeric_cols[1:min(5, length(numeric_cols))], collapse = " + ")))
  f1 <- update(rhs, y1 ~ .)
  f2 <- update(rhs, y2 ~ .)
  list(f1 = f1, f2 = f2, random_vars = numeric_cols[1:min(2, length(numeric_cols))])
}

formula_dense <- get_formula(data_dense)
formula_open <- get_formula(data_open)

cat("Dense formulas:\n")
cat("  Formula 1:", deparse(formula_dense$f1), "\n")
cat("  Formula 2:", deparse(formula_dense$f2), "\n")
cat("  Random vars:", paste(formula_dense$random_vars, collapse = ", "), "\n\n")

cat("Open formulas:\n")
cat("  Formula 1:", deparse(formula_open$f1), "\n")
cat("  Formula 2:", deparse(formula_open$f2), "\n")
cat("  Random vars:", paste(formula_open$random_vars, collapse = ", "), "\n\n")

# ---- Run comparison suite -----------------------------------------------
results <- list()

# Dense dataset models
cat("\n\n### DENSE DATASET MODELS ###\n")
results[[1]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  dependence_name = "Famoye (SML)",
  dependence_obj = "famoye",
  method = "sml"
)

results[[2]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  dependence_name = "Frank Copula (SML)",
  dependence_obj = copula("frank"),
  method = "sml"
)

results[[3]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  dependence_name = "Gaussian Copula (SML)",
  dependence_obj = copula("normal"),
  method = "sml"
)

results[[4]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  dependence_name = "Independence (SML)",
  dependence_obj = "independence",
  method = "sml"
)

# Open dataset models
cat("\n\n### OPEN DATASET MODELS ###\n")
results[[5]] <- fit_and_report(
  "open", data_open, formula_open$f1, formula_open$f2,
  dependence_name = "Famoye (SML)",
  dependence_obj = "famoye",
  method = "sml"
)

results[[6]] <- fit_and_report(
  "open", data_open, formula_open$f1, formula_open$f2,
  dependence_name = "Frank Copula (SML)",
  dependence_obj = copula("frank"),
  method = "sml"
)

results[[7]] <- fit_and_report(
  "open", data_open, formula_open$f1, formula_open$f2,
  dependence_name = "Clayton Copula (SML)",
  dependence_obj = copula("clayton"),
  method = "sml"
)

results[[8]] <- fit_and_report(
  "open", data_open, formula_open$f1, formula_open$f2,
  dependence_name = "Independence (SML)",
  dependence_obj = "independence",
  method = "sml"
)

# Open dataset with Laplace method
cat("\n\n### OPEN DATASET WITH LAPLACE APPROXIMATION ###\n")
results[[9]] <- fit_and_report(
  "open", data_open, formula_open$f1, formula_open$f2,
  random_1 = formula_open$random_vars,
  random_2 = formula_open$random_vars,
  dependence_name = "Famoye (Laplace)",
  dependence_obj = "famoye",
  method = "laplace"
)

# ---- Summary table -----------------------------------------------
cat("\n\n", paste(rep("=", 100), collapse = ""), "\n", sep = "")
cat("SUMMARY TABLE - ALL MODELS\n")
cat(paste(rep("=", 100), collapse = ""), "\n")

comparison_df <- do.call(rbind, results)
print(comparison_df, digits = 3)

# ---- Model comparison within each dataset -----------------------------------
cat("\n\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("WITHIN-DATASET MODEL COMPARISON (by AIC)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

for (dataset in c("dense", "open")) {
  cat("\n", dataset, " dataset:\n", sep = "")
  subset_df <- comparison_df[comparison_df$data_name == dataset, ]
  subset_df <- subset_df[order(subset_df$AIC, na.last = TRUE), ]
  print(subset_df[, c("model", "method", "logLik", "AIC", "BIC", "npar", "convergence")],
        digits = 3, row.names = FALSE)
}

cat("\n\nComparison suite complete.\n")
