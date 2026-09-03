#!/usr/bin/env Rscript
# =============================================================================
# Compare TMB vs non-TMB engines for rpbnb
# Fits same model with both engines to benchmark timing and verify agreement
# Uses export_dense_all.csv and export_open_all.csv
# =============================================================================

devtools::load_all()

library(rpbnb)

# ---- Helper function --------------------------------------------------------
fit_and_report <- function(data_name, data, formula_1, formula_2,
                          random_1 = NULL, random_2 = NULL,
                          engine_name, engine_func, ...) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
  cat(sprintf("%s | Data: %s\n", engine_name, data_name))
  cat(paste(rep("=", 80), collapse = ""), "\n")

  tryCatch({
    t_fit <- system.time(
      fit <- do.call(engine_func, c(
        list(formula_1 = formula_1, formula_2 = formula_2, data = data,
             random_1 = random_1, random_2 = random_2),
        list(...)
      ))
    )[["elapsed"]]

    result <- data.frame(
      data = data_name,
      engine = engine_name,
      n_obs = nrow(data),
      time_sec = t_fit,
      logLik = fit$logLik,
      AIC = if (is.null(fit$AIC)) AIC(fit) else fit$AIC,
      BIC = if (is.null(fit$BIC)) BIC(fit) else fit$BIC,
      npar = fit$npar,
      convergence = fit$convergence,
      row.names = NULL
    )

    cat(sprintf("Time: %.1f s | logLik: %.2f | AIC: %.2f | BIC: %.2f | npar: %d\n",
                t_fit, fit$logLik, result$AIC, result$BIC, fit$npar))

    return(result)
  }, error = function(e) {
    cat("ERROR:", e$message, "\n")
    return(data.frame(
      data = data_name,
      engine = engine_name,
      n_obs = nrow(data),
      time_sec = NA_real_,
      logLik = NA_real_,
      AIC = NA_real_,
      BIC = NA_real_,
      npar = NA_integer_,
      convergence = NA_integer_,
      row.names = NULL
    ))
  })
}

# ---- Load both datasets -----------------------------------------------------
data_dense <- read.csv(system.file(
  "extdata", "export_dense_all.csv",
  package = "rpbnb", mustWork = TRUE
))

data_open <- read.csv(system.file(
  "extdata", "export_open_all.csv",
  package = "rpbnb", mustWork = TRUE
))

cat("=== TMB vs Non-TMB Engine Comparison ===\n")
cat("Dense dataset observations:", nrow(data_dense), "\n")
cat("Open dataset observations:", nrow(data_open), "\n\n")

# ---- Setup formulas -----------------------------------------------
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

# Dense dataset: Non-TMB Famoye
cat("\n### DENSE DATASET: FAMOYE DEPENDENCE ###\n")
results[[1]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  random_1 = formula_dense$random_vars[1],
  random_2 = formula_dense$random_vars[1],
  engine_name = "Non-TMB (BFGS)",
  engine_func = fit_rpbnb,
  dependence = "famoye",
  draws = 300,
  seed = 20260809,
  control = rpbnb_control(n_cores = parallel::detectCores() - 2, compute_se = TRUE)
)

# Dense dataset: TMB Famoye
results[[2]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  random_1 = formula_dense$random_vars[1],
  random_2 = formula_dense$random_vars[1],
  engine_name = "TMB (SML)",
  engine_func = fit_rpbnb_tmb,
  dependence = "famoye",
  draws = 300,
  seed = 20260809,
  method = "sml",
  control = rpbnb_tmb_control(n_cores = parallel::detectCores() - 2)
)

# Open dataset: Non-TMB Famoye
cat("\n### OPEN DATASET: FAMOYE DEPENDENCE ###\n")
results[[3]] <- fit_and_report(
  "open", data_open, formula_open$f1, formula_open$f2,
  random_1 = formula_open$random_vars[1],
  random_2 = formula_open$random_vars[1],
  engine_name = "Non-TMB (BFGS)",
  engine_func = fit_rpbnb,
  dependence = "famoye",
  draws = 300,
  seed = 20260809,
  control = rpbnb_control(n_cores = parallel::detectCores() - 2, compute_se = TRUE)
)

# Open dataset: TMB Famoye
results[[4]] <- fit_and_report(
  "open", data_open, formula_open$f1, formula_open$f2,
  random_1 = formula_open$random_vars[1],
  random_2 = formula_open$random_vars[1],
  engine_name = "TMB (SML)",
  engine_func = fit_rpbnb_tmb,
  dependence = "famoye",
  draws = 300,
  seed = 20260809,
  method = "sml",
  control = rpbnb_tmb_control(n_cores = parallel::detectCores() - 2)
)

# Dense dataset: Non-TMB Frank copula
cat("\n### DENSE DATASET: FRANK COPULA ###\n")
results[[5]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  random_1 = NULL,
  random_2 = NULL,
  engine_name = "Non-TMB (BFGS)",
  engine_func = fit_rpbnb,
  dependence = copula("frank"),
  draws = 250,
  seed = 20260809,
  control = rpbnb_control(n_cores = parallel::detectCores() - 2, compute_se = TRUE)
)

# Dense dataset: TMB Frank copula
results[[6]] <- fit_and_report(
  "dense", data_dense, formula_dense$f1, formula_dense$f2,
  random_1 = NULL,
  random_2 = NULL,
  engine_name = "TMB (SML)",
  engine_func = fit_rpbnb_tmb,
  dependence = copula("frank"),
  draws = 250,
  seed = 20260809,
  method = "sml",
  control = rpbnb_tmb_control(n_cores = parallel::detectCores() - 2)
)

# ---- Summary table -----------------------------------------------
cat("\n\n", paste(rep("=", 110), collapse = ""), "\n", sep = "")
cat("SUMMARY TABLE - TMB VS NON-TMB COMPARISON\n")
cat(paste(rep("=", 110), collapse = ""), "\n")

comparison_df <- do.call(rbind, results)
print(comparison_df, digits = 3)

# ---- Speedup analysis -----------------------------------------------
cat("\n\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
cat("SPEEDUP ANALYSIS (TMB time / Non-TMB time)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

for (dataset in c("dense", "open")) {
  for (model in c("Famoye", "Frank")) {
    subset_df <- comparison_df[(comparison_df$data == dataset), ]

    if (model == "Famoye") {
      rows <- subset_df[1:2, ]
    } else if (model == "Frank") {
      rows <- subset_df[5:6, ]
    } else {
      next
    }

    if (nrow(rows) == 2 && !is.na(rows$time_sec[1]) && !is.na(rows$time_sec[2])) {
      speedup <- rows$time_sec[1] / rows$time_sec[2]
      cat(sprintf("\n%s | %s: Non-TMB %.1f s, TMB %.1f s → %.2fx speedup\n",
                  dataset, model, rows$time_sec[1], rows$time_sec[2], speedup))
    }
  }
}

cat("\nNote: Speedup < 1 means TMB is slower; > 1 means TMB is faster\n")
cat("      Both engines should produce similar logLik values (within numerical precision)\n")
