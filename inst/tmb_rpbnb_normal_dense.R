library(rpbnb)

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("/home/wonstran/repos/truck")
# detectCores() reports the hardware CPU count and ignores OMP_NUM_THREADS, so
# on this box it returns 32 while OpenMP -- and therefore TMB -- will only ever
# grant 8.  Asking for 32 just produced a "using 8 supported threads" warning on
# every run; take the binding limit instead.
n_cores <- local({
  hardware <- parallel::detectCores()
  omp <- suppressWarnings(as.integer(Sys.getenv("OMP_NUM_THREADS")))
  if (is.na(omp) || omp < 1L) hardware else min(hardware, omp)
})
draws <- 500L
dependence <- copula("normal")
optimizer <- "laplace"

# Path components stay separate so the same call resolves on POSIX and Windows.
truck_data <- read.csv(file.path("data", "export_dense_all.csv"))

cat("=== RP-BNB on truck all crashes (Laplace) ===\n")
cat("Dependence   : copula(\"", dependence$family, "\")\n", sep = "")
cat("Cores asked  :", n_cores, "\n")
cat("Optimizer    :", optimizer, "\n")
cat("Draws (if SML):", draws, "\n\n")

f1 <- ALL_3  ~ LNAADT_3+SR40_MI3+MPD_ME+MPD_STD+IRI_ME+RUT_L+SP50GE+ACCPNTS+SIGNAL1+NEAR_SIG+AUXLNUM+DP10_ME
f2 <- C_HV ~ LNAADT_3+SR40_MI3+MPD_ME+MPD_STD+IRI_ME+RUT_7+RUT_9+SP50GE+ACCPNTS+SIGNAL1+NEAR_SIG+AUXLNUM+DP01_ME
r1 <- c("SR40_MI3")
r2 <- c("SR40_MI3")

cat("Equation 1   :", deparse(f1), "\n")
cat("Equation 2   :", deparse(f2), "\n")
cat("Random Parameters 1:", deparse(r1), "\n")
cat("Random Parameters 2:", deparse(r2), "\n\n")

t_fit <- system.time(
  fit <- fit_rpbnb_tmb(
    formula_1  = f1,
    formula_2  = f2,
    data       = truck_data,
    random_1   = r1,
    random_2   = r2,
    dependence = dependence,
    seed       = 20240712,
    method     = optimizer,
    draws = draws,
    # No max_workload override: the header's claim that the default suffices is
    # only true if the guard is actually left on.  This fit's workload is
    # nrow(data) * draws = 1.74e6, comfortably inside the default (which
    # rpbnb_tmb_max_workload() sizes from available memory, so it is not a fixed
    # number), so the guard costs nothing here and still catches an accidentally
    # oversized respecification.
    control    = rpbnb_tmb_control(
      print_level = 1,
      n_cores     = n_cores,
      max_threads = n_cores
    )
  )
)[["elapsed"]]

cat(sprintf("\nEstimation finished in %.2f s\n", t_fit))
# Deliberately not reporting gc() figures: the TMB tape lives on the C++ heap
# and is invisible to R's garbage collector, so gc() would understate exactly
# the quantity this script exists to test.  Watch the process working set
# externally (Task Manager, or inst/benchmark_memory.R) if a number is needed.
cat(sprintf(
  "TMB threads: requested=%d, realized=%d\n",
  fit$parallel$requested, fit$parallel$realized
))
cat(sprintf("Optimizer: code=%d, message=%s\n",
            fit$optimizer$convergence, fit$optimizer$message))
cat("sdreport positive-definite Hessian:",
    if (isTRUE(fit$sdreport$pdHess)) "yes" else "no", "\n")

# ---- Persist the fit ---------------------------------------------------
# This fit costs ~55 min.  Without it on disk every follow-up diagnostic means
# paying that again, so write it before any post-estimation step can fail.
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
fit_path <- file.path("results", paste0("fit_normal_dense_", stamp, ".rds"))
saveRDS(fit, fit_path)
cat("Fit object saved to:", fit_path, "\n")

# ---- Convergence diagnostics -------------------------------------------
# print_level = 1 buries these under thousands of TMB inner-iteration lines, so
# restate them loudly: a boundary-bound dispersion or a non-PD Hessian makes the
# standard errors below meaningless, and that must not be easy to scroll past.
diagnostics <- character(0)
if (!isTRUE(fit$sdreport$pdHess)) {
  diagnostics <- c(diagnostics,
                   "Hessian is not positive definite: standard errors are unreliable.")
}
if (length(fit$boundary_report)) {
  diagnostics <- c(diagnostics,
                   paste0("Parameters at a constraint bound: ",
                          paste(paste0(fit$boundary_report,
                                       " (", fit$boundary_sides, ")"),
                                collapse = ", ")))
}
if (length(diagnostics)) {
  sep(); cat("CONVERGENCE WARNINGS\n"); sep()
  cat(paste0("  * ", diagnostics, collapse = "\n"), "\n")
} else {
  cat("No boundary or Hessian warnings.\n")
}

# ---- Model summary -----------------------------------------------------
sep(); cat("MODEL SUMMARY\n"); sep()
model_summary_output <- capture.output(summary(fit))
cat(model_summary_output, sep = "\n")
cat("\n")

# ---- Fitted means (predict) ---------------------------------------------
sep(); cat("FITTED MEANS (predict) -- first 6 observations\n"); sep()
print(head(predict(fit)))

# ---- Dependence -----------------------------------------------------------
sep(); cat("DEPENDENCE (sdreport)\n"); sep()
if (!is.null(fit$sdreport)) print(summary(fit$sdreport, "report"))

# ---- Marginal effects (AME) ------------------------------------------------
sep(); cat("AVERAGE MARGINAL EFFECTS (AME)\n"); sep()
marginal_effects_output <- capture.output(
  marginal_effects <- rpbnb_tmb_marginal_effects(fit, which = "both")
)
cat(marginal_effects_output, sep = "\n")
cat("\n")

# ---- Elasticities / semi-elasticities (AME) --------------------------------
sep(); cat("ELASTICITIES / SEMI-ELASTICITIES (AME)\n"); sep()
elasticities_output <- capture.output(
  elasticities <- rpbnb_tmb_elasticities(fit, which = "both")
)
cat(elasticities_output, sep = "\n")
cat("\n")

# ---- Export requested results ----------------------------------------------
results_path <- rpbnb:::.write_truck_results_markdown(
  model_summary = model_summary_output,
  marginal_effects = marginal_effects_output,
  elasticities = elasticities_output,
  dependence = fit$dependence,
  method = fit$method
)
cat("Results written to:", results_path, "\n")
