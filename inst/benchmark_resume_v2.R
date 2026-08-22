#!/usr/bin/env Rscript
# Resume benchmark_open_v2 from checkpoint 2026-08-20-230615
# Only runs the 4 pending configs; does not touch the 2 completed fits.

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

n_cores        <- 16L
draws          <- 300L
seed           <- 20240712L
boundary_tests <- "all"
boundary_draws <- NULL

data <- read.csv(file.path("inst", "extdata", "export_open_all.csv"))
cat("Resume benchmark data rows:", nrow(data), "\n")

f1 <- ALL_3 ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + G_ABG2 + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME + RUT_L
f2 <- C_HV  ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME

ctrl <- rpbnb_control(
  print_level   = 1,
  n_cores       = n_cores,
  reltol        = 1e-8,
  halton_burn   = 300L,
  gradtol       = 1e-5,
  restarts      = 10L,
  max_threads   = n_cores,
  max_workload  = Inf,
  parallel_tape = FALSE
)

summary_path <- "results/benchmark_open_v2_summary_2026-08-20-230615.rds"
results <- readRDS(summary_path)
cat("Loaded checkpoint:", summary_path, "\n")
cat("Status before resume:\n")
for (i in seq_along(results)) {
  cat(sprintf("  %s: elapsed=%s fit=%s error=%s\n",
              names(results)[i],
              if(is.null(results[[i]]$elapsed) || is.na(results[[i]]$elapsed)) "NA" else sprintf("%.1f", results[[i]]$elapsed),
              !is.null(results[[i]]$fit),
              if(is.null(results[[i]]$error)) "none" else results[[i]]$error))
}

# which to run: fit is NULL and no error (pending), OR explicitly failed normal could be retried
# sml_normal/laplace_* are pending (elapsed is NULL/NA)
pending_idx <- which(sapply(results, function(x) is.null(x$fit)))
cat("Pending indices:", paste(pending_idx, collapse=", "), " -> ", paste(names(results)[pending_idx], collapse=", "), "\n")

for (i in pending_idx) {
  tmb_method <- results[[i]]$tmb_method
  # fallback if stored tmb_method missing (should not happen)
  if (is.null(tmb_method)) tmb_method <- strsplit(names(results)[i], "_")[[1]][1]
  dep_fam <- results[[i]]$dep_fam
  if (is.null(dep_fam)) dep_fam <- sub("^[a-z]+_", "", names(results)[i])
  dependence <- copula(dep_fam)
  label <- paste0(tmb_method, " + ", dep_fam)
  # Gaussian needs force_parallel to avoid 1-thread cap; harmless for other families
  use_force <- dep_fam == "normal"
  cat("\n", paste(rep("=", 72), collapse=""), "\n", sep="")
  cat(sprintf("[%d/%d] RESUME %s  (dependence=%s, method=%s, force_parallel_gaussian=%s)\n",
              i, length(results), label, dep_fam, tmb_method, use_force))
  cat(paste(rep("=", 72), collapse=""), "\n", sep="")
  flush.console()

  t_elapsed <- system.time({
    fit <- tryCatch(
      rpbnb(
        formula_1      = f1,
        formula_2      = f2,
        data           = data,
        engine         = "tmb",
        method         = tmb_method,
        boundary_draws = boundary_draws,
        random_1       = c("SR40_MI3", "MPD_ME"),
        random_2       = c("SR40_MI3"),
        dependence     = dependence,
        seed           = seed,
        draws          = draws,
        standardize    = TRUE,
        boundary_tests = boundary_tests,
        control        = ctrl,
        force_parallel_gaussian = use_force
      ),
      error = function(e) { cat("ERROR:", conditionMessage(e), "\n"); print(traceback()); e }
    )
  })[["elapsed"]]

  is_err <- inherits(fit, "error")
  if (is_err) {
    cat(sprintf("FAILED in %.1fs: %s\n", t_elapsed, conditionMessage(fit)))
    results[[i]]$elapsed <- t_elapsed
    results[[i]]$error <- conditionMessage(fit)
    results[[i]]$fit <- NULL
  } else {
    fit$benchmark_elapsed <- t_elapsed
    fit$benchmark_config <- results[[i]]$config
    stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
    fit_path <- file.path("results", sprintf("fit_benchmark_open_v2_%s_%s_%s.rds", tmb_method, dep_fam, stamp))
    saveRDS(fit, fit_path)
    cat(sprintf("Saved: %s\n", fit_path))
    cat(sprintf("Elapsed: %.1f s (%.1f min) | logLik=%.2f AIC=%.1f BIC=%.1f | conv=%d %s | max|grad|=%.2e | npar=%d\n",
                t_elapsed, t_elapsed/60,
                as.numeric(logLik(fit)), AIC(fit), BIC(fit),
                fit$optimizer$convergence, fit$optimizer$message,
                if (is.null(fit$optimizer$max_abs_gradient)) NA_real_ else fit$optimizer$max_abs_gradient,
                length(stats::coef(fit))))
    if (!is.null(fit$boundary_tests)) {
      cat("Boundary LR tests:\n"); print(fit$boundary_tests)
    }
    cat(sprintf("Control ignored: %s\n",
                if (length(fit$control_ignored)) paste(fit$control_ignored, collapse=", ") else "(none)"))
    results[[i]]$elapsed  <- t_elapsed
    results[[i]]$error    <- NULL
    results[[i]]$fit_path <- fit_path
    results[[i]]$fit      <- fit
  }
  saveRDS(results, summary_path)
  cat(sprintf("Checkpoint saved: %s\n", summary_path))
  flush.console()
  gc()
}

cat("\nAll pending configs done. Summary:", summary_path, "\n")
cat("DONE\n")
