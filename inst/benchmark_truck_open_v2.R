#!/usr/bin/env Rscript
# Benchmark harness for rpbnb_truck_open_v2.R
# Sweeps tmb_method = c("sml","laplace") x dependence = c("frank","kimeldorf","normal")
# Mirrors inst/rpbnb_truck_open_v2.R exactly -- does NOT edit that file.
# Same f1/f2, random_1/random_2, draws/seed/standardize/boundary_tests/control.

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

n_cores        <- 16L
draws          <- 300L
seed           <- 20240712L
boundary_tests <- "all"
boundary_draws <- NULL
max_rows       <- NULL   # NULL = full 2321 rows; set to e.g. 400L for smoke

data <- read.csv(file.path("inst", "extdata", "export_open_all.csv"))
if (!is.null(max_rows)) {
  data <- utils::head(data, max_rows)
  cat("*** SMOKE TEST: using first", max_rows, "rows only ***\n")
}
cat("Benchmark data rows:", nrow(data), "\n")

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
cat("Control:\n"); print(ctrl, engine = "tmb")

configs <- expand.grid(
  tmb_method = c("sml", "laplace"),
  dependence = c("frank", "kimeldorf", "normal"),
  stringsAsFactors = FALSE
)
# Stable order: sml first then laplace, frank/kimeldorf/normal within each
configs <- configs[order(match(configs$tmb_method, c("sml","laplace")),
                         match(configs$dependence, c("frank","kimeldorf","normal"))), ]

dir.create("results", showWarnings = FALSE, recursive = TRUE)
stamp_global <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
summary_path <- file.path("results", paste0("benchmark_open_v2_summary_", stamp_global, ".rds"))
log_path     <- file.path("results", paste0("benchmark_open_v2_log_", stamp_global, ".txt"))
cat("Global stamp:", stamp_global, "\n")
cat("Configs:\n"); print(configs)

sink(log_path, split = TRUE)

results <- vector("list", nrow(configs))
names(results) <- paste0(configs$tmb_method, "_", configs$dependence)

for (i in seq_len(nrow(configs))) {
  tmb_method <- configs$tmb_method[i]
  dep_fam    <- configs$dependence[i]
  dependence <- copula(dep_fam)
  label      <- paste0(tmb_method, " + ", dep_fam)
  cat("\n", paste(rep("=", 72), collapse=""), "\n", sep="")
  cat(sprintf("[%d/%d] %s  (dependence=%s, method=%s)\n", i, nrow(configs), label, dep_fam, tmb_method))
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
        control        = ctrl
      ),
      error = function(e) { cat("ERROR:", conditionMessage(e), "\n"); e }
    )
  })[["elapsed"]]

  is_err <- inherits(fit, "error")
  if (is_err) {
    cat(sprintf("FAILED in %.1fs: %s\n", t_elapsed, conditionMessage(fit)))
    results[[i]] <- list(
      config = configs[i,], label = label, tmb_method = tmb_method, dep_fam = dep_fam,
      elapsed = t_elapsed, error = conditionMessage(fit), fit = NULL
    )
  } else {
    fit$benchmark_elapsed <- t_elapsed
    fit$benchmark_config  <- configs[i,]
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
    # Also save individual summary snapshot
    results[[i]] <- list(
      config = configs[i,], label = label, tmb_method = tmb_method, dep_fam = dep_fam,
      elapsed = t_elapsed, error = NULL, fit_path = fit_path, fit = fit
    )
    # Incremental save of summary list
    saveRDS(results, summary_path)
    cat(sprintf("Incremental summary saved: %s\n", summary_path))
  }
  flush.console()
  gc()
}

sink()
cat("\nAll configs done. Summary:", summary_path, "\n")
cat("Log:", log_path, "\n")
# Final save (fits stripped for size? keep fit_path refs)
saveRDS(results, summary_path)
cat("DONE\n")
