#!/usr/bin/env Rscript
# Benchmark harness for rpbnb_truck_dense_v2.R
# Sweeps tmb_method = c("sml","laplace") x dependence = c("frank","kimeldorf","normal")
# Mirrors inst/rpbnb_truck_dense_v2.R exactly -- does NOT edit that file.
# Same f1/f2, random_1/random_2, draws/seed/standardize/boundary_tests/control.
#
# The only deliberate divergence from rpbnb_truck_dense_v2.R is that Gaussian
# copula runs pass force_parallel_gaussian = TRUE, matching the established
# benchmark_resume_v2.R precedent (the single-thread default cap is a known
# SIGSEGV in the registered atomic and is markedly slower).

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

n_cores        <- 12L            # as in rpbnb_truck_dense_v2.R (not the open harness's 16)
draws          <- 300
seed           <- 20240712
boundary_tests <- "all"
boundary_draws <- NULL
max_rows       <- NULL   # NULL = all 3487 dense rows; set to e.g. 400L for smoke

data <- read.csv(file.path("inst", "extdata", "export_dense_all.csv"))
if (!is.null(max_rows)) {
  data <- utils::head(data, max_rows)
  cat("*** SMOKE TEST: using first", max_rows, "rows only ***\n")
}
cat("Benchmark data rows:", nrow(data), "\n")

# --- Formulas / random coefficients from rpbnb_truck_dense_v2.R -------------
f1 <- ALL_3   ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + IRI_ME + RUT_L + SP50GE + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP10_ME
f2 <- C_DISTR ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + RUT_9 + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP01_ME + CS_MINAB

ctrl <- rpbnb_control(
  print_level    = 1,
  n_cores        = n_cores,
  reltol         = 1e-8,
  halton_burn    = 300,
  gradtol        = 1e-5,
  restarts       = 10,
  max_threads    = n_cores,
  max_workload   = Inf,
  parallel_tape  = FALSE
)
cat("Control:\n"); print(ctrl, engine = "tmb")

configs <- expand.grid(
  tmb_method = c("sml", "laplace"),
  dependence = c("frank", "kimeldorf", "normal"),
  stringsAsFactors = FALSE
)
configs <- configs[order(match(configs$tmb_method, c("sml","laplace")),
                         match(configs$dependence, c("frank","kimeldorf","normal"))), ]

dir.create("results", showWarnings = FALSE, recursive = TRUE)
stamp_global <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
summary_path <- file.path("results", paste0("benchmark_dense_v2_summary_", stamp_global, ".rds"))
log_path     <- file.path("results", paste0("benchmark_dense_v2_log_", stamp_global, ".txt"))
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
  use_force  <- dep_fam == "normal"
  cat("\n", paste(rep("=", 72), collapse=""), "\n", sep="")
  cat(sprintf("[%d/%d] %s  (dependence=%s, method=%s, force_parallel_gaussian=%s)\n",
              i, nrow(configs), label, dep_fam, tmb_method, use_force))
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
        random_1       = c("SR40_MI3"),
        random_2       = c("SR40_MI3"),
        dependence     = dependence,
        seed           = seed,
        draws          = draws,
        standardize    = TRUE,
        boundary_tests = boundary_tests,
        control        = ctrl,
        force_parallel_gaussian = use_force
      ),
      error = function(e) { cat("ERROR:", conditionMessage(e), "\n"); e }
    )
  })[["elapsed"]]

  is_err <- inherits(fit, "error")
  if (is_err) {
    cat(sprintf("FAILED in %.1fs: %s\n", t_elapsed, conditionMessage(fit)))
    results[[i]] <- list(config = configs[i,], label = label, tmb_method = tmb_method,
                         dep_fam = dep_fam, elapsed = t_elapsed,
                         error = conditionMessage(fit), fit = NULL)
  } else {
    fit$benchmark_elapsed <- t_elapsed
    fit$benchmark_config  <- configs[i,]
    stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
    fit_path <- file.path("results", sprintf("fit_benchmark_dense_v2_%s_%s_%s.rds", tmb_method, dep_fam, stamp))
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
    results[[i]] <- list(config = configs[i,], label = label, tmb_method = tmb_method,
                         dep_fam = dep_fam, elapsed = t_elapsed, error = NULL,
                         fit_path = fit_path, fit = fit)
  }
  saveRDS(results, summary_path)
  cat(sprintf("Checkpoint saved: %s\n", summary_path))
  cat("========================================\n")
  flush.console()
  gc()
}

sink()
cat("\nAll configs done. Summary:", summary_path, "\n")
cat("Log:", log_path, "\n")
saveRDS(results, summary_path)
cat("DONE\n")