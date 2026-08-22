#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() on the DENSE-section truck-crash data, run through BOTH engines from
# ONE control object -- the dense-data counterpart of inst/rpbnb_truck_open.R.
#
# Three things distinguish this script from rpbnb_truck_open.R:
#
#   1. DATA.  inst/extdata/export_dense_all.csv (dense sections) rather than
#      export_open_all.csv (open sections).
#
#   2. FORMULAS.  Taken from inst/tmb_truck_rpbnb_diff_famoye_dense.R: the two
#      equations use DIFFERENT covariate sets (ALL_3 in eq 1, C_DISTR in
#      eq 2) and DIFFERENT random coefficients per equation (SR40_MI3,
#      AUXLNUM, MPD_ME, MPD_STD in eq 1; SR40_MI3, RUT_9, MPD_ME, MPD_STD in
#      eq 2) -- four random slopes per equation, all on continuous
#      regressors, none shared with the open-section scripts.
#
#   3. DEPENDENCE.  Famoye/Sarmanov (dependence = "famoye"), matching the
#      referenced dense script, rather than a copula.
#
# Everything else -- ONE rpbnb_control() object handed unchanged to both
# engines, `boundary_tests` as a grouped LR-test switch ("sd" / "dispersion"
# / "dependence" / "all"), standardize = TRUE with automatic back-transform,
# and the side-by-side engine comparison at the end -- is exactly
# rpbnb_truck_open.R's pattern; see that script's header for the full
# rationale behind the unified control object and the boundary-test groups.
#
# COST.  Eight random-coefficient scales (4 + 4) plus 2 dispersions plus 1
# dependence parameter = 11 restricted refits per engine if boundary_tests =
# "all", i.e. roughly twelve full fits per engine. Narrow `boundary_tests`
# below, or set `max_rows` for a smoke test, before running the full thing.
#
# devtools::load_all() (not library()): the script must run against the
# current source tree. Run from the package root:
#     Rscript inst/rpbnb_truck_dense.R
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

# ---- Knobs ------------------------------------------------------------------
engines <- c("classic", "tmb")
n_cores <- 12L
draws   <- 300L
seed    <- 20240712L
# TMB-only. "sml" keeps the ESTIMATOR fixed across the two engines so the
# comparison at the bottom varies only the engine.
tmb_method <- "sml"
# Famoye/Sarmanov, matching inst/tmb_truck_rpbnb_diff_famoye_dense.R.
dependence <- "famoye"
# Which groups to LR-test. "all" = c("sd", "dispersion", "dependence").
# FALSE or "none" skips them entirely. Set to e.g. c("dispersion",
# "dependence") for a much cheaper run (8 fewer refits: the sd group here has
# 8 random-coefficient scales, not 3 as in rpbnb_truck_open.R).
boundary_tests <- "all"
# TMB-only: draws for the restricted refits. NULL reuses the main fit's
# `draws`.
boundary_draws <- NULL
# Smoke-test knob: NULL uses every row. Set to e.g. 400L to rehearse the
# whole script cheaply before committing to a full run.
max_rows <- NULL

data <- read.csv(file.path("inst", "extdata", "export_dense_all.csv"))
if (!is.null(max_rows)) {
  data <- utils::head(data, max_rows)
  cat("*** SMOKE TEST: using the first", max_rows, "rows only ***\n")
}

is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else as.character(dependence)
cat("=== rpbnb() on truck all crashes, dense sections (", dep_label, ") ===\n",
    sep = "")
cat("Observations   :", nrow(data), "\n")
cat("Engines        :", paste(engines, collapse = ", "), "\n")
cat("Cores asked    :", n_cores, "\n")
cat("Draws          :", draws, "\n")
cat("LR test groups :",
    if (isFALSE(boundary_tests)) "none" else paste(boundary_tests, collapse = ", "),
    "\n")

# Different covariate sets AND different random coefficients per equation --
# see inst/tmb_truck_rpbnb_diff_famoye_dense.R's header for why (random
# coefficients sit on continuous regressors only, each must appear in its own
# equation's formula).
f1 <- ALL_3   ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + IRI_ME + RUT_L + SP50GE + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP10_ME
f2 <- C_DISTR ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + RUT_9 + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP01_ME + CS_MINAB

cat("Equation 1     :", deparse(f1), "\n")
cat("Equation 2     :", deparse(f2), "\n")

random_1 <- c("SR40_MI3")
random_2 <- c("SR40_MI3")

# ---- One control object for both engines ------------------------------------
# Deliberately a superset of what either engine reads (see
# rpbnb_truck_open.R's header for the full rationale). `max_workload` is
# raised past its default calibration for the same reason
# tmb_truck_rpbnb_diff_famoye_dense.R raises it: this data's per-draw cost
# under-runs the guard's default estimate.
ctrl <- rpbnb_control(
  # --- read by both engines ---
  print_level  = 1,
  n_cores      = n_cores,
  reltol       = 1e-8,
  halton_burn  = 300L,
  # --- maxLik side only (ignored, and reported as ignored, by the TMB fit) ---
  se_method    = "opg",
  hess_eps     = 1e-5,
  # --- TMB side only (ignored, and reported as ignored, by the classic fit) ---
  gradtol      = 1e-5,
  restarts     = 10L,
  max_threads  = n_cores,
  max_workload = Inf,
  parallel_tape = FALSE
)

sep(); cat("CONTROL OBJECT (one object, both engines)\n"); sep()
print(ctrl)
cat("\nEach fit below names the fields IT did not read, in its own\n")
cat("print()/summary() -- nothing is dropped silently.\n")

# ---- Fit both engines -------------------------------------------------------
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
dir.create("results", recursive = TRUE, showWarnings = FALSE)
fits    <- list()
elapsed <- numeric(0)

for (eng in engines) {
  sep(); cat("FIT: engine = \"", eng, "\"\n", sep = ""); sep()

  extra <- if (eng == "tmb") {
    list(method = tmb_method, boundary_draws = boundary_draws)
  } else {
    list()
  }

  t_fit <- system.time(
    fit <- do.call(rpbnb, c(list(
      formula_1      = f1,
      formula_2      = f2,
      data           = data,
      engine         = eng,
      random_1       = random_1,
      random_2       = random_2,
      dependence     = dependence,
      seed           = seed,
      draws          = draws,
      standardize    = TRUE,
      boundary_tests = boundary_tests,
      control        = ctrl
    ), extra))
  )[["elapsed"]]

  fits[[eng]]    <- fit
  elapsed[[eng]] <- t_fit

  cat(sprintf("\nFinished in %.2f s%s\n", t_fit,
              if (isFALSE(boundary_tests) || !length(boundary_tests)) ""
              else " (includes the restricted LR refits)"))
  if (!is.null(fit$convergence)) {
    cat(sprintf("Convergence  : code=%d, %s (iterations=%d)\n",
                fit$convergence$code, fit$convergence$message,
                fit$convergence$iterations))
  } else {
    cat(sprintf("Convergence  : nlminb code=%d, %s\n",
                fit$optimizer$convergence, fit$optimizer$message))
  }

  fit_path <- file.path("results",
                        paste0("fit_truck_dense_", eng, "_", stamp, ".rds"))
  saveRDS(fit, fit_path)
  cat("Saved to     :", fit_path, "\n")

  cat("Control settings this engine did not read: ",
      if (length(fit$control_ignored)) paste(fit$control_ignored, collapse = ", ")
      else "(none)", "\n", sep = "")
}

# ---- Summaries --------------------------------------------------------------
show_summary <- function(fit) {
  if (inherits(fit, "rpbnb_tmb_fit")) summary(fit) else print(summary(fit))
  invisible(NULL)
}
for (eng in engines) {
  sep(); cat("MODEL SUMMARY -- engine = \"", eng,
             "\" (original covariate units)\n", sep = ""); sep()
  show_summary(fits[[eng]])
  cat("\n")
}

# ---- LR tests, standalone ---------------------------------------------------
for (eng in engines) {
  bt <- fits[[eng]]$boundary_tests
  sep(); cat("LR TESTS -- engine = \"", eng, "\"\n", sep = ""); sep()
  if (is.null(bt)) {
    cat("Skipped: `boundary_tests` named no group.  Set it at the top of this\n")
    cat("script -- \"all\", or e.g. c(\"dispersion\", \"dependence\") for the\n")
    cat("cheap subset -- to run them.\n")
  } else {
    print(bt)
  }
  cat("\n")
}

# ---- Engine comparison ------------------------------------------------------
if (length(engines) > 1L) {
  sep(); cat("ENGINE COMPARISON\n"); sep()

  fit_row <- function(eng) {
    f <- fits[[eng]]
    data.frame(engine = eng,
               logLik = as.numeric(stats::logLik(f)),
               npar = f$npar,
               AIC = AIC(f), BIC = BIC(f),
               seconds = round(elapsed[[eng]], 1),
               row.names = NULL)
  }
  print(do.call(rbind, lapply(engines, fit_row)), row.names = FALSE)

  bts <- lapply(fits, `[[`, "boundary_tests")
  if (all(vapply(bts, Negate(is.null), logical(1)))) {
    cat("\nLR statistic and p-value by parameter:\n")
    params <- unique(unlist(lapply(bts, `[[`, "Parameter")))
    pick <- function(eng, col) {
      b <- bts[[eng]]
      b[[col]][match(params, b$Parameter)]
    }
    cmp <- data.frame(Parameter = params, stringsAsFactors = FALSE)
    for (eng in engines) {
      cmp[[paste0("LR.", eng)]] <- round(pick(eng, "LR"), 3)
      cmp[[paste0("p.", eng)]]  <- signif(pick(eng, "p.value"), 3)
    }
    print(cmp, row.names = FALSE)
    cat("\nA blank cell means that engine produced no row for the parameter\n")
    cat("(different scale label, or a restricted refit that did not converge).\n")
  }
}

sep(); cat("DONE\n"); sep()
