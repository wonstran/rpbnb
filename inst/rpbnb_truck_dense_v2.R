#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() on the dense-section truck-crash data, TMB engine only -- a trimmed
# version of inst/rpbnb_truck_dense.R with the classic-engine side removed.
#
# rpbnb_truck_dense.R runs BOTH engines from one rpbnb_control() object to
# demonstrate that the control object is a superset either engine can accept.
# This script keeps that same dense model structure but fits only
# engine = "tmb", so there is no engine loop, no engine-comparison section,
# and the maxLik-only control knobs (se_method, hess_eps) are dropped since
# nothing here reads them.  It is the dense-data counterpart of
# inst/rpbnb_truck_open_v2.R; see that script's header for the trimming
# rationale.
#
# DATA and MODEL STRUCTURE are taken straight from rpbnb_truck_dense.R:
#
#   * inst/extdata/export_dense_all.csv (dense sections), not the open-section
#     export_open_all.csv.
#   * The two equations use DIFFERENT covariate sets -- ALL_3 in eq 1,
#     C_DISTR in eq 2 -- and one random coefficient on SR40_MI3 in each
#     equation (the same random coefficient, weakly identified, is shared as
#     in the open scripts so every truck script fits the same style of model).
#   * Dependence = "famoye" (Famoye/Sarmanov), matching the referenced dense
#     script, rather than a copula.
#
# standardize = TRUE (same as rpbnb_truck_dense.R) because the dense data
# carries the same strictly-positive continuous covariates (SR40_MI3, MPD_ME,
# MPD_STD, etc.) that would otherwise make a random-coefficient carrier a
# random intercept in disguise, and would inflate the design-matrix condition
# number.  The printed table is back-transformed to original units.
#
# COST.  One full fit plus one restricted refit per tested parameter: 2
# random-coefficient scales (SR40_MI3 in each equation) + 2 dispersions + 1
# dependence = 5 refits, so roughly six full fits total.  Drop `boundary_tests`
# to c("dispersion", "dependence") for a much cheaper run, or set `max_rows`
# below for a smoke test.
#
# devtools::load_all() (not library()): the script must run against the
# current source tree.  Run from the package root:
#     Rscript inst/rpbnb_truck_dense_v2.R
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

# ---- Knobs ------------------------------------------------------------------
n_cores <- 12L
draws   <- 300L
seed    <- 20240712L
# "sml" (simulated ML) or "laplace" (TMB's memory-saving alternative).
tmb_method <- "sml"
# Same knob as rpbnb_truck_dense.R: "famoye" or a copula() object.  Note
# copula("normal") is capped at one thread unless force_parallel_gaussian =
# TRUE (a known SIGSEGV in the registered atomic), so a Gaussian run here
# would be markedly slower.
dependence <- "famoye"
# Which groups to LR-test.  "all" = c("sd", "dispersion", "dependence").
# FALSE or "none" skips them entirely.
boundary_tests <- "all"
# Draws for the restricted refits.  NULL reuses the main fit's `draws`, which
# is what keeps the restricted and full simulated likelihoods on common
# random numbers -- diverge from it only deliberately.
boundary_draws <- NULL
# Smoke-test knob: NULL uses every row.  Set to e.g. 400L to rehearse the
# whole script cheaply before committing to a full run.
max_rows <- NULL

data <- read.csv(file.path("inst", "extdata", "export_dense_all.csv"))
if (!is.null(max_rows)) {
  data <- utils::head(data, max_rows)
  cat("*** SMOKE TEST: using the first", max_rows, "rows only ***\n")
}

is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else as.character(dependence)
cat("=== rpbnb(engine = \"tmb\") on truck all crashes, dense sections (",
    dep_label, ") ===\n", sep = "")
cat("Observations   :", nrow(data), "\n")
cat("Cores asked    :", n_cores, "\n")
cat("Draws          :", draws, "\n")
cat("LR test groups :",
    if (isFALSE(boundary_tests)) "none" else paste(boundary_tests, collapse = ", "),
    "\n")

# Formulas and random coefficients taken from rpbnb_truck_dense.R: different
# covariate sets per equation (ALL_3 vs C_DISTR), one random coefficient on
# SR40_MI3 in each equation.
f1 <- ALL_3   ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + IRI_ME + RUT_L + SP50GE + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP10_ME
f2 <- C_DISTR ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + RUT_9 + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP01_ME + CS_MINAB

cat("Equation 1     :", deparse(f1), "\n")
cat("Equation 2     :", deparse(f2), "\n")

random_1 <- c("SR40_MI3")
random_2 <- c("SR40_MI3")

# ---- Control object (TMB knobs only) -----------------------------------------
ctrl <- rpbnb_control(
  print_level  = 1,          # also un-silences the per-refit LR-test messages
  n_cores      = n_cores,
  reltol       = 1e-8,
  halton_burn  = 300L,
  gradtol      = 1e-5,
  restarts     = 10L,
  max_threads  = n_cores,
  # The default workload guard's calibration under-estimates this data's
  # per-draw cost (see inst/rpbnb_truck_dense.R's header), so leaving it
  # engaged would block the fit rather than warn.
  max_workload = Inf,
  parallel_tape = FALSE
)

sep(); cat("CONTROL OBJECT\n"); sep()
print(ctrl, engine = "tmb", method = tmb_method, draws = draws)

# ---- Fit ----------------------------------------------------------------
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
dir.create("results", recursive = TRUE, showWarnings = FALSE)

sep(); cat("FIT: engine = \"tmb\"\n", sep = ""); sep()

t_fit <- system.time(
  fit <- rpbnb(
    formula_1      = f1,
    formula_2      = f2,
    data           = data,
    engine         = "tmb",
    method         = tmb_method,
    boundary_draws = boundary_draws,
    random_1       = random_1,
    random_2       = random_2,
    dependence     = dependence,
    seed           = seed,
    draws          = draws,
    standardize    = TRUE,
    boundary_tests = boundary_tests,
    control        = ctrl
  )
)[["elapsed"]]

cat(sprintf("\nFinished in %.2f s%s\n", t_fit,
            if (isFALSE(boundary_tests) || !length(boundary_tests)) ""
            else " (includes the restricted LR refits)"))
cat(sprintf("Convergence  : nlminb code=%d, %s\n",
            fit$optimizer$convergence, fit$optimizer$message))

fit_path <- file.path("results", paste0("fit_truck_dense_tmb_", stamp, ".rds"))
saveRDS(fit, fit_path)
cat("Saved to     :", fit_path, "\n")

cat("Control settings this engine did not read: ",
    if (length(fit$control_ignored)) paste(fit$control_ignored, collapse = ", ")
    else "(none)", "\n", sep = "")

# ---- Summary ------------------------------------------------------------
# standardize = TRUE back-transforms the coefficient table to original units
# automatically.  With the LR tests attached, the natural-scale block carries
# LR/df/p for the scale and dispersion rows in place of the NA they would
# otherwise show, AND the dependence row shows its LR test instead of a Wald z.
sep(); cat("MODEL SUMMARY (original covariate units)\n"); sep()
summary(fit)
cat("\n")

# ---- LR tests, standalone -------------------------------------------------
sep(); cat("LR TESTS\n"); sep()
if (is.null(fit$boundary_tests)) {
  cat("Skipped: `boundary_tests` named no group.  Set it at the top of this\n")
  cat("script -- \"all\", or e.g. c(\"dispersion\", \"dependence\") for the\n")
  cat("cheap subset -- to run them.\n")
} else {
  print(fit$boundary_tests)
}

sep(); cat("DONE\n"); sep()