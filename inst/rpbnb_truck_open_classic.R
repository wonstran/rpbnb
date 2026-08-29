#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() on the open-section truck-crash data, classic engine with the
# simulated-ML (sml) estimator -- a trimmed version of inst/rpbnb_truck_open.R.
#
# rpbnb_truck_open.R runs BOTH engines from one rpbnb_control() object to
# demonstrate that the control object is a superset either engine can accept.
# This script keeps that grouped `boundary_tests` switch ("sd" / "dispersion"
# / "dependence" / "all") but fits only engine = "classic" (simulated ML,
# maxLik BFGS), so there is no engine loop or engine-comparison section.
#
# NOTE (rpbnb >= 0.4.2): method = "BFGS" is the only implemented optimizer
# knob of rpbnb_control(), so the old method = "sml" line was dropped; the
# classic engine is simulated ML (maxLik BFGS) regardless. boundary_draws is
# TMB-engine only and is not passed to the classic fit; the TMB-only control
# knobs stay on the object but are reported as ignored in fit$control_ignored.
#
# rpbnb_frank_open.R's header explains WHY this data needs standardizing:
# SR40_MI3 and MPD_ME are strictly positive and bounded away from zero, so as
# random-coefficient carriers x * (b + sd * u_i) is a random INTERCEPT in
# disguise unless centred; and IRI_ME (29-380) next to 0/1 indicators gives a
# design-matrix condition number near 1e7 unless scaled.  standardize = TRUE
# handles both and back-transforms the printed table to original units.
#
# COST.  The classic engine draws `draws` Halton points per evaluation, so
# this is the expensive path; one full fit plus one restricted refit per
# tested parameter.
# Drop `boundary_tests` to c("dispersion", "dependence") for a cheaper run, or
# set `max_rows` below for a smoke test.
#
# devtools::load_all() (not library()): the script must run against the
# current source tree.  Run from the package root:
#     Rscript inst/rpbnb_truck_open_classic.R
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

# ---- Knobs ------------------------------------------------------------------
n_cores <- 20L
draws   <- 1000L
seed    <- 20240712L
# "famoye" or a copula() object.
dependence <- copula("frank") #copula("normal") #copula("kimeldorf")
# Which groups to LR-test.  "all" = c("sd", "dispersion", "dependence").
# FALSE or "none" skips them entirely.
boundary_tests <- c("sd")
# Draws for the restricted refits.  NULL reuses the main fit's `draws`, which
# keeps the restricted and full simulated likelihoods on common random numbers.
boundary_draws <- NULL # TMB-engine only as of 0.4.2; not passed to the classic fit
# Smoke-test knob: NULL uses every row.  Set to e.g. 400L to rehearse the
# whole script cheaply before committing to a full run.
max_rows <- NULL

data <- read.csv("https://its.cutr.usf.edu/ftp/data/export_open_all.csv")
if (!is.null(max_rows)) {
  data <- utils::head(data, max_rows)
  cat("*** SMOKE TEST: using the first", max_rows, "rows only ***\n")
}

is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else as.character(dependence)
cat("=== rpbnb(engine = \"classic\") on truck all crashes, open sections (",
    dep_label, ") ===\n", sep = "")
cat("Observations   :", nrow(data), "\n")
cat("Cores asked    :", n_cores, "\n")
cat("Draws          :", draws, "\n")
cat("LR test groups :",
    if (isFALSE(boundary_tests)) "none" else paste(boundary_tests, collapse = ", "),
    "\n")

# Random slope on SR40_MI3 in eq 1 only -- NOT also in eq 2. The same covariate
# random in both equations of a copula-linked bivariate model is weakly
# identified, and on this dataset it is not just "weak": empirically, adding
# random_2 = "SR40_MI3" sends sd1:SR40_MI3 to a runaway boundary value (order
# 1e3 at draws = 300, order 1e4 at draws = 1000) instead of the well-behaved
# ~0.01 this single-equation spec converges to. A small subset of the data does
# NOT reproduce this -- it only shows up at the full sample size -- so do not
# reintroduce random_2 here without re-checking sd1 for a boundary run-away.
f1 <- ALL_3  ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + G_ABG2 + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME + RUT_L
f2 <- C_HV ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME

cat("Equation 1     :", deparse(f1), "\n")
cat("Equation 2     :", deparse(f2), "\n")

# ---- Control object (classic-engine knobs only) ------------------------------
ctrl <- rpbnb_control(
  # method = "BFGS" is the only implemented optimizer (the default), and the
  # classic engine is simulated ML (maxLik BFGS) regardless -- no method knob.
  print_level  = 1,            # also un-silences the per-refit LR-test messages
  n_cores      = n_cores,
  reltol       = 1e-8,
  halton_burn  = 300L,         # discard the first Halton points per draw
  se_method    = "opg",       # BHHH information: fastest SE path (unreliable at boundary)
  # TMB-only knobs (gradtol, restarts, max_threads, max_workload,
  # parallel_tape): the classic engine does not read them; they are listed
  # in fit$control_ignored, which the script prints after the fit.
  gradtol      = 1e-5,
  restarts     = 10L,
  max_threads  = n_cores,
  max_workload = Inf,
  parallel_tape = FALSE
)

sep(); cat("CONTROL OBJECT\n"); sep()
print(ctrl)

# ---- Fit ----------------------------------------------------------------
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
dir.create("results", recursive = TRUE, showWarnings = FALSE)

sep(); cat("FIT: engine = \"classic\"\n", sep = ""); sep()

t_fit <- system.time(
  fit <- rpbnb(
    formula_1      = f1,
    formula_2      = f2,
    data           = data,
    engine         = "classic",
    random_1       = c("SR40_MI3"),
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

fit_path <- file.path("results", paste0("fit_truck_open_classic_", stamp, ".rds"))
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
# print() explicitly: summary() returns invisibly, so a bare call is not
# auto-printed when the script runs from the RStudio console.
print(summary(fit))
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
