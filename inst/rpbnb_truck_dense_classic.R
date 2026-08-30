#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() on the open-section truck-crash data, classic engine only -- a
# trimmed version of inst/rpbnb_truck_open.R with the TMB-engine side removed.
#
# rpbnb_truck_open.R runs BOTH engines from one rpbnb_control() object to
# demonstrate that the control object is a superset either engine can accept.
# This script keeps that same control object and the grouped `boundary_tests`
# switch ("sd" / "dispersion" / "dependence" / "all"), but fits only
# engine = "classic", so there is no engine loop, no engine-comparison section,
# and the TMB-only control knobs (gradtol, restarts, max_threads) are dropped
# since nothing here reads them.
#
# rpbnb_frank_open.R's header explains WHY this data needs standardizing:
# SR40_MI3 and MPD_ME are strictly positive and bounded away from zero, so as
# random-coefficient carriers x * (b + sd * u_i) is a random INTERCEPT in
# disguise unless centred; and IRI_ME (29-380) next to 0/1 indicators gives a
# design-matrix condition number near 1e7 unless scaled.  standardize = TRUE
# handles both and back-transforms the printed table to original units.
#
# COST.  One full fit plus one restricted refit per tested parameter: 3
# random-coefficient scales + 2 dispersions + 1 dependence = 6 refits, so
# roughly seven full fits total.  Drop `boundary_tests` to
# c("dispersion", "dependence") for a much cheaper run, or set `max_rows`
# below for a smoke test.
#
# Run from the package root:
#     Rscript inst/rpbnb_truck_dense_classic.R
# =============================================================================

library(rpbnb)

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")

# ---- Knobs ------------------------------------------------------------------
n_cores <- 31L
draws   <- 300L
seed    <- 20240712L
# "famoye" or a copula() object.  The classic engine has no method/estimator
# knob (always maxLik BFGS with Halton draws) and no Gaussian-thread cap, so
# the TMB-specific knobs from rpbnb_truck_open_v2.R (tmb_method,
# force_parallel_gaussian, gradtol, restarts, max_threads, ...) do not apply.
dependence <- copula("frank")
# Which groups to LR-test.  "all" = c("sd", "dispersion", "dependence").
# FALSE or "none" skips them entirely.
boundary_tests <- c("sd")
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

# Same formulas and random-coefficient specification as rpbnb_truck.R: random
# slopes on SR40_MI3 and MPD_ME in eq 1, SR40_MI3 only in eq 2 (weakly
# identified there; kept to match the other truck scripts so all of them fit
# the same model).
f1 <- ALL_3  ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + G_ABG2 + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME + RUT_L
f2 <- C_HV ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME

cat("Equation 1     :", deparse(f1), "\n")
cat("Equation 2     :", deparse(f2), "\n")

# ---- Control object (classic-engine knobs) ----------------------------------
ctrl <- rpbnb_control(
  print_level  = 1,          # also un-silences the per-refit LR-test messages
  n_cores      = n_cores,
  reltol       = 1e-8,
  halton_burn  = 300L,
  # The default workload guard's calibration under-estimates this data's
  # per-draw cost (see inst/rpbnb_truck_tmb.R's header), so leaving it engaged
  # would block the fit rather than warn.
  max_workload = Inf,
  parallel_tape = FALSE
)

sep(); cat("CONTROL OBJECT\n"); sep()
print(ctrl, engine = "classic")

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
    random_1       = c("SR40_MI3", "MPD_ME"),
    random_2       = c("SR40_MI3"),
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

fit_path <- file.path("results", paste0("fit_truck_open_tmb_", stamp, ".rds"))
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
