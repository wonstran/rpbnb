#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() on the open-section truck-crash data, TMB engine only -- a trimmed
# version of inst/rpbnb_truck_open.R with the classic-engine side removed.
#
# rpbnb_truck_open.R runs BOTH engines from one rpbnb_control() object to
# demonstrate that the control object is a superset either engine can accept.
# This script keeps that same control object and the grouped `boundary_tests`
# switch ("sd" / "dispersion" / "dependence" / "all"), but fits only
# engine = "tmb", so there is no engine loop, no engine-comparison section,
# and the maxLik-only control knobs (se_method, hess_eps) are dropped since
# nothing here reads them.
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
# devtools::load_all() (not library()): the script must run against the
# current source tree.  Run from the package root:
#     Rscript inst/rpbnb_truck_open_v2.R
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

# ---- Knobs ------------------------------------------------------------------
n_cores <- 24L
draws   <- 1000L
seed    <- 20240712L
# Draw-chunk count for the TMB engine's SML tape (see ?rpbnb_control's
# tape_chunks and docs/TMB_SML_large_draws_OOM_guide.md). At draws = 1000 this
# script exists specifically to exercise draw chunking: pinned explicitly
# because max_workload is disabled below (its calibration doesn't fit this
# data), so the auto-chunking resolver would otherwise never engage and this
# would build one full [n x 1000] tape instead. NULL falls back to the
# resolver's C = 1 default under max_workload = Inf -- i.e. no chunking.
tape_chunks <- 10L
# "sml" (simulated ML) or "laplace" (TMB's memory-saving alternative).
tmb_method <- "sml"
# Same knob as rpbnb_truck.R / rpbnb_truck_tmb.R: "famoye" or a copula()
# object.  Note copula("normal") is capped at one thread unless
# force_parallel_gaussian = TRUE (a known SIGSEGV in the registered atomic),
# so a Gaussian run here will be markedly slower.
dependence <- copula("normal") #copula("kimeldorf") 
# Which groups to LR-test.  "all" = c("sd", "dispersion", "dependence").
# FALSE or "none" skips them entirely.
boundary_tests <- FALSE #"all"
# Draws for the restricted refits.  NULL reuses the main fit's `draws`, which
# is what keeps the restricted and full simulated likelihoods on common
# random numbers -- diverge from it only deliberately.
boundary_draws <- NULL
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
cat("=== rpbnb(engine = \"tmb\") on truck all crashes, open sections (",
    dep_label, ") ===\n", sep = "")
cat("Observations   :", nrow(data), "\n")
cat("Cores asked    :", n_cores, "\n")
cat("Draws          :", draws, "\n")
cat("Tape chunks    :",
    if (is.null(tape_chunks)) "NULL (auto; C = 1 under max_workload = Inf)"
    else tape_chunks, "\n")
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
  # per-draw cost (see inst/rpbnb_truck_tmb.R's header), so leaving it engaged
  # would block the fit rather than warn. tape_chunks (set above) stands in
  # for it: at draws = 1000 this pins the layout directly instead of relying
  # on a workload estimate that is Inf (disabled) for exactly that reason.
  max_workload = Inf,
  tape_chunks  = tape_chunks,
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
    random_1       = c("SR40_MI3", "MPD_ME"),
    random_2       = c("SR40_MI3"),
    dependence     = dependence,
    seed           = seed,
    draws          = draws,
    standardize    = TRUE,
    boundary_tests = boundary_tests,
    force_parallel_gaussian = TRUE,
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
