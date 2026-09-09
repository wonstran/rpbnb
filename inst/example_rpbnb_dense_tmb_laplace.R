#!/usr/bin/env Rscript
# =============================================================================
# rpbnb(engine = "tmb", method = "laplace") on the dense-section truck-crash data.
#
# The method = "laplace" twin of inst/example_rpbnb_dense_tmb_sml.R: the same
# dense model structure (same data, equations, and random coefficient) run
# through the TMB engine's Laplace approximation instead of its SML estimator.
# Laplace is the TMB engine's memory-saving analytic alternative to SML: it
# works over one latent vector per observation, so the tape -- and peak
# memory -- scale with nrow(data) alone rather than nrow(data) * draws (see
# ?fit_rpbnb_tmb's `method` argument). The two estimators approximate the same
# random-coefficient integral DIFFERENTLY; they agree asymptotically but need
# not agree closely on any given dataset, so this is not a drop-in
# reproduction of the SML script's fit.
#
# Under method = "laplace" the SML-only parameters have no effect and are
# therefore not set anywhere in this script:
#   * `draws` / `seed` -- the Halton draw grid. Under Laplace `draws` does not
#     affect the likelihood, the tape, or any chunking; its only remaining job
#     is sizing the averaging grid for predict() and the marginal-effect
#     functions. This script calls none of those, so draws/seed are left at
#     their package defaults.
#   * `boundary_draws` -- the draws for the restricted LR refits. SML-only for
#     the same reason; the LR refits here are Laplace fits too, so there is no
#     draw dimension to keep on common random numbers.
#
# DATA and MODEL STRUCTURE (taken from rpbnb_truck_dense.R, same as the SML
# twin):
#
#   * inst/extdata/export_dense_all.csv (dense sections), not the
#     open-section export_open_all.csv.
#   * The two equations use DIFFERENT covariate sets -- ALL_3 in eq 1,
#     C_DISTR in eq 2 -- and one random coefficient on SR40_MI3 in each
#     equation. The same random coefficient, weakly identified, is shared as
#     in the open scripts so every truck script fits the same style of model.
#   * dependence = "famoye" (Famoye/Sarmanov), matching the referenced dense
#     script, rather than a copula.
#
# standardize = TRUE (same as rpbnb_truck_dense.R): the dense data carries
# strictly-positive continuous covariates (SR40_MI3, MPD_ME, MPD_STD, ...)
# that would otherwise make a random-coefficient carrier a random intercept in
# disguise and would inflate the design-matrix condition number. summary()
# back-transforms the coefficient table to original units automatically.
#
# COST. One full fit plus one restricted refit per tested parameter: 2
# random-coefficient scales (SR40_MI3 in each equation) + 2 dispersions +
# 1 dependence = 5 refits, i.e. roughly six full fits total. Drop
# boundary_tests to c("dispersion", "dependence") for a much cheaper run, or
# set max_rows below for a smoke test.
#
# The script loads via library(rpbnb) by default (see the load block below);
# to run against the current source tree instead, comment out the library()
# line and uncomment devtools::load_all(). Run from the package root:
#     Rscript inst/example_rpbnb_dense_tmb_laplace.R
# =============================================================================

# Load the package. Use library(rpbnb) once the package is installed; while
# developing against the current source tree (not yet installed), comment the
# library() line out and uncomment devtools::load_all() instead -- it rebuilds
# and loads the package from this working copy.
library(rpbnb)
# devtools::load_all()

# Prints a horizontal separator line between output sections.
sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")

# ---- Knobs ------------------------------------------------------------------
# n_cores: OpenMP threads the TMB engine uses. This is the ONE parallel knob
#         the TMB engine reads; max_threads (a cap on it, defaulting to
#         n_cores) is not set here because requesting n_cores and capping at
#         n_cores is the same thing.
n_cores <- 24L

# tmb_method: the TMB engine's estimator. "laplace" here -- the memory-saving
#             Laplace approximation, as opposed to "sml" (the SML twin's
#             estimator), which maximizes the simulated likelihood over
#             Halton draws.
tmb_method <- "laplace"

# dependence: the inter-equation dependence. "famoye" (Famoye/Sarmanov) here.
#             A copula() object is also accepted, one of three families:
#               copula("frank")       -- Frank copula (theta; symmetric)
#               copula("normal")      -- Gaussian copula (rho in (-1, 1))
#               copula("kimeldorf")   -- Clayton copula (theta > 0)
#             (an optional par argument fixes the dependence parameter for
#             simulation). Note copula("normal") runs multithreaded by default
#             (since 0.4.6 -- the SIGSEGV the old single-thread cap guarded
#             against was fixed in 0.4.4), so set disable_parallel_gaussian =
#             TRUE on the fit to pin a Gaussian run to one thread.
dependence <- copula("kimeldorf")

# boundary_tests: which parameter groups to LR-test against their boundary
#                 restriction. "all" = c("sd", "dispersion", "dependence") --
#                 one restricted refit per tested parameter. FALSE or "none"
#                 skips the LR tests entirely.
boundary_tests <- FALSE

# max_rows: smoke-test cap. NULL uses every row; set e.g. 400L to rehearse the
#           whole script cheaply before committing to a full run.
max_rows <- NULL

data_path <- "https://its.cutr.usf.edu/ftp/data/export_dense_all.csv"
data <- tryCatch(
  read.csv(data_path),
  error = function(e) {
    stop(
      "Could not read export_dense_all.csv from ", data_path, ". This is ",
      "local research data hosted on CUTR's internal FTP server rather than ",
      "shipped with the package -- confirm that server is reachable (e.g. ",
      "VPN/network access) before running this script.\n",
      "Underlying error: ", conditionMessage(e),
      call. = FALSE
    )
  }
)
if (!is.null(max_rows)) {
  data <- utils::head(data, max_rows)
  cat("*** SMOKE TEST: using the first", max_rows, "rows only ***\n")
}

# is_cop / dep_label: build the header label from `dependence` -- a copula
# object reports its family name, a bare string ("famoye") reports itself.
is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else as.character(dependence)
cat("=== rpbnb(engine = \"tmb\", method = \"laplace\") on truck all crashes, dense sections (",
    dep_label, ") ===\n", sep = "")
cat("Observations   :", nrow(data), "\n")
cat("Cores asked    :", n_cores, "\n")
cat("LR test groups :",
    if (isFALSE(boundary_tests)) "none" else paste(boundary_tests, collapse = ", "),
    "\n")

# f1, f2: the two marginal equations. Each is a count response regressed on a
# covariate set; the equations deliberately use DIFFERENT covariate sets
# (ALL_3 in eq 1, C_DISTR in eq 2), taken from rpbnb_truck_dense.R.
f1 <- ALL_3   ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + IRI_ME + RUT_L + SP50GE + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP10_ME
f2 <- C_DISTR ~ LNAADT_3 + SR40_MI3 + MPD_ME + MPD_STD + RUT_9 + ACCPNTS + SIGNAL1 + NEAR_SIG + AUXLNUM + DP01_ME + CS_MINAB

cat("Equation 1     :", deparse(f1), "\n")
cat("Equation 2     :", deparse(f2), "\n")

# random_1, random_2: the covariates given a random (varying) slope in each
# equation, i.e. whose coefficient varies across observations from a random
# distribution. Both equations put the one random slope on SR40_MI3.
random_1 <- c("SR40_MI3")
random_2 <- c("SR40_MI3")

# ---- Control object (TMB knobs only) -----------------------------------------
# One TMB setting is named here -- n_cores, the OpenMP thread count for the
# Laplace fit; the single parallel knob the TMB engine reads. print_level is
# left at the TMB default of 1 (it is also the switch that silences the
# per-refit LR-test messages, none of which fire while boundary_tests is
# FALSE). Everything else the TMB engine reads -- iterlim, reltol, gradtol,
# restarts, max_threads, parallel_tape -- is left at its package default and
# therefore not named here. The SML-only memory knobs (max_workload,
# tape_chunks) are likewise omitted: under method = "laplace" the tape scales
# with nrow(data) alone, so there is no draw dimension to chunk and the
# weighted-workload guard bounds nothing.
ctrl <- rpbnb_control(
  n_cores = n_cores
)

# The banner carries the same engine/method/dependence identity the print
# below is narrowed to, so the header names the parameters it is about to show
# (the print itself has no dependence field to echo).
sep(); cat(sprintf(
  "CONTROL OBJECT (engine = \"tmb\", method = \"%s\", dependence = \"%s\")\n",
  tmb_method, dep_label)); sep()
# engine=/method= narrow this print to the settings the TMB + Laplace fit
# actually reads.
print(ctrl, engine = "tmb", method = tmb_method)

# ---- Fit ----------------------------------------------------------------
# stamp / results: a timestamped run label and the directory the fit is saved
# to.
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
dir.create("results", recursive = TRUE, showWarnings = FALSE)

sep(); cat("FIT: engine = \"tmb\", method = \"laplace\"\n", sep = ""); sep()

# t_fit: wall-clock seconds for the whole fit, INCLUDING the restricted LR
# refits boundary_tests triggers.
t_fit <- system.time(
  fit <- rpbnb(
    formula_1      = f1,
    formula_2      = f2,
    data           = data,
    engine         = "tmb",
    method         = tmb_method,
    random_1       = random_1,
    random_2       = random_2,
    dependence     = dependence,
    standardize    = TRUE,
    boundary_tests = boundary_tests,
    control        = ctrl
  )
)[["elapsed"]]

cat(sprintf("\nFinished in %.2f s%s\n", t_fit,
            if (isFALSE(boundary_tests) || !length(boundary_tests)) ""
            else " (includes the restricted LR refits)"))
# Convergence of the TMB engine's nlminb optimizer (code + message).
cat(sprintf("Convergence  : nlminb code=%d, %s\n",
            fit$optimizer$convergence, fit$optimizer$message))

fit_path <- file.path("results", paste0("fit_truck_dense_tmb_", stamp, ".rds"))
saveRDS(fit, fit_path)
cat("Saved to     :", fit_path, "\n")

# Names any supplied control setting the TMB engine did not read (none are
# expected here, since ctrl sets only TMB-applicable fields).
cat("Control settings this engine did not read: ",
    if (length(fit$control_ignored)) paste(fit$control_ignored, collapse = ", ")
    else "(none)", "\n", sep = "")

# ---- Summary ------------------------------------------------------------
# standardize = TRUE back-transforms the coefficient table to original units
# automatically. With the LR tests attached, the natural-scale block carries
# LR/df/p for the scale and dispersion rows in place of the NA they would
# otherwise show, and the dependence row shows its LR test instead of a Wald z.
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
