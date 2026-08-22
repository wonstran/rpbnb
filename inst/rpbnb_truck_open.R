#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() on the open-section truck-crash data, run through BOTH engines from
# ONE control object, with the grouped LR-test switch turned all the way up.
#
# This script exists to exercise the two 0.4.1 changes end to end on real data:
#
#   1. ONE control object.  rpbnb_control() now carries the union of both
#      engines' tuning knobs and either engine accepts it; rpbnb_tmb_control()
#      is a retained alias for the same thing.  The `ctrl` built below is a
#      deliberate SUPERSET -- it sets maxLik-only knobs (se_method, hess_eps)
#      and TMB-only knobs (gradtol, restarts, max_threads, max_workload,
#      parallel_tape) in the same call -- and is then handed to both engines
#      unchanged.  Each fit reads the fields that apply to it, IGNORES the
#      rest, and names the ignored ones in its own print()/summary():
#
#          Control settings ignored (not used by the TMB engine): se_method, ...
#
#      That is the whole point: a script can flip `engine` without rewriting
#      its control call, and nothing is silently dropped.  Two fields whose
#      defaults differ between the engines -- iterlim (300 maxLik / 500 nlminb)
#      and print_level (2 / 0) -- are left unset here, so each engine resolves
#      its own; set them explicitly and both engines honor the number.
#
#   2. The grouped LR-test switch.  `boundary_tests` is no longer a bare
#      logical: it names which groups to test -- "sd" (random-coefficient
#      scales), "dispersion" (the NB2 overdispersions m1/m2), "dependence"
#      (the association parameter), plus "all"/"none".  TRUE still means
#      c("sd", "dispersion") exactly as it always did.  The DEPENDENCE test is
#      new and is what this script is really here to show: H0 is the
#      independence model, and the correction is applied per family -- Famoye
#      lambda, Frank theta, and Gaussian rho all have an INTERIOR null at 0
#      and get an ordinary chi-square(1); only Clayton/Kimeldorf (theta > 0)
#      takes the 50:50 boundary mixture.  Requesting it replaces the
#      dependence row's Wald z/p in summary() with the LR test, on both
#      engines.
#
# Relationship to the neighbouring scripts: inst/rpbnb_truck.R is the
# classic-engine-only version of this fit and inst/rpbnb_truck_tmb.R the
# TMB-only one.  Both remain the place to look for the post-estimation
# sections (marginal effects, elasticities, residual checks).  This script
# deliberately stops at the LR tables and instead puts the two engines side by
# side, because agreement between them on the same restriction is the thing
# worth reading here.
#
# rpbnb_frank_open.R's header explains WHY this data needs standardizing:
# SR40_MI3 and MPD_ME are strictly positive and bounded away from zero, so as
# random-coefficient carriers x * (b + sd * u_i) is a random INTERCEPT in
# disguise unless centred; and IRI_ME (29-380) next to 0/1 indicators gives a
# design-matrix condition number near 1e7 unless scaled.  standardize = TRUE
# handles both and back-transforms the printed table to original units.
#
# COST.  Each engine pays one full fit plus one restricted refit per tested
# parameter: 3 random-coefficient scales + 2 dispersions + 1 dependence = 6
# refits, so roughly seven full fits per engine.  Drop `boundary_tests` to
# c("dispersion", "dependence") for a much cheaper run that still shows the
# new test, or set `max_rows` below for a smoke test.
#
# devtools::load_all() (not library()): the script must run against the
# current source tree.  Run from the package root:
#     Rscript inst/rpbnb_truck_open.R
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

# ---- Knobs ------------------------------------------------------------------
engines <- c("classic", "tmb")
n_cores <- 16L
draws   <- 300L
# Held equal across the two engines on purpose: the comparison at the bottom
# is only meaningful if both simulated likelihoods use the same number of
# draws (the seed is shared too, so the Halton prefixes match).
seed    <- 20240712L
# TMB-only.  "sml" keeps the ESTIMATOR fixed across the two engines so the
# comparison varies only the engine; "laplace" would change what is being
# approximated as well.
tmb_method <- "sml"
# Same knob as rpbnb_truck.R / rpbnb_truck_tmb.R: "famoye" or a copula()
# object.  Note copula("normal") is capped at one thread on both engines
# unless force_parallel_gaussian = TRUE (a known SIGSEGV in the registered
# atomic), so a Gaussian run here will be markedly slower.
dependence <- copula("frank")
# Which groups to LR-test.  "all" = c("sd", "dispersion", "dependence").
# FALSE or "none" skips them entirely.
boundary_tests <- "all"
# TMB-only: draws for the restricted refits.  NULL reuses the main fit's
# `draws`, which is what keeps the restricted and full simulated likelihoods
# on common random numbers -- diverge from it only deliberately.
boundary_draws <- NULL
# Smoke-test knob: NULL uses every row.  Set to e.g. 400L to rehearse the
# whole script cheaply before committing to a full run.
max_rows <- NULL

data <- read.csv(file.path("inst", "extdata", "export_open_all.csv"))
if (!is.null(max_rows)) {
  data <- utils::head(data, max_rows)
  cat("*** SMOKE TEST: using the first", max_rows, "rows only ***\n")
}

is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else as.character(dependence)
cat("=== rpbnb() on truck all crashes, open sections (", dep_label, ") ===\n",
    sep = "")
cat("Observations   :", nrow(data), "\n")
cat("Engines        :", paste(engines, collapse = ", "), "\n")
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

# ---- One control object for both engines ------------------------------------
# Deliberately a superset of what either engine reads.  Nothing here is
# translated between engines: `n_cores` happens to mean OpenMP threads on both
# paths, but `iterlim` would mean a maxLik BFGS limit to one and an nlminb
# limit to the other, which is exactly why the two estimator-dependent
# defaults are left unset rather than pinned to one number.
ctrl <- rpbnb_control(
  # --- read by both engines ---
  print_level  = 1,          # also un-silences the per-refit LR-test messages
  n_cores      = n_cores,
  reltol       = 1e-8,
  halton_burn  = 300L,
  # --- maxLik side only (ignored, and reported as ignored, by the TMB fit) ---
  # Observed-information Hessian on the same draws that produced the estimate.
  # "opg" is the faster BHHH alternative; the boundary refits force
  # compute_se = FALSE internally regardless, since an LR test needs only
  # logLik and df.
  se_method    = "opg",
  hess_eps     = 1e-5,
  # --- TMB side only (ignored, and reported as ignored, by the classic fit) ---
  gradtol      = 1e-5,
  restarts     = 10L,
  max_threads  = n_cores,
  # The default workload guard's calibration under-estimates this data's
  # per-draw cost (see inst/rpbnb_truck_tmb.R's header), so leaving it engaged
  # would block the fit rather than warn.
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

  # method / force_parallel_gaussian / boundary_draws are TMB-only.  rpbnb()
  # would drop the first two with a warning under engine = "classic" and
  # ERROR on boundary_draws, so they are added per engine rather than passed
  # unconditionally -- the argument matrix in ?rpbnb has the full list.
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
      random_1       = c("SR40_MI3", "MPD_ME"),
      random_2       = c("SR40_MI3"),
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
  # Convergence is reported differently by the two engines (maxLik code vs
  # nlminb code), so read whichever record this fit carries.
  if (!is.null(fit$convergence)) {
    cat(sprintf("Convergence  : code=%d, %s (iterations=%d)\n",
                fit$convergence$code, fit$convergence$message,
                fit$convergence$iterations))
  } else {
    cat(sprintf("Convergence  : nlminb code=%d, %s\n",
                fit$optimizer$convergence, fit$optimizer$message))
  }

  fit_path <- file.path("results",
                        paste0("fit_truck_open_", eng, "_", stamp, ".rds"))
  saveRDS(fit, fit_path)
  cat("Saved to     :", fit_path, "\n")

  # The ignored-settings line the unification exists to produce.  It is part
  # of summary() below too; surfaced on its own first so it is not lost in the
  # coefficient tables.
  cat("Control settings this engine did not read: ",
      if (length(fit$control_ignored)) paste(fit$control_ignored, collapse = ", ")
      else "(none)", "\n", sep = "")
}

# ---- Summaries --------------------------------------------------------------
# standardize = TRUE back-transforms the coefficient table to original units
# automatically.  With the LR tests attached, the natural-scale block carries
# LR/df/p for the scale and dispersion rows in place of the NA they would
# otherwise show, AND the dependence row shows its LR test instead of a Wald z.
#
# The two engines' summary() methods differ in HOW they emit: the classic one
# returns a summary.rpbnb_fit object that has to be print()ed, while the TMB
# one prints as it goes and returns the fit invisibly. Wrapping both in
# print() would re-print the whole TMB fit underneath its own summary, so
# dispatch on the class rather than assuming one convention.
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
# The payoff of running both from one control object.  The two engines
# maximize different approximations of the same likelihood, so logLik will not
# match to the last digit; what should agree is the SUBSTANCE -- which
# restrictions the data rejects, and roughly how strongly.  A parameter the
# two engines disagree about (one significant, one not) is worth investigating
# before it is reported.
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
    # Full outer join on Parameter: the two engines can legitimately produce
    # different row sets (a scale label follows the random-coefficient
    # distribution, and a refit that failed to converge is NA), so a
    # positional cbind() would silently misalign them.
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
