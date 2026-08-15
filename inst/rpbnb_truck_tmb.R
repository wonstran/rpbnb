#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() (engine = "tmb", standardize = TRUE) on the open-section truck-crash
# data. The TMB counterpart of inst/rpbnb_truck.R: same data, formulas,
# dependence, and draws, dispatched through the SAME rpbnb() front end --
# only `engine` differs, plus MPD_ME's random coefficient below is Uniform
# here rather than rpbnb_truck.R's Normal (see the random_1 comment). Aside
# from that one distribution, comparing the two scripts' output is comparing
# the two engines on an otherwise identical specification.
#
# boundary_tests = TRUE works under engine = "tmb" too
# (rpbnb_tmb_boundary_tests()), covering the same parameters the classic
# engine's rpbnb_boundary_tests() does: every random-coefficient scale AND
# both NB2 dispersions. The two engines differ only in how each restricted
# fit holds a scale at zero while preserving common random numbers -- the
# classic engine zeroes that coefficient's draw column, the TMB engine pins
# its log_sd and maps it out of the free parameters.
#
# boundary_draws below is rpbnb()'s pass-through to
# rpbnb_tmb_boundary_tests()'s own `draws` argument, so the restricted
# refits can use a different `draws` than the main fit without a separate
# manual rpbnb_tmb_boundary_tests() call. Their default control reuses the
# main fit's n_cores (fit$parallel$requested), so the refits get the same
# thread budget as the main fit. rpbnb() also forwards force_parallel_gaussian
# (set below) to the boundary refits -- needed here since dependence is a
# Gaussian copula (see below): without it, every restricted refit would
# re-cap itself to one thread regardless of n_cores, same as the main fit
# would without the override (see ?fit_rpbnb_tmb, `force_parallel_gaussian`).
#
# method = "sml" is used below (not TMB's memory-saving "laplace") so the
# comparison against rpbnb_truck.R holds the ESTIMATOR fixed and varies only
# the engine (Rcpp/OpenMP maxLik BFGS vs TMB automatic differentiation +
# nlminb). inst/tmb_rpbnb_frank_open.R uses "laplace" instead -- see its
# header for why (memory scales with n rather than n * draws) if that
# trade-off is what's being compared instead.
#
# rpbnb_frank_open.R's header explains WHY this data needs standardizing:
# SR40_MI3 and MPD_ME are strictly positive and bounded away from zero, so as
# random-coefficient carriers x * (b + sd * u_i) is a random INTERCEPT in
# disguise unless centred; and IRI_ME (29-380) next to 0/1 indicators gives a
# design-matrix condition number near 1e7 unless scaled. Read that script's
# header for the full argument -- it applies unchanged here.
#
# devtools::load_all() (not library()): the script must run against the
# current source tree -- it reads internal helpers (rpbnb:::) an installed
# build may not carry. Run from the package root:
#     Rscript inst/rpbnb_truck_tmb.R
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

# detectCores() returns 24 on this box (31.5 GiB RAM); see
# inst/tmb_rpbnb_frank_open.R's header for the memory-vs-draws calibration
# behind these numbers on this exact dataset/model (TMB checkpoints its
# per-draw Gaussian-copula kernel, so peak working set grows roughly linearly
# in draws rather than the pre-checkpoint quadratic blow-up that comment
# documents).
n_cores <- 16L
draws <- 300L
method <- "sml"
# Same knob as rpbnb_truck.R: "famoye" or a copula() object. Kept identical
# between the two scripts so their output is directly comparable.
dependence <- copula("normal") #copula("kimeldorf")
# One restricted refit per random-coefficient scale (3 here) plus one per
# dispersion (2), each run at boundary_draws (see the header), so this adds
# real time on top of the main fit.
boundary_tests <- TRUE
# rpbnb(boundary_draws = )'s own knob for the restricted refits -- NULL would
# fall back to the main fit's `draws` (500 here); set independently so the
# restricted refits can trade precision for speed (fewer draws, cheaper) or
# vice versa (more draws) without re-running the main fit at a different
# `draws`. Left equal to `draws` here so this script's boundary LR statistics
# stay directly comparable to rpbnb_truck.R's (same draws throughout).
boundary_draws <- 200L

data <- read.csv(file.path("inst", "extdata", "export_open_all.csv"))
cat("Observations :", nrow(data), "\n")

is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else as.character(dependence)
dep_desc <- if (is_cop) {
  sprintf("%s copula joining two NB margins", dependence$family)
} else {
  "Famoye/Sarmanov bivariate NB"
}
cat(sprintf("=== rpbnb(engine = \"tmb\", method = \"%s\") on truck all crashes (%s) ===\n",
            method, dep_label))
cat("Dependence   :", dep_desc, "\n")
cat("Cores asked  :", n_cores, "\n")
cat("Draws        :", draws, "\n")
cat("Boundary draws:", if (boundary_tests) boundary_draws else "n/a (boundary_tests = FALSE)", "\n")

# Same formulas as rpbnb_truck.R / rpbnb_frank_open.R: random slopes on
# SR40_MI3 and MPD_ME in eq 1, SR40_MI3 only in eq 2 (weakly identified
# there; kept only to match the other two scripts so all three fit the exact
# same model). Random-coefficient DISTRIBUTIONS differ from those scripts,
# though: MPD_ME is Uniform here (random_1 below), not Normal -- see its
# comment for why that changes the summary()/boundary-test row label
# (w1:MPD_ME, a half-width, not sd1:MPD_ME).
f1 <- ALL_3  ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + G_ABG2 + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME + RUT_L
f2 <- C_HV ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME

cat("Equation 1   :", deparse(f1), "\n")
cat("Equation 2   :", deparse(f2), "\n\n")

t_fit <- system.time(
  fit <- rpbnb(
    formula_1      = f1,
    formula_2      = f2,
    data           = data,
    engine         = "tmb",
    # MPD_ME as Uniform rather than the default Normal: list form lets each
    # variable pick its own distribution ("normal"/"lognormal"/"uniform"/
    # "triangular" -- see ?fit_rpbnb_tmb's random_1/random_2 or
    # R/rand_dist.R's registry). SR40_MI3 spelled out as "normal" rather than
    # left as a bare name for the same reason MPD_ME needs the list form: one
    # random_1 argument, so every entry uses the same (named list) syntax.
    random_1       = c("SR40_MI3", "MPD_ME"),
    random_2       = c("SR40_MI3"),
    dependence     = dependence,
    seed           = 20240712,
    #draws          = draws,
    standardize    = TRUE,
    method         = method,
    # No max_workload override beyond Inf: the guard's default calibration
    # under-estimates this data's per-draw cost (see the header), so leaving
    # it engaged would block the fit rather than just warn.
    force_parallel_gaussian = TRUE,
    boundary_tests = boundary_tests,
    #boundary_draws = boundary_draws,
    control        = rpbnb_tmb_control(
      print_level  = 1,
      n_cores      = n_cores,
      max_threads  = n_cores,
      max_workload = Inf
    )
  )
)[["elapsed"]]

cat(sprintf("\nEstimation finished in %.2f s%s\n", t_fit,
            if (boundary_tests) " (includes the boundary LR refits)" else ""))
# Deliberately not reporting gc() figures: the TMB tape lives on the C++ heap
# and is invisible to R's garbage collector, so gc() would understate exactly
# the quantity this script exists to test. Watch the process working set
# externally (Task Manager, or inst/tmb_benchmark_memory.R) if a number is
# needed.
cat(sprintf("TMB threads: requested=%d, realized=%d\n",
            fit$parallel$requested, fit$parallel$realized))
cat(sprintf("Optimizer: code=%d, message=%s\n",
            fit$optimizer$convergence, fit$optimizer$message))
cat("sdreport positive-definite Hessian:",
    if (isTRUE(fit$sdreport$pdHess)) "yes" else "no", "\n")
cat("Standardized (centred and scaled) predictors (auto-detected):\n")
print(round(do.call(rbind, fit$scaling), 4))

# ---- Persist the fit --------------------------------------------------------
# standardize = TRUE stores the scaling ON the fit ($scaling/$continuous_vars),
# so unlike tmb_rpbnb_frank_open.R there is no separate `scaling` object to
# save alongside it -- one RDS is the whole story.
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
fit_path <- file.path("results", paste0("fit_truck_tmb_", stamp, ".rds"))
dir.create("results", recursive = TRUE, showWarnings = FALSE)
saveRDS(fit, fit_path)
cat("Fit object saved to:", fit_path, "\n")

# ---- Convergence diagnostics -------------------------------------------
# print_level = 1 buries these under thousands of TMB inner-iteration lines. A
# non-PD Hessian makes every standard error below meaningless, and that must
# not be easy to scroll past.
diagnostics <- character(0)
if (!isTRUE(fit$sdreport$pdHess)) {
  diagnostics <- c(diagnostics,
                   "Hessian is not positive definite: standard errors are unreliable.")
}
if (length(fit$boundary_report)) {
  diagnostics <- c(diagnostics,
                   paste0("Parameters at a constraint bound: ",
                          paste(paste0(fit$boundary_report,
                                       " (", fit$boundary_sides, ")"),
                                collapse = ", ")))
}
if (length(diagnostics)) {
  sep(); cat("CONVERGENCE WARNINGS\n"); sep()
  cat(paste0("  * ", diagnostics, collapse = "\n"), "\n")
} else {
  cat("No boundary or Hessian warnings.\n")
}

# ---- Model summary (original covariate units, boundary LR tests) -----------
# rpbnb(standardize = TRUE) back-transforms the print()/summary() coefficient
# table automatically for BOTH engines (see R/rpbnb_scaling.R and the
# NEWS.md entry "rpbnb(standardize = TRUE)") -- no coef_orig_units()/
# sd_orig_units() helper code is needed here, unlike tmb_rpbnb_frank_open.R.
# The dependence parameter (lambda, or the copula's native theta and Kendall's
# tau, from ADREPORT) is part of summary()'s own "--- Dependence ---" section,
# so this script has no separate DEPENDENCE block either. With
# fit$boundary_tests attached above, BOTH the "Random-coefficient scales"
# blocks and the "Dispersion (m1, m2)" block carry a real LR/df/Pr(>chisq)
# instead of NA -- see the NEWS.md entry "rpbnb_tmb_boundary_tests()".
sep(); cat("MODEL SUMMARY (original covariate units, boundary LR tests)\n"); sep()
print(summary(fit))
cat("\n")

# ---- Boundary LR tests (standalone table) -----------------------------------
# fit$boundary_tests is the same rpbnb_tmb_boundary_tests() result already
# folded into the summary above (attached automatically by boundary_tests =
# TRUE, run at boundary_draws draws via rpbnb()'s boundary_draws argument);
# printed again here on its own for the raw LR/df/p table with its Signif
# stars, matching rpbnb_truck.R's equivalent section.
if (!is.null(fit$boundary_tests)) {
  sep(); cat("BOUNDARY LR TESTS (random-coefficient scales and dispersions)\n"); sep()
  print(fit$boundary_tests)
} else {
  sep(); cat("BOUNDARY LR TESTS (random-coefficient scales and dispersions)\n"); sep()
  cat("Skipped (boundary_tests = FALSE). Set boundary_tests <- TRUE at the top\n")
  cat("of this script to run them (folded into the rpbnb() call above via\n")
  cat("boundary_tests = TRUE, at boundary_draws draws).\n")
}
cat("\n")

# ---- Fitted means (predict) ---------------------------------------------
sep(); cat("FITTED MEANS (predict) -- first 6 observations\n"); sep()
print(head(predict(fit)))

# ---- Marginal effects (AME) and elasticities (original covariate units) ----
# rpbnb_tmb_marginal_effects()/rpbnb_tmb_elasticities() have taken `scaling =`/
# `log_vars =` since the rpbnb.tmb merge (the classic engine's counterparts
# gained the same arguments later -- see rpbnb_truck.R and the NEWS.md entry
# "original-units marginal effects/elasticities for engine = \"classic\"").
# Passing `fit$scaling` restates both in original units directly; without it,
# elasticities of the centred continuous predictors would print as ~0 (a
# centred regressor has x-bar = 0, so the elasticity's leading x-bar factor
# vanishes) -- that reads as "no effect" but means "these units are arbitrary".
#
# LNAADT_3 is log(AADT); `log_vars` reports its AME/elasticity per unit of
# AADT itself rather than per unit of log(AADT) -- see
# rpbnb_tmb_marginal_effects()'s `log_vars` documentation for why that
# distinction is otherwise an order-of-magnitude silent overstatement.
log_vars <- "LNAADT_3"
sep(); cat("AVERAGE MARGINAL EFFECTS (AME, original covariate units)\n"); sep()
marginal_effects <- rpbnb_tmb_marginal_effects(fit, which = "both",
                                               scaling = fit$scaling,
                                               log_vars = log_vars)
cat("\n")

#sep(); cat("ELASTICITIES / SEMI-ELASTICITIES (AME, original covariate units)\n"); sep()
#elasticities <- rpbnb_tmb_elasticities(fit, which = "both",
#                                       scaling = fit$scaling,
#                                       log_vars = log_vars)

cat("\nFit saved to:", fit_path, "\n")
