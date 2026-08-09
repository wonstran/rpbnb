#!/usr/bin/env Rscript
# =============================================================================
# rpbnb() (engine = "classic", standardize = TRUE, boundary_tests = TRUE) on
# the open-section truck-crash data. Same data, formulas, random-coefficient
# specification, dependence, and draws as inst/rpbnb_frank_open.R -- but fit
# through the rpbnb() dispatcher instead of calling fit_rpbnb() directly, with
# standardize = TRUE instead of the by-hand centring/scaling + original-units
# back-transform that script builds itself (~150 lines of coef_orig_units()/
# sd_orig_units() code, replaced here by two arguments), and boundary_tests =
# TRUE instead of a separate rpbnb_boundary_tests() call that reconstructs the
# standardized data by hand. summary(fit) below shows both automatically: the
# main coefficient table in original units, and a real LR test (not NA) for
# the random-coefficient SDs and NB2 dispersions in its natural-scale block.
#
# rpbnb_frank_open.R's header explains WHY this data needs standardizing:
# SR40_MI3 and MPD_ME are strictly positive and bounded away from zero, so as
# random-coefficient carriers x * (b + sd * u_i) is a random INTERCEPT in
# disguise unless centred; and IRI_ME (29-380) next to 0/1 indicators gives a
# design-matrix condition number near 1e7 unless scaled. Read that script's
# header for the full argument -- it applies unchanged here.
#
# devtools::load_all() (not library()): the script must run against the
# current source tree. Run from the package root:
#     Rscript inst/rpbnb_truck.R
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

n_cores <- 20L
draws <- 500L
boundary_tests <- TRUE
# Same knob as rpbnb_frank_open.R: "famoye" or a copula() object.
dependence <- copula("frank")

data <- read.csv(file.path("inst", "extdata", "export_open_all.csv"))
cat("Observations :", nrow(data), "\n")

is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else as.character(dependence)
dep_desc <- if (is_cop) {
  sprintf("%s copula joining two NB margins", dependence$family)
} else {
  "Famoye/Sarmanov bivariate NB"
}
cat(sprintf("=== rpbnb(engine = \"classic\") on truck all crashes (%s) ===\n", dep_label))
cat("Dependence   :", dep_desc, "\n")
cat("Cores asked  :", n_cores, "\n")
cat("Draws        :", draws, "\n")

# Same formulas and random-coefficient specification as rpbnb_frank_open.R:
# random slopes on SR40_MI3 and MPD_ME in eq 1, SR40_MI3 only in eq 2 (that
# script's comment on the eq-2 SR40_MI3 SD applies unchanged: it is weakly
# identified, kept here only to match rpbnb_frank_open.R / rpbnb_tmb_frank_open.R).
f1 <- ALL_3  ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + G_ABG2 + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME + RUT_L
f2 <- C_HV ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME

cat("Equation 1   :", deparse(f1), "\n")
cat("Equation 2   :", deparse(f2), "\n\n")

# boundary_tests = TRUE folds every restricted refit (rpbnb_boundary_tests(),
# one per random-coefficient SD plus one per NB2 dispersion) into this same
# call, run against the same standardized design the full fit uses -- rpbnb()
# derives that data itself (see "Boundary LR tests" in ?rpbnb), so unlike
# rpbnb_frank_open.R there is no separate rpbnb:::.apply_scaling() step here.
# Cost is real: per rpbnb_boundary_tests()'s own timing note, a copula
# dependence's restricted refits run markedly more expensive than Famoye's, so
# t_fit below now includes that cost too, not just the full fit's.
t_fit <- system.time(
  fit <- rpbnb(
    formula_1      = f1,
    formula_2      = f2,
    data           = data,
    engine         = "classic",
    random_1       = c("SR40_MI3", "MPD_ME"),
    random_2       = c("SR40_MI3"),
    dependence     = dependence,
    seed           = 20240712,
    draws          = draws,
    standardize    = TRUE,
    boundary_tests = boundary_tests,
    control        = rpbnb_control(
      print_level = 1,
      n_cores     = n_cores,
      # Observed-information Hessian on the same draws that produced the
      # estimate. "opg" is a faster BHHH alternative; "numeric" is the
      # robust default (rpbnb_frank_open.R uses "opg" too). The boundary
      # refits force compute_se = FALSE internally regardless of this
      # setting -- the LR test needs only logLik and df.
      se_method   = "opg"
    )
  )
)[["elapsed"]]

cat(sprintf("\nEstimation finished in %.2f s%s\n", t_fit,
            if (boundary_tests) " (includes the boundary LR test refits)" else ""))
# Memory is not reported: the C++ core keeps its working set outside R's
# allocator, so gc() would understate exactly the quantity this script tests.
cat(sprintf("Convergence : code=%d, message=%s (iterations=%d)\n",
            fit$convergence$code, fit$convergence$message,
            fit$convergence$iterations))
cat("Standardized (centred and scaled) predictors (auto-detected):\n")
print(round(do.call(rbind, fit$scaling), 4))

# ---- Persist the fit --------------------------------------------------------
# standardize = TRUE stores the scaling ON the fit ($scaling/$continuous_vars),
# so unlike rpbnb_frank_open.R there is no separate `scaling` object to save
# alongside it -- one RDS is the whole story.
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
fit_path <- file.path("results", paste0("fit_truck_classic_", stamp, ".rds"))
dir.create("results", recursive = TRUE, showWarnings = FALSE)
saveRDS(fit, fit_path)
cat("Fit object saved to:", fit_path, "\n")

# ---- Convergence diagnostics -------------------------------------------
# print_level = 1 still buries the key convergence facts; a non-PD information
# matrix makes every standard error below meaningless, so surface it here.
diagnostics <- character(0)
if (!isTRUE(fit$convergence$converged)) {
  diagnostics <- c(diagnostics, sprintf(
    "Optimizer did not converge (code %d: %s).",
    fit$convergence$code, fit$convergence$message))
}
if (!is.null(fit$hessian_diag) && !isTRUE(fit$hessian_diag$positive_definite)) {
  diagnostics <- c(diagnostics, sprintf(
    "Information matrix is not positive definite (min eigenvalue %s); a ridge of %s was added -- SEs are regularized, not observed-information.",
    formatC(fit$hessian_diag$min_eigenvalue, format = "g"),
    formatC(fit$hessian_diag$ridge, format = "g")))
}
# Famoye/Sarmanov ONLY -- a copula fit carries `cop_family` and leaves
# `lambda`/`bounds` NULL. Gate on the family first.
if (is.null(fit$cop_family) && length(fit$lambda) == 1L &&
    length(fit$bounds) == 2L) {
  lam_span <- diff(fit$bounds)
  if (isTRUE(is.finite(lam_span) && lam_span > 0 &&
             min(abs(fit$lambda - fit$bounds)) < 1e-4 * lam_span)) {
    diagnostics <- c(diagnostics, sprintf(
      "lambda = %.6f is on its data-adaptive bound [%.6f, %.6f]: the near-zero SE and huge z printed for it below are boundary artifacts, not evidence of precision.",
      fit$lambda, fit$bounds[1], fit$bounds[2]))
  }
}
if (any(!is.finite(fit$se))) {
  diagnostics <- c(diagnostics, paste0(
    "Non-finite SEs: ",
    paste(names(fit$se)[!is.finite(fit$se)], collapse = ", "), "."))
}
if (length(diagnostics)) {
  sep(); cat("CONVERGENCE WARNINGS\n"); sep()
  cat(paste0("  * ", diagnostics, collapse = "\n"), "\n")
} else {
  cat("No boundary or Hessian warnings.\n")
}

# ---- Model summary (original covariate units + boundary LR tests) ----------
# rpbnb(standardize = TRUE) back-transforms the print()/summary() coefficient
# table automatically (see R/rpbnb_scaling.R and the NEWS.md entry
# "rpbnb(standardize = TRUE)") -- no coef_orig_units()/sd_orig_units() helper
# code is needed here, unlike rpbnb_frank_open.R. The dependence parameter
# (lambda, or the copula's native theta and Kendall's tau) is included in the
# "Natural-scale dispersion / dependence" block below, so this script has no
# separate DEPENDENCE section either. With boundary_tests = TRUE, that same
# block ALSO carries the boundary-corrected LR test (LR/df/p) for the
# random-coefficient SDs and NB2 dispersions, in place of the NA those rows
# would otherwise show -- see the NEWS.md entry "rpbnb(boundary_tests = TRUE)".
sep(); cat("MODEL SUMMARY (original covariate units, boundary LR tests)\n"); sep()
print(summary(fit))
cat("\n")

# ---- Boundary LR tests (standalone table) -----------------------------------
# fit$boundary_tests is the same rpbnb_boundary_tests() result already folded
# into the summary above (attached automatically by boundary_tests = TRUE);
# printed again here on its own for the raw LR/df/p table with its Signif
# stars, matching what rpbnb_frank_open.R prints from its separate call.
if (!is.null(fit$boundary_tests)) {
  sep(); cat("BOUNDARY LR TESTS (SDs and dispersions)\n"); sep()
  print(fit$boundary_tests)
} else {
  sep(); cat("BOUNDARY LR TESTS (SDs and dispersions)\n"); sep()
  cat("Skipped (boundary_tests = FALSE). Set boundary_tests <- TRUE at the top\n")
  cat("of this script to run them (one restricted refit per boundary parameter,\n")
  cat("folded into the rpbnb() call above via boundary_tests = TRUE).\n")
}
cat("\n")

# ---- Fitted means (predict) ---------------------------------------------
sep(); cat("FITTED MEANS (predict) -- first 6 observations\n"); sep()
print(head(predict(fit)))

# ---- Marginal effects (AME) and elasticities --------------------------------
# rpbnb_marginal_effects()/rpbnb_elasticities() (the classic engine's
# post-estimation functions) have no `scaling =` argument -- unlike their TMB
# counterparts -- so their output below is on the STANDARDIZED scale: AMEs are
# per SD, and the elasticities of the centred continuous predictors print as
# ~0 (a centred regressor has x-bar = 0, so its elasticity's leading x-bar
# factor vanishes; this reads as "no effect" but means "these units are
# arbitrary"). rpbnb_frank_open.R's `raw_diag()` restates both in original
# units by hand using exactly `fit$scaling`/`fit$continuous_vars` (now
# attached to any standardize = TRUE fit) -- see that script if the original-
# units versions are needed here too.
sep(); cat("AVERAGE MARGINAL EFFECTS (AME, standardized scale)\n"); sep()
marginal_effects <- rpbnb_marginal_effects(fit, which = "both", type = "AME",
                                           n_cores = n_cores)
cat("\n")

sep(); cat("ELASTICITIES / SEMI-ELASTICITIES (AME, standardized scale)\n"); sep()
elasticities <- rpbnb_elasticities(fit, which = "both", type = "AME",
                                   n_cores = n_cores)
cat("\nNote: continuous-predictor elasticities above are ~0 -- see the note\n")
cat("above the AME section. Restate in original units via fit$scaling,\n")
cat("following rpbnb_frank_open.R's raw_diag() pattern.\n")

cat("\nFit saved to:", fit_path, "\n")
