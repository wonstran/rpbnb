#!/usr/bin/env Rscript
# =============================================================================
# rpbnb estimation demo -- SAME formulas, Famoye/Sarmanov dependence.
#
# Fits a random-parameter bivariate NB (RP-BNB) model to the German health-care
# counts in data/rwm1984_bnb.csv:
#   * y1 = docvis  (doctor visits), y2 = hospvis (hospital visits)
#   * BOTH equations use the SAME covariate set
#   * a fixed + random coefficient mix: a random slope on one CONTINUOUS
#     covariate per equation (age in eq 1, hhninc in eq 2); all others fixed
#   * Famoye/Sarmanov dependence between the two counts
#
# Usage demo on real data -- there is no known ground truth, so the script
# reports the fitted model, predictions, marginal effects, and residual
# diagnostics rather than an estimate-vs-truth comparison. Run from the package
# root:  Rscript inst/fit_rpbnb_same_famoye.R
# =============================================================================

devtools::load_all(quiet = TRUE)

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")

# ---- 1. Data ----------------------------------------------------------------
data <- read.csv(file.path("data", "rwm1984_bnb.csv"))
cat("=== RP-BNB (same formulas, Famoye) on rwm1984 health counts ===\n")
cat("Observations :", nrow(data), "\n")
cat("Threads      :", rpbnb_threads(), "\n")

# ---- 2. Model specification -------------------------------------------------
# The same covariates enter both equations. Random coefficients sit on
# CONTINUOUS regressors only: a random slope on a 0/1 dummy is weakly identified
# here (the NB dispersion and the random-coefficient scale trade off), so keep
# the dummies as ordinary fixed effects.
rhs <- ~ age + hhninc + educ + female + married + kids + outwork
f1  <- update(rhs, docvis  ~ .)
f2  <- update(rhs, hospvis ~ .)
cat("Equation 1   :", deparse(f1), "\n")
cat("Equation 2   :", deparse(f2), "\n\n")

t_fit <- system.time(
  fit <- fit_rpbnb(
    formula_1  = f1,
    formula_2  = f2,
    data       = data,
    random_1   = "age",      # random slope on age    (eq 1), continuous
    random_2   = "hhninc",   # random slope on hhninc (eq 2), continuous
    dependence = "famoye",
    draws      = 200,
    seed       = 20240712,
    control    = rpbnb_control(
      print_level = 1,
      n_cores     = rpbnb_threads(),
      se_method   = "opg"    # fast BHHH/OPG SEs; the robust default is "numeric"
    )
  )
)[["elapsed"]]
cat(sprintf("\nEstimation finished in %.1f s\n", t_fit))

# ---- 3. Model summary -------------------------------------------------------
sep(); cat("MODEL SUMMARY\n"); sep()
print(summary(fit))

# ---- 4. Fitted means (predict) ----------------------------------------------
sep(); cat("FITTED MEANS (predict) -- first 6 observations\n"); sep()
print(head(predict(fit)))

# ---- 5. Marginal effects & elasticities (AME) -------------------------------
# Built on the Monte-Carlo integrated mean E[exp(x'beta)] (the estimand predict()
# uses); the random-coefficient covariate uses the draw-integrated formula while
# the fixed covariates reduce to the classic result. See ?rpbnb_marginal_effects.
sep(); cat("AVERAGE MARGINAL EFFECTS (AME)\n"); sep()
me <- rpbnb_marginal_effects(fit, which = "both", type = "AME",
                             n_cores = rpbnb_threads())

sep(); cat("ELASTICITIES / SEMI-ELASTICITIES (AME)\n"); sep()
el <- rpbnb_elasticities(fit, which = "both", type = "AME",
                         n_cores = rpbnb_threads())

# ---- 6. Residual diagnostics ------------------------------------------------
# Randomized quantile residuals (Dunn-Smyth) are the primary count-model
# residual (~ N(0,1) under a correct fit). plot() writes four panels per margin
# (residuals-vs-fitted, normal QQ of the RQR, RQR histogram, scale-location);
# bnb_residual_checks() reports normality, dispersion, the cross-margin residual
# correlation, outliers, and a composite misspecification verdict. The RQR
# randomization is seeded so the diagnostics reproduce across runs.
sep(); cat("RESIDUAL DIAGNOSTICS\n"); sep()
dir.create("results", showWarnings = FALSE)
resid_pdf <- file.path("results", "fit_rpbnb_same_famoye_residuals.pdf")
grDevices::pdf(resid_pdf, width = 9, height = 7)
plot(fit, margin = "both", seed = 20240712)
grDevices::dev.off()
cat("Residual diagnostic plots written to", resid_pdf, "\n\n")
bnb_residual_checks(fit, seed = 20240712)
