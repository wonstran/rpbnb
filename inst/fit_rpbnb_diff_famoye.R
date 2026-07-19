#!/usr/bin/env Rscript
# =============================================================================
# rpbnb estimation demo -- DIFFERENT formulas, Famoye/Sarmanov dependence.
#
# Fits a random-parameter bivariate NB (RP-BNB) model to the German health-care
# counts in data/rwm1984_bnb.csv:
#   * y1 = docvis  (doctor visits), y2 = hospvis (hospital visits)
#   * the two equations use DIFFERENT covariate sets (the package does not
#     require the margins to share a design)
#   * a fixed + random coefficient mix: a random slope on one CONTINUOUS
#     covariate per equation (hhninc in eq 1, educ in eq 2); all others fixed
#   * Famoye/Sarmanov dependence between the two counts
#
# Usage demo on real data -- there is no known ground truth, so the script
# reports the fitted model, predictions, marginal effects, and residual
# diagnostics rather than an estimate-vs-truth comparison. Run from the package
# root:  Rscript inst/fit_rpbnb_diff_famoye.R
# =============================================================================

devtools::load_all(quiet = TRUE)

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")

# ---- 1. Data ----------------------------------------------------------------
data <- read.csv(file.path("data", "rwm1984_bnb.csv"))
cat("=== RP-BNB (different formulas, Famoye) on rwm1984 health counts ===\n")
cat("Observations :", nrow(data), "\n")
cat("Threads      :", rpbnb_threads(), "\n")

# ---- 2. Model specification -------------------------------------------------
# The two equations carry DIFFERENT covariates. Random coefficients sit on
# CONTINUOUS regressors only (dummies would be weakly identified: NB dispersion
# vs random-coefficient scale), and each random covariate must appear in its
# own equation's formula.
f1 <- docvis  ~ age + hhninc + educ + female + married + kids
f2 <- hospvis ~ age + educ + outwork + female + self
cat("Equation 1   :", deparse(f1), "\n")
cat("Equation 2   :", deparse(f2), "\n\n")

t_fit <- system.time(
  fit <- fit_rpbnb(
    formula_1  = f1,
    formula_2  = f2,
    data       = data,
    random_1   = "hhninc",   # random slope on hhninc (eq 1), continuous, in f1
    random_2   = "educ",     # random slope on educ   (eq 2), continuous, in f2
    dependence = "famoye",
    draws      = 500,
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

# ---- 3b. Significance of the boundary parameters (LR tests) -----------------
# The natural-scale table shows no Wald z/p for the random-coefficient SDs
# (sd1:hhninc, sd2:educ) or the NB2 dispersions (m1, m2): those are positive
# parameters whose null sits on the boundary of the parameter space (sd = 0, or
# m = 0 = Poisson), where the Wald ratio does not test that null. The correct
# test is a likelihood-ratio test against a properly nested restricted fit, with
# the 50:50 chi-square boundary correction (Self & Liang 1987). Each restricted
# refit reuses the SAME draws / seed / control as the full model, so the two
# simulated log-likelihoods are compared on common random numbers.
#   * SD tests:         drop the random slope (random_1 / random_2).
#   * Dispersion tests: fit that margin at its Poisson limit (poisson_1 /
#                       poisson_2 pins m = 0), testing for overdispersion.
sep(); cat("BOUNDARY-PARAMETER SIGNIFICANCE (boundary-corrected LR tests)\n"); sep()

# SEs are not needed for the restricted fits (lr_test uses logLik + df only), so
# skip them to save time.
lr_ctrl <- rpbnb_control(print_level = 0, n_cores = rpbnb_threads(),
                         compute_se = FALSE)

# Random-coefficient SDs: drop the random slope.
rest_no_sd1 <- fit_rpbnb(f1, f2, data = data, random_2 = "educ",
                         dependence = "famoye", draws = 500, seed = 20240712,
                         control = lr_ctrl)                  # drops sd1:hhninc
rest_no_sd2 <- fit_rpbnb(f1, f2, data = data, random_1 = "hhninc",
                         dependence = "famoye", draws = 500, seed = 20240712,
                         control = lr_ctrl)                  # drops sd2:educ

# NB2 dispersions: fit the margin at its Poisson limit (m = 0).
rest_pois1 <- fit_rpbnb(f1, f2, data = data, random_1 = "hhninc", random_2 = "educ",
                        dependence = "famoye", draws = 500, seed = 20240712,
                        control = lr_ctrl, poisson_1 = TRUE) # pins m1 = 0
rest_pois2 <- fit_rpbnb(f1, f2, data = data, random_1 = "hhninc", random_2 = "educ",
                        dependence = "famoye", draws = 500, seed = 20240712,
                        control = lr_ctrl, poisson_2 = TRUE) # pins m2 = 0

cat("\nH0: sd1:hhninc = 0  (no random slope on hhninc, equation 1)\n")
print(lr_test(rest_no_sd1, fit, boundary = TRUE))
cat("\nH0: sd2:educ = 0  (no random slope on educ, equation 2)\n")
print(lr_test(rest_no_sd2, fit, boundary = TRUE))
cat("\nH0: m1 = 0  (equation 1 is Poisson, no overdispersion)\n")
print(lr_test(rest_pois1, fit, boundary = TRUE))
cat("\nH0: m2 = 0  (equation 2 is Poisson, no overdispersion)\n")
print(lr_test(rest_pois2, fit, boundary = TRUE))

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
resid_pdf <- file.path("results", "fit_rpbnb_diff_famoye_residuals.pdf")
grDevices::pdf(resid_pdf, width = 9, height = 7)
plot(fit, margin = "both", seed = 20240712)
grDevices::dev.off()
cat("Residual diagnostic plots written to", resid_pdf, "\n\n")
bnb_residual_checks(fit, seed = 20240712)
