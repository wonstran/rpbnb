#!/usr/bin/env Rscript
# =============================================================================
# example_bnb.R -- fixed-coefficient bivariate negative binomial (BNB) model
#
# Estimates a bivariate NB2 model for the two count outcomes in the German
# health-care utilization panel (rwm1984.csv): number of doctor visits
# (docvis) and number of hospital visits (hospvis). fit_bnb() is the
# fixed-coefficient estimator -- every covariate has one coefficient shared
# across observations (contrast with fit_rpbnb()/fit_rpbnb_tmb(), which let
# selected coefficients vary by observation; see the README's "Two
# random-parameter engines" section).
#
# Run with:   Rscript inst/example_bnb.R
# or, after install:
#   Rscript -e 'source(system.file("example_bnb.R", package = "rpbnb"))'
# =============================================================================

#library(rpbnb)
devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

# ---- Load data ---------------------------------------------------------------
# rwm1984.csv is the raw panel (see rwm1984_clean.csv for a version with
# derived dummy variables used elsewhere in the package's own examples).
d <- read.csv(system.file("extdata", "rwm1984.csv", package = "rpbnb", mustWork = TRUE))

cat("=== rwm1984.csv ===\n")
cat("Observations:", nrow(d), "\n")
cat("Columns     :", paste(names(d), collapse = ", "), "\n\n")
cat("docvis  (doctor visits) : mean =", round(mean(d$docvis), 2),
    " var =", round(var(d$docvis), 2), "\n")
cat("hospvis (hospital visits): mean =", round(mean(d$hospvis), 2),
    " var =", round(var(d$hospvis), 2), "\n\n")

# ---- Dummy variables from the raw categorical columns --------------------------
# rwm1984.csv already carries edlevel1..edlevel4 (one dummy per raw edlevel
# category) but leaves the categorical covariates themselves for the user to
# turn into model-ready dummies -- this is the same derivation
# rwm1984_clean.csv ships pre-computed, done by hand here to show the pattern.
#
# postHS: schooling beyond the lowest track (edlevel == 1, no Hauptschule
# degree), i.e. edlevel 2, 3, or 4.
d$postHS <- as.integer(d$edlevel >= 2)

# Sex x marital-status interaction, as four mutually exclusive dummies. Only
# three enter the formula below -- SinM (single male) is left out as the
# reference category, so its effect is absorbed into the intercept.
d$MarM <- as.integer(d$married == 1 & d$female == 0)  # married male
d$MarF <- as.integer(d$married == 1 & d$female == 1)  # married female
d$SinM <- as.integer(d$married == 0 & d$female == 0)  # single male (reference)
d$SinF <- as.integer(d$married == 0 & d$female == 1)  # single female

f1 <- docvis  ~ outwork + kids + age + postHS + MarM + MarF + SinF
f2 <- hospvis ~ outwork + kids + age + postHS + MarM + MarF + SinF

# ---- Baseline: independent margins -------------------------------------------
# Two separate NB2 fits with no dependence between docvis and hospvis.
fit_indep <- fit_bnb(f1, f2, data = d, dependence = "independence")

# ---- Famoye/Sarmanov bivariate NB ---------------------------------------------
# Adds one dependence parameter (lambda) linking the two margins.
# hessian is a control-object field, not a fit_bnb() argument; pass it via
# rpbnb_control(). "analytic" uses the package's closed-form Hessian.
fit_famoye <- fit_bnb(f1, f2, data = d,
                      dependence = "famoye",
                      boundary_tests = TRUE,
                      control = rpbnb_control(hessian = "analytic"))

cat("=== Famoye/Sarmanov bivariate NB ===\n")
print(summary(fit_famoye))

# Likelihood-ratio test for dependence (H0: lambda = 0, i.e. independence).
# boundary = TRUE applies the correct mixture-chi-squared reference
# distribution for a null on the boundary of the parameter space.
cat("\n=== LR test: independence vs. Famoye/Sarmanov dependence ===\n")
print(lr_test(fit_indep, fit_famoye, boundary = TRUE))

# ---- Goodness of fit ----------------------------------------------------------
cat("\n=== Goodness of fit (Famoye/Sarmanov) ===\n")
bnb_gof(fit_famoye)

# ---- Copula dependence, for comparison -----------------------------------------
# Note: bnb_boundary_tests() (the dispersion boundary test) is NOT applicable
# here -- it only tests the NB2 dispersions m1/m2 against the Poisson limit,
# and fit_bnb() refuses to combine that Poisson restriction with a copula
# dependence, so there is no dispersion boundary test to run for these fits.
# The dependence null is instead tested below with lr_test(boundary = FALSE).
# Three copula families with a single dependence parameter (native copula
# parameter + the induced Kendall's tau): Gaussian (normal), Frank, and
# Kimeldorf. (The copula() registry supports exactly these three; there is no
# Clayton family, so a Clayton comparison is not available in this package.)
fit_cop_norm <- fit_bnb(f1, f2, data = d, dependence = copula("normal"))
cat("=== Copula normal bivariate NB ===\n")
print(summary(fit_famoye))

fit_cop_frank <- fit_bnb(f1, f2, data = d, dependence = copula("frank"))
cat("=== Copula frank bivariate NB ===\n")
print(summary(fit_famoye))

fit_cop_kim  <- fit_bnb(f1, f2, data = d, dependence = copula("kimeldorf"))
cat("=== Copula kimeldorf bivariate NB ===\n")
print(summary(fit_famoye))

# One row per dependence structure. Copula rows show the induced Kendall's tau;
# the non-copula rows show NA there.
dep_rows <- list(
  list(model = "independence",  ll = fit_indep$logLik,    aic = fit_indep$AIC,    bic = fit_indep$BIC,    tau = NA_real_),
  list(model = "famoye",        ll = fit_famoye$logLik,   aic = fit_famoye$AIC,   bic = fit_famoye$BIC,   tau = NA_real_),
  list(model = "copula/normal", ll = fit_cop_norm$logLik, aic = fit_cop_norm$AIC, bic = fit_cop_norm$BIC, tau = fit_cop_norm$cop_tau),
  list(model = "copula/frank",  ll = fit_cop_frank$logLik,aic = fit_cop_frank$AIC,bic = fit_cop_frank$BIC,tau = fit_cop_frank$cop_tau),
  list(model = "copula/kimeldorf", ll = fit_cop_kim$logLik, aic = fit_cop_kim$AIC, bic = fit_cop_kim$BIC, tau = fit_cop_kim$cop_tau)
)

cat("\n=== Dependence structure comparison ===\n")
for (r in dep_rows) {
  cat(sprintf("%-14s  logLik = %10.2f   AIC = %10.2f   BIC = %10.2f",
              r$model, r$ll, r$aic, r$bic))
  if (!is.na(r$tau)) cat(sprintf("   (tau = %.3f)", r$tau))
  cat("\n")
}

# ---- LR tests: independence vs. each copula ------------------------------------
# Each copula adds one dependence parameter, so each is a 1-df test against the
# independence fit. boundary = FALSE: the null (dependence = 0, i.e. tau = 0)
# is an interior point of the copula parameter space, so the ordinary
# chi-squared(1) reference applies -- not the mixture distribution reserved for
# a null on a support boundary (the Famoye lambda >= 0 case above).
cat("\n=== LR tests: independence vs. copula dependence ===\n")
for (f in list(list("copula/normal",    fit_cop_norm),
               list("copula/frank",     fit_cop_frank),
               list("copula/kimeldorf", fit_cop_kim))) {
  cat(sprintf("\n--- %s ---\n", f[[1]]))
  print(lr_test(fit_indep, f[[2]], boundary = FALSE))
}

# ---- Predicted means for a couple of profiles ----------------------------------
# Two married women, one working, one not, otherwise identical (age 40, no
# kids, post-Hauptschule schooling); MarM/SinM/SinF all 0 -> MarF = 1.
newdata <- data.frame(outwork = c(0, 1), kids = c(0, 0), age = c(40, 40),
                       postHS = c(1, 1), MarM = c(0, 0), MarF = c(1, 1),
                       SinF = c(0, 0))

cat("\n=== Predicted means (Famoye/Sarmanov fit) ===\n")
print(cbind(newdata, predict(fit_famoye, newdata = newdata)))
