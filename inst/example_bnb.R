#!/usr/bin/env Rscript
# =============================================================================
# example_bnb.R -- bivariate negative binomial regression (fit_bnb()),
# Famoye/Sarmanov OR copula dependence
#
# Estimates a (non-random-parameter) bivariate NB2 model for the two count
# outcomes in the German health-care utilization panel (rwm1984.csv), via
# fit_bnb(): an exact-likelihood maxLik::maxLik(method = "BFGS") fit with an
# analytic gradient (no simulation, no draws, no engine choice -- unlike the
# random-parameter RP-BNB models in inst/example_rpbnb_*.R, which all refer
# back to this script for the dummy-variable derivation below).
#
# DEPENDENCE below picks exactly ONE dependence structure to fit:
#   "famoye" -- Famoye/Sarmanov (fit_bnb()'s default `dependence`)
#   "copula" -- a copula() object (COPULA_FAMILY below; "normal" = Gaussian)
# Flip DEPENDENCE and re-run to fit the other; the two are not fit together
# in one run of this script. bnb_boundary_tests() (a Poisson-limit,
# `m = 0`, LR test on each margin's dispersion) is only defined for Famoye
# (and independence) -- fit_bnb() itself refuses poisson_1/poisson_2 with a
# copula() dependence, so there is no Poisson-limit restriction to test; this
# script skips it when DEPENDENCE == "copula" rather than erroring.
#
# `kids` (and every other regressor here) is an ordinary fixed coefficient --
# fit_bnb() has no random-coefficient argument at all (that is what
# random_1/random_2 and rpbnb()/fit_rpbnb() add on top of this model).
#
# Run with:   Rscript inst/example_bnb.R
# or, after install:
#   Rscript -e 'source(system.file("example_bnb.R", package = "rpbnb"))'
# =============================================================================

# Load the package. Use library(rpbnb) once the package is installed; while
# developing against the current source tree (not yet installed), comment the
# library() line out and uncomment devtools::load_all() instead -- it rebuilds
# and loads the package from this working copy.
library(rpbnb)
# devtools::load_all()

# ---- Load data ---------------------------------------------------------------
d <- read.csv(system.file("extdata", "rwm1984.csv", package = "rpbnb", mustWork = TRUE))

cat("=== rwm1984.csv ===\n")
cat("Observations:", nrow(d), "\n")
cat("docvis  (doctor visits) : mean =", round(mean(d$docvis), 2),
    " var =", round(var(d$docvis), 2), "\n")
cat("hospvis (hospital visits): mean =", round(mean(d$hospvis), 2),
    " var =", round(var(d$hospvis), 2), "\n\n")

# ---- Dummy variables from the raw categorical columns --------------------------
# edlevel is a 1..4 schooling-track code; postHS collapses it to "beyond the
# lowest track" (edlevel >= 2), since the lowest track (edlevel == 1, no more
# than a Hauptschule/basic-secondary leaving certificate) is the modal group
# and finer track distinctions are not of interest here.
#
# married and female are each already 0/1 in the raw data. Combining them
# gives four mutually exclusive sex x marital-status dummies (every
# observation falls in exactly one): MarM (married male), MarF (married
# female), SinM (single male), SinF (single female, where "single" here
# covers not-currently-married generally, including divorced/widowed, per the
# rwm1984 coding). SinM is left out of the formulas below as the reference
# category, so MarM/MarF/SinF are read relative to a single man.
d$postHS <- as.integer(d$edlevel >= 2)
d$MarM <- as.integer(d$married == 1 & d$female == 0)  # married male
d$MarF <- as.integer(d$married == 1 & d$female == 1)  # married female
d$SinM <- as.integer(d$married == 0 & d$female == 0)  # single male (reference)
d$SinF <- as.integer(d$married == 0 & d$female == 1)  # single female

f1 <- docvis  ~ outwork + kids + postHS + MarM + MarF + SinF
f2 <- hospvis ~ outwork + kids + postHS + MarM + MarF + SinF

# Which dependence structure to fit -- see this script's header. Flip to
# "copula" (and, optionally, COPULA_FAMILY below) to switch.
DEPENDENCE <- "copula"  # "famoye" or "copula"

# Which copula to fit when DEPENDENCE == "copula"; ignored otherwise.
# copula() accepts exactly these three families; everything downstream
# (labels, the reported native parameter, Kendall's tau) adapts to whichever
# is set here.
#   "frank"     -- theta on the whole real line, no tail dependence
#   "normal"    -- Gaussian, rho in (-1, 1)
#   "kimeldorf" -- Kimeldorf-Sampson (Clayton), theta > 0, lower-tail dependence
COPULA_FAMILY <- "normal"

dependence_arg <- if (identical(DEPENDENCE, "copula")) copula(COPULA_FAMILY) else "famoye"
dependence_label <- if (identical(DEPENDENCE, "copula")) paste(COPULA_FAMILY, "copula") else "Famoye/Sarmanov"

# Run bnb_boundary_tests() (Poisson-limit LR test on log_m1/log_m2) after the
# fit -- Famoye only; see this script's header. copula's fit_bnb() call below
# instead sets boundary_tests = FALSE and the print block further down is
# skipped for it.
BOUNDARY_TESTS <- !identical(DEPENDENCE, "copula")

# fit_bnb() ignores rpbnb_control()'s se_method/n_cores/TMB knobs entirely
# (see ?fit_bnb) -- the exact likelihood needs neither simulation draws nor
# OpenMP threads -- so the default control() is enough here.
ctrl <- rpbnb_control()

# ---- BNB, Famoye/Sarmanov or copula dependence ----------------------------------
# boundary_tests = BOUNDARY_TESTS runs bnb_boundary_tests() on the converged
# fit and attaches it as fit$boundary_tests, so summary() below fills in the
# m1/m2 LR/df/p columns instead of leaving them blank -- no separate
# bnb_boundary_tests() call needed.
cat(sprintf("=== Fitting BNB (%s, fixed coefficients) ===\n", dependence_label))
t_fit <- system.time(
  fit <- fit_bnb(f1, f2, data = d,
                dependence = dependence_arg,
                control = ctrl,
                boundary_tests = BOUNDARY_TESTS)
)[["elapsed"]]
cat(sprintf("Estimation finished in %.1f s\n\n", t_fit))
print(summary(fit))

if (BOUNDARY_TESTS) {
  cat(sprintf("\n=== Boundary LR tests (%s fit) ===\n", dependence_label))
  print(fit$boundary_tests)
}

# ---- Goodness of fit -------------------------------------------------------------
cat(sprintf("\n=== Goodness of fit (%s fit) ===\n", dependence_label))
bnb_gof(fit)

# ---- Marginal effects and elasticities -------------------------------------------
cat(sprintf("\n=== Average marginal effects (%s fit) ===\n", dependence_label))
bnb_marginal_effects(fit, which = "both")

cat(sprintf("\n=== Elasticities / semi-elasticities (%s fit) ===\n", dependence_label))
bnb_elasticities(fit, which = "both")

# ---- Predicted means for a couple of profiles ------------------------------------
# Two married women, one working, one not, otherwise identical (one kid,
# post-Hauptschule schooling); MarM/SinF both 0 -> MarF = 1.
# predict.bnb_fit() returns a data frame with columns mu1/mu2.
newdata <- data.frame(outwork = c(0, 1), kids = c(1, 1), postHS = c(1, 1),
                      MarM = c(0, 0), MarF = c(1, 1), SinF = c(0, 0))
cat(sprintf("\n=== Predicted means (%s fit) ===\n", dependence_label))
print(cbind(newdata, predict(fit, newdata = newdata)))

print(rpbnb_build_info())
