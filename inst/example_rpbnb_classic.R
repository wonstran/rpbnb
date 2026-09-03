#!/usr/bin/env Rscript
# =============================================================================
# example_rpbnb_classic.R -- random-parameter bivariate NB (RP-BNB), classic
# engine
#
# Estimates a random-parameter bivariate NB2 model for the two count outcomes
# in the German health-care utilization panel (rwm1984.csv): number of doctor
# visits (docvis) and number of hospital visits (hospvis), via rpbnb() -- the
# front end that dispatches to fit_rpbnb() (this script's engine = "classic")
# or fit_rpbnb_tmb() (engine = "tmb"), validates arguments against whichever
# engine is selected, and (with boundary_tests = TRUE, used below) runs and
# attaches the boundary LR tests itself instead of a separate manual call.
# See the README's "Two random-parameter engines" section for how the engines
# compare, and inst/example_bnb.R for the fixed-coefficient counterpart this
# extends: `kids`' coefficient is normal-distributed and varies across
# individuals in BOTH equations, every other coefficient is fixed.
#
# `kids` is loaded as random in both margins deliberately, not just one: under
# Famoye/Sarmanov dependence, a normal/lognormal random coefficient present in
# BOTH margins makes the admissible lambda interval the constant c(-1, 1) (see
# ?fit_rpbnb's Return section), which is guaranteed never to be escaped by the
# optimizer. A single-margin random coefficient does not have that guarantee --
# an earlier version of this script with random_1 = "age" (equation 2 fixed)
# hit exactly this: the fitted lambda ended up outside the interval recomputed
# at the optimum, an un-interpretable fit flagged by a warning.
#
# This is a genuinely more expensive model class than inst/example_bnb.R's
# fixed-coefficient one -- the likelihood is simulated over `draws` Halton
# points per observation, evaluated many times during optimization -- so this
# script uses OpenMP multithreading (control$n_cores). boundary_tests = TRUE
# adds four more full restricted refits PER fit (sd1:kids, sd2:kids, m1, m2),
# so expect this script to take a while at DRAWS = 1000; lower DRAWS for a
# quick check.
#
# Run with:   Rscript inst/example_rpbnb_classic.R
# or, after install:
#   Rscript -e 'source(system.file("example_rpbnb_classic.R", package = "rpbnb"))'
# =============================================================================

#library(rpbnb)
devtools::load_all()

# ---- Load data ---------------------------------------------------------------
d <- read.csv(system.file("extdata", "rwm1984.csv", package = "rpbnb", mustWork = TRUE))

cat("=== rwm1984.csv ===\n")
cat("Observations:", nrow(d), "\n")
cat("docvis  (doctor visits) : mean =", round(mean(d$docvis), 2),
    " var =", round(var(d$docvis), 2), "\n")
cat("hospvis (hospital visits): mean =", round(mean(d$hospvis), 2),
    " var =", round(var(d$hospvis), 2), "\n\n")

# ---- Dummy variables from the raw categorical columns --------------------------
# Same derivation as inst/example_bnb.R -- see that script for the full
# explanation. postHS: schooling beyond the lowest track (edlevel >= 2). MarM/
# MarF/SinM/SinF: sex x marital-status, four mutually exclusive dummies; SinM
# (single male) is left out of the formulas below as the reference category.
d$postHS <- as.integer(d$edlevel >= 2)
d$MarM <- as.integer(d$married == 1 & d$female == 0)  # married male
d$MarF <- as.integer(d$married == 1 & d$female == 1)  # married female
d$SinM <- as.integer(d$married == 0 & d$female == 0)  # single male (reference)
d$SinF <- as.integer(d$married == 0 & d$female == 1)  # single female

f1 <- docvis  ~ outwork + kids + postHS + MarM + MarF + SinF
f2 <- hospvis ~ outwork + kids + postHS + MarM + MarF + SinF

DRAWS <- 1000
SEED  <- 1234
# rpbnb_control()'s n_cores is the OpenMP thread count for this simulated
# likelihood (see ?rpbnb_control); leave 1 core free for the rest of the
# system.
N_CORES <- max(1L, parallel::detectCores() - 1L)
# se_method is a classic-engine-only knob (fit_bnb uses `hessian` instead; see
# ?rpbnb_control). "analytic" -- the closed-form Louis-mixture Hessian -- is
# exact and much faster than the "numeric" default for a model this size, but
# it is Famoye-only: the copula path has no analytic Hessian and errors if
# asked for one, so its control keeps "opg" (fast and the family's own
# default since 0.4.6, unreliable only very near a boundary).
ctrl_famoye <- rpbnb_control(n_cores = N_CORES, se_method = "analytic")
ctrl_copula <- rpbnb_control(n_cores = N_CORES, se_method = "opg")
cat("Using", N_CORES, "OpenMP threads,", DRAWS, "draws.\n\n")

# ---- Famoye/Sarmanov RP-BNB ----------------------------------------------------
# boundary_tests = TRUE runs rpbnb_boundary_tests() on the converged fit and
# attaches it as fit$boundary_tests, so summary() below fills in the sd1:kids/
# sd2:kids/m1/m2 LR/df/p columns instead of leaving them blank -- no separate
# rpbnb_boundary_tests() call needed.
cat("=== Fitting RP-BNB (classic engine, Famoye/Sarmanov, random kids) ===\n")
t_famoye <- system.time(
  fit_famoye <- rpbnb(f1, f2, data = d, engine = "classic",
                      random_1 = "kids", random_2 = "kids",
                      draws = DRAWS,
                      seed = SEED,
                      dependence = "famoye",
                      control = ctrl_famoye,
                      boundary_tests = TRUE)
)[["elapsed"]]
cat(sprintf("Estimation finished in %.1f s\n\n", t_famoye))
print(summary(fit_famoye))

# The same table summary() just read from, printed standalone: H0: sd(kids) = 0
# in each equation (kids' coefficient is not actually random) and H0: m1/m2 = 0
# (Poisson limit). All four nulls sit on the boundary of the parameter space,
# which is why summary() reports no Wald z/p for them without this test.
cat("\n=== Boundary LR tests (sd1:kids, sd2:kids, m1, m2) ===\n")
print(fit_famoye$boundary_tests)

# ---- Copula dependence, for comparison ------------------------------------------
# Same random-coefficient specification and boundary_tests = TRUE, Gaussian-
# copula dependence instead of Famoye/Sarmanov. The copula likelihood is more
# expensive per evaluation (discrete-copula pmf + per-draw NB CDF corners), so
# this -- and its four boundary-test refits -- takes noticeably longer than
# the Famoye block above at the same draws/n.
cat("\n=== Fitting RP-BNB (classic engine, Gaussian copula, random kids) ===\n")
t_copula <- system.time(
  fit_copula <- rpbnb(f1, f2, data = d, engine = "classic",
                      random_1 = "kids", random_2 = "kids",
                      dependence = copula("normal"),
                      draws = DRAWS, seed = SEED,
                      control = ctrl_copula,
                      boundary_tests = TRUE)
)[["elapsed"]]
cat(sprintf("Estimation finished in %.1f s\n\n", t_copula))
print(summary(fit_copula))
cat("Estimated copula rho:", round(tanh(coef(fit_copula)[["z_theta"]]), 4), "\n")

cat("\n=== Boundary LR tests (sd1:kids, sd2:kids, m1, m2) ===\n")
print(fit_copula$boundary_tests)

# ---- Dependence structure comparison --------------------------------------------
cat("\n=== Dependence structure comparison ===\n")
cat(sprintf("%-14s  logLik = %10.2f   AIC = %10.2f   BIC = %10.2f   (%.1fs)\n",
            "famoye", fit_famoye$logLik, fit_famoye$AIC, fit_famoye$BIC, t_famoye))
cat(sprintf("%-14s  logLik = %10.2f   AIC = %10.2f   BIC = %10.2f   (%.1fs)\n",
            "copula/normal", fit_copula$logLik, fit_copula$AIC, fit_copula$BIC, t_copula))

# ---- Predicted means for a couple of profiles ----------------------------------
# Two married women, one working, one not, otherwise identical (one kid,
# post-Hauptschule schooling); MarM/SinF both 0 -> MarF = 1.
newdata <- data.frame(outwork = c(0, 1), kids = c(1, 1), postHS = c(1, 1),
                      MarM = c(0, 0), MarF = c(1, 1), SinF = c(0, 0))
cat("\n=== Predicted means (Famoye/Sarmanov fit) ===\n")
print(cbind(newdata, predict(fit_famoye, newdata = newdata)))
