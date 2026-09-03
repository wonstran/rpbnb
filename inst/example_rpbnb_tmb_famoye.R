#!/usr/bin/env Rscript
# =============================================================================
# example_rpbnb_tmb_famoye.R -- random-parameter bivariate NB (RP-BNB), TMB
# engine, method = "sml", Famoye/Sarmanov dependence
#
# Estimates the same random-parameter bivariate NB2 model as
# inst/example_rpbnb_classic.R's Famoye fit -- same data (rwm1984.csv), same
# dummy variables (see inst/example_bnb.R), same formulas, same random
# coefficient (kids, in both equations) -- but via the TMB engine
# (fit_rpbnb_tmb() / rpbnb(engine = "tmb")) instead of the classic engine, and
# with method = "sml" explicit rather than relying on it being the default.
#
# This is the Famoye/Sarmanov side of the TMB engine's SML estimator; see
# inst/example_rpbnb_tmb_sml.R for the copula-dependence counterpart (the two
# were one script until split so each dependence structure -- and its own
# boundary-test cost -- can be run independently rather than paying for both
# every time).
#
# The TMB engine has two estimators for the random-coefficient integral:
# "sml" (simulated maximum likelihood over Halton draws, the same integral
# the classic engine approximates, but with an exact automatic-differentiation
# gradient instead of a numerical one) and "laplace" (a sparse-Hessian
# approximation that removes `draws` from the memory cost -- see
# ?fit_rpbnb_tmb's `method` argument). This script is the "sml" side of that
# choice: it should give estimates close to inst/example_rpbnb_classic.R's
# (same integral, same data, same draws), not close to a Laplace fit of the
# same model, which approximates a different way and need not agree closely
# on any given dataset (see ?fit_rpbnb_tmb). The two scripts' boundary tests
# do NOT match, though: this one narrows LR_TEST to "sd" (2 refits), while
# the classic script uses boundary_tests = TRUE (sd + dispersion, 4 refits)
# -- see LR_TEST below to widen this one to match.
#
# `kids` is loaded as random in BOTH equations: under Famoye/Sarmanov
# dependence, a normal/lognormal random coefficient present in both margins
# makes the admissible lambda interval the constant c(-1, 1) (see
# ?fit_rpbnb_tmb's Return section, `lambda_bounds`), which the optimizer can
# never escape. A single-margin random coefficient does not have that
# guarantee.
#
# Run with:   Rscript inst/example_rpbnb_tmb_famoye.R
# or, after install:
#   Rscript -e 'source(system.file("example_rpbnb_tmb_famoye.R", package = "rpbnb"))'
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
# rpbnb_control()'s n_cores is the OpenMP thread count for both TMB estimators
# (see ?rpbnb_control); leave 1 core free for the rest of the system. TMB's
# se_method-equivalent is the `inference` argument to fit_rpbnb_tmb() (full
# covariance by default), not a control field -- control$se_method is a
# classic-engine-only knob and is ignored here.
N_CORES <- max(1L, parallel::detectCores() - 1L)
# Which boundary_tests group(s) to run -- see ?rpbnb's boundary_tests table.
# "sd" tests only the random-coefficient scales (sd1:kids, sd2:kids); add
# "dispersion" (or pass TRUE for both) to also test m1/m2, at the cost of two
# more restricted refits.
LR_TEST <- "sd"
ctrl <- rpbnb_control(n_cores = N_CORES)
cat("Using", N_CORES, "OpenMP threads,", DRAWS, "draws.\n\n")

# ---- Famoye/Sarmanov RP-BNB, TMB engine, SML ------------------------------------
# boundary_tests = LR_TEST runs rpbnb_tmb_boundary_tests() on the converged fit
# and attaches it as fit$boundary_tests, so summary() below fills in the
# sd1:kids/sd2:kids LR/df/p columns instead of leaving them blank -- no
# separate rpbnb_tmb_boundary_tests() call needed.
cat("=== Fitting RP-BNB (TMB engine, method = sml, Famoye/Sarmanov, random kids) ===\n")
t_famoye <- system.time(
  fit_famoye <- rpbnb(f1, f2, data = d, engine = "tmb", method = "sml",
                      random_1 = "kids", random_2 = "kids",
                      draws = DRAWS,
                      seed = SEED,
                      dependence = "famoye",
                      control = ctrl,
                      boundary_tests = LR_TEST)
)[["elapsed"]]
cat(sprintf("Estimation finished in %.1f s\n\n", t_famoye))
print(summary(fit_famoye))

# The same table summary() just read from, printed standalone: H0: sd(kids) = 0
# in each equation (kids' coefficient is not actually random). This null sits
# on the boundary of the parameter space, which is why summary() reports no
# Wald z/p for it without this test.
cat("\n=== Boundary LR tests (sd1:kids, sd2:kids) ===\n")
print(fit_famoye$boundary_tests)

# ---- Predicted means for a couple of profiles ----------------------------------
# Two married women, one working, one not, otherwise identical (one kid,
# post-Hauptschule schooling); MarM/SinF both 0 -> MarF = 1. Unlike the
# classic engine's predict.rpbnb_fit() (columns mu1/mu2), predict.rpbnb_tmb_fit()
# returns a matrix with columns y1/y2.
newdata <- data.frame(outwork = c(0, 1), kids = c(1, 1), postHS = c(1, 1),
                      MarM = c(0, 0), MarF = c(1, 1), SinF = c(0, 0))
cat("\n=== Predicted means (Famoye/Sarmanov fit) ===\n")
print(cbind(newdata, predict(fit_famoye, newdata = newdata)))
