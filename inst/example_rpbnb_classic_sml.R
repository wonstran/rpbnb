#!/usr/bin/env Rscript
# =============================================================================
# example_rpbnb_classic_sml.R -- random-parameter bivariate NB (RP-BNB),
# classic engine, Famoye/Sarmanov OR copula dependence
#
# Estimates a random-parameter bivariate NB2 model for the two count outcomes
# in the German health-care utilization panel (rwm1984.csv), via the classic
# engine (fit_rpbnb() / rpbnb(engine = "classic")): a multithreaded (OpenMP)
# Rcpp simulated likelihood optimized by maxLik::maxLik(method = "BFGS") with
# a numerical gradient. Same data, dummy variables (see inst/example_bnb.R),
# formulas, and random coefficient (kids, in both equations) as
# inst/example_rpbnb_tmb_famoye.R and inst/example_rpbnb_tmb_sml.R -- those
# two scripts fit the TMB-engine counterparts of this script's Famoye and
# copula settings respectively (and refer back to this script as their
# classic-engine baseline).
#
# DEPENDENCE below picks exactly ONE dependence structure to fit:
#   "famoye" -- Famoye/Sarmanov (the classic engine's default `dependence`)
#   "copula" -- a copula() object (COPULA_FAMILY below; "normal" = Gaussian,
#               to match inst/example_rpbnb_tmb_sml.R's default)
# Flip DEPENDENCE and re-run to fit the other; the two are not fit together
# in one run of this script. The copula path is more numerically expensive
# per iteration than Famoye at the same draws/n (discrete-copula pmf plus a
# per-draw NB CDF corner; see ?fit_rpbnb's `dependence` argument).
#
# At the same draws and seed (so both engines draw the same Halton sequence),
# this script's Famoye fit should give estimates close to
# inst/example_rpbnb_tmb_famoye.R's, and this script's copula fit close to
# inst/example_rpbnb_tmb_sml.R's -- both engines approximate the same
# simulated-likelihood integral, the TMB engine just differentiates it exactly
# instead of numerically. Neither comparison holds against
# inst/example_rpbnb_tmb_laplace.R, which approximates the integral a
# different way (Laplace, not simulation) and need not agree closely on any
# given dataset (see ?fit_rpbnb_tmb).
#
# `kids` is loaded as random in BOTH equations, for consistency with the TMB
# scripts and to avoid identifying the random coefficient off only one
# margin's variation. As those scripts note, this cuts against ?fit_rpbnb's
# own advice for the copula path specifically: `kids` is a 0/1 dummy in this
# data, and a random coefficient on a dummy regressor is weakly identified
# under copula dependence (NB dispersion trades off against the
# random-coefficient scale) -- kept anyway here so the specification lines up
# across engines and dependence structures; a continuous regressor would
# identify more cleanly.
#
# boundary_tests = TRUE runs both the "sd" (random-coefficient scales) and
# "dispersion" (NB2 overdispersions m1/m2) LR-test groups on the converged fit
# via rpbnb_boundary_tests() -- 4 restricted refits (2 scales + 2
# dispersions), roughly quintupling the fit's total cost. This is
# deliberately wider than inst/example_rpbnb_tmb_famoye.R and
# inst/example_rpbnb_tmb_sml.R, which narrow LR_TEST to "sd" (2 refits) to
# keep their own boundary-test cost down; set BOUNDARY_TESTS to "sd" below to
# match those scripts instead.
#
# Run with:   Rscript inst/example_rpbnb_classic_sml.R
# or, after install:
#   Rscript -e 'source(system.file("example_rpbnb_classic_sml.R", package = "rpbnb"))'
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

# Which boundary_tests group(s) to run after the fit -- see ?rpbnb's
# boundary_tests table. TRUE = c("sd", "dispersion"), 4 restricted refits;
# narrow to "sd" to match inst/example_rpbnb_tmb_famoye.R and
# inst/example_rpbnb_tmb_sml.R's cheaper 2-refit boundary test.
BOUNDARY_TESTS <- "sd"

# rpbnb_control()'s n_cores is the OpenMP thread count for the classic
# engine's Rcpp simulated likelihood (see ?rpbnb_control); leave 1 core free
# for the rest of the system. se_method is set explicitly per dependence
# structure rather than left at its NULL default: "analytic" for Famoye (the
# closed-form Famoye (2010) Hessian -- exact and much faster than "numeric"),
# "opg" for copula (no analytic Hessian is implemented for the copula path --
# it errors -- and "opg" is markedly faster than "numeric" there; see
# ?rpbnb_control's se_method for the boundary-parameter caveat on "opg").
N_CORES <- max(1L, parallel::detectCores() - 1L)
se_method <- if (identical(DEPENDENCE, "copula")) "opg" else "analytic"
ctrl <- rpbnb_control(n_cores = N_CORES, se_method = se_method)
cat("Using", N_CORES, "OpenMP threads,", DRAWS, "draws.\n\n")

# ---- RP-BNB, classic engine -----------------------------------------------------
# boundary_tests = BOUNDARY_TESTS runs rpbnb_boundary_tests() on the converged
# fit and attaches it as fit$boundary_tests, so summary() below fills in the
# sd1:kids/sd2:kids (and, at TRUE, log_m1/log_m2) LR/df/p columns instead of
# leaving them blank -- no separate rpbnb_boundary_tests() call needed.
cat(sprintf(
  "=== Fitting RP-BNB (classic engine, %s, random kids) ===\n", dependence_label))
t_fit <- system.time(
  fit <- rpbnb(f1, f2, data = d, engine = "classic",
              random_1 = "kids", random_2 = "kids",
              draws = DRAWS,
              seed = SEED,
              dependence = dependence_arg,
              control = ctrl,
              boundary_tests = BOUNDARY_TESTS)
)[["elapsed"]]
cat(sprintf("Estimation finished in %.1f s\n\n", t_fit))
print(summary(fit))

cat(sprintf("\n=== Boundary LR tests (%s fit) ===\n", dependence_label))
print(fit$boundary_tests)

# ---- Predicted means for a couple of profiles ----------------------------------
# Two married women, one working, one not, otherwise identical (one kid,
# post-Hauptschule schooling); MarM/SinF both 0 -> MarF = 1.
# predict.rpbnb_fit() (classic engine) returns a data frame with columns
# mu1/mu2 -- unlike predict.rpbnb_tmb_fit()'s y1/y2 (see
# inst/example_rpbnb_tmb_famoye.R).
newdata <- data.frame(outwork = c(0, 1), kids = c(1, 1), postHS = c(1, 1),
                      MarM = c(0, 0), MarF = c(1, 1), SinF = c(0, 0))
cat(sprintf("\n=== Predicted means (%s fit) ===\n", dependence_label))
print(cbind(newdata, predict(fit, newdata = newdata)))

print(rpbnb_build_info())
