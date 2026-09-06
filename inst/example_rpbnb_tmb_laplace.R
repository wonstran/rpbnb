#!/usr/bin/env Rscript
# =============================================================================
# example_rpbnb_tmb_laplace.R -- random-parameter bivariate NB (RP-BNB), TMB
# engine, method = "laplace", copula dependence
#
# Estimates a random-parameter bivariate NB2 model for the two count outcomes
# in the German health-care utilization panel (rwm1984.csv) under copula
# dependence (COPULA_FAMILY below), via the TMB engine's Laplace-approximation
# estimator. Same data, dummy variables (see inst/example_bnb.R), formulas,
# and random coefficient (kids, in both equations) as
# inst/example_rpbnb_tmb_sml.R -- this one is the "laplace" side of that
# script's estimator choice; run the two side by side to compare the two
# approximations on the same model.
#
# The TMB engine has two estimators for the random-coefficient integral:
# "sml" (simulated maximum likelihood over Halton draws, the same integral
# the classic engine approximates, but with an exact automatic-differentiation
# gradient instead of a numerical one) and "laplace" (a sparse-Hessian
# approximation over one latent vector per observation that removes `draws`
# from the memory cost -- tape size scales with nrow(data) alone, see
# ?fit_rpbnb_tmb's `method` argument). This script is the "laplace" side of
# that choice. The two are DIFFERENT approximations to the same integral:
# they agree asymptotically but need not agree closely on any given dataset,
# so a Laplace fit is not a drop-in reproduction of an SML fit. summary()
# still reports AIC/BIC, but under "laplace" they are computed from an
# approximated marginal likelihood, so an AIC from a Laplace fit is not
# meaningful to compare against an AIC from an SML fit of the same model.
# "laplace" also supports "normal" and "lognormal" random coefficients only
# and requires at least one random coefficient -- `kids` is Normal in both
# equations, so both hold here.
#
# `kids` is loaded as random in BOTH equations, for consistency with
# inst/example_rpbnb_tmb_sml.R and to avoid identifying the random
# coefficient off only one margin's variation. Note this cuts against
# ?fit_rpbnb_tmb's own advice for the copula path specifically: `kids` is a
# 0/1 dummy in this data, and a random coefficient on a dummy regressor is
# weakly identified under copula dependence (NB dispersion trades off against
# the random-coefficient scale) -- kept anyway here so the two scripts'
# specifications line up; a continuous regressor would identify more cleanly.
#
# Run with:   Rscript inst/example_rpbnb_tmb_laplace.R
# or, after install:
#   Rscript -e 'source(system.file("example_rpbnb_tmb_laplace.R", package = "rpbnb"))'
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

# `draws`/`seed` (fit_rpbnb_tmb()'s Halton-grid arguments) are left at their
# defaults here on purpose. Under method = "laplace" `draws` does NOT affect
# the likelihood, the TMB tape, or any chunking (tape size scales with
# nrow(data) alone); its only remaining job is sizing the Halton grid used by
# predict() and the marginal-effect functions for post-estimation averaging
# (the frozen Famoye admissible-lambda interval is derived from the random
# coefficients' support instead, independent of draws/seed under either
# estimator -- see ?fit_rpbnb_tmb's `draws` argument). This script calls
# neither predict() nor a marginal-effect function, so draws/seed have no
# observable effect on anything it prints; set them explicitly only once you
# add such a call and want its averaging grid reproducible or sized to taste.
# Which copula to fit. copula() accepts exactly these three families;
# everything downstream (labels, the reported native parameter, Kendall's
# tau) adapts to whichever is set here.
#   "frank"     -- theta on the whole real line, no tail dependence
#   "normal"    -- Gaussian, rho in (-1, 1)
#   "kimeldorf" -- Kimeldorf-Sampson (Clayton), theta > 0, lower-tail dependence
COPULA_FAMILY <- "normal"

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

# ---- TMB engine control ------------------------------------------------------
# One rpbnb_control() object drives every estimator; here only the thread
# fields matter for this fit (n_cores, max_threads). The SML-only memory
# knobs (max_workload, tape_chunks) are NOT set: under method = "laplace" the
# tape scales with nrow(data) alone, so there is no draw dimension to chunk
# and the weighted-workload guard bounds nothing -- .resolve_tape_chunks()
# reports them as ignored rather than erroring (see ?rpbnb_control).
ctrl <- rpbnb_control(
  n_cores = N_CORES,
  # OpenMP thread count for both TMB estimators (see ?rpbnb_control). Left 1
  # core free above via N_CORES.
  max_threads = N_CORES,
  # Ceiling on OpenMP threads permitted for one TMB fit. NULL would default to
  # n_cores; set equal here so the ceiling never binds below the request.
  # (max_workload / tape_chunks deliberately omitted: SML-only, ignored for
  # method = "laplace".)
)
cat("Using", N_CORES, "OpenMP threads.\n\n")

# ---- Copula RP-BNB, TMB engine, Laplace -----------------------------------------
# method = "laplace" uses TMB's Laplace approximation with a sparse Hessian
# over one latent vector per observation; tape size scales with nrow(data)
# alone, so this fit uses far less peak memory than the SML script's.
# boundary_tests = LR_TEST runs rpbnb_tmb_boundary_tests() on the converged fit
# and attaches it as fit$boundary_tests, so summary() below fills in the
# sd1:kids/sd2:kids LR/df/p columns instead of leaving them blank -- no
# separate rpbnb_tmb_boundary_tests() call needed. (Gaussian-copula evaluation
# runs multithreaded by default since 0.4.6 -- see NEWS.md -- so
# COPULA_FAMILY = "normal" needs no extra argument either.)
cat(sprintf(
  "=== Fitting RP-BNB (TMB engine, method = laplace, %s copula, random kids) ===\n",
  COPULA_FAMILY))
t_copula <- system.time(
  fit_copula <- rpbnb(f1, f2, data = d, engine = "tmb", method = "laplace",
                      random_1 = "kids",
                      random_2 = "kids",
                      dependence = copula(COPULA_FAMILY),
                      control = ctrl,
                      boundary_tests = LR_TEST)
)[["elapsed"]]
cat(sprintf("Estimation finished in %.1f s\n\n", t_copula))
print(summary(fit_copula))

# ---- Threading diagnostics ----------------------------------------------------
# fit$parallel$realized can come in below $requested for reasons that are easy
# to conflate: no OpenMP in this build (openmp: FALSE below -- affects every
# family, e.g. Apple clang with no libomp linked), fewer threads actually
# supported than requested (TMB::openmp(max = TRUE) returned less than
# max_threads), or -- Gaussian copula only, and only on a pre-0.4.6 install --
# the single-thread safety cap that used to apply unconditionally (see
# ?fit_rpbnb_tmb's `disable_parallel_gaussian`) -- directly relevant here
# since COPULA_FAMILY = "normal" above is exactly the case that cap affected.
cat(sprintf("Threads requested: %d, realized: %d\n",
            fit_copula$parallel$requested, fit_copula$parallel$realized))
print(rpbnb_build_info())
