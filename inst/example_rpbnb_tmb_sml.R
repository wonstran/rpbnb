#!/usr/bin/env Rscript
# =============================================================================
# example_rpbnb_tmb_sml.R -- random-parameter bivariate NB (RP-BNB), TMB
# engine, method = "sml", copula dependence
#
# Estimates a random-parameter bivariate NB2 model for the two count outcomes
# in the German health-care utilization panel (rwm1984.csv) under copula
# dependence (COPULA_FAMILY below), via the TMB engine's SML estimator. Same
# data, dummy variables (see inst/example_bnb.R), formulas, and random
# coefficient (kids, in both equations) as inst/example_rpbnb_tmb_famoye.R --
# that script covers Famoye/Sarmanov dependence; this one was split out of it
# so each dependence structure -- and its own boundary-test cost -- can be run
# independently rather than paying for both every time.
#
# The TMB engine has two estimators for the random-coefficient integral:
# "sml" (simulated maximum likelihood over Halton draws, the same integral
# the classic engine approximates, but with an exact automatic-differentiation
# gradient instead of a numerical one) and "laplace" (a sparse-Hessian
# approximation that removes `draws` from the memory cost -- see
# ?fit_rpbnb_tmb's `method` argument). This script is the "sml" side of that
# choice, not close to a Laplace fit of the same model, which approximates a
# different way and need not agree closely on any given dataset (see
# ?fit_rpbnb_tmb). At COPULA_FAMILY = "normal" it should also give estimates
# close to inst/example_rpbnb_classic.R's copula fit (same integral, same
# data, same draws) -- that script only demonstrates the Gaussian copula, so
# the comparison holds only for that one family, not for whichever
# COPULA_FAMILY happens to be set below. The two scripts' boundary tests do
# NOT match, either: this one narrows LR_TEST to "sd" (2 refits), while the
# classic script uses boundary_tests = TRUE (sd + dispersion, 4 refits) -- see
# LR_TEST below to widen this one to match.
#
# `kids` is loaded as random in BOTH equations, for consistency with
# inst/example_rpbnb_tmb_famoye.R and to avoid identifying the random
# coefficient off only one margin's variation. Note this cuts against
# ?fit_rpbnb_tmb's own advice for the copula path specifically: `kids` is a
# 0/1 dummy in this data, and a random coefficient on a dummy regressor is
# weakly identified under copula dependence (NB dispersion trades off against
# the random-coefficient scale) -- kept anyway here so the two scripts'
# specifications line up; a continuous regressor would identify more cleanly.
#
# Run with:   Rscript inst/example_rpbnb_tmb_sml.R
# or, after install:
#   Rscript -e 'source(system.file("example_rpbnb_tmb_sml.R", package = "rpbnb"))'
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
# One rpbnb_control() object drives every estimator; here only the TMB fields
# below matter (n_cores, max_threads, max_workload, tape_chunks), the rest are
# resolved or ignored for this fit. See ?rpbnb_control for the full field
# reference and ?fit_rpbnb_tmb for the memory/chunking model.
ctrl <- rpbnb_control(
  n_cores = N_CORES,
  # OpenMP thread count for both TMB estimators (see ?rpbnb_control). Left 1
  # core free above via N_CORES.
  max_threads = N_CORES,
  # Ceiling on OpenMP threads permitted for one TMB fit. NULL would default to
  # n_cores; set equal here so the ceiling never binds below the request.
  max_workload = rpbnb_tmb_max_workload(budget_gib = 32),
  # Memory guard: the largest weighted observation-draw count the TMB tape may
  # build before the fit is refused or chunked. Weighted workload is
  # n * draws * family_weight (frank = 3.6, the heaviest family) with peak at
  # ~12083 bytes/unit, so this budgets a ~32 GiB peak for one fit. Inf disables
  # the guard entirely.
  tape_chunks = NULL,
  # SML-only: split the draws into this many chunks, replayed over one smaller
  # TMB tape, cutting peak tape memory to nrow * ceiling(DRAWS / chunks) at the
  # cost of somewhat slower gradient evaluations. A fixed integer must be 1L or
  # greater and not exceed draws (validated at fit time in .resolve_tape_chunks()).
  #
  # NOTE: NULL here (the default) auto-selects the smallest sufficient chunk
  # count when the weighted workload exceeds max_workload, or 1L (no chunking)
  # when it does not -- so with max_workload pinned to the 32 GiB budget above,
  # NULL picks the layout on its own from that budget. Pass a fixed integer
  # (e.g. 4L) to pin the layout regardless of the auto-threshold.
  #
  # Any chunked fit (value > 1L, including an auto-picked one) has no taped
  # Hessian, so profile-based inference (confint(method = "profile"),
  # rpbnb_tmb_dependence_profile()) falls back to Wald with a warning; default
  # Wald/optimHess is unaffected.
)
cat("Using", N_CORES, "OpenMP threads,", DRAWS, "draws.\n\n")

# ---- Copula RP-BNB, TMB engine, SML ---------------------------------------------
# boundary_tests = LR_TEST runs rpbnb_tmb_boundary_tests() on the converged fit
# and attaches it as fit$boundary_tests, so summary() below fills in the
# sd1:kids/sd2:kids LR/df/p columns instead of leaving them blank -- no
# separate rpbnb_tmb_boundary_tests() call needed. (Gaussian-copula evaluation
# runs multithreaded by default since 0.4.6 -- see NEWS.md -- so
# COPULA_FAMILY = "normal" needs no extra argument either.)
cat(sprintf(
  "=== Fitting RP-BNB (TMB engine, method = sml, %s copula, random kids) ===\n",
  COPULA_FAMILY))
t_copula <- system.time(
  fit_copula <- rpbnb(f1, f2, data = d, engine = "tmb", method = "sml",
                      random_1 = "kids", 
                      random_2 = "kids",
                      dependence = copula(COPULA_FAMILY),
                      draws = DRAWS, 
                      seed = SEED,
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
# ?fit_rpbnb_tmb's `disable_parallel_gaussian`). Printed here because
# COPULA_FAMILY = "normal" above is exactly the case that cap used to affect.
cat(sprintf("Threads requested: %d, realized: %d\n",
            fit_copula$parallel$requested, fit_copula$parallel$realized))
print(rpbnb_build_info())
