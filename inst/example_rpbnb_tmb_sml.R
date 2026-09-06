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
# If the full model does not converge -- "false convergence (8)" with a
# non-positive-definite Hessian and NA scale/dispersion SEs -- that is the
# weak-identification signature of a 0/1 dummy (kids) as the random
# coefficient in both equations under a Gaussian copula, NOT an
# optimizer-settings problem. A convergence gate after the fit (the
# "Convergence gate" block) stops with an actionable diagnostic: make the
# random coefficient a CONTINUOUS regressor in at least one equation, or load
# kids in only one equation, or use a dependence structure that identifies the
# scale better. No restarts/iterlim/rel.tol change manufactures the missing
# curvature.
#
# Run with:   Rscript inst/example_rpbnb_tmb_sml.R
# or, after install:
#   Rscript -e 'source(system.file("example_rpbnb_tmb_sml.R", package = "rpbnb"))'
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

DRAWS <- 500
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

# ---- Convergence gate before the boundary tests ----------------------------
# rpbnb_tmb_boundary_tests() refuses to run on a full model that nlminb did
# not drive to stationarity (code 0), so check it here and explain WHY rather
# than surfacing the cryptic stop from the boundary tester.
#
# "false convergence (8)" on THIS specification is a symptom of weak
# identification, not of the optimizer giving up too early: the score is far
# from stationarity AND the Hessian is not positive definite. Verified by a
# keep = "full" re-fit and eigendecomposition of the Hessian AT THE OPTIMUM
# (last.par.best, not the MakeADFun zero start): 3 negative eigenvalues
# (-58.8, -43.2, -24.7) out of 19 free parameters, i.e. the "optimum" is a
# saddle, not a maximum. The two smallest negative directions load almost
# entirely on log_sd1:kids (0.99) and log_sd2:kids (0.95) -- both random-
# coefficient scales curve the wrong way -- and the largest (-58.8) mixes
# log_m2 (0.85) with log_sd2:kids (0.26). The gradient is dominated by the
# dispersion/scale/dependence block (log_m1, log_m2, z_dep) rather than clean
# coefficient signal. kids is a 0/1 dummy in both equations, and a random
# coefficient on a dummy is weakly identified under a Gaussian copula (NB
# dispersion trades off against the random-coefficient scale) -- the exact
# caveat in this script's header. No optimizer knob (more restarts, a bigger
# iterlim, a tighter rel.tol) can converge to a saddle; it will burn hours and
# return the same code. See the "Fix" below for what does.
if (!identical(fit_copula$optimizer$convergence, 0L)) {
  stop(
    "The full model did not converge (nlminb code ", fit_copula$optimizer$convergence,
    ": ", fit_copula$optimizer$message,
    "; max|gradient| = ", signif(fit_copula$optimizer$max_abs_gradient, 4),
    " vs tolerance ", signif(fit_copula$optimizer$gradient_tolerance, 4),
    "), and the Hessian is not positive definite (sdreport$pdHess = ",
    fit_copula$sdreport$pdHess, "). This is the weak-identification signature ",
    "of a 0/1 dummy (kids) as the random coefficient in both equations under a ",
    "Gaussian copula -- not an optimizer-settings problem.\n",
    "Fix: put the random coefficient on a CONTINUOUS regressor (kids is 0/1 in ",
    "both equations, and every other regressor here is also 0/1 -- so a ",
    "continuous column must come from the data), or load kids in only ONE ",
    "equation, or switch to a dependence structure that identifies the scale ",
    "better. Once the score is near stationarity, the boundary tests below run ",
    "unconditionally.",
    call. = FALSE)
}
print(summary(fit_copula))

# ---- (Optional) Saddle-point diagnostic -------------------------------------
# Un-comment to reproduce the weak-identification verification above. It
# confirms, from the curvature itself, that a non-converged fit is a saddle
# (negative curvature in the random-scale block) rather than a stalled trust
# region. Costs one more full Hessian evaluation at the optimum (o$he()), which
# is expensive on this data (n = 3874 x 500 draws), so it is off by default.
#
# Three requirements:
#   (1) keep = "full" so the TMB objective o <- fit$obj is stored (the
#       default keep = "postfit" does not retain it -- o$he()/o$gr() then
#       error). Re-fit identically except for keep, or refit this one.
#   (2) Evaluate at o$env$last.par.best (the true optimum, objective =
#       -logLik), NOT o$par -- o$par is the all-zero MakeADFun START, where the
#       objective (~3e4 here) and gradient (hundreds-to-thousands everywhere)
#       are meaningless and would masquerade as "the whole model is bad".
#   (3) Symmetric eigen() for both the eigenvalues (saddle test) and the
#       eigenvector loadings (which parameters the negative directions live in).
#
# Expected on this weakly-identified spec: 3 negative eigenvalues out of 19
# (the two smallest loading ~entirely on log_sd1:kids / log_sd2:kids, the
# largest mixing log_m2 with log_sd2:kids), confirming a saddle, not a maximum.
# A healthy, well-identified fit instead shows all-positive eigenvalues and a
# small max|gradient| at last.par.best.
#
# Runs only when the fit did not converge (the case worth inspecting). To
# check a fit that DID converge (a sanity check that all eigenvalues are
# positive), drop the `if` guard and run the body unconditionally.
#
# if (!identical(fit_copula$optimizer$convergence, 0L)) {
#   if (is.null(fit_copula$obj))
#     stop("Refit with keep = \"full\" first: fit$obj (the TMB objective) is ",
#          "not stored under the default keep = \"postfit\".")
#   o <- fit_copula$obj
#   p <- o$env$last.par.best
#   par_names <- names(coef(fit_copula))
#   H <- o$he(p)
#   g <- as.numeric(as.matrix(o$gr(p)))
#   cat("objective at optimum:", o$fn(p), " (== -logLik = ",
#       -as.numeric(fit_copula$logLik), ")\n")
#   cat("max|gradient| at optimum:", max(abs(g)), "\n")
#   ev <- eigen(H, symmetric = TRUE)
#   cat("Hessian eigenvalues (ascending):\n"); print(round(ev$values, 3))
#   cat("negative eigenvalues:", sum(ev$values < 0), " of", length(ev$values),
#       "\n")
#   # Which parameters the negative (saddle) directions live in:
#   for (i in which(ev$values < 0)) {
#     v <- ev$vectors[, i]
#     top <- order(abs(v), decreasing = TRUE)[1:6]
#     cat(sprintf("\neigenvalue %s:\n", signif(ev$values[i], 4)))
#     print(cbind(name = par_names[top], load = v[top]))
#   }
# }

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
