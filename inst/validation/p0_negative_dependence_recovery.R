# =====================================================================
# P0 validation: negative-dependence parameter recovery for the corrected
# Famoye/Sarmanov lower lambda bound.
#
# Context: the P0 fix changes the lower bound to
#   lambda_min = -1 / max((1 - c1)(1 - c2), c1*c2)
# so it binds on BOTH positive corners of h1*h2. The c1*c2 corner dominates at
# low means (large c), exactly where the pre-fix bound admitted invalid,
# pmf-negative lambda values.
#
# This script is the reproducible artifact behind the recovery claim in
# comments/response_2026-07-15-23-07-35.md. Run from the package root:
#   Rscript inst/validation/p0_negative_dependence_recovery.R
#
# Data-generating process, seeds, optimizer, and diagnostics are all fixed
# below so the summary is reproducible.
# =====================================================================

suppressMessages({
  if (requireNamespace("pkgload", quietly = TRUE)) {
    tryCatch(pkgload::load_all(quiet = TRUE), error = function(e) library(rpbnb))
  } else {
    library(rpbnb)
  }
})

## --- Fixed DGP / configuration --------------------------------------
beta1       <- c("(Intercept)" = 0.10, x = 0.30)   # low intercept -> low mean -> large c
beta2       <- c("(Intercept)" = -0.10, x = -0.20)
dispersion  <- c(m1 = 0.5, m2 = 0.5)
lambda_true <- -1.2                                # negative dependence, admissible post-fix
n           <- 4000
reps        <- 25
seeds       <- 1000 + seq_len(reps)                # one dataset seed per replication
# Optimizer: fit_bnb() famoye path uses BFGS with the default rpbnb_control().

## --- Admissible bound at the design means ---------------------------
mu1 <- exp(beta1[["(Intercept)"]]); mu2 <- exp(beta2[["(Intercept)"]])
c1  <- rpbnb:::c_val(mu1, dispersion[["m1"]])
c2  <- rpbnb:::c_val(mu2, dispersion[["m2"]])
b   <- rpbnb:::lambda_bounds_vec(c1, c2)
old_lo <- -1 / ((1 - c1) * (1 - c2))               # pre-fix bound (ignores c1*c2)
cat(sprintf("c1=%.4f c2=%.4f | c1*c2=%.4f (1-c1)(1-c2)=%.4f\n",
            c1, c2, c1 * c2, (1 - c1) * (1 - c2)))
cat(sprintf("corrected lower bound=%.4f | pre-fix bound=%.4f | lambda_true=%.2f (%s)\n\n",
            b[1], old_lo, lambda_true,
            if (lambda_true > b[1]) "admissible" else "INVALID"))

## --- Monte Carlo recovery -------------------------------------------
est   <- matrix(NA_real_, reps, 4,
                dimnames = list(NULL, c("b1_x", "b2_x", "m1", "lambda")))
nconv <- 0L
nfloor <- 0L   # simulator flooring warnings (indicate an out-of-support lambda)
for (r in seq_len(reps)) {
  sim <- withCallingHandlers(
    simulate_bnb(n, beta1, beta2, dispersion = dispersion,
                 lambda = lambda_true, seed = seeds[r]),
    warning = function(w) { nfloor <<- nfloor + 1L; invokeRestart("muffleWarning") })
  fit <- tryCatch(suppressWarnings(
    fit_bnb(y1 ~ x, y2 ~ x, data = sim$data, dependence = "famoye")),
    error = function(e) NULL)
  if (is.null(fit)) next
  if (isTRUE(fit$convergence$converged)) nconv <- nconv + 1L
  cf <- coef(fit)
  est[r, ] <- c(cf[["b1:x"]], cf[["b2:x"]], exp(cf[["log_m1"]]), fit$lambda)
}

## --- Summary --------------------------------------------------------
truth <- c(b1_x = 0.30, b2_x = -0.20, m1 = 0.5, lambda = lambda_true)
mest  <- colMeans(est, na.rm = TRUE)
sdest <- apply(est, 2, sd, na.rm = TRUE)
cat(sprintf("n=%d reps=%d | converged=%d/%d | simulator flooring warnings=%d\n\n",
            n, reps, nconv, reps, nfloor))
cat("param     truth     mean_est   sd_est     bias\n")
for (k in names(truth))
  cat(sprintf("%-8s %8.3f  %8.3f  %8.3f  %+8.3f\n",
              k, truth[[k]], mest[[k]], sdest[[k]], mest[[k]] - truth[[k]]))
