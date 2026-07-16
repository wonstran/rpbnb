# =====================================================================
# Start-sensitivity check for the fixed Famoye/Sarmanov fitter.
#
# The famoye analytic gradient freezes the lambda-bounds, so the BFGS objective
# is start-sensitive. This script compares two defensible starts -- an all-zero
# mean-coefficient start and marginal glm.nb starts -- across several DGPs and
# the rwm1984 data, reporting which reaches the higher converged log-likelihood.
# It is the evidence behind the start policy documented in
# comments/response_2026-07-15-23-07-35.md.
#
# Run from the package root:
#   Rscript inst/validation/start_sensitivity_famoye.R
# =====================================================================

suppressMessages({
  if (requireNamespace("pkgload", quietly = TRUE)) {
    tryCatch(pkgload::load_all(quiet = TRUE), error = function(e) library(rpbnb))
  } else library(rpbnb)
})

# SEs are irrelevant to the log-likelihood comparison, and the optimizer output
# is noise here -- fit quietly without computing standard errors.
silent <- rpbnb_control(compute_se = FALSE, print_level = 0)

fit_from <- function(f1, f2, data, start) {
  fit <- tryCatch(suppressWarnings(
    fit_bnb(f1, f2, data = data, dependence = "famoye", start = start,
            control = silent)),
    error = function(e) NULL)
  if (is.null(fit)) return(c(ll = NA_real_, conv = 0))
  c(ll = as.numeric(logLik(fit)), conv = as.numeric(isTRUE(fit$convergence$converged)))
}

# Winner gated on convergence: a converged fit beats a non-converged one; among
# equally-converged fits the higher log-likelihood wins.
pick_winner <- function(rz, rg) {
  cz <- isTRUE(rz["conv"] == 1); cg <- isTRUE(rg["conv"] == 1)
  if (cz && !cg) return("zero")
  if (cg && !cz) return("glmnb")
  if (!is.finite(rz["ll"]) || !is.finite(rg["ll"])) return("?")
  if (rz["ll"] >= rg["ll"]) "zero" else "glmnb"
}

# Zero vs glm.nb starts for a p1=p2=2 (intercept + one covariate) model.
starts_for <- function(data, f1, f2) {
  X1 <- model.matrix(f1, data); X2 <- model.matrix(f2, data)
  Y1 <- data[[all.vars(f1)[1]]]; Y2 <- data[[all.vars(f2)[1]]]
  zero <- c(rep(0, ncol(X1) + ncol(X2)), log(0.5), log(0.5), 0)
  g1 <- suppressWarnings(tryCatch(MASS::glm.nb(Y1 ~ X1 - 1), error = function(e) NULL))
  g2 <- suppressWarnings(tryCatch(MASS::glm.nb(Y2 ~ X2 - 1), error = function(e) NULL))
  glmnb <- c(if (!is.null(g1)) unname(coef(g1)) else rep(0, ncol(X1)),
             if (!is.null(g2)) unname(coef(g2)) else rep(0, ncol(X2)),
             if (!is.null(g1)) log(1 / g1$theta) else log(0.5),
             if (!is.null(g2)) log(1 / g2$theta) else log(0.5), 0)
  list(zero = zero, glmnb = glmnb)
}

scenarios <- list(
  list(name = "low-mean neg-dep",  b1 = c("(Intercept)" = 0.1, x = 0.3),
       b2 = c("(Intercept)" = -0.1, x = -0.2), lam = -1.0),
  list(name = "mid-mean pos-dep",  b1 = c("(Intercept)" = 0.8, x = 0.4),
       b2 = c("(Intercept)" = 0.6, x = -0.3), lam = 0.5),
  list(name = "high-mean pos-dep", b1 = c("(Intercept)" = 1.4, x = 0.2),
       b2 = c("(Intercept)" = 1.2, x = -0.2), lam = 0.8)
)

cat(sprintf("%-20s %12s %12s   winner\n", "scenario", "ll_zero", "ll_glmnb"))
for (sc in scenarios) {
  sim <- simulate_bnb(3000, sc$b1, sc$b2, dispersion = c(m1 = 0.5, m2 = 0.5),
                      lambda = sc$lam, seed = 202)
  st  <- starts_for(sim$data, y1 ~ x, y2 ~ x)
  rz  <- fit_from(y1 ~ x, y2 ~ x, sim$data, st$zero)
  rg  <- fit_from(y1 ~ x, y2 ~ x, sim$data, st$glmnb)
  cat(sprintf("%-20s %12.3f %12.3f   %s\n", sc$name, rz["ll"], rg["ll"],
              pick_winner(rz, rg)))
}

# rwm1984 reference (docvis/hospvis ~ outwork + kids).
f <- system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")
if (nzchar(f)) {
  d <- read.csv(f)
  st <- starts_for(d, docvis ~ outwork + kids, hospvis ~ outwork + kids)
  rz <- fit_from(docvis ~ outwork + kids, hospvis ~ outwork + kids, d, st$zero)
  rg <- fit_from(docvis ~ outwork + kids, hospvis ~ outwork + kids, d, st$glmnb)
  cat(sprintf("%-20s %12.3f %12.3f   %s\n", "rwm1984", rz["ll"], rg["ll"],
              pick_winner(rz, rg)))
}
