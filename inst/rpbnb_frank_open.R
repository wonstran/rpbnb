#!/usr/bin/env Rscript
# =============================================================================
# rpbnb on the open-section truck-crash data.  The `dependence` knob below picks
# the family; every section works under either, branching where they differ.
#
# Fits the bivariate random-parameter negative binomial (RP-BNB) model with
# rpbnb::fit_rpbnb() (maximum simulated likelihood, randomized Halton draws,
# OpenMP-multithreaded C++ core) to inst/extdata/export_open_all.csv:
#   * y1 = ALL_3 (all crashes), y2 = C_HV (heavy-vehicle crashes)
#   * random coefficients on SR40_MI3 and MPD_ME in eq 1, SR40_MI3 in eq 2
#   * dependence between the two margins: Famoye/Sarmanov or a copula()
#
# devtools::load_all() (not library()): the script must run against the current
# source tree -- it reads a couple of internal helpers (rpbnb:::) that an
# installed build may not carry.  Run from the package root:
#     Rscript inst/rpbnb_frank_open.R
#
# Standardization:
#   Both random-coefficient carriers are strictly positive and bounded away from
#   zero -- SR40_MI3 over [17.9, 65.1] and MPD_ME over [0.46, 3.62] -- so
#   x * (b + sd * u_i) is not a random slope but a random INTERCEPT in disguise,
#   with a per-observation SD of sd * x.  That absorbs the overdispersion the NB
#   dispersion exists to carry, and the two then compete for the same variance.
#   Centring makes the latent variance zero at the mean of the carrier and
#   nonzero only in its tails: a slope again.  This data is overdispersed too
#   (var/mean = 32.1 for ALL_3, 2.65 for C_HV), so there is real dispersion to
#   lose.  Scaling is the second half: IRI_ME spans 29-380 next to 0/1
#   indicators, giving kappa(X1) = 3055 and a Hessian condition number near 1e7.
#   Binary regressors stay 0/1 -- centring them only moves the intercept, and
#   it costs the "one unit = present" reading the marginal effects rely on.
#
# Draws / threads:
#   detectCores() returns 24 on this box (31.5 GiB RAM).  The C++ core
#   parallelises the per-draw likelihood across OpenMP threads (shared memory,
#   no per-call serialization), so 20 threads buy speed on the draw loop and the
#   finite-difference Hessian.  500 draws is a round MSL choice: more draws
#   reduce simulation noise in the objective and its same-draw Hessian at a
#   linear runtime cost.
# =============================================================================

devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")

sep <- function() cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

n_cores <- 20L
draws <- 500L
# Boundary LR tests for the random-coefficient SDs and the NB2 dispersions.
# These are the only valid tests of those parameters (their null sits on the
# boundary, so no Wald z/p exists).  Each refits a restricted model, warm-started
# from the full fit on its stored draws and skipping the Hessian.
#
# Cost depends strongly on the dependence family (measured here, 20 threads,
# 500 draws, n = 2321):
#   famoye : ~170 s on top of a ~510 s fit  (+33%)
#   frank  : ~2090 s on top of a ~765 s fit (+273%, ~48 min end to end)
# The Famoye margin likelihood has the OpenMP C++ core; the copula path is much
# more expensive per restricted refit, so budget accordingly before enabling
# this on a copula fit.
boundary_tests <- TRUE
# fit_rpbnb() accepts "famoye" or a copula() object (copula("frank"),
# copula("kimeldorf"), ...).  The two store their dependence differently -- a
# copula fit leaves fit$lambda/fit$bounds NULL and carries fit$cop_family plus a
# z_theta coefficient -- so the diagnostics and DEPENDENCE sections branch on
# fit$cop_family rather than assuming lambda exists.
dependence <- copula("frank")

# setwd() is at the project root; file.path() builds the platform-native
# separator, so this resolves on both Windows and POSIX.
data <- read.csv(file.path("inst", "extdata", "export_open_all.csv"))
cat("Observations :", nrow(data), "\n")

# ---- Standardize the continuous predictors ---------------------------------
# See the header: centring turns the bounded random carriers back into random
# slopes (so dispersion and random variance stop competing), and scaling fixes
# the conditioning.  `scaling` is saved with the fit so the original-unit
# tables below can be regenerated without refitting.
continuous_vars <- c("SR40_MI3", "MPD_ME", "LNAADT_3", "IRI_ME",
                     "ACCPNTS", "CS_MINAB", "DP10_ME")
scaling <- lapply(data[continuous_vars], function(x) {
  c(center = mean(x), scale = stats::sd(x))
})
for (v in continuous_vars) {
  data[[v]] <- (data[[v]] - scaling[[v]]["center"]) / scaling[[v]]["scale"]
}
cat("Standardized (centred and scaled) predictors:\n")
print(round(do.call(rbind, scaling), 4))

# `dependence` is either the string "famoye" or a copula() object -- a two-element
# list, which sprintf() vectorizes over, so "%s" on it printed the header TWICE
# ("(frank)" then "(NULL)").  Collapse it to one label first.
is_cop <- inherits(dependence, "rpbnb_copula")
dep_label <- if (is_cop) paste0(dependence$family, " copula") else
  as.character(dependence)
dep_desc <- if (is_cop) {
  sprintf("%s copula joining two NB margins", dependence$family)
} else {
  "Famoye/Sarmanov bivariate NB"
}
cat(sprintf("=== RP-BNB on truck all crashes (%s) ===\n", dep_label))
cat("Dependence   :", dep_desc, "\n")
cat("Cores asked  :", n_cores, "\n")
cat("Draws        :", draws, "\n")

f1 <- ALL_3  ~ SR40_MI3 + MPD_ME + LNAADT_3 + IRI_ME + G_ABG2 + SP50LE + ACCPNTS + SIGNAL1 + NEAR_SIG + CS_MINAB + DP10_ME + RUT_L
f2 <- C_HV ~SR40_MI3+MPD_ME+LNAADT_3+IRI_ME+SP50LE+ACCPNTS+SIGNAL1+NEAR_SIG+CS_MINAB+DP10_ME

cat("Equation 1   :", deparse(f1), "\n")
cat("Equation 2   :", deparse(f2), "\n\n")

t_fit <- system.time(
  fit <- fit_rpbnb(
    formula_1  = f1,
    formula_2  = f2,
    data       = data,
    random_1   = c("SR40_MI3", "MPD_ME"),
    # The earlier 40-draw fit estimated equation 2's SR40_MI3 random SD at
    # exp(-12.4) ~= 0 with SE 156: C_HV shows no detectable slope heterogeneity,
    # so the parameter is weakly identified and only costs a Halton dimension
    # and a Hessian row.  Kept here to match the rpbnb.tmb script; set
    # random_2 = NULL to drop it (the same draw count then integrates a 2-D
    # latent instead of 3-D).
    random_2   = c("SR40_MI3"),
    dependence = dependence,
    seed       = 20240712,
    draws      = draws,
    control    = rpbnb_control(
      print_level = 1,
      n_cores     = n_cores,
      # Observed-information Hessian on the same draws that produced the
      # estimate -- the counterpart of the rpbnb.tmb sdreport.  "opg" is a
      # faster BHHH alternative; "numeric" is the robust default.
      se_method   = "opg"
    )
  )
)[["elapsed"]]

cat(sprintf("\nEstimation finished in %.2f s\n", t_fit))
# Memory is not reported: the C++ core keeps its working set outside R's
# allocator, so gc() would understate exactly the quantity this script tests.
# Watch the process working set externally if a number is needed.
cat(sprintf("Convergence : code=%d, message=%s (iterations=%d)\n",
            fit$convergence$code, fit$convergence$message,
            fit$convergence$iterations))

# ---- Persist the fit and the transform --------------------------------------
# `scaling` has to outlive the session or the raw-unit tables below can never be
# regenerated without paying for the fit again, so the two are saved together.
stamp <- format(Sys.time(), "%Y-%m-%d-%H%M%S")
fit_path <- file.path("results", paste0("fit_open_centered_", stamp, ".rds"))
dir.create("results", recursive = TRUE, showWarnings = FALSE)
saveRDS(list(fit = fit, scaling = scaling), fit_path)
cat("Fit object saved to:", fit_path, "\n")

# ---- Convergence diagnostics -------------------------------------------
# print_level = 1 still buries the key convergence facts; a non-PD information
# matrix makes every standard error below meaningless, so surface it here.
diagnostics <- character(0)
if (!isTRUE(fit$convergence$converged)) {
  diagnostics <- c(diagnostics, sprintf(
    "Optimizer did not converge (code %d: %s).",
    fit$convergence$code, fit$convergence$message))
}
if (!is.null(fit$hessian_diag) && !isTRUE(fit$hessian_diag$positive_definite)) {
  diagnostics <- c(diagnostics, sprintf(
    "Information matrix is not positive definite (min eigenvalue %s); a ridge of %s was added -- SEs are regularized, not observed-information.",
    formatC(fit$hessian_diag$min_eigenvalue, format = "g"),
    formatC(fit$hessian_diag$ridge, format = "g")))
}

# lambda is optimized through a squashing reparameterization onto data-adaptive
# bounds, so a lambda sitting on a bound is an interior-optimum failure, not a
# converged estimate: the map's derivative vanishes there, the delta-method SE
# collapses to ~0, and the printed z blows up (2.9e6 on the Famoye path).  Flag
# it rather than let the dependence table read as an overwhelmingly precise
# result.
#
# Famoye/Sarmanov ONLY.  A copula fit carries `cop_family` and leaves
# `lambda`/`bounds` NULL, which made diff(NULL) -> numeric(0) and turned the
# whole `&&` chain into NA ("missing value where TRUE/FALSE needed").  Gate on
# the family first, and keep the comparison inside isTRUE() so a future
# non-finite bound degrades to "no warning" instead of an error.
if (is.null(fit$cop_family) && length(fit$lambda) == 1L &&
    length(fit$bounds) == 2L) {
  lam_span <- diff(fit$bounds)
  if (isTRUE(is.finite(lam_span) && lam_span > 0 &&
             min(abs(fit$lambda - fit$bounds)) < 1e-4 * lam_span)) {
    diagnostics <- c(diagnostics, sprintf(
      "lambda = %.6f is on its data-adaptive bound [%.6f, %.6f]: the dependence sits at the edge of the Famoye/Sarmanov admissible region, so the near-zero SE and huge z printed for it below are boundary artifacts, not evidence of precision.",
      fit$lambda, fit$bounds[1], fit$bounds[2]))
  }
}
if (any(!is.finite(fit$se))) {
  diagnostics <- c(diagnostics, paste0(
    "Non-finite SEs: ",
    paste(names(fit$se)[!is.finite(fit$se)], collapse = ", "), "."))
}
if (length(diagnostics)) {
  sep(); cat("CONVERGENCE WARNINGS\n"); sep()
  cat(paste0("  * ", diagnostics, collapse = "\n"), "\n")
} else {
  cat("No boundary or Hessian warnings.\n")
}

# ---- Model summary -----------------------------------------------------
sep(); cat("MODEL SUMMARY\n"); sep()
print(summary(fit))
cat("\n")

# ---- Coefficients in original units (display) -------------------------------
# The fit above is on standardized predictors, so each continuous coefficient
# is per standard deviation.  Because the standardization is an affine column
# transform, the original-unit coefficients are exact and need no refit:
# slopes divide by the scale, the intercept absorbs the centring shift
# -sum_j (c_j / s_j) * b_j, and binary 0/1 coefficients are unchanged.  SEs are
# delta-method on the full covariance, so the intercept's cross-covariances
# with the slopes count.  Random-coefficient SDs rescale the same way as
# slopes (Estimate_log = log_sd - log scale; the SE is unchanged by the
# constant shift).  This section is for display only; the fitted design itself
# stays standardized (see the note at the end of the original-units section).
sep(); cat("COEFFICIENTS IN ORIGINAL UNITS\n"); sep()
sc_mat <- do.call(rbind, scaling)

# Fixed-decimal table printer matching summary()'s .print_coef_matrix format.
print_tbl <- function(df, digits = 4) {
  num <- vapply(df, is.numeric, logical(1))
  df[num] <- lapply(df[num], formatC, format = "f", digits = digits)
  print(df, row.names = FALSE, right = TRUE)
}

coef_orig_units <- function(eq) {
  nm <- names(fit$coef)
  idx <- grep(paste0("^b", eq, ":"), nm)
  var <- sub(paste0("^b", eq, ":"), "", nm[idx])
  A <- diag(length(idx))
  dimnames(A) <- list(nm[idx], nm[idx])
  cont <- which(var %in% continuous_vars & var != "(Intercept)")
  A[cont, cont] <- diag(1 / sc_mat[var[cont], "scale"])
  A[var == "(Intercept)", cont] <-
    -sc_mat[var[cont], "center"] / sc_mat[var[cont], "scale"]
  V <- fit$vcov
  if (is.null(rownames(V))) dimnames(V) <- list(names(fit$coef), names(fit$coef))
  b_orig  <- as.vector(A %*% fit$coef[idx])
  se_orig <- sqrt(pmax(diag(A %*% V[nm[idx], nm[idx]] %*% t(A)), 0))
  z_orig  <- b_orig / se_orig
  p_orig  <- 2 * pnorm(-abs(z_orig))
  data.frame(
    # print_tbl() drops row names, so the label has to be a column or the table
    # prints as an unlabelled block of numbers.  `var` (not nm[idx]) matches the
    # Parameter column summary() prints for the same rows.
    Parameter = var,
    Estimate = b_orig,
    `Std. Error` = se_orig,
    `z value` = z_orig,
    `Pr(>|z|)` = p_orig,
    Signif = rpbnb:::signif_stars(p_orig),
    row.names = NULL, check.names = FALSE
  )
}
cat("\n--- Equation 1 (y1) ---\n")
print_tbl(coef_orig_units(1L))
cat("\n--- Equation 2 (y2) ---\n")
print_tbl(coef_orig_units(2L))
# Random-coefficient SDs, per equation and in the model summary's format.
# (All random coefficients here are Normal, hence the log_sd prefix.)
cat("\n--- Random-coefficient SDs (equation 1) ---\n")
sd_orig_units <- function(prefix) {
  nm <- names(fit$coef)
  idx <- grep(paste0("^", prefix, ":"), nm)
  var <- sub(paste0("^", prefix, ":"), "", nm[idx])
  log_orig <- fit$coef[idx] - log(sc_mat[var, "scale"])
  # No Wald p / stars here, matching summary()'s natural-scale block.  Two
  # separate reasons, either one fatal:
  #   1. The null for an SD sits on the boundary of the parameter space (sd = 0),
  #      where the Wald statistic has no standard normal reference.  That is why
  #      summary() prints NA for these rows -- the valid test is the 50:50
  #      chi-square boundary LR test in rpbnb_boundary_tests() (below).
  #   2. log_orig / se does not even test sd = 0.  It tests log_orig = 0, i.e.
  #      sd_orig = 1 -- and because log_orig = log_sd - log(scale), that null
  #      moves with the arbitrary standardization scale.  Rescaling SR40_MI3
  #      would change the p-value of a hypothesis about the same data.
  data.frame(
    # As above: a column, not row names.  "log_sd1:X" -> "sd1:X" is the label
    # summary()'s natural-scale block uses.
    Parameter = sub("^log_", "", nm[idx]),
    Estimate_log = log_orig,
    Estimate = exp(log_orig),
    `Std. Error` = fit$se[idx],
    row.names = NULL, check.names = FALSE
  )
}
print_tbl(sd_orig_units("log_sd1"))
cat("\n--- Random-coefficient SDs (equation 2) ---\n")
if (any(grepl("^log_sd2:", names(fit$coef)))) {
  print_tbl(sd_orig_units("log_sd2"))
} else {
  cat("  (no random coefficients in equation 2)\n")
}
cat("\nSD standard errors are on the log scale (the constant -log(scale) shift\n")
cat("leaves them unchanged).  No p-values: sd = 0 is a boundary null, so the\n")
cat("Wald statistic has no standard normal reference -- see the LR tests below.\n")
cat("\n")

# ---- Boundary LR tests ------------------------------------------------------
# The one valid test for the SDs and dispersions: rpbnb_boundary_tests() refits
# each properly nested restricted model (SD -> 0, m -> 0/Poisson) on the full
# fit's stored draws (common random numbers), warm-started from its coefficients,
# and applies lr_test()'s 50:50 chi-square boundary correction.  One restricted
# refit per boundary parameter -- five here, ~160 s total.
#
# A parameter already at its boundary (sd2:SR40_MI3, at exp(-18.6) ~= 0) makes
# the restricted fit tie or nominally beat the full one; lr_test() clamps the
# negative statistic to 0 and warns.  That warning is the expected reading for
# "this SD is indistinguishable from zero", not a failure.
if (boundary_tests) {
  sep(); cat("BOUNDARY LR TESTS (SDs and dispersions)\n"); sep()
  t_bt <- system.time(
    bt <- rpbnb_boundary_tests(fit, data = data,
                               control = rpbnb_control(n_cores = n_cores,
                                                       compute_se = FALSE))
  )[["elapsed"]]
  print(bt)
  cat(sprintf("\nBoundary tests finished in %.2f s\n", t_bt))
} else {
  sep(); cat("BOUNDARY LR TESTS (SDs and dispersions)\n"); sep()
  cat("Skipped (boundary_tests = FALSE).  These are the only valid tests of the\n")
  cat("SD and dispersion rows above; set boundary_tests <- TRUE at the top of\n")
  cat("this script to run them (one restricted refit per boundary parameter).\n")
}
cat("\n")

# ---- Fitted means (predict) ---------------------------------------------
sep(); cat("FITTED MEANS (predict) -- first 6 observations\n"); sep()
print(head(predict(fit)))

# ---- Dependence -----------------------------------------------------------
# The two dependence families store different things, and the Famoye fields are
# NULL on a copula fit.  sprintf() with a zero-length argument returns
# character(0), so the old lambda-only version printed a header and then NOTHING
# under a copula -- a silent hole rather than an error.  Branch explicitly.
sep(); cat("DEPENDENCE\n"); sep()
if (is.null(fit$cop_family)) {
  cat(sprintf("lambda = %.6f   data-adaptive bounds at the optimum = [%.6f, %.6f]\n",
              fit$lambda, fit$bounds[1], fit$bounds[2]))
} else {
  # Optimized as an unconstrained z_theta; report the native parameter and
  # Kendall's tau on their natural scales, delta-method SEs from se(z_theta) --
  # the same transform summary() uses for its dependence rows.
  z    <- fit$coef[["z_theta"]]
  se_z <- fit$se[["z_theta"]]
  nat  <- rpbnb:::z_to_native(fit$cop_family, z)
  td   <- rpbnb:::copula_tau_and_deriv(fit$cop_family, z)
  nat_se <- if (is.finite(se_z)) abs(rpbnb:::dnative_dz(fit$cop_family, z)) * se_z
            else NA_real_
  tau_se <- if (is.finite(se_z)) abs(td$dtau_dz) * se_z else NA_real_
  cat(sprintf("copula        : %s\n", fit$cop_family))
  cat(sprintf("theta         : %.6f   (SE %.6f)\n", nat, nat_se))
  cat(sprintf("Kendall's tau : %.6f   (SE %.6f)\n", td$tau, tau_se))
  cat("The copula dependence parameter is unbounded, so the data-adaptive\n")
  cat("bounds -- and the boundary caveat -- are Famoye/Sarmanov-specific and do\n")
  cat("not apply here.\n")
}

# ---- Marginal effects (AME) ------------------------------------------------
# Built on the Monte-Carlo integrated mean E[exp(x'beta)] over the fit's stored
# draws; random-coefficient covariates use the draw-integrated formula.  The
# `n_cores` here parallelise the delta-method jacobian across a PSOCK cluster.
sep(); cat("AVERAGE MARGINAL EFFECTS (AME)\n"); sep()
marginal_effects <- rpbnb_marginal_effects(fit, which = "both", type = "AME",
                                           n_cores = n_cores)
cat("\n")

# ---- Elasticities / semi-elasticities (AME) --------------------------------
sep(); cat("ELASTICITIES / SEMI-ELASTICITIES (AME)\n"); sep()
elasticities <- rpbnb_elasticities(fit, which = "both", type = "AME",
                                   n_cores = n_cores)
cat("\n")

# ---- Results in the covariates' original units ------------------------------
# The two sections above are per standard deviation, and the elasticities of the
# centred regressors are identically zero because their sample mean is zero --
# they print as 0.0000, which reads as "no effect" rather than "this number
# means nothing".  `scaling` restates both: AMEs divide by the scale (an exact
# chain rule -- the estimand is the same function of the same draws, only the
# units change, so the delta-method SEs divide too), and elasticities recover
# the x-bar factor via x_orig = scale * x_std + center.  The fitted design is
# NOT rebuilt: substituting raw x back into the random term would re-add the
# (center/scale) * dev random intercept that centring just removed.
#
# LNAADT_3 is log(AADT).  The elasticity column reports the elasticity with
# respect to the LOG of traffic (x_orig * dlnmu/dx_orig); the elasticity with
# respect to traffic itself is dlnmu/d ln(AADT), which for a fixed coefficient
# is exactly the coefficient.
sep(); cat("MARGINAL EFFECTS AND ELASTICITIES IN ORIGINAL UNITS\n"); sep()

# Recompute the RP integrated means and their derivatives from the
# STANDARDIZED fit (same draws, same capping as the package), then apply the
# affine chain rule.  One row per model term (intercept excluded).
#
# SEs for the original-unit elasticities are delta-method: the estimand for a
# continuous term is mean_i x_orig_ij * (d mu_i / d x_orig_ij) / mu_i, with
# d mu_i / d x_orig_ij = (d mu_i / d x_std_ij) / scale_j -- a smooth function of
# the equation's coefficients and log random scales (draws held fixed), so its
# jacobian G gives SE = sqrt(G V G').  For a FIXED coefficient the estimand
# collapses to xbar_orig_j * beta_j / scale_j, so the jacobian row is exactly
# (0, ..., xbar_orig_j/scale_j, ..., 0) and the delta method reproduces the
# closed form; RANDOM coefficients need the full jacobian because
# dln(mu_i)/dx_orig varies by observation.  This is the same G V G' machinery
# the package's rpbnb_elasticities() SEs come from -- under an identity
# standardization the two reproduce each other bit-for-bit.  Binary terms are
# untouched: the package's AME and semi-elasticity (and both SEs) are already
# in original units.
raw_diag <- function(eq, me_tab, el_tab) {
  X  <- if (eq == 1L) fit$X1 else fit$X2
  cn <- colnames(X)
  rand_idx <- if (eq == 1L) fit$rand_idx1 else fit$rand_idx2
  dist <- if (eq == 1L) fit$rp_meta$dist1 else fit$rp_meta$dist2
  sign <- if (eq == 1L) fit$rp_meta$sign1 else fit$rp_meta$sign2
  Z    <- if (eq == 1L) fit$rp_meta$Z1 else fit$rp_meta$Z2
  b  <- fit$coef[paste0("b", eq, ":", cn)]; names(b) <- cn
  # All random coefficients here are Normal, so the scale prefix is log_sd.
  sds <- vapply(rand_idx,
                function(j) exp(fit$coef[[paste0("log_sd", eq, ":", cn[j])]]),
                numeric(1))
  rr <- if (length(rand_idx)) rpbnb:::rand_realize(Z, dist, sign, b[rand_idx], sds)
        else NULL
  dev <- if (is.null(rr)) matrix(0, nrow(Z), 0) else rr$dev
  coef_mat <- if (is.null(rr)) NULL else rr$coef   # R x q realized coefficients
  mu_mat <- rpbnb:::.rp_g_matrix(X, b, rand_idx, dev)   # n x R capped per-draw means
  mu <- rowMeans(mu_mat)

  # --- Delta-method SEs for the original-unit elasticities --------------------
  # theta = the equation's b's then log random scales, the exact order the
  # package's vcov is indexed by (see .rp_diag_meta()/rpbnb_elasticities()).
  # lambda and the dispersion parameters never enter the mean estimand.
  theta_names <- c(paste0("b", eq, ":", cn),
                   paste0("log_sd", eq, ":", cn[rand_idx]))
  V <- fit$vcov[theta_names, theta_names, drop = FALSE]

  is_bin <- vapply(seq_len(ncol(X)), function(j)
    !identical(cn[j], "(Intercept)") && all(X[, j] %in% c(0, 1)), logical(1))
  cont <- which(!is_bin & cn != "(Intercept)")

  # Raw-unit elasticity vector for the continuous columns as a function of
  # theta.  numDeriv::jacobian() drops names on its evaluation grid, so names
  # are re-set inside the closure (the same step .rp_diag_one() takes).
  raw_el <- function(th, cols) {
    names(th) <- theta_names
    bb <- th[paste0("b", eq, ":", cn)]; names(bb) <- cn
    ss <- exp(th[paste0("log_sd", eq, ":", cn[rand_idx])])
    r  <- if (length(rand_idx))
            rpbnb:::rand_realize(Z, dist, sign, bb[rand_idx], ss) else NULL
    devm <- if (is.null(r)) matrix(0, nrow(Z), 0) else r$dev
    cmat <- if (is.null(r)) NULL else r$coef
    gm   <- rpbnb:::.rp_g_matrix(X, bb, rand_idx, devm)
    mui  <- rowMeans(gm)
    out  <- numeric(length(cols))
    for (k in seq_along(cols)) {
      j <- cols[k]
      rk <- match(j, rand_idx)
      cw <- if (!is.na(rk)) cmat[, rk] else rep(bb[[j]], nrow(Z))
      dmu <- rowMeans(sweep(gm, 2, cw, `*`))
      s <- sc_mat[cn[j], "scale"]
      out[k] <- mean((s * X[, j] + sc_mat[cn[j], "center"]) * dmu / (s * mui))
    }
    out
  }

  # One full Richardson jacobian per equation, sequential (~15-30 s here, small
  # next to the ~600 s fit; numerically identical to the package's per-param
  # columns, which it reproduces bit-for-bit under an identity scaling).
  el_se <- rep(NA_real_, length(cn))
  if (length(cont)) {
    G <- numDeriv::jacobian(function(th) raw_el(th, cont), fit$coef[theta_names])
    el_se[cont] <- sqrt(pmax(diag(G %*% V %*% t(G)), 0))
  }

  rows <- lapply(seq_along(cn), function(j) {
    if (cn[j] == "(Intercept)") return(NULL)
    me_row <- me_tab[me_tab$Name == cn[j], ]
    if (!nrow(me_row)) return(NULL)
    if (is_bin[j]) {
      # Binary columns are never standardized: the package's AME and
      # semi-elasticity -- and both their SEs -- are already in original units.
      el_row <- el_tab[el_tab$Name == cn[j], ]
      data.frame(Variable = cn[j],
                 AME = me_row$Estimate, `AME Std. Error` = me_row$StdErr,
                 Elasticity = el_row$Estimate,
                 `Elasticity Std. Error` = el_row$StdErr, check.names = FALSE)
    } else {
      rk <- match(j, rand_idx)
      coef_row <- if (!is.na(rk)) coef_mat[, rk] else rep(b[[j]], nrow(Z))
      dmu <- rowMeans(sweep(mu_mat, 2, coef_row, `*`))   # dE[Y]/dx in SD units
      s <- sc_mat[cn[j], "scale"]; c <- sc_mat[cn[j], "center"]
      x_orig <- s * X[, j] + c
      data.frame(Variable = cn[j],
                 AME = me_row$Estimate / s,
                 `AME Std. Error` = me_row$StdErr / s,
                 # elasticity wrt the original covariate: mean x_orig * dlnmu/dx_orig
                 Elasticity = mean(x_orig * dmu / (s * mu)),
                 `Elasticity Std. Error` = el_se[j], check.names = FALSE)
    }
  })
  do.call(rbind, Filter(Negate(is.null), rows))
}

cat("\n--- Equation 1 (y1) ---\n")
print_tbl(raw_diag(1L, marginal_effects$y1, elasticities$y1))
cat("\n--- Equation 2 (y2) ---\n")
print_tbl(raw_diag(2L, marginal_effects$y2, elasticities$y2))
cat("\nNotes:\n")
cat("  * AME standard errors rescale exactly: AME_orig = AME_sd / scale, so the\n")
cat("    SE divides by the same constant.  Binary terms are unchanged.\n")
cat("  * Elasticity standard errors are delta-method: G V G' on the raw-unit\n")
cat("    elasticity estimand (same draws as the fit, same vcov).  For FIXED\n")
cat("    continuous terms the estimand is exactly xbar_orig * beta_orig and the\n")
cat("    jacobian collapses to one non-zero entry, so this reproduces the closed\n")
cat("    form |xbar_orig| * se(beta)/scale.  RANDOM-coefficient terms get the\n")
cat("    full jacobian: dln(mu_i)/dx_orig varies by observation, so no\n")
cat("    constant-times-coefficient collapse exists there.\n")
cat("  * Elasticities in THIS table are wrt the original covariate; the per-SD\n")
cat("    tables above are wrt the standardized one.  LNAADT_3 is log(AADT), so\n")
cat("    both are wrt log-traffic; wrt traffic itself the elasticity is\n")
cat("    dln(mu)/d ln(AADT), equal to the coefficient for a fixed effect.\n")
cat("  * The fitted design stays standardized -- this table is a units restatement.\n")

# ---- Persistence -----------------------------------------------------------
# The rpbnb.tmb markdown exporter has no rpbnb counterpart; the RDS written
# above (fit + scaling) is the persistence mechanism, and every printed table
# can be regenerated from it without refitting.
cat("\nFit and raw-unit transforms saved to:", fit_path, "\n")
