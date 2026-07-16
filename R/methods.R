# S3 methods for bnb_fit and rpbnb_fit. All model methods live here.

#' @export
coef.bnb_fit <- function(object, ...) object$coef

#' @export
vcov.bnb_fit <- function(object, ...) object$vcov

#' @export
logLik.bnb_fit <- function(object, ...) {
  structure(as.numeric(object$logLik), df = object$npar,
            nobs = object$nobs, class = "logLik")
}

# Predicted mean for one equation. For models with random coefficients the
# `sd_prefix` log-SD entries add the closed-form lognormal correction
# 0.5 * sum_j sd_j^2 * x_j^2; for fixed-coefficient models no such entries exist
# and this reduces to exp(x'b). Aligns newdata design columns to the fitted
# coefficients and errors on a factor-level / column mismatch. Internal.
.bnb_predict_mu <- function(formula, coef, beta_prefix, sd_prefix, newdata) {
  X <- stats::model.matrix(formula[-2L], newdata)
  b <- coef[grep(paste0("^", beta_prefix, ":"), names(coef))]
  names(b) <- sub(paste0("^", beta_prefix, ":"), "", names(b))
  b <- b[colnames(X)]                       # align coefficients to design columns
  if (anyNA(b)) {
    stop("newdata produces design columns absent from the fitted model: ",
         paste(colnames(X)[is.na(b)], collapse = ", "),
         ". Ensure factor levels match the training data.", call. = FALSE)
  }
  lp <- as.vector(X %*% b)
  sd_log <- coef[grep(paste0("^", sd_prefix, ":"), names(coef))]
  if (length(sd_log)) {
    rand_cols <- sub(paste0("^", sd_prefix, ":"), "", names(sd_log))
    sds <- exp(sd_log)
    for (k in seq_along(rand_cols)) {
      col <- rand_cols[k]
      if (col %in% colnames(X)) lp <- lp + 0.5 * (sds[[k]]^2) * (X[, col]^2)
    }
  }
  as.vector(exp(lp))
}

#' @export
predict.bnb_fit <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) return(data.frame(mu1 = object$mu1, mu2 = object$mu2))
  data.frame(
    mu1 = .bnb_predict_mu(object$formula_1, object$coef, "b1", "log_sd1", newdata),
    mu2 = .bnb_predict_mu(object$formula_2, object$coef, "b2", "log_sd2", newdata)
  )
}

# Shared builder: coefficient matrix Estimate/SE/z/p with stars. The z/p/stars
# are suppressed (NA) for positive-scale/dispersion nuisance parameters on the
# raw (log) scale -- log_sd, log_w, log_s, log_m -- because testing e.g.
# log_sd = 0 tests that the native SD equals one, not that the variance is zero;
# it is not a zero-null Wald test. Regression coefficients and the unconstrained
# dependence parameters keep their tests.
.coef_matrix <- function(object) {
  est <- object$coef
  se  <- if (is.null(object$se)) rep(NA_real_, length(est)) else object$se[names(est)]
  z   <- est / se
  p   <- 2 * stats::pnorm(-abs(z))
  no_test <- grepl("^log_(sd|w|s|m)", names(est))
  z[no_test] <- NA_real_
  p[no_test] <- NA_real_
  data.frame(Parameter = names(est), Estimate = as.numeric(est),
             StdErr = as.numeric(se), z = as.numeric(z), p = as.numeric(p),
             Signif = signif_stars(p), row.names = NULL, check.names = FALSE)
}

.print_coef_matrix <- function(tab, digits = 4) {
  num <- vapply(tab, is.numeric, logical(1))
  tab[num] <- lapply(tab[num], formatC, format = "f", digits = digits)
  print(tab, row.names = FALSE, right = TRUE)
}

.split_coef_by_equation <- function(coef_matrix) {
  b1_idx <- grepl("^b1:", coef_matrix$Parameter)
  b2_idx <- grepl("^b2:", coef_matrix$Parameter)
  list(
    b1 = coef_matrix[b1_idx, , drop = FALSE],
    b2 = coef_matrix[b2_idx, , drop = FALSE]
  )
}

# Natural-scale view of the transformed parameters: random-coefficient SDs
# (exp(log_sd)), NB2 dispersions m = exp(log_m), and the Famoye dependence lambda
# (stored on the object), each with a delta-method standard error. Positive scale
# components (SDs, dispersions) report no z/p/stars because their Wald ratio does
# not test the boundary null a = 0; dependence parameters keep the Wald test.
# Returns a list with $random (random SDs) and $dispersion (m1, m2, lambda) components.
.natural_scale_table <- function(object) {
  cf <- object$coef
  se <- object$se
  se_of <- function(nm) if (!is.null(se) && nm %in% names(se) && is.finite(se[[nm]])) se[[nm]] else NA_real_

  random_rows <- list()
  dispersion_rows <- list()

  # `test = FALSE` for positive scale/dispersion parameters a = exp(eta): there
  # z = est/SE reduces to 1/SE(eta), which is not a Wald test of a = 0 (a = 0 is
  # eta = -Inf, a boundary). Report the estimate and delta-method SE but no
  # z/p/stars. Dependence parameters (lambda, copula native/tau) have an
  # interior zero and keep the regular Wald test (test = TRUE).
  add_random <- function(name, est, stderr) {
    random_rows[[length(random_rows) + 1L]] <<-
      data.frame(Parameter = name, Estimate = est, StdErr = stderr,
                 z = NA_real_, p = NA_real_, Signif = signif_stars(NA_real_),
                 check.names = FALSE)
  }

  add_dispersion <- function(name, est, stderr, test = TRUE) {
    if (test) {
      z_val <- if (is.na(est) || is.na(stderr) || stderr == 0) NA_real_ else est / stderr
      p_val <- if (is.na(z_val)) NA_real_ else 2 * stats::pnorm(-abs(z_val))
    } else {
      z_val <- NA_real_; p_val <- NA_real_
    }
    dispersion_rows[[length(dispersion_rows) + 1L]] <<-
      data.frame(Parameter = name, Estimate = est, StdErr = stderr,
                 z = z_val, p = p_val, Signif = signif_stars(p_val),
                 check.names = FALSE)
  }

  # Random-coefficient scales, all distributions: normal SD (log_sd -> sd),
  # uniform/triangular half-width (log_w -> w), lognormal log-scale (log_s -> s).
  scale_pfx <- c(log_sd = "sd", log_w = "w", log_s = "s")
  for (pfx in names(scale_pfx)) {
    for (nm in grep(paste0("^", pfx, "[12]:"), names(cf), value = TRUE)) {
      val <- exp(cf[[nm]])
      add_random(sub(paste0("^", pfx), scale_pfx[[pfx]], nm), val, val * se_of(nm))
    }
  }

  # Dispersion parameters
  if ("log_m1" %in% names(cf)) {
    m1 <- exp(cf[["log_m1"]])
    add_dispersion("m1 (dispersion)", m1, m1 * se_of("log_m1"), test = FALSE)
  }
  if ("log_m2" %in% names(cf)) {
    m2 <- exp(cf[["log_m2"]])
    add_dispersion("m2 (dispersion)", m2, m2 * se_of("log_m2"), test = FALSE)
  }

  # Only models with an estimated dependence parameter (z_lambda present, i.e.
  # famoye) get a lambda row; an independence fit has no dependence parameter.
  if ("z_lambda" %in% names(cf) && !is.null(object$lambda)) {
    lam_se <- NA_real_
    if (!is.null(object$bounds) && all(is.finite(object$bounds)) &&
        is.finite(se_of("z_lambda"))) {
      eps <- 1e-6; sig <- stats::plogis(cf[["z_lambda"]])
      dlam_dz <- (object$bounds[2] - object$bounds[1]) * (1 - 2 * eps) * sig * (1 - sig)
      lam_se <- abs(dlam_dz) * se_of("z_lambda")
    }
    add_dispersion("lambda (dependence)", object$lambda, lam_se)
  }

  # Copula models have z_theta; report native param and Kendall's tau
  if ("z_theta" %in% names(cf) && !is.null(object$cop_family)) {
    z     <- cf[["z_theta"]]
    se_z  <- se_of("z_theta")
    fam   <- object$cop_family
    nat   <- z_to_native(fam, z)
    dn_dz <- dnative_dz(fam, z)
    nat_se <- if (is.finite(se_z)) abs(dn_dz) * se_z else NA_real_
    param_label <- switch(fam, frank="theta (Frank)", normal="rho (Gaussian)", kimeldorf="theta (Clayton)")
    add_dispersion(param_label, nat, nat_se)
    td <- copula_tau_and_deriv(fam, z)
    tau_se <- if (is.finite(se_z)) abs(td$dtau_dz) * se_z else NA_real_
    add_dispersion("Kendall's tau", td$tau, tau_se)
  }

  # Return list with both components (NULL if empty)
  list(
    random = if (length(random_rows)) {
      out <- do.call(rbind, random_rows)
      rownames(out) <- NULL
      out
    } else NULL,
    dispersion = if (length(dispersion_rows)) {
      out <- do.call(rbind, dispersion_rows)
      rownames(out) <- NULL
      out
    } else NULL
  )
}

# Flat natural-scale table (random SDs + dispersion/dependence in one data frame)
# for summary() objects, which expose `$natural` as a single data frame with a
# Parameter column. print() uses the split view (.print_natural_scale) instead.
.natural_scale_flat <- function(object) {
  nat   <- .natural_scale_table(object)
  parts <- Filter(Negate(is.null), list(nat$random, nat$dispersion))
  if (!length(parts)) return(NULL)
  out <- do.call(rbind, parts)
  rownames(out) <- NULL
  out
}

.print_natural_scale <- function(object, digits = 4) {
  nat <- .natural_scale_table(object)
  if (is.null(nat$random) && is.null(nat$dispersion)) return(invisible(NULL))

  # Print random coefficient standard deviations
  if (!is.null(nat$random)) {
    cat("\nRandom coefficient standard deviations:\n")
    .print_coef_matrix(nat$random, digits)
  }

  # Print dispersion and dependence parameters with separator
  if (!is.null(nat$dispersion)) {
    cat("\n")
    cat(paste(rep("-", 70), collapse = ""), "\n")
    cat("Natural-scale dispersion / dependence (delta-method SE):\n")
    .print_coef_matrix(nat$dispersion, digits)
    cat("Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  }

  invisible(nat)
}

#' @export
print.bnb_fit <- function(x, digits = 4, ...) {
  cat("Bivariate NB (", x$dependence, ") fit\n", sep = "")
  cat("Call: "); print(x$call)

  coef_matrix <- .coef_matrix(x)
  split_coef <- .split_coef_by_equation(coef_matrix)

  y1_name <- as.character(x$formula_1[[2]])
  if (nrow(split_coef$b1) > 0) {
    cat("\n--- Equation 1: ", y1_name, " ---\n", sep = "")
    split_coef$b1$Parameter <- sub("^b1:", "", split_coef$b1$Parameter)
    .print_coef_matrix(split_coef$b1, digits)
  }

  y2_name <- as.character(x$formula_2[[2]])
  if (nrow(split_coef$b2) > 0) {
    cat("\n--- Equation 2: ", y2_name, " ---\n", sep = "")
    split_coef$b2$Parameter <- sub("^b2:", "", split_coef$b2$Parameter)
    .print_coef_matrix(split_coef$b2, digits)
  }

  .print_natural_scale(x, digits)
  cat(sprintf("\nlogLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              as.numeric(x$logLik), x$AIC, x$BIC))
  cat("Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  invisible(x)
}

#' @export
summary.bnb_fit <- function(object, ...) {
  structure(list(coefficients = .coef_matrix(object),
                 natural = .natural_scale_flat(object),
                 logLik = as.numeric(object$logLik), AIC = object$AIC,
                 BIC = object$BIC, nobs = object$nobs, npar = object$npar,
                 dependence = object$dependence, call = object$call,
                 formula_1 = object$formula_1, formula_2 = object$formula_2),
            class = "summary.bnb_fit")
}

#' @export
print.summary.bnb_fit <- function(x, digits = 4, ...) {
  cat("Bivariate NB (", x$dependence, ") - summary\n", sep = "")

  split_coef <- .split_coef_by_equation(x$coefficients)

  # Extract variable names from the formulas for labeling
  y1_name <- if (!is.null(x$formula_1)) {
    as.character(x$formula_1[[2]])
  } else if (!is.null(x$call$formula_1)) {
    as.character(x$call$formula_1[[2]])
  } else {
    "y1"
  }
  y2_name <- if (!is.null(x$formula_2)) {
    as.character(x$formula_2[[2]])
  } else if (!is.null(x$call$formula_2)) {
    as.character(x$call$formula_2[[2]])
  } else {
    "y2"
  }

  if (nrow(split_coef$b1) > 0) {
    cat("\n--- Equation 1: ", y1_name, " ---\n", sep = "")
    split_coef$b1$Parameter <- sub("^b1:", "", split_coef$b1$Parameter)
    .print_coef_matrix(split_coef$b1, digits)
  }

  if (nrow(split_coef$b2) > 0) {
    cat("\n--- Equation 2: ", y2_name, " ---\n", sep = "")
    split_coef$b2$Parameter <- sub("^b2:", "", split_coef$b2$Parameter)
    .print_coef_matrix(split_coef$b2, digits)
  }

  if (!is.null(x$natural)) {
    cat("\nNatural-scale dispersion / dependence (delta-method SE):\n")
    .print_coef_matrix(x$natural, digits)
  }
  cat(sprintf("\nn = %d   k = %d   logLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              x$nobs, x$npar, x$logLik, x$AIC, x$BIC))
  invisible(x)
}

# ---- rpbnb_fit (delegate the generic readers to the bnb_fit versions) ----

#' @export
coef.rpbnb_fit <- function(object, ...) object$coef

#' @export
vcov.rpbnb_fit <- function(object, ...) object$vcov

#' @export
logLik.rpbnb_fit <- function(object, ...) {
  structure(as.numeric(object$logLik), df = object$npar,
            nobs = object$nobs, class = "logLik")
}

# Population mean E[exp(x'beta)] for one RP equation on a design X, ESTIMATED by
# a Monte Carlo average over the stored optimization draws (RP_PRED_CAP caps each
# per-draw exp() as a numerical guard). Distribution-aware via the same
# rand_realize() deviations the fitter uses, so normal, uniform, triangular, and
# sign-constrained lognormal random coefficients are all handled (the previous
# code applied a normal-only 0.5*sd^2*x^2 correction and ignored log_w/log_s).
#
# The population mean is analytically INFINITE for a lognormal random coefficient
# j on observation i whenever sign_j * X_ij > 0 (the coefficient exp() term makes
# the linear predictor grow without bound over the mixing density). Those rows
# are returned as Inf with a warning rather than a finite, cap-dependent average.
RP_PRED_CAP <- 1e15
.rp_integrated_mu <- function(X, b, rand_idx, dist, sign, Z, scales) {
  xb <- as.vector(X %*% b)
  if (length(rand_idx) == 0 || is.null(Z) || ncol(Z) == 0) return(exp(xb))
  dev <- rand_realize(Z, dist, sign, b[rand_idx], scales)$dev   # R x q deviations
  XR  <- X[, rand_idx, drop = FALSE]
  mu_mat <- vapply(seq_len(nrow(Z)),
                   function(r) pmin(exp(xb + as.vector(XR %*% dev[r, ])), RP_PRED_CAP),
                   numeric(nrow(X)))
  # vapply simplifies to a vector when nrow(X) == 1; force an nrow(X) x R matrix
  # so rowMeans() (per-observation mean over draws) works for a single row too.
  mu_mat <- matrix(mu_mat, nrow = nrow(X))
  mu <- rowMeans(mu_mat)

  # Analytic infinities: a lognormal coefficient with sign_j * X_ij > 0.
  inf_rows <- logical(nrow(X))
  for (j in seq_along(rand_idx)) {
    if (identical(dist[j], "lognormal")) {
      inf_rows <- inf_rows | (sign[j] * X[, rand_idx[j]] > 0)
    }
  }
  if (any(inf_rows)) {
    mu[inf_rows] <- Inf
    warning(sum(inf_rows), " observation(s) have an analytically infinite ",
            "population mean (a lognormal random coefficient with sign * covariate ",
            "> 0); returning Inf for those rows.", call. = FALSE)
  }
  mu
}

# Build the integrated mean for equation `eq` (1 or 2) on design matrix X, using
# the fit's stored random-distribution metadata and draws.
.rp_predict_mu <- function(object, eq, X) {
  bpfx <- paste0("b", eq)
  b <- object$coef[grep(paste0("^", bpfx, ":"), names(object$coef))]
  names(b) <- sub(paste0("^", bpfx, ":"), "", names(b))
  b <- b[colnames(X)]                       # align coefficients to design columns
  if (anyNA(b)) {
    stop("newdata produces design columns absent from the fitted model: ",
         paste(colnames(X)[is.na(b)], collapse = ", "),
         ". Ensure factor levels match the training data.", call. = FALSE)
  }
  meta     <- object$rp_meta
  rand_idx <- if (eq == 1) object$rand_idx1 else object$rand_idx2
  if (is.null(meta) || length(rand_idx) == 0) return(exp(as.vector(X %*% b)))
  dist <- if (eq == 1) meta$dist1 else meta$dist2
  sgn  <- if (eq == 1) meta$sign1 else meta$sign2
  Z    <- if (eq == 1) meta$Z1 else meta$Z2
  cols <- colnames(X)[rand_idx]
  scales <- vapply(seq_along(rand_idx), function(j) {
    lbl <- rand_dist_registry[[dist[j]]]$scale_label
    exp(object$coef[[paste0(lbl, eq, ":", cols[j])]])
  }, numeric(1))
  .rp_integrated_mu(X, b, rand_idx, dist, sgn, Z, scales)
}

# Assemble the predict() data frame and tag it with the estimand definition and
# the simulation settings that produced it, so the quantity is self-describing.
.rp_pred_df <- function(mu1, mu2, object) {
  n_draws <- if (!is.null(object$rp_meta$Z1)) nrow(object$rp_meta$Z1) else object$draws
  structure(
    data.frame(mu1 = mu1, mu2 = mu2),
    estimand = paste0("population mean E[exp(x'beta)] over the random-coefficient ",
                      "distribution, estimated by Monte Carlo over the stored ",
                      "optimization draws (Inf where analytically infinite)"),
    n_draws  = n_draws,
    per_draw_cap = RP_PRED_CAP
  )
}

#' @export
predict.rpbnb_fit <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    if (!is.null(object$mu1)) return(.rp_pred_df(object$mu1, object$mu2, object))
    # Copula fits do not cache fitted means; recompute from the stored design.
    if (is.null(object$rp_meta)) {
      stop("predict() without 'newdata' is unavailable for this fit ",
           "(no fitted means or draws stored); pass newdata.", call. = FALSE)
    }
    return(.rp_pred_df(.rp_predict_mu(object, 1, object$X1),
                       .rp_predict_mu(object, 2, object$X2), object))
  }
  .rp_pred_df(
    .rp_predict_mu(object, 1, stats::model.matrix(object$formula_1[-2L], newdata)),
    .rp_predict_mu(object, 2, stats::model.matrix(object$formula_2[-2L], newdata)),
    object
  )
}

#' @export
print.rpbnb_fit <- function(x, digits = 4, ...) {
  cat("Random-parameter bivariate NB fit (draws = ", x$draws,
      ", draw_type = ", x$draw_type, ")\n", sep = "")
  cat("Call: "); print(x$call)

  coef_matrix <- .coef_matrix(x)
  split_coef <- .split_coef_by_equation(coef_matrix)

  y1_name <- as.character(x$formula_1[[2]])
  if (nrow(split_coef$b1) > 0) {
    cat("\n--- Equation 1: ", y1_name, " ---\n", sep = "")
    split_coef$b1$Parameter <- sub("^b1:", "", split_coef$b1$Parameter)
    .print_coef_matrix(split_coef$b1, digits)
  }

  y2_name <- as.character(x$formula_2[[2]])
  if (nrow(split_coef$b2) > 0) {
    cat("\n--- Equation 2: ", y2_name, " ---\n", sep = "")
    split_coef$b2$Parameter <- sub("^b2:", "", split_coef$b2$Parameter)
    .print_coef_matrix(split_coef$b2, digits)
  }

  .print_natural_scale(x, digits)
  cat(sprintf("\nlogLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              as.numeric(x$logLik), x$AIC, x$BIC))
  invisible(x)
}

#' @export
summary.rpbnb_fit <- function(object, ...) {
  structure(list(coefficients = .coef_matrix(object),
                 natural = .natural_scale_flat(object),
                 logLik = as.numeric(object$logLik), AIC = object$AIC,
                 BIC = object$BIC, nobs = object$nobs, npar = object$npar,
                 draws = object$draws, call = object$call,
                 formula_1 = object$formula_1, formula_2 = object$formula_2),
            class = "summary.rpbnb_fit")
}

#' @export
print.summary.rpbnb_fit <- function(x, digits = 4, ...) {
  cat("Random-parameter bivariate NB - summary (draws = ", x$draws, ")\n", sep = "")

  split_coef <- .split_coef_by_equation(x$coefficients)

  y1_name <- if (!is.null(x$formula_1)) {
    as.character(x$formula_1[[2]])
  } else if (!is.null(x$call$formula_1)) {
    as.character(x$call$formula_1[[2]])
  } else {
    "y1"
  }
  y2_name <- if (!is.null(x$formula_2)) {
    as.character(x$formula_2[[2]])
  } else if (!is.null(x$call$formula_2)) {
    as.character(x$call$formula_2[[2]])
  } else {
    "y2"
  }

  if (nrow(split_coef$b1) > 0) {
    cat("\n--- Equation 1: ", y1_name, " ---\n", sep = "")
    split_coef$b1$Parameter <- sub("^b1:", "", split_coef$b1$Parameter)
    .print_coef_matrix(split_coef$b1, digits)
  }

  if (nrow(split_coef$b2) > 0) {
    cat("\n--- Equation 2: ", y2_name, " ---\n", sep = "")
    split_coef$b2$Parameter <- sub("^b2:", "", split_coef$b2$Parameter)
    .print_coef_matrix(split_coef$b2, digits)
  }

  if (!is.null(x$natural)) {
    cat("\nNatural-scale dispersion / dependence (delta-method SE):\n")
    .print_coef_matrix(x$natural, digits)
  }
  cat(sprintf("\nn = %d   k = %d   logLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              x$nobs, x$npar, x$logLik, x$AIC, x$BIC))
  invisible(x)
}
