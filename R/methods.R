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

# Build a column-stable design matrix and the newdata offset for equation `eq`
# from `newdata`. Uses the stored training terms/xlevels/contrasts so that absent
# factor levels reproduce the TRAINING columns (as zeros) rather than shifting
# positions -- standard model-method behavior, and a precondition for aligning
# fitted coefficients and random-coefficient columns by name. Falls back to the
# bare formula RHS for fits made before `predict_meta` was stored (offset 0).
.newdata_design <- function(object, eq, newdata) {
  pm   <- object$predict_meta
  form <- if (eq == 1) object$formula_1 else object$formula_2
  if (is.null(pm)) {
    X <- stats::model.matrix(form[-2L], newdata)
    return(list(X = X, offset = numeric(nrow(X))))
  }
  tt <- stats::delete.response(if (eq == 1) pm$terms1 else pm$terms2)
  xl <- if (eq == 1) pm$xlevels1 else pm$xlevels2
  cc <- if (eq == 1) pm$contrasts1 else pm$contrasts2
  mf <- stats::model.frame(tt, newdata, xlev = xl, na.action = stats::na.pass)
  X  <- stats::model.matrix(tt, mf, contrasts.arg = cc)
  list(X = X, offset = .as_offset(stats::model.offset(mf), nrow(X)))
}

# Predicted mean for one equation. For models with random coefficients the
# `sd_prefix` log-SD entries add the closed-form lognormal correction
# 0.5 * sum_j sd_j^2 * x_j^2; for fixed-coefficient models no such entries exist
# and this reduces to exp(x'b + offset). Aligns newdata design columns to the
# fitted coefficients and errors on a factor-level / column mismatch. Internal.
.bnb_predict_mu <- function(coef, beta_prefix, sd_prefix, X, offset) {
  b <- coef[grep(paste0("^", beta_prefix, ":"), names(coef))]
  names(b) <- sub(paste0("^", beta_prefix, ":"), "", names(b))
  b <- b[colnames(X)]                       # align coefficients to design columns
  if (anyNA(b)) {
    stop("newdata produces design columns absent from the fitted model: ",
         paste(colnames(X)[is.na(b)], collapse = ", "),
         ". Ensure factor levels match the training data.", call. = FALSE)
  }
  lp <- as.vector(X %*% b) + offset
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
  # The cached mu1/mu2 already include the training offset (glm.nb / the fitted
  # linear predictor), so the no-newdata path stays consistent with explicit
  # newdata = training_data.
  if (is.null(newdata)) return(data.frame(mu1 = object$mu1, mu2 = object$mu2))
  d1 <- .newdata_design(object, 1, newdata)
  d2 <- .newdata_design(object, 2, newdata)
  data.frame(
    mu1 = .bnb_predict_mu(object$coef, "b1", "log_sd1", d1$X, d1$offset),
    mu2 = .bnb_predict_mu(object$coef, "b2", "log_sd2", d2$X, d2$offset)
  )
}

# Shared builder: coefficient matrix Estimate/SE/z/p with stars. The z/p/stars
# are suppressed (NA) for positive-scale/dispersion nuisance parameters on the
# raw (log) scale -- log_sd, log_w, log_s, log_m -- because testing e.g.
# log_sd = 0 tests that the native SD equals one, not that the variance is zero;
# it is not a zero-null Wald test. Regression coefficients and the unconstrained
# dependence parameters keep their tests.
.coef_matrix <- function(object) {
  orig <- .rpbnb_orig_units(object)
  est <- if (!is.null(orig)) orig$coef else object$coef
  se  <- if (!is.null(orig)) orig$se
         else if (is.null(object$se)) rep(NA_real_, length(est))
         else object$se[names(est)]
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
  # A "df" column (boundary LR test degrees of freedom) is an integer count,
  # not a decimal estimate -- format it as one rather than "1.0000".
  if ("df" %in% names(tab)) num[["df"]] <- FALSE
  tab[num] <- lapply(tab[num], formatC, format = "f", digits = digits)
  if ("df" %in% names(tab)) {
    tab[["df"]] <- ifelse(is.na(tab[["df"]]), "NA", formatC(tab[["df"]], format = "d"))
  }
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
# components (SDs, dispersions) report no Wald z/p because their ratio does not
# test the boundary null a = 0 (a = 0 is eta = -Inf); dependence parameters have
# an interior zero and keep the regular Wald test.
#
# When `object$boundary_tests` is present (attached by rpbnb(boundary_tests =
# TRUE, engine = "classic"), or by assigning a manual rpbnb_boundary_tests()
# result), the SD and dispersion rows report that boundary-corrected LR test
# instead of leaving the gap unfilled: LR/df columns, and `p`/`Signif` from the
# LR p-value rather than NA. Rows without a matching boundary-test parameter
# (no `$boundary_tests`, or a Poisson-pinned margin, which is fixed rather than
# estimated) keep LR/df = NA. Returns a list with $random (random SDs) and
# $dispersion (m1, m2, lambda) components.
.natural_scale_table <- function(object) {
  orig <- .rpbnb_orig_units(object)
  cf <- if (!is.null(orig)) orig$coef else object$coef
  se <- object$se
  se_of <- function(nm) if (!is.null(se) && nm %in% names(se) && is.finite(se[[nm]])) se[[nm]] else NA_real_

  bt <- object$boundary_tests
  bt_row <- function(param) {
    if (is.null(bt)) return(NULL)
    hit <- which(bt$Parameter == param)
    if (!length(hit)) return(NULL)
    bt[hit[1L], , drop = FALSE]
  }

  random_rows <- list()
  dispersion_rows <- list()

  # SD rows are always boundary parameters (never a plain Wald z/p); a matching
  # boundary-test row supplies LR/df/p instead of leaving them NA.
  add_random <- function(name, est, stderr) {
    b <- bt_row(name)
    random_rows[[length(random_rows) + 1L]] <<-
      data.frame(Parameter = name, Estimate = est, StdErr = stderr,
                 LR = if (is.null(b)) NA_real_ else b$LR,
                 df = if (is.null(b)) NA_integer_ else b$df,
                 z = NA_real_,
                 p = if (is.null(b)) NA_real_ else b$p.value,
                 Signif = signif_stars(if (is.null(b)) NA_real_ else b$p.value),
                 check.names = FALSE)
  }

  # `test = FALSE` for positive scale/dispersion parameters a = exp(eta) without
  # a `boundary_param` match: their z = est/SE reduces to 1/SE(eta), which is
  # not a Wald test of a = 0. Dependence parameters (lambda, copula native/tau)
  # are never boundary-tested (`boundary_param = NULL`) and keep the ordinary
  # Wald test (`test = TRUE`).
  add_dispersion <- function(name, est, stderr, test = TRUE, boundary_param = NULL) {
    b <- if (is.null(boundary_param)) NULL else bt_row(boundary_param)
    if (!is.null(b)) {
      z_val <- NA_real_; p_val <- b$p.value
    } else if (test) {
      z_val <- if (is.na(est) || is.na(stderr) || stderr == 0) NA_real_ else est / stderr
      p_val <- if (is.na(z_val)) NA_real_ else 2 * stats::pnorm(-abs(z_val))
    } else {
      z_val <- NA_real_; p_val <- NA_real_
    }
    dispersion_rows[[length(dispersion_rows) + 1L]] <<-
      data.frame(Parameter = name, Estimate = est, StdErr = stderr,
                 LR = if (is.null(b)) NA_real_ else b$LR,
                 df = if (is.null(b)) NA_integer_ else b$df,
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

  # Dispersion parameters. A Poisson-restricted margin (poisson_1/poisson_2) is
  # exactly m = 0 by the public contract; its pinned log_m holds only the
  # numerical POISSON_M display placeholder, which must NOT leak into the
  # natural-scale table. Report exactly 0 with an NA SE (a fixed parameter, so
  # it has no boundary counterpart either -- `boundary_param` stays unset).
  if ("log_m1" %in% names(cf)) {
    if (isTRUE(object$poisson_1)) {
      add_dispersion("m1 (dispersion)", 0, NA_real_, test = FALSE)
    } else {
      m1 <- exp(cf[["log_m1"]])
      add_dispersion("m1 (dispersion)", m1, m1 * se_of("log_m1"), test = FALSE,
                     boundary_param = "m1")
    }
  }
  if ("log_m2" %in% names(cf)) {
    if (isTRUE(object$poisson_2)) {
      add_dispersion("m2 (dispersion)", 0, NA_real_, test = FALSE)
    } else {
      m2 <- exp(cf[["log_m2"]])
      add_dispersion("m2 (dispersion)", m2, m2 * se_of("log_m2"), test = FALSE,
                     boundary_param = "m2")
    }
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
    # `boundary_param = "lam"` only matters when the caller asked for the
    # dependence LR test (which = "dependence"); without such a row this
    # behaves exactly as before and keeps the ordinary Wald z/p, which is
    # valid here -- lambda = 0 is an INTERIOR null.
    add_dispersion("lambda (dependence)", object$lambda, lam_se,
                   boundary_param = "lam")
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
    # As for lambda above: picks up the dependence LR test when one was run,
    # otherwise unchanged. Kendall's tau below is a monotone transform of the
    # same parameter and is deliberately left on its Wald row -- one LR row per
    # restriction, not one per way of displaying it.
    add_dispersion(param_label, nat, nat_se,
                   boundary_param = .dep_boundary_param(fam))
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

# Footnote for a natural-scale table. Two independent notes, either or both
# printed: one for rows carrying an actual boundary-corrected LR test (`LR`
# column non-NA -- from `rpbnb(boundary_tests = TRUE)` or a manually attached
# rpbnb_boundary_tests() result), explaining what LR/df/p mean there; and one
# for rows with neither a Wald z/p nor a boundary LR/p (positive scale/
# dispersion parameters left untested), pointing to rpbnb_boundary_tests() to
# fill the gap. `tab` is a natural-scale data frame with `z`/`p` columns and,
# when boundary tests are present, `LR`/`df` columns.
.print_natural_scale_footnote <- function(tab) {
  if (is.null(tab) || !"z" %in% names(tab)) return(invisible(NULL))
  lr_tested <- if ("LR" %in% names(tab)) !is.na(tab$LR) else rep(FALSE, nrow(tab))
  has_lr <- any(lr_tested)
  untested <- is.na(tab$z) & !lr_tested
  if (has_lr) {
    cat("Note: LR/df/p for rows with a likelihood-ratio test (H0: parameter = 0;\n",
        "      see rpbnb_boundary_tests()). Scale and dispersion nulls sit on the\n",
        "      boundary and use the 50:50 chi-square mixture; a dependence row's\n",
        "      null is interior (except Clayton) and uses chi-square(1). Where an\n",
        "      LR test is shown the Wald z is suppressed -- one test per row.\n",
        sep = "")
  }
  if (any(untested)) {
    cat("Note: no Wald z/p or boundary LR test for positive scale/dispersion\n",
        "      parameters (SDs, m) above without one; their null is a boundary.\n",
        "      Use rpbnb_boundary_tests() (or rpbnb(boundary_tests = TRUE)) to\n",
        "      test these.\n", sep = "")
  }
  invisible(NULL)
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
  .print_natural_scale_footnote(.natural_scale_flat(object))

  invisible(nat)
}

#' @export
print.bnb_fit <- function(x, digits = 4, ...) {
  cat("Bivariate NB (", x$dependence, ") fit\n", sep = "")
  cat("Call: "); print(x$call)
  .print_control_ignored(x)

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
                 formula_1 = object$formula_1, formula_2 = object$formula_2,
                 control_ignored = object$control_ignored,
                 control_engine = object$control_engine),
            class = "summary.bnb_fit")
}

#' @export
print.summary.bnb_fit <- function(x, digits = 4, ...) {
  cat("Bivariate NB (", x$dependence, ") - summary\n", sep = "")
  .print_control_ignored(x)

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
    .print_natural_scale_footnote(x$natural)
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
.rp_integrated_mu <- function(X, b, rand_idx, dist, sign, Z, scales, offset = NULL) {
  xb <- as.vector(X %*% b) + .as_offset(offset, nrow(X))
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

# Build the integrated mean for equation `eq` (1 or 2) on design matrix X (with
# newdata offset), using the fit's stored random-distribution metadata and draws.
#
# The random-coefficient columns are identified by NAME and remapped against the
# rebuilt design with match(): the stored rand_idx are positions in the TRAINING
# design, which need not coincide with newdata column positions (e.g. a factor
# whose unused level disappears shifts every later column). We recover the random
# column names from the stored training design and locate them in X by name.
.rp_predict_mu <- function(object, eq, X, offset = NULL) {
  bpfx <- paste0("b", eq)
  b <- object$coef[grep(paste0("^", bpfx, ":"), names(object$coef))]
  names(b) <- sub(paste0("^", bpfx, ":"), "", names(b))
  b <- b[colnames(X)]                       # align coefficients to design columns
  if (anyNA(b)) {
    stop("newdata produces design columns absent from the fitted model: ",
         paste(colnames(X)[is.na(b)], collapse = ", "),
         ". Ensure factor levels match the training data.", call. = FALSE)
  }
  meta       <- object$rp_meta
  train_cn   <- colnames(if (eq == 1) object$X1 else object$X2)
  train_idx  <- if (eq == 1) object$rand_idx1 else object$rand_idx2
  rand_names <- train_cn[train_idx]
  if (is.null(meta) || length(train_idx) == 0) {
    return(exp(as.vector(X %*% b) + .as_offset(offset, nrow(X))))
  }
  rand_idx <- match(rand_names, colnames(X))   # positions in the REBUILT design
  if (anyNA(rand_idx)) {
    stop("newdata design is missing random-coefficient column(s): ",
         paste(rand_names[is.na(rand_idx)], collapse = ", "),
         ". Ensure factor levels match the training data.", call. = FALSE)
  }
  dist <- if (eq == 1) meta$dist1 else meta$dist2
  sgn  <- if (eq == 1) meta$sign1 else meta$sign2
  Z    <- if (eq == 1) meta$Z1 else meta$Z2
  scales <- vapply(seq_along(rand_idx), function(j) {
    lbl <- rand_dist_registry[[dist[j]]]$scale_label
    exp(object$coef[[paste0(lbl, eq, ":", rand_names[j])]])
  }, numeric(1))
  .rp_integrated_mu(X, b, rand_idx, dist, sgn, Z, scales, offset)
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
  # Training offsets (all-zero when the model has none), used for the no-newdata
  # path so it matches predict(object, newdata = training_data).
  off1 <- if (!is.null(object$predict_meta)) object$predict_meta$off1 else NULL
  off2 <- if (!is.null(object$predict_meta)) object$predict_meta$off2 else NULL
  if (is.null(newdata)) {
    # Recompute from the stored design so the SAME estimand (including the
    # lognormal Inf correction in .rp_integrated_mu) is applied as with an
    # explicit newdata. The cached object$mu1/mu2 are the capped finite-draw
    # means and would silently disagree on analytically-infinite rows.
    if (is.null(object$rp_meta)) {
      if (!is.null(object$mu1)) return(.rp_pred_df(object$mu1, object$mu2, object))
      stop("predict() without 'newdata' is unavailable for this fit ",
           "(no fitted means or draws stored); pass newdata.", call. = FALSE)
    }
    return(.rp_pred_df(.rp_predict_mu(object, 1, object$X1, off1),
                       .rp_predict_mu(object, 2, object$X2, off2), object))
  }
  d1 <- .newdata_design(object, 1, newdata)
  d2 <- .newdata_design(object, 2, newdata)
  .rp_pred_df(
    .rp_predict_mu(object, 1, d1$X, d1$offset),
    .rp_predict_mu(object, 2, d2$X, d2$offset),
    object
  )
}

#' @export
print.rpbnb_fit <- function(x, digits = 4, ...) {
  cat("Random-parameter bivariate NB fit (draws = ", x$draws,
      ", draw_type = ", x$draw_type, ")\n", sep = "")
  cat("Call: "); print(x$call)
  .print_standardize_note(x)
  .print_control_ignored(x)

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
                 formula_1 = object$formula_1, formula_2 = object$formula_2,
                 scaling = object$scaling, continuous_vars = object$continuous_vars,
                 control_ignored = object$control_ignored,
                 control_engine = object$control_engine),
            class = "summary.rpbnb_fit")
}

#' @export
print.summary.rpbnb_fit <- function(x, digits = 4, ...) {
  cat("Random-parameter bivariate NB - summary (draws = ", x$draws, ")\n", sep = "")
  .print_standardize_note(x)
  .print_control_ignored(x)

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
    .print_natural_scale_footnote(x$natural)
  }
  cat(sprintf("\nn = %d   k = %d   logLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              x$nobs, x$npar, x$logLik, x$AIC, x$BIC))
  invisible(x)
}
