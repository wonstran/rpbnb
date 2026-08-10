#' @export
coef.rpbnb_tmb_fit <- function(object, ...) object$coef

#' @export
vcov.rpbnb_tmb_fit <- function(object, ...) {
  if (!is.null(object$vcov)) return(object$vcov)
  nm <- names(object$coef)
  out <- matrix(NA_real_, length(nm), length(nm), dimnames = list(nm, nm))
  if (!is.null(object$vcov_diag)) diag(out) <- object$vcov_diag
  out
}

#' @export
logLik.rpbnb_tmb_fit <- function(object, ...) {
  structure(object$logLik, df = object$npar, nobs = object$nobs, class = "logLik")
}

#' @export
print.rpbnb_tmb_fit <- function(x, ...) {
  cat("rpbnb_tmb fit\n")
  cat("  Log-likelihood:", format(x$logLik, digits = 6), "\n")
  cat("  Nobs:", x$nobs, "  Npar:", x$npar, "\n")
  if (!is.null(x$parallel$realized)) {
    cat(sprintf("  TMB threads: %d\n", as.integer(x$parallel$realized)))
  }
  cat("  Estimator:", if (is.null(x$method)) "sml" else x$method, "\n")
  cat("  Dependence:", deparse(x$dependence), "\n")
  .print_standardize_note(x)
  orig <- .rpbnb_orig_units(x)
  cat("\nCoefficients:\n")
  print(round(if (!is.null(orig)) orig$coef else x$coef, 4))
  invisible(x)
}

# Significance stars helper
.signif_stars <- function(p) {
  symnum(p, corr = FALSE, na = FALSE,
         cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
         symbols = c("***", "**", "*", ".", " "))
}

# Print a coefficient table: controlled decimal places, no quotes on column names.
# digits = 4 (default): 4 decimal places. digits < 0: full precision.
.print_tbl <- function(x, digits = 4) {
  x <- as.data.frame(x)
  # A "df" column (boundary LR test degrees of freedom) is an integer count,
  # not a decimal estimate -- format it as one rather than "1.0000".
  has_df <- "df" %in% names(x)
  if (has_df) {
    df_col <- x[["df"]]
    x[["df"]] <- ifelse(is.na(df_col), "NA", sprintf("%d", as.integer(df_col)))
  }
  if (!is.null(digits) && digits >= 0) {
    fmt <- sprintf("%%.%df", digits)
    for (j in seq_len(ncol(x))) {
      if (is.numeric(x[[j]])) x[[j]] <- sprintf(fmt, x[[j]])
    }
  }
  print(x, print.gap = 3)
}

#' Summarize a fitted TMB-engine rpbnb model
#'
#' @param object A fitted model object of class rpbnb_tmb_fit.
#' @param digits Number of decimal places for numeric columns in coefficient
#'   tables. Default 4. Use a negative value (e.g. -1) for full precision.
#' @param ... Not used.
#' @return The fitted object, invisibly.
#' @export
summary.rpbnb_tmb_fit <- function(object, digits = 4L, ...) {
  cat("Summary: rpbnb_tmb fit\n")
  cat("  Log-likelihood:", format(object$logLik, digits = 6),
      "  AIC:", format(AIC(object), digits = 6),
      "  BIC:", format(BIC(object), digits = 6), "\n")
  iterations <- object$optimizer$iterations
  cat("  Nobs:", object$nobs, "  Npar:", object$npar,
      if (!is.null(iterations)) c("  Iterations:", iterations), "\n")
  cat("  Estimator:", if (is.null(object$method)) "sml" else object$method,
      "\n")
  .print_standardize_note(object)
  cat("\n")

  orig <- .rpbnb_orig_units(object)
  cf <- if (!is.null(orig)) orig$coef else object$coef
  cse <- if (!is.null(orig)) orig$se else object$se

  # ---- Equation 1 coefficients (b1:*) ----
  nm <- names(object$coef)
  eq1 <- grep("^b1:", nm)
  if (length(eq1)) {
    cat("--- Equation 1 (y1) ---\n")
    p1 <- 2 * pnorm(-abs(cf[eq1] / cse[eq1]))
    tbl1 <- data.frame(Estimate = cf[eq1],
                       `Std. Error` = cse[eq1],
                       `z value` = cf[eq1] / cse[eq1],
                       `Pr(>|z|)` = p1,
                       Signif = .signif_stars(p1),
                       row.names = nm[eq1], check.names = FALSE)
    .print_tbl(tbl1, digits)
    cat("\n")
  }

  # ---- Equation 2 coefficients (b2:*) ----
  eq2 <- grep("^b2:", nm)
  if (length(eq2)) {
    cat("--- Equation 2 (y2) ---\n")
    p2 <- 2 * pnorm(-abs(cf[eq2] / cse[eq2]))
    tbl2 <- data.frame(Estimate = cf[eq2],
                       `Std. Error` = cse[eq2],
                       `z value` = cf[eq2] / cse[eq2],
                       `Pr(>|z|)` = p2,
                       Signif = .signif_stars(p2),
                       row.names = nm[eq2], check.names = FALSE)
    .print_tbl(tbl2, digits)
    cat("\n")
  }

  # ---- Random-coefficient scales ----
  #
  # No p-values and no significance stars on these rows, deliberately.
  #
  # The natural-scale quantity is a positive scale, and the interesting null is
  # "this random coefficient has no variance", i.e. scale = 0. On the working
  # (log) scale that null sits at -Inf, which is not a point an ordinary
  # two-sided Wald test can address. A test of `log_scale / SE(log_scale)` is a
  # test of log_scale = 0, i.e. natural scale = 1 -- an arbitrary value with no
  # modelling meaning, which is nonetheless what a starred row under a heading
  # reading "Random-coefficient SDs" will be read as.
  #
  # This is the same conclusion already reached for the Rcpp engine (NEWS,
  # "Review fixes (2026-07-15 model review)"; see .print_natural_scale_footnote()
  # in R/methods.R) and it is deliberately mirrored here so the two engines do
  # not disagree about what they will assert. Use lr_test() or
  # rpbnb_boundary_tests() for a boundary-aware test.
  #
  # Both scales are shown side by side and labelled, rather than pairing a
  # natural-scale estimate with a log-scale SE in unlabelled columns. The
  # natural-scale SE is the delta-method transform, scale * SE(log_scale).
  #
  # The heading says "scales", not "SDs", and each row keeps its
  # distribution-specific name. Only `sd` IS a standard deviation: `w` is the
  # half-width of a uniform or triangular coefficient and `s` is the log-scale
  # of a lognormal one, so calling exp() of either an "SD" reports a quantity
  # the model does not estimate. The label comes from stripping `log_` off the
  # parameter name, which is itself built from rand_dist_registry's
  # `scale_label`, so a new distribution's label follows automatically.
  # When `object$boundary_tests` carries a row for a scale (from
  # rpbnb(boundary_tests = TRUE, engine = "tmb") or a manually attached
  # rpbnb_tmb_boundary_tests() result), that boundary-corrected LR test fills
  # LR/df/p here instead of leaving the gap unfilled. Row labels match the
  # boundary table's Parameter values -- both come from .sd_label().
  bt_all <- object$boundary_tests
  bt_lookup <- function(param) {
    if (is.null(bt_all)) return(NULL)
    hit <- which(bt_all$Parameter == param)
    if (!length(hit)) return(NULL)
    bt_all[hit[1L], , drop = FALSE]
  }
  scale_block <- function(idx, eq) {
    if (!length(idx)) return(invisible(NULL))
    cat("--- Random-coefficient scales (equation ", eq, ") ---\n", sep = "")
    log_est <- cf[idx]
    log_se  <- object$se[idx]
    nat_est <- exp(log_est)
    labels <- sub("^log_", "", nm[idx])
    lr_v <- vapply(labels, function(p) {
      b <- bt_lookup(p); if (is.null(b)) NA_real_ else b$LR
    }, numeric(1))
    df_v <- vapply(labels, function(p) {
      b <- bt_lookup(p); if (is.null(b)) NA_integer_ else b$df
    }, integer(1))
    p_v <- vapply(labels, function(p) {
      b <- bt_lookup(p); if (is.null(b)) NA_real_ else b$p.value
    }, numeric(1))
    tbl <- data.frame(
      `Estimate (log)`   = log_est,
      `Std. Error (log)` = log_se,
      Estimate           = nat_est,
      `Std. Error`       = nat_est * log_se,   # delta method
      LR = lr_v, df = df_v,
      `Pr(>chisq)` = p_v,
      Signif = .signif_stars(p_v),
      row.names = labels,
      check.names = FALSE
    )
    .print_tbl(tbl, digits)
    cat("Note: sd = standard deviation, w = half-width (uniform/triangular),\n",
        "      s = lognormal log-scale. These are the distributions' own scale\n",
        "      parameters, not all standard deviations.\n", sep = "")
    if (any(!is.na(lr_v))) {
      cat("      LR/df/Pr(>chisq): boundary-corrected LR test (H0: scale = 0,\n",
          "      50:50 chi-square mixture; see rpbnb_tmb_boundary_tests()).\n",
          sep = "")
    }
    if (any(is.na(lr_v))) {
      cat("      No Wald z/p or boundary LR test for the row(s) without one:\n",
          "      the null (scale = 0) is a boundary. Use\n",
          "      rpbnb_tmb_boundary_tests() (or rpbnb(boundary_tests = TRUE)).\n",
          sep = "")
    }
    cat("\n")
  }
  scale_block(grep("^(log_sd1|log_s1|log_w1):", nm), 1L)
  scale_block(grep("^(log_sd2|log_s2|log_w2):", nm), 2L)

  # ---- Dispersion parameters (natural scale from fit object) ----
  # m1/m2 are ADREPORTed in the template (src/rpbnb_tmb.cpp), so their
  # delta-method Std. Error is already in `sdreport` -- read it the same way
  # the dependence block below does, rather than leaving it unset. No Wald
  # z/p: m = 0 (the Poisson limit) is a boundary null, so an ordinary Wald
  # ratio does not test it (same reasoning as the random-coefficient scale
  # blocks above). When `object$boundary_tests` is present (from
  # rpbnb(boundary_tests = TRUE, engine = "tmb") or a manually attached
  # rpbnb_tmb_boundary_tests() result), that boundary-corrected LR test fills
  # LR/df/p instead of leaving the gap unfilled -- mirrors the classic
  # engine's .natural_scale_table() (R/methods.R).
  cat("--- Dispersion (m1, m2) ---\n")
  sdr <- object$sdreport
  disp_se <- c(NA_real_, NA_real_)
  if (!is.null(sdr)) {
    sdr_sum_disp <- suppressWarnings(try(summary(sdr, "report"), silent = TRUE))
    if (!inherits(sdr_sum_disp, "try-error")) {
      for (i in seq_along(c("m1", "m2"))) {
        nm_i <- c("m1", "m2")[i]
        if (nm_i %in% rownames(sdr_sum_disp)) {
          disp_se[i] <- sdr_sum_disp[nm_i, "Std. Error"]
        }
      }
    }
  }
  bt <- object$boundary_tests
  bt_row <- function(param) {
    if (is.null(bt)) return(NULL)
    hit <- which(bt$Parameter == param)
    if (!length(hit)) return(NULL)
    bt[hit[1L], , drop = FALSE]
  }
  disp_lr <- vapply(c("m1", "m2"), function(p) {
    b <- bt_row(p); if (is.null(b)) NA_real_ else b$LR
  }, numeric(1))
  disp_df <- vapply(c("m1", "m2"), function(p) {
    b <- bt_row(p); if (is.null(b)) NA_integer_ else b$df
  }, integer(1))
  disp_p <- vapply(c("m1", "m2"), function(p) {
    b <- bt_row(p); if (is.null(b)) NA_real_ else b$p.value
  }, numeric(1))
  disp <- data.frame(
    Parameter  = c("m1", "m2"),
    Estimate   = c(object$m1, object$m2),
    `Std. Error` = disp_se,
    LR = disp_lr, df = disp_df,
    `Pr(>chisq)` = disp_p,
    Signif = .signif_stars(disp_p),
    row.names = NULL, check.names = FALSE
  )
  .print_tbl(disp, digits)
  if (any(!is.na(disp_lr))) {
    cat("LR/df/Pr(>chisq) for rows with a boundary-corrected LR test (H0:\n",
        "m = 0, 50:50 chi-square mixture; see rpbnb_tmb_boundary_tests()).\n", sep = "")
  }
  if (any(is.na(disp_lr))) {
    cat("No Wald z/p and no boundary LR test for the row(s) above without\n",
        "one; the null (m = 0, the Poisson limit) is a boundary. Use\n",
        "rpbnb_tmb_boundary_tests() (or rpbnb(boundary_tests = TRUE)) to\n",
        "test it.\n", sep = "")
  }
  cat("\n")

  # ---- Dependence parameter (from sdreport) ----
  cat("--- Dependence ---\n")
  dep <- object$dependence
  dep_name <- "z_dep"
  if (inherits(dep, "rpbnb_copula")) {
    dep_name <- switch(dep$family, frank = "theta", normal = "rho", kimeldorf = "theta")
  } else if (identical(dep, "famoye")) {
    dep_name <- "lam"
  }
  if (!is.null(sdr)) {
    sdr_sum <- suppressWarnings(try(summary(sdr, "report"), silent = TRUE))
    if (!inherits(sdr_sum, "try-error") && dep_name %in% rownames(sdr_sum)) {
      dep_val <- sdr_sum[dep_name, "Estimate"]
      dep_se  <- sdr_sum[dep_name, "Std. Error"]
      cat("  ", dep_name, " =", format(dep_val, digits = if (digits >= 0) digits else 6),
          "  Std. Error =", format(dep_se, digits = if (digits >= 0) digits else 6), "\n")
    } else {
      cat("  (dependence parameter not in sdreport)\n")
    }
  }
  cat("\n")

  invisible(object)
}

#' Predict from a fitted bivariate count model
#'
#' Predictions integrate over the retained simulation draws when random
#' coefficients are present. Link predictions are the log of the integrated
#' response mean, so \code{exp(predict(fit, type = "link"))} equals response
#' predictions.
#'
#' @param object A fitted \code{rpbnb_tmb_fit} object.
#' @param newdata Optional data frame. If omitted, the retained fitting design
#'   is used; compact fits require \code{newdata}.
#' @param type Either \code{"response"} or \code{"link"}.
#' @param which Return both margins or only \code{"y1"} or \code{"y2"}.
#' @param ... Reserved for future use.
#' @return A two-column numeric matrix for \code{which = "both"}, otherwise a
#'   numeric vector.
#' @export
predict.rpbnb_tmb_fit <- function(
  object, newdata = NULL,
  type = c("response", "link"),
  which = c("both", "y1", "y2"), ...
) {
  type <- match.arg(type)
  which <- match.arg(which)
  equations <- if (which == "both") 1:2 else if (which == "y1") 1L else 2L

  predictions <- lapply(equations, function(eq) {
    X <- .prediction_design(object, eq, newdata)
    response <- .prediction_margin(object, eq, X)
    if (type == "link") log(response) else response
  })

  if (which != "both") return(predictions[[1L]])
  out <- do.call(cbind, predictions)
  colnames(out) <- c("y1", "y2")
  out
}

.prediction_design <- function(object, eq, newdata) {
  if (is.null(newdata)) {
    X <- object[[paste0("X", eq)]]
    if (is.null(X)) {
      stop(
        'This compact fit does not retain fitting rows; supply `newdata`.',
        call. = FALSE
      )
    }
    return(X)
  }
  if (!is.data.frame(newdata)) {
    stop("`newdata` must be a data frame.", call. = FALSE)
  }
  meta <- object$model_meta[[paste0("eq", eq)]]
  frame <- stats::model.frame(
    meta$terms, data = newdata, na.action = stats::na.pass,
    xlev = meta$xlevels
  )
  X <- stats::model.matrix(
    meta$terms, data = frame, contrasts.arg = meta$contrasts
  )
  expected <- sub(paste0("^b", eq, ":"), "", grep(
    paste0("^b", eq, ":"), names(object$coef), value = TRUE
  ))
  if (!identical(colnames(X), expected)) {
    stop("`newdata` produced an incompatible design matrix.", call. = FALSE)
  }
  X
}

.prediction_margin <- function(object, eq, X) {
  cn <- colnames(X)
  beta_names <- paste0("b", eq, ":", cn)
  beta <- object$coef[beta_names]
  rand_idx <- object[[paste0("rand_idx", eq)]]

  if (!length(rand_idx)) {
    return(exp(pmin(pmax(as.vector(X %*% beta), -700), 700)))
  }
  if (is.null(object$rp_meta)) {
    stop(
      "Random-coefficient prediction requires retained simulation draws.",
      call. = FALSE
    )
  }

  Z <- object$rp_meta[[paste0("Z", eq)]]
  dist <- object$rp_meta[[paste0("dist", eq)]]
  sign <- object$rp_meta[[paste0("sign", eq)]]
  scale_labels <- vapply(
    dist, function(name) rand_dist_registry[[name]]$scale_label,
    character(1)
  )
  scale_names <- paste0(scale_labels, eq, ":", cn[rand_idx])
  scales <- exp(object$coef[scale_names])
  realized <- rand_realize(
    Z, dist, sign, beta[rand_idx], scales
  )
  xb <- as.vector(X %*% beta)
  xr <- X[, rand_idx, drop = FALSE]
  .draw_mean_exp(xb, xr, realized$dev)
}
