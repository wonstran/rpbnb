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

# Shared builder: coefficient matrix Estimate/SE/z/p with stars.
.coef_matrix <- function(object) {
  est <- object$coef
  se  <- if (is.null(object$se)) rep(NA_real_, length(est)) else object$se[names(est)]
  z   <- est / se
  p   <- 2 * stats::pnorm(-abs(z))
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
# (stored on the object), each with a delta-method standard error. Returns NULL
# when there is nothing to report (e.g. an independence fit shows only m1, m2).
.natural_scale_table <- function(object) {
  cf <- object$coef
  se <- object$se
  se_of <- function(nm) if (!is.null(se) && nm %in% names(se) && is.finite(se[[nm]])) se[[nm]] else NA_real_
  rows <- list()
  add <- function(name, est, stderr) rows[[length(rows) + 1L]] <<-
    data.frame(Parameter = name, Estimate = est, StdErr = stderr,
               check.names = FALSE)

  for (nm in grep("^log_sd[12]:", names(cf), value = TRUE)) {
    val <- exp(cf[[nm]])
    add(sub("^log_sd", "sd", nm), val, val * se_of(nm))  # d(exp)/d. = exp
  }
  if ("log_m1" %in% names(cf)) { m1 <- exp(cf[["log_m1"]]); add("m1 (dispersion)", m1, m1 * se_of("log_m1")) }
  if ("log_m2" %in% names(cf)) { m2 <- exp(cf[["log_m2"]]); add("m2 (dispersion)", m2, m2 * se_of("log_m2")) }

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
    add("lambda (dependence)", object$lambda, lam_se)
  }
  if (!length(rows)) return(NULL)
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

.print_natural_scale <- function(object, digits = 4) {
  nat <- .natural_scale_table(object)
  if (is.null(nat)) return(invisible(NULL))
  cat("\nNatural-scale dispersion / dependence (delta-method SE):\n")
  .print_coef_matrix(nat, digits)
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
                 natural = .natural_scale_table(object),
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

#' @export
predict.rpbnb_fit <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) return(data.frame(mu1 = object$mu1, mu2 = object$mu2))
  data.frame(
    mu1 = .bnb_predict_mu(object$formula_1, object$coef, "b1", "log_sd1", newdata),
    mu2 = .bnb_predict_mu(object$formula_2, object$coef, "b2", "log_sd2", newdata)
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
                 natural = .natural_scale_table(object),
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
