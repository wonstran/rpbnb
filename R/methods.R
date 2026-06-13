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

#' @export
predict.bnb_fit <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) return(data.frame(mu1 = object$mu1, mu2 = object$mu2))
  X1 <- stats::model.matrix(object$formula_1[-2L], newdata)
  X2 <- stats::model.matrix(object$formula_2[-2L], newdata)
  b1 <- object$coef[grep("^b1:", names(object$coef))]
  b2 <- object$coef[grep("^b2:", names(object$coef))]
  data.frame(mu1 = as.vector(exp(X1 %*% b1)),
             mu2 = as.vector(exp(X2 %*% b2)))
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

#' @export
print.bnb_fit <- function(x, digits = 4, ...) {
  cat("Bivariate NB (", x$dependence, ") fit\n", sep = "")
  cat("Call: "); print(x$call)
  .print_coef_matrix(.coef_matrix(x), digits)
  cat(sprintf("\nlogLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              as.numeric(x$logLik), x$AIC, x$BIC))
  cat("Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  invisible(x)
}

#' @export
summary.bnb_fit <- function(object, ...) {
  structure(list(coefficients = .coef_matrix(object),
                 logLik = as.numeric(object$logLik), AIC = object$AIC,
                 BIC = object$BIC, nobs = object$nobs, npar = object$npar,
                 dependence = object$dependence, call = object$call),
            class = "summary.bnb_fit")
}

#' @export
print.summary.bnb_fit <- function(x, digits = 4, ...) {
  cat("Bivariate NB (", x$dependence, ") - summary\n", sep = "")
  .print_coef_matrix(x$coefficients, digits)
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
  X1 <- stats::model.matrix(object$formula_1[-2L], newdata)
  X2 <- stats::model.matrix(object$formula_2[-2L], newdata)
  b1 <- object$coef[grep("^b1:", names(object$coef))]
  b2 <- object$coef[grep("^b2:", names(object$coef))]
  data.frame(mu1 = as.vector(exp(X1 %*% b1)),
             mu2 = as.vector(exp(X2 %*% b2)))
}

#' @export
print.rpbnb_fit <- function(x, digits = 4, ...) {
  cat("Random-parameter bivariate NB fit (draws = ", x$draws,
      ", draw_type = ", x$draw_type, ")\n", sep = "")
  cat("Call: "); print(x$call)
  .print_coef_matrix(.coef_matrix(x), digits)
  cat(sprintf("\nlogLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              as.numeric(x$logLik), x$AIC, x$BIC))
  invisible(x)
}

#' @export
summary.rpbnb_fit <- function(object, ...) {
  structure(list(coefficients = .coef_matrix(object),
                 logLik = as.numeric(object$logLik), AIC = object$AIC,
                 BIC = object$BIC, nobs = object$nobs, npar = object$npar,
                 draws = object$draws, call = object$call),
            class = "summary.rpbnb_fit")
}

#' @export
print.summary.rpbnb_fit <- function(x, digits = 4, ...) {
  cat("Random-parameter bivariate NB - summary (draws = ", x$draws, ")\n", sep = "")
  .print_coef_matrix(x$coefficients, digits)
  cat(sprintf("\nn = %d   k = %d   logLik = %.4f   AIC = %.4f   BIC = %.4f\n",
              x$nobs, x$npar, x$logLik, x$AIC, x$BIC))
  invisible(x)
}
