# Likelihood-ratio test for two nested model fits (bnb_fit / rpbnb_fit).
#
# Motivation: the natural-scale summary reports no Wald z/p for the positive
# scale/dispersion parameters (random-coefficient SDs, NB2 dispersions m),
# because their Wald ratio does not test the boundary null a = 0. The correct
# test for those parameters is a likelihood-ratio test against a fit that omits
# the term -- with a boundary correction when the null pins a variance-type
# parameter to zero.

# Reject a package fit (bnb_fit / rpbnb_fit) whose optimizer did not converge:
# an LR test needs maximized likelihoods, and a non-converged fit invalidates the
# comparison. Generic logLik-only objects carry no convergence record and pass
# through (documented as unvalidated).
.lr_chk_converged <- function(fit, role) {
  if (inherits(fit, c("bnb_fit", "rpbnb_fit")) &&
      !is.null(fit$convergence) && !isTRUE(fit$convergence$converged)) {
    stop("The ", role, " model did not converge (code ",
         fit$convergence$code, ": ", fit$convergence$message,
         "). A likelihood-ratio test requires maximized likelihoods; refit it to ",
         "convergence before calling lr_test().", call. = FALSE)
  }
  invisible(TRUE)
}

#' Likelihood-ratio test between two nested model fits
#'
#' Compares a restricted fit against a full (nesting) fit by the likelihood-ratio
#' statistic. Works for any [fit_bnb()] / [fit_rpbnb()] objects, since both carry
#' a `logLik()` with a `"df"` attribute equal to the number of estimated
#' parameters.
#'
#' The restricted model is one you fit yourself with a term removed -- for
#' example dropping a name from `random_1` (testing a random-coefficient SD), or
#' a plain NB / independence fit (testing an NB2 dispersion). This is the
#' statistically appropriate replacement for the Wald z/p that the natural-scale
#' summary suppresses on positive scale/dispersion parameters.
#'
#' A likelihood-ratio test requires two *maximized* likelihoods. When either
#' argument is a package fit (`bnb_fit` / `rpbnb_fit`) that records a failed
#' optimization (`convergence$converged = FALSE`), `lr_test()` errors rather than
#' returning a p-value from an unfinished fit -- the sign of the statistic cannot
#' establish convergence. Generic objects that only carry a `logLik()` (no
#' convergence record) are still accepted, but their convergence cannot be
#' validated and is the caller's responsibility.
#'
#' @param restricted The smaller (restricted) fit -- fewer estimated parameters.
#' @param full The larger (full) fit that nests `restricted`.
#' @param boundary Logical. When the restriction pins a variance/dispersion-type
#'   parameter (a random-coefficient SD, or NB2 dispersion `m`) to its zero
#'   boundary, the null distribution is not a plain chi-square. Set `boundary =
#'   TRUE` to use the 50:50 mixture of `chisq(df)` and `chisq(df - 1)` (Self &
#'   Liang, 1987); for a single boundary parameter (`df = 1`) this halves the
#'   naive p-value. Default `FALSE` (ordinary interior restriction, e.g. dropping
#'   a fixed covariate). The mixture is exact only for a single parameter on the
#'   boundary; simultaneous boundary restrictions of several parameters need
#'   different weights and are not handled.
#' @return An object of class `rpbnb_lrtest` with the LR `statistic`, degrees of
#'   freedom `df`, `p.value`, the two log-likelihoods and their df, and the
#'   `boundary` flag. Has a `print` method.
#' @references Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of
#'   maximum likelihood estimators and likelihood ratio tests under nonstandard
#'   conditions. \emph{JASA} 82(398), 605--610.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 600,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' ctrl <- rpbnb_control(compute_se = FALSE)
#' full <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                   draws = 100, control = ctrl)
#' rest <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
#'                   draws = 100, control = ctrl)   # no random coefficient
#' lr_test(rest, full, boundary = TRUE)             # test sd(x1) = 0
lr_test <- function(restricted, full, boundary = FALSE) {
  .lr_chk_converged(restricted, "restricted")
  .lr_chk_converged(full, "full")

  ll_f <- stats::logLik(full)
  ll_r <- stats::logLik(restricted)
  df_f <- attr(ll_f, "df")
  df_r <- attr(ll_r, "df")

  df <- df_f - df_r
  if (!isTRUE(df > 0)) {
    stop("`full` must have more parameters than `restricted` (df = ", df_f,
         " - ", df_r, " = ", df, "); check the argument order and that the ",
         "models are nested.", call. = FALSE)
  }

  stat <- 2 * (as.numeric(ll_f) - as.numeric(ll_r))
  if (stat < 0) {
    warning("Restricted model has the higher log-likelihood (LR statistic = ",
            formatC(stat, format = "f", digits = 4), " < 0); the fits may not ",
            "be nested or one did not converge. Clamping the statistic to 0.",
            call. = FALSE)
    stat <- 0
  }

  p <- if (boundary) {
    # 50:50 chisq(df)/chisq(df-1) mixture. pchisq(stat, 0) is a point mass at 0,
    # so for df = 1 the second term vanishes and the mixture halves chisq(1).
    0.5 * stats::pchisq(stat, df, lower.tail = FALSE) +
      0.5 * stats::pchisq(stat, df - 1, lower.tail = FALSE)
  } else {
    stats::pchisq(stat, df, lower.tail = FALSE)
  }

  structure(
    list(statistic = stat, df = df, p.value = p,
         logLik_full = as.numeric(ll_f), logLik_restricted = as.numeric(ll_r),
         df_full = df_f, df_restricted = df_r, boundary = boundary),
    class = "rpbnb_lrtest"
  )
}

#' @export
print.rpbnb_lrtest <- function(x, digits = 4, ...) {
  fmt <- function(v) formatC(v, format = "f", digits = digits)
  cat("Likelihood-ratio test\n")
  cat(sprintf("  full model:       logLik = %s  (df = %d)\n",
              fmt(x$logLik_full), as.integer(x$df_full)))
  cat(sprintf("  restricted model: logLik = %s  (df = %d)\n",
              fmt(x$logLik_restricted), as.integer(x$df_restricted)))
  cat("  ", paste(rep("-", 50), collapse = ""), "\n", sep = "")
  cat(sprintf("  LR statistic = %s  on %d df   p = %s  %s\n",
              fmt(x$statistic), as.integer(x$df), fmt(x$p.value),
              signif_stars(x$p.value)))
  if (isTRUE(x$boundary)) {
    cat("  (boundary-corrected 50:50 chi-square mixture)\n")
  }
  invisible(x)
}
