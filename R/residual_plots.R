# Base-graphics residual diagnostic plots for bnb_fit and rpbnb_fit. Four panels
# per margin: residuals-vs-fitted, QQ (RQR vs N(0,1)), histogram of RQR with an
# N(0,1) overlay, and scale-location. Panels 2 and 3 always use randomized
# quantile residuals (only RQR are ~N(0,1) under a correct count model); panels
# 1 and 4 use `resid_type`.

# Draw up to four panels for one margin. `rqr` is the randomized quantile
# residual (panels 2-3); `r_used` is the `resid_type` residual (panels 1, 4).
.residual_plot_margin <- function(mu, r_used, rqr, which, resp_name, resid_label) {
  fin_ru <- is.finite(mu) & is.finite(r_used)
  fin_rq <- is.finite(rqr)
  if (1 %in% which) {
    if (sum(fin_ru) == 0) {
      graphics::plot.new()
      graphics::title(main = paste0(resp_name, ": Residuals vs fitted"))
      graphics::text(0.5, 0.5, "no finite residuals")
    } else {
      graphics::plot(mu[fin_ru], r_used[fin_ru],
                     xlab = "Fitted mean", ylab = resid_label,
                     main = paste0(resp_name, ": Residuals vs fitted"))
      graphics::abline(h = 0, lty = 3)
      if (sum(fin_ru) >= 3) graphics::lines(stats::lowess(mu[fin_ru], r_used[fin_ru]), col = "red")
    }
  }
  if (2 %in% which) {
    if (sum(fin_rq) == 0) {
      graphics::plot.new()
      graphics::title(main = paste0(resp_name, ": Normal QQ (RQR)"))
      graphics::text(0.5, 0.5, "no finite residuals")
    } else {
      stats::qqnorm(rqr[fin_rq], main = paste0(resp_name, ": Normal QQ (RQR)"))
      stats::qqline(rqr[fin_rq])
    }
  }
  if (3 %in% which) {
    if (sum(fin_rq) == 0) {
      graphics::plot.new()
      graphics::title(main = paste0(resp_name, ": Histogram (RQR)"))
      graphics::text(0.5, 0.5, "no finite residuals")
    } else {
      graphics::hist(rqr[fin_rq], freq = FALSE, breaks = "FD",
                     xlab = "Randomized quantile residual",
                     main = paste0(resp_name, ": Histogram (RQR)"))
      # Pass the bare NAME dnorm rather than the expression stats::dnorm(x):
      # curve() substitutes its first argument and, for a name, builds the call
      # itself -- so no free `x` appears here for R CMD check to report as an
      # undefined global variable. It must be the unqualified name: curve()
      # accepts a name or a call mentioning `x`, and `stats::dnorm` is a call
      # that mentions neither, which curve() rejects. The bare `dnorm` resolves
      # through this package's @importFrom stats dnorm (rpbnb-package.R).
      graphics::curve(dnorm, add = TRUE, col = "red")
    }
  }
  if (4 %in% which) {
    if (sum(fin_ru) == 0) {
      graphics::plot.new()
      graphics::title(main = paste0(resp_name, ": Scale-location"))
      graphics::text(0.5, 0.5, "no finite residuals")
    } else {
      sl <- sqrt(abs(r_used))
      graphics::plot(mu[fin_ru], sl[fin_ru],
                     xlab = "Fitted mean", ylab = paste0("sqrt|", resid_label, "|"),
                     main = paste0(resp_name, ": Scale-location"))
      if (sum(fin_ru) >= 3) graphics::lines(stats::lowess(mu[fin_ru], sl[fin_ru]), col = "red")
    }
  }
  invisible(NULL)
}

# Shared driver for both classes: pulls the two residual vectors per margin via
# the residuals() generic and lays panels out 2x2 per margin.
.residual_plot <- function(x, margin, which, resid_type, seed) {
  margin <- match.arg(margin, c("both", "y1", "y2"))
  eqs    <- if (margin == "both") c("y1", "y2") else margin
  op <- graphics::par(mfrow = c(2, 2))
  on.exit(graphics::par(op), add = TRUE)
  for (nm in eqs) {
    mu  <- .rp_fitted_mean(x, if (nm == "y1") 1L else 2L)
    rq  <- residuals(x, type = "quantile", margin = nm, seed = seed)
    ru  <- if (identical(resid_type, "quantile")) rq
           else residuals(x, type = resid_type, margin = nm)
    .residual_plot_margin(mu, ru, rq, which, nm, resid_type)
  }
  invisible(NULL)
}

#' Residual diagnostic plots for a bivariate NB model
#'
#' Four base-graphics panels per margin: residuals-vs-fitted, a normal QQ plot of
#' the randomized quantile residuals, a histogram of the RQR with an N(0,1)
#' overlay, and a scale-location plot. The QQ and histogram panels always use
#' RQR (only these are approximately N(0,1) under a correct count model).
#'
#' @param x A `bnb_fit` object from [fit_bnb()].
#' @param margin Which margin to plot: `"both"` (default), `"y1"`, or `"y2"`.
#' @param which Integer subset of panels `1:4` (1 = residuals-vs-fitted,
#'   2 = QQ, 3 = histogram, 4 = scale-location).
#' @param resid_type Residual type for panels 1 and 4: `"quantile"` (default),
#'   `"pearson"`, `"deviance"`, or `"response"`.
#' @param seed Optional integer seed for the RQR randomization.
#' @param ... Unused.
#' @return `NULL`, invisibly (called for the side effect of drawing).
#' @export
plot.bnb_fit <- function(x, margin = c("both", "y1", "y2"), which = 1:4,
                         resid_type = "quantile", seed = NULL, ...) {
  .residual_plot(x, match.arg(margin), which, resid_type, seed)
}

#' Residual diagnostic plots for a random-parameter bivariate NB model
#'
#' Four base-graphics panels per margin, as for [plot.bnb_fit()], built on the
#' mixture-based randomized quantile residuals. `resid_type = "deviance"` is not
#' available for `rpbnb_fit`.
#'
#' @param x An `rpbnb_fit` object from [fit_rpbnb()].
#' @param margin Which margin to plot: `"both"` (default), `"y1"`, or `"y2"`.
#' @param which Integer subset of panels `1:4`.
#' @param resid_type Residual type for panels 1 and 4: `"quantile"` (default),
#'   `"pearson"`, or `"response"`.
#' @param seed Optional integer seed for the RQR randomization.
#' @param ... Unused.
#' @return `NULL`, invisibly.
#' @export
plot.rpbnb_fit <- function(x, margin = c("both", "y1", "y2"), which = 1:4,
                           resid_type = "quantile", seed = NULL, ...) {
  .residual_plot(x, match.arg(margin), which, resid_type, seed)
}
