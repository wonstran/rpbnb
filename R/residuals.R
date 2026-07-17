# Residual extraction for bnb_fit and rpbnb_fit count models. Randomized
# quantile residuals (Dunn & Smyth 1996) are the primary count-model residual:
# ~ N(0,1) under a correct NB2 model, so QQ/histogram/normality checks are
# meaningful. Pearson/deviance/response residuals are also provided.

# Run `thunk()` under set.seed(seed) without disturbing the caller's RNG stream.
# With seed = NULL, runs thunk() directly (the RQR randomization is then drawn
# from the caller's current stream).
.with_seed <- function(seed, thunk) {
  if (is.null(seed)) return(thunk())
  has_old <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (has_old) {
    old <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    on.exit(assign(".Random.seed", old, envir = .GlobalEnv), add = TRUE)
  } else {
    on.exit(if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
              rm(".Random.seed", envir = .GlobalEnv), add = TRUE)
  }
  set.seed(seed)
  thunk()
}

# Dunn-Smyth randomized quantile residual from CDF corners Flo = F(y-1),
# Fhi = F(y): qnorm of a uniform draw on (Flo, Fhi). Draws from the current RNG
# stream (seed handling is the caller's job, via .with_seed()).
.rqr <- function(Flo, Fhi) {
  n <- length(Fhi)
  stats::qnorm(Flo + stats::runif(n) * (Fhi - Flo))
}

# NB2 Pearson residual. m is the NB2 dispersion (size = 1/m), variance mu+m*mu^2.
.nb2_pearson_resid <- function(y, mu, m) {
  (y - mu) / sqrt(mu + m * mu^2)
}

# NB2 signed deviance residual. size r = 1/m; the y*log(y/mu) term -> 0 at y=0.
.nb2_deviance_resid <- function(y, mu, m) {
  r      <- 1 / m
  term_y <- ifelse(y == 0, 0, y * log(y / mu))
  d      <- 2 * (term_y - (y + r) * log((y + r) / (mu + r)))
  d      <- pmax(d, 0)                     # guard tiny negatives from rounding
  sign(y - mu) * sqrt(d)
}

#' Residuals for a bivariate NB model
#'
#' Per-margin residuals for a fixed-coefficient [fit_bnb()] model. Each margin is
#' NB2 with fitted mean `mu` and dispersion `m` (`size = 1/m`). `"quantile"`
#' returns randomized quantile residuals (Dunn & Smyth 1996), which are
#' approximately N(0,1) under a correct model and are the recommended residual
#' for normality-style diagnostics on count data.
#'
#' @param object A `bnb_fit` object from [fit_bnb()].
#' @param type Residual type: `"quantile"` (default), `"pearson"`, `"deviance"`,
#'   or `"response"`.
#' @param margin Which margin: `"both"` (default), `"y1"`, or `"y2"`.
#' @param seed Optional integer seed for the quantile-residual randomization
#'   (ignored for other types); does not disturb the caller's RNG stream.
#' @param ... Unused.
#' @return A numeric vector for a single margin, or a two-column data frame
#'   (`y1`, `y2`) for `margin = "both"`.
#' @export
residuals.bnb_fit <- function(object,
                              type   = c("quantile", "pearson", "deviance", "response"),
                              margin = c("both", "y1", "y2"),
                              seed   = NULL, ...) {
  type   <- match.arg(type)
  margin <- match.arg(margin)
  m1 <- exp(object$coef[["log_m1"]]); m2 <- exp(object$coef[["log_m2"]])

  one <- function(y, mu, m) {
    switch(type,
      response = y - mu,
      pearson  = .nb2_pearson_resid(y, mu, m),
      deviance = .nb2_deviance_resid(y, mu, m),
      quantile = {
        r   <- 1 / m
        Fhi <- stats::pnbinom(y, size = r, mu = mu)
        Flo <- ifelse(y > 0, stats::pnbinom(y - 1, size = r, mu = mu), 0)
        .rqr(Flo, Fhi)
      })
  }

  build <- function() {
    if (margin == "y1") return(one(object$Y1, object$mu1, m1))
    if (margin == "y2") return(one(object$Y2, object$mu2, m2))
    data.frame(y1 = one(object$Y1, object$mu1, m1),
               y2 = one(object$Y2, object$mu2, m2))
  }
  if (type == "quantile") .with_seed(seed, build) else build()
}
