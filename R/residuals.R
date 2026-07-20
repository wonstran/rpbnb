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

# Poisson signed deviance residual: the m = 0 limit of .nb2_deviance_resid, where
# the NB2 (y+r)log((y+r)/(mu+r)) term -> (y - mu). Used for a Poisson-restricted
# margin (r = Inf would make the NB2 form a NaN).
.pois_deviance_resid <- function(y, mu) {
  term_y <- ifelse(y == 0, 0, y * log(y / mu))
  d      <- pmax(2 * (term_y - (y - mu)), 0)
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
  # A Poisson-restricted margin (m = 0) uses the exact Poisson variance (mu) and
  # CDF (ppois), not NB2 at r = 1/POISSON_M. Older fits without the stored flag
  # default to the NB2 path via isTRUE().
  pois1 <- isTRUE(object$poisson_1); pois2 <- isTRUE(object$poisson_2)
  m1 <- if (pois1) 0 else exp(object$coef[["log_m1"]])
  m2 <- if (pois2) 0 else exp(object$coef[["log_m2"]])

  one <- function(y, mu, m, pois) {
    switch(type,
      response = y - mu,
      # Pearson variance mu + m*mu^2 reduces to the Poisson variance mu at m = 0.
      pearson  = .nb2_pearson_resid(y, mu, m),
      deviance = if (pois) .pois_deviance_resid(y, mu) else .nb2_deviance_resid(y, mu, m),
      quantile = {
        if (pois) {
          Fhi <- stats::ppois(y, mu)
          Flo <- ifelse(y > 0, stats::ppois(y - 1, mu), 0)
        } else {
          r   <- 1 / m
          Fhi <- stats::pnbinom(y, size = r, mu = mu)
          Flo <- ifelse(y > 0, stats::pnbinom(y - 1, size = r, mu = mu), 0)
        }
        .rqr(Flo, Fhi)
      })
  }

  build <- function() {
    if (margin == "y1") return(one(object$Y1, object$mu1, m1, pois1))
    if (margin == "y2") return(one(object$Y2, object$mu2, m2, pois2))
    data.frame(y1 = one(object$Y1, object$mu1, m1, pois1),
               y2 = one(object$Y2, object$mu2, m2, pois2))
  }
  if (type == "quantile") .with_seed(seed, build) else build()
}

# Extract the per-equation pieces needed for RP mixture residuals, mirroring
# .rp_predict_mu() in R/methods.R: mean coefs aligned to design columns, the
# random-coefficient distribution metadata, the stored draws, and native scales.
.rp_margin_parts <- function(object, eq) {
  X    <- if (eq == 1L) object$X1 else object$X2
  y    <- if (eq == 1L) object$Y1 else object$Y2
  # A Poisson-restricted margin uses the exact m = 0 limit: variance mu (not
  # mu + m*mu^2) and CDF ppois (r = Inf). Older fits without the flag keep NB2.
  pois <- if (eq == 1L) isTRUE(object$poisson_1) else isTRUE(object$poisson_2)
  m    <- if (pois) 0 else if (eq == 1L) object$m1 else object$m2
  bpfx <- paste0("b", eq)
  b    <- object$coef[grep(paste0("^", bpfx, ":"), names(object$coef))]
  names(b) <- sub(paste0("^", bpfx, ":"), "", names(b))
  b        <- b[colnames(X)]
  rand_idx <- if (eq == 1L) object$rand_idx1 else object$rand_idx2
  meta     <- object$rp_meta
  has_rand <- !is.null(meta) && length(rand_idx) > 0
  parts <- list(X = X, y = y, m = m, r = if (pois) Inf else 1 / m, pois = pois,
                b = b, rand_idx = rand_idx,
                xb = as.vector(X %*% b), has_rand = has_rand,
                dist = NULL, sgn = NULL, Z = NULL, scales = NULL)
  if (!has_rand) return(parts)
  parts$dist <- if (eq == 1L) meta$dist1 else meta$dist2
  parts$sgn  <- if (eq == 1L) meta$sign1 else meta$sign2
  parts$Z    <- if (eq == 1L) meta$Z1 else meta$Z2
  cols       <- colnames(X)[rand_idx]
  parts$scales <- vapply(seq_along(rand_idx), function(j) {
    lbl <- rand_dist_registry[[parts$dist[j]]]$scale_label
    exp(object$coef[[paste0(lbl, eq, ":", cols[j])]])
  }, numeric(1))
  parts
}

# Per-observation R x n matrix column of per-draw means mu_ir (capped), for one
# equation. Returns an n x R matrix.
.rp_margin_mu_draws <- function(p) {
  if (!p$has_rand) return(matrix(exp(p$xb), nrow = length(p$xb), ncol = 1L))
  dev <- rand_realize(p$Z, p$dist, p$sgn, p$b[p$rand_idx], p$scales)$dev
  XR  <- p$X[, p$rand_idx, drop = FALSE]
  mu  <- vapply(seq_len(nrow(p$Z)),
                function(r) pmin(exp(p$xb + as.vector(XR %*% dev[r, ])), RP_PRED_CAP),
                numeric(nrow(p$X)))
  matrix(mu, nrow = nrow(p$X))
}

# Exact mixture CDF corners Fbar(y-1), Fbar(y) for one equation, averaging the
# NB2 CDF over the stored draws. Analytic-Inf rows (lognormal, sign*x>0) -> NA.
.rp_mixture_cdf <- function(object, eq) {
  p   <- .rp_margin_parts(object, eq)
  mu  <- .rp_margin_mu_draws(p)                       # n x R
  Rn  <- ncol(mu)
  # A Poisson-restricted margin averages ppois over the draws (pnbinom with
  # size = 1/m = Inf would crash); the NB2 margin uses pnbinom(size = 1/m).
  cdf_hi <- if (p$pois) function(r) stats::ppois(p$y, mu[, r])
            else        function(r) stats::pnbinom(p$y, size = p$r, mu = mu[, r])
  cdf_lo <- if (p$pois) function(r) ifelse(p$y > 0, stats::ppois(p$y - 1, mu[, r]), 0)
            else        function(r) ifelse(p$y > 0, stats::pnbinom(p$y - 1, size = p$r, mu = mu[, r]), 0)
  Fhi <- rowMeans(vapply(seq_len(Rn), cdf_hi, numeric(nrow(mu))))
  Flo <- rowMeans(vapply(seq_len(Rn), cdf_lo, numeric(nrow(mu))))
  if (p$has_rand) {
    inf <- .rp_inf_rows(p$X, p$rand_idx, p$dist, p$sgn)
    if (any(inf)) {
      Fhi[inf] <- NA_real_; Flo[inf] <- NA_real_
      warning(sum(inf), " observation(s) have an analytically infinite mixture ",
              "mean (a lognormal random coefficient with sign * covariate > 0); ",
              "residuals set to NA for those rows.", call. = FALSE)
    }
  }
  list(Flo = Flo, Fhi = Fhi)
}

# Exact mixture marginal variance per observation (law of total variance over
# the draws): mean_r(NB2 var at mu_ir) + population var_r(mu_ir). Analytic-Inf
# rows (lognormal, sign*x>0) -> NA.
.rp_mixture_var <- function(object, eq) {
  p  <- .rp_margin_parts(object, eq)
  mu <- .rp_margin_mu_draws(p)                        # n x R
  Rn <- ncol(mu)
  mu_bar    <- rowMeans(mu)
  nbvar_bar <- rowMeans(mu + p$m * mu^2)              # mean of per-draw NB2 vars
  var_mu    <- rowMeans(mu^2) - mu_bar^2              # population var over draws
  v <- nbvar_bar + var_mu
  if (p$has_rand) {
    inf <- .rp_inf_rows(p$X, p$rand_idx, p$dist, p$sgn)
    if (any(inf)) {
      v[inf] <- NA_real_
      warning(sum(inf), " observation(s) have an analytically infinite mixture ",
              "mean (a lognormal random coefficient with sign * covariate > 0); ",
              "residuals set to NA for those rows.", call. = FALSE)
    }
  }
  v
}

# Per-observation fitted mean for one equation. Uses the stored mean when
# present (Famoye path); for copula fits (mu1/mu2 = NULL) it is the
# draw-integrated mixture mean, consistent with predict() and the residuals.
.rp_fitted_mean <- function(object, eq) {
  mu <- if (eq == 1L) object$mu1 else object$mu2
  if (is.null(mu)) mu <- rowMeans(.rp_margin_mu_draws(.rp_margin_parts(object, eq)))
  mu
}

#' Residuals for a random-parameter bivariate NB model
#'
#' Per-margin residuals for an [fit_rpbnb()] model. Each margin is a mixture of
#' NB2 distributions over the random-coefficient draws. `"quantile"` returns
#' randomized quantile residuals (Dunn & Smyth 1996) from the exact mixture
#' predictive CDF (the recommended residual); `"pearson"` uses the exact mixture
#' marginal variance. `"deviance"` is not defined for the mixture and errors.
#'
#' @param object An `rpbnb_fit` object from [fit_rpbnb()].
#' @param type Residual type: `"quantile"` (default), `"pearson"`, or
#'   `"response"`. `"deviance"` is not supported for `rpbnb_fit`.
#' @param margin Which margin: `"both"` (default), `"y1"`, or `"y2"`.
#' @param seed Optional integer seed for the quantile-residual randomization;
#'   does not disturb the caller's RNG stream.
#' @param ... Unused.
#' @return A numeric vector for a single margin, or a two-column data frame
#'   (`y1`, `y2`) for `margin = "both"`.
#' @export
residuals.rpbnb_fit <- function(object,
                                type   = c("quantile", "pearson", "deviance", "response"),
                                margin = c("both", "y1", "y2"),
                                seed   = NULL, ...) {
  type   <- match.arg(type)
  margin <- match.arg(margin)
  if (type == "deviance") {
    stop("deviance residuals are not defined for a random-parameter (rpbnb_fit) ",
         "mixture model; use type = \"quantile\" (recommended) or \"pearson\".",
         call. = FALSE)
  }

  one <- function(eq) {
    y  <- if (eq == 1L) object$Y1 else object$Y2
    mu <- .rp_fitted_mean(object, eq)
    switch(type,
      response = {
        # Match predict()/Pearson/quantile: a lognormal random coefficient with
        # sign * covariate > 0 makes the population mean analytically infinite, so
        # y - mu is undefined on those rows even though the cached mu1/mu2 is a
        # finite draw-capped average. Return NA there with the same warning.
        r <- y - mu
        p <- .rp_margin_parts(object, eq)
        if (p$has_rand) {
          inf <- .rp_inf_rows(p$X, p$rand_idx, p$dist, p$sgn)
          if (any(inf)) {
            r[inf] <- NA_real_
            warning(sum(inf), " observation(s) have an analytically infinite ",
                    "mixture mean (a lognormal random coefficient with sign * ",
                    "covariate > 0); residuals set to NA for those rows.",
                    call. = FALSE)
          }
        }
        r
      },
      pearson  = (y - mu) / sqrt(.rp_mixture_var(object, eq)),
      quantile = {
        cc <- .rp_mixture_cdf(object, eq)
        out <- rep(NA_real_, length(y))
        ok  <- is.finite(cc$Flo) & is.finite(cc$Fhi)
        out[ok] <- .rqr(cc$Flo[ok], cc$Fhi[ok])
        out
      })
  }

  build <- function() {
    if (margin == "y1") return(one(1L))
    if (margin == "y2") return(one(2L))
    data.frame(y1 = one(1L), y2 = one(2L))
  }
  if (type == "quantile") .with_seed(seed, build) else build()
}
