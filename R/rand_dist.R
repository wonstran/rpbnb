# Random-coefficient distribution registry: the single source of truth for the
# per-distribution transforms and analytic-gradient factors used by the
# random-parameter BNB likelihood, its gradient, the lambda-bounds, the fitted
# means, and the simulator.
#
# A realized random coefficient for a column is
#   beta = location + scale * base                 (normal, uniform, triangular)
#   beta = sign * exp(location + scale * base)      (lognormal)
# where `base` is the inverse-CDF-ready base draw: a standard normal z for
# normal/lognormal, the symmetric triangular variate for triangular, and the
# uniform u in (0,1) itself for uniform. `u_to_base()` maps a uniform draw to
# the base variate; the simulator may instead draw the base directly.

#' Symmetric triangular inverse-CDF on `[-1, 1]`
#' @keywords internal
#' @noRd
tri_icdf <- function(u) {
  ifelse(u < 0.5, -1 + sqrt(2 * u), 1 - sqrt(2 * (1 - u)))
}

#' Registry of supported random-coefficient distributions
#' @keywords internal
#' @noRd
rand_dist_registry <- list(
  normal = list(
    base        = "normal",
    u_to_base   = function(u) stats::qnorm(u),
    coef        = function(b, s, base, sign) b + s * base,
    dev         = function(b, s, base, sign) s * base,
    dloc_factor = function(b, s, base, coef) rep(1, length(base)),
    dscale      = function(b, s, base, coef) s * base,
    scale_label = "log_sd"
  ),
  uniform = list(
    base        = "uniform",
    u_to_base   = function(u) u,
    coef        = function(b, s, base, sign) b + s * (2 * base - 1),
    dev         = function(b, s, base, sign) s * (2 * base - 1),
    dloc_factor = function(b, s, base, coef) rep(1, length(base)),
    dscale      = function(b, s, base, coef) s * (2 * base - 1),
    scale_label = "log_w"
  ),
  triangular = list(
    base        = "uniform",
    u_to_base   = function(u) tri_icdf(u),
    coef        = function(b, s, base, sign) b + s * base,
    dev         = function(b, s, base, sign) s * base,
    dloc_factor = function(b, s, base, coef) rep(1, length(base)),
    dscale      = function(b, s, base, coef) s * base,
    scale_label = "log_w"
  ),
  lognormal = list(
    base        = "normal",
    u_to_base   = function(u) stats::qnorm(u),
    coef        = function(b, s, base, sign) sign * exp(b + s * base),
    dev         = function(b, s, base, sign) sign * exp(b + s * base) - b,
    dloc_factor = function(b, s, base, coef) coef,
    dscale      = function(b, s, base, coef) coef * base * s,
    scale_label = "log_s"
  )
)

#' Normalize a random-coefficient spec to aligned name/dist/sign/scale vectors
#'
#' Accepts NULL, a character vector of column names (all Normal), or a named
#' list whose values are either a distribution-name string or a list with
#' `dist`, optional `sign` (lognormal only), and optional `scale` (or `sd`).
#' @keywords internal
#' @noRd
parse_rand_spec <- function(spec) {
  if (is.null(spec) || length(spec) == 0) {
    return(list(names = character(0), dist = character(0),
                sign = numeric(0), scale = numeric(0)))
  }
  valid <- names(rand_dist_registry)
  if (is.character(spec) && is.null(names(spec))) {
    return(list(names = spec,
                dist  = rep("normal", length(spec)),
                sign  = rep(1, length(spec)),
                scale = rep(NA_real_, length(spec))))
  }
  if (!is.list(spec) || is.null(names(spec)) || any(!nzchar(names(spec)))) {
    stop("random spec must be a character vector of names or a named list.",
         call. = FALSE)
  }
  nm    <- names(spec)
  dist  <- character(length(nm))
  sgn   <- numeric(length(nm))
  scale <- numeric(length(nm))
  for (i in seq_along(nm)) {
    v <- spec[[i]]
    if (is.character(v) && length(v) == 1L) {
      d <- v; this_sign <- 1; this_scale <- NA_real_
    } else if (is.list(v)) {
      d          <- if (is.null(v$dist)) "normal" else v$dist
      this_sign  <- if (is.null(v$sign)) 1 else v$sign
      this_scale <- if (!is.null(v$scale)) v$scale
                    else if (!is.null(v$sd)) v$sd else NA_real_
      if (!is.null(v$sign) && d != "lognormal") {
        stop("`sign` is only meaningful for lognormal (got '", d,
             "' for '", nm[i], "').", call. = FALSE)
      }
    } else {
      stop("random spec value for '", nm[i],
           "' must be a distribution name or a list.", call. = FALSE)
    }
    if (!d %in% valid) {
      stop("unknown distribution '", d, "' for '", nm[i], "'. Valid: ",
           paste(valid, collapse = ", "), ".", call. = FALSE)
    }
    if (!this_sign %in% c(-1, 1)) {
      stop("`sign` must be -1 or 1 for '", nm[i], "'.", call. = FALSE)
    }
    dist[i] <- d; sgn[i] <- this_sign; scale[i] <- this_scale
  }
  list(names = nm, dist = dist, sign = sgn, scale = scale)
}

#' Per-draw realized coefficients and gradient factors for one equation
#'
#' @param U An R x q matrix of uniform Halton draws (one column per random coef).
#' @param dist,sign Length-q distribution names and signs.
#' @param b,s Length-q location and native-scale (exp of the log-scale param).
#' @return A list of R x q matrices: `base`, `coef`, `dev`, `dloc`, `dscale`.
#' @keywords internal
#' @noRd
rand_realize <- function(U, dist, sign, b, s) {
  R <- nrow(U); q <- ncol(U)
  base <- coef <- dev <- dloc <- dscale <- matrix(0, nrow = R, ncol = q)
  for (j in seq_len(q)) {
    reg <- rand_dist_registry[[dist[j]]]
    bj  <- reg$u_to_base(U[, j])
    cj  <- reg$coef(b[j], s[j], bj, sign[j])
    base[, j]   <- bj
    coef[, j]   <- cj
    dev[, j]    <- reg$dev(b[j], s[j], bj, sign[j])
    dloc[, j]   <- reg$dloc_factor(b[j], s[j], bj, cj)
    dscale[, j] <- reg$dscale(b[j], s[j], bj, cj)
  }
  list(base = base, coef = coef, dev = dev, dloc = dloc, dscale = dscale)
}
