#' Generate uniform draws from a (randomized) Halton sequence
#'
#' Builds a Halton low-discrepancy sequence, discards the first `burn` points,
#' then applies a Cranley-Patterson rotation (a per-dimension uniform shift drawn
#' from the R RNG) and clamps away from the open endpoints. The rotation makes
#' the draw set depend on the current RNG state, so a caller's [set.seed()] /
#' `seed` choice produces a different but fully reproducible low-discrepancy
#' point set (randomized quasi-Monte Carlo).
#'
#' @param n_draws Number of draws (rows).
#' @param d Dimension (columns). If 0, a 0-column matrix is returned.
#' @param burn Number of leading Halton points to discard.
#' @return An `n_draws` x `d` numeric matrix of uniforms in (0, 1).
#' @keywords internal
#' @noRd
halton_uniform <- function(n_draws, d, burn = 200) {
  if (d <= 0) return(matrix(0, nrow = n_draws, ncol = 0))
  n <- burn + n_draws
  U <- randtoolbox::halton(n = n, dim = d, normal = FALSE,
                           usetime = FALSE, init = TRUE)
  U <- matrix(U, nrow = n, ncol = d)
  U <- U[(burn + 1):n, , drop = FALSE]
  shift <- stats::runif(d)
  U <- sweep(U, 2, shift, `+`)
  U <- U - floor(U)
  pmin(pmax(U, 1e-12), 1 - 1e-12)
}

#' Standard-normal Halton draws (qnorm of `halton_uniform()`)
#' @inheritParams halton_uniform
#' @return An `n_draws` x `d` numeric matrix of standard-normal values.
#' @keywords internal
#' @noRd
halton_normal <- function(n_draws, d, burn = 200) {
  if (d <= 0) return(matrix(0, nrow = n_draws, ncol = 0))
  stats::qnorm(halton_uniform(n_draws, d, burn = burn))
}

#' Generate default standard-normal covariates for a simulator
#'
#' Used by `simulate_bnb()` / `simulate_rpbnb()` when `covariates = NULL`.
#' @keywords internal
#' @noRd
.sim_default_covariates <- function(beta1, beta2, n) {
  vars <- setdiff(union(names(beta1), names(beta2)), "(Intercept)")
  if (length(vars) == 0L) return(data.frame(row.names = seq_len(n)))
  as.data.frame(stats::setNames(lapply(vars, function(v) stats::rnorm(n)), vars))
}

#' Validate a user-supplied covariate data frame for a simulator
#'
#' Checks that every non-intercept coefficient name has a matching column and
#' that the frame has exactly `n` rows (an nrow mismatch would otherwise be
#' silently recycled by `cbind()`/`data.frame()` in `.build_sim_X()`).
#' @keywords internal
#' @noRd
.check_sim_covariates <- function(covariates, beta1, beta2, n) {
  vars <- setdiff(union(names(beta1), names(beta2)), "(Intercept)")
  missing_cov <- setdiff(vars, names(covariates))
  if (length(missing_cov)) {
    stop("covariates is missing required column(s): ",
         paste(missing_cov, collapse = ", "), ".", call. = FALSE)
  }
  if (nrow(covariates) != n) {
    stop("`covariates` must have exactly n = ", n, " rows (got ",
         nrow(covariates), ").", call. = FALSE)
  }
}

#' Build a design matrix aligned to a named coefficient vector
#'
#' `(Intercept)` is always placed first; `beta` is returned reordered to match
#' the resulting columns so `X %*% beta` is correct regardless of the order
#' the caller supplied the coefficients in (positional matrix multiplication
#' would otherwise silently pair the wrong coefficient with each column).
#' @keywords internal
#' @noRd
.build_sim_X <- function(beta, covariates, n) {
  nms <- c("(Intercept)", setdiff(names(beta), "(Intercept)"))
  X <- cbind(`(Intercept)` = rep(1, n))
  for (nm in nms[-1]) X <- cbind(X, covariates[[nm]])
  colnames(X) <- nms
  list(X = X, beta = beta[nms])
}
