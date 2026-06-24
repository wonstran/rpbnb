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
