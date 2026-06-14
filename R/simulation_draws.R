#' Generate standard-normal draws from a (randomized) Halton sequence
#'
#' Builds a Halton low-discrepancy sequence, discards the first `burn` points,
#' then applies a Cranley-Patterson rotation (a per-dimension uniform shift drawn
#' from the R RNG). The rotation makes the draw set depend on the current RNG
#' state, so a caller's [set.seed()] / `seed` choice produces a different but
#' fully reproducible low-discrepancy point set (randomized quasi-Monte Carlo),
#' instead of the same fixed sequence every time.
#'
#' @param n_draws Number of draws (rows).
#' @param d Dimension (columns). If 0, a 0-column matrix is returned.
#' @param burn Number of leading Halton points to discard (mitigates the
#'   correlation among early points across dimensions).
#' @return An `n_draws` x `d` numeric matrix of standard-normal values.
#' @keywords internal
#' @noRd
halton_normal <- function(n_draws, d, burn = 200) {
  if (d <= 0) return(matrix(0, nrow = n_draws, ncol = 0))
  n <- burn + n_draws
  U <- randtoolbox::halton(n = n, dim = d, normal = FALSE,
                           usetime = FALSE, init = TRUE)
  U <- matrix(U, nrow = n, ncol = d)
  U <- U[(burn + 1):n, , drop = FALSE]
  # Cranley-Patterson rotation: shift each dimension by a seed-dependent uniform,
  # wrap into [0, 1), then clamp away from the open endpoints so qnorm is finite.
  shift <- stats::runif(d)
  U <- sweep(U, 2, shift, `+`)
  U <- U - floor(U)
  U <- pmin(pmax(U, 1e-12), 1 - 1e-12)
  stats::qnorm(U)
}
