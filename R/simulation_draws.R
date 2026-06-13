#' Generate standard-normal draws from a Halton sequence
#'
#' @param n_draws Number of draws (rows).
#' @param d Dimension (columns). If 0, a 0-column matrix is returned.
#' @param skip,burn Halton skip and burn-in.
#' @return An `n_draws` x `d` numeric matrix of standard-normal values.
#' @keywords internal
#' @noRd
halton_normal <- function(n_draws, d, skip = 100, burn = 200) {
  if (d <= 0) return(matrix(0, nrow = n_draws, ncol = 0))
  # `skip` retained for API symmetry; randtoolbox::halton has no skip arg.
  n <- burn + n_draws
  U <- randtoolbox::halton(n = n, dim = d, normal = FALSE,
                           usetime = FALSE, init = TRUE)
  U <- matrix(U, nrow = n, ncol = d)
  qnorm(U[(burn + 1):n, , drop = FALSE])
}
