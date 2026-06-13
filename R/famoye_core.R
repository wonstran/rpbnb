# Famoye/Sarmanov bivariate NB (NB2) core math. Internal. Single source of truth.

#' Famoye's d constant: 1 - exp(-1)
#' @keywords internal
#' @noRd
d_const <- function() 1 - exp(-1)

#' E(exp(-Y)) under NB2(mu, m); c(mu, m) = (1 + d*m*mu)^(-1/m)
#' @keywords internal
#' @noRd
c_val <- function(mu, m) (1 + d_const() * m * mu)^(-1 / m)

#' NB2 log pmf with mean mu and size r = 1/m
#' @keywords internal
#' @noRd
nb_logpmf_y_mu_r <- function(y, mu, r) {
  p <- r / (r + mu)
  lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r * log(p) + y * log1p(-p)
}

#' Conservative global lambda-bounds from per-obs c1, c2
#' @keywords internal
#' @noRd
lambda_bounds_vec <- function(c1, c2) {
  lam_min <- -1 / ((1 - c1) * (1 - c2))
  lam_max <-  1 / pmax(c1 * (1 - c2), c2 * (1 - c1))
  c(max(lam_min), min(lam_max))
}

#' d c(mu,m) / d m
#' @keywords internal
#' @noRd
dct_dm <- function(mu, m, c) {
  m_inv <- 1 / m
  denom <- 1 + d_const() * m * mu
  term  <- m_inv * (m_inv * log(denom) - (d_const() * mu) / denom)
  term * c
}

#' d c(mu,m) / d beta_j as an n x p matrix
#' @keywords internal
#' @noRd
dc_dbeta_mat <- function(mu, m, c, X) {
  denom <- 1 + d_const() * m * mu
  row_factor <- -(d_const() * c * mu) / denom
  sweep(X, 1, row_factor, `*`)
}
