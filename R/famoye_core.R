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
  # h_t(y) = exp(-y) - c_t ranges over (-c_t, 1 - c_t]. For 1 + lambda*h1*h2 >= 0
  # over the full support, the binding positive corner of h1*h2 is
  # max((1 - c1)(1 - c2), c1*c2) and the binding negative corner is
  # -max(c1(1 - c2), c2(1 - c1)). At low means (large c) the c1*c2 corner
  # dominates the lower bound, so it must be included via pmax().
  lam_min <- -1 / pmax((1 - c1) * (1 - c2), c1 * c2)
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

# Second derivatives of c(mu, m) -- Famoye (2010) Appendix. The beta-blocks are
# rank-one in the covariates: d2c/dbeta_j dbeta_s = d2ct_dbeta2_factor * x_j * x_s
# and d2c/dm dbeta_j = d2ct_dmdbeta_factor * x_j, so these helpers return the
# scalar (per-observation) factor and the caller contracts with the design.

#' d^2 c(mu,m) / d m^2
#' @keywords internal
#' @noRd
d2ct_dm2 <- function(mu, m, c) {
  d <- d_const(); G <- 1 + d * mu * m
  A <- m^(-2) * log(G) - d * mu / (m * G)
  (A^2 + 2 * d * mu / (m^2 * G) - 2 * m^(-3) * log(G) + (1 / m) * (d * mu / G)^2) * c
}

#' Per-obs factor b with d^2 c(mu,m) / d beta_j d beta_s = b * x_j * x_s
#' @keywords internal
#' @noRd
d2ct_dbeta2_factor <- function(mu, m, c) {
  d <- d_const(); G <- 1 + d * mu * m
  -d * c * mu * (1 - d * mu) / G^2
}

#' Per-obs factor e with d^2 c(mu,m) / d m d beta_j = e * x_j
#'
#' Derived directly as d/dm of dc/dbeta = -d*c*mu*x/G (with dc/dm = A*c,
#' dG/dm = d*mu): d2c/dm dbeta = -d*mu*c*(A*G - d*mu)/G^2 * x. (The printed
#' Appendix cross-term in Famoye (2010) carries a spurious m^-2; this closed
#' form is the one that matches the numerical derivative of c_val.)
#' @keywords internal
#' @noRd
d2ct_dmdbeta_factor <- function(mu, m, c) {
  d <- d_const(); G <- 1 + d * mu * m
  A <- m^(-2) * log(G) - d * mu / (m * G)
  -d * mu * c * (A * G - d * mu) / G^2
}
