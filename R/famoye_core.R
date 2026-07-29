# Famoye/Sarmanov bivariate NB (NB2) core math. Internal. Single source of truth.

#' Famoye's d constant: 1 - exp(-1)
#' @keywords internal
#' @noRd
d_const <- function() 1 - exp(-1)

#' E(exp(-Y)) under NB2(mu, m); c(mu, m) = (1 + d*m*mu)^(-1/m)
#'
#' At m = 0 the NB2 margin collapses to Poisson(mu), whose E(exp(-Y)) is the
#' exact limit exp(-d*mu) (the generic (1 + d*m*mu)^(-1/m) form is a 0/0 that
#' evaluates to a wrong 1 at m = 0). `m = 0` therefore selects the true Poisson
#' branch, used by the poisson_1/poisson_2 restricted margins.
#' @keywords internal
#' @noRd
c_val <- function(mu, m) {
  if (m == 0) return(exp(-d_const() * mu))
  (1 + d_const() * m * mu)^(-1 / m)
}

#' NB2 log pmf with mean mu and size r = 1/m
#'
#' A size of r = Inf (m = 0) selects the exact Poisson log-pmf dpois(y, mu, log).
#' The finite-r NB2 form converges to it as r -> Inf but never equals it (the
#' error is governed by m*mu = mu/r), so a Poisson-restricted margin must take
#' this branch rather than a large-but-finite r.
#'
#' Leave that leading "r = Inf" unbackticked. DESCRIPTION sets
#' `Roxygen: list(markdown = TRUE)`, and a backticked span opening with "r "
#' is markdown's inline-R-code syntax, so roxygen tries to evaluate `= Inf`
#' as R and aborts the block on every document() run.
#' @keywords internal
#' @noRd
nb_logpmf_y_mu_r <- function(y, mu, r) {
  if (any(is.infinite(r))) return(stats::dpois(y, mu, log = TRUE))
  p <- r / (r + mu)
  # y * log(1 - p) with 1 - p = mu / (r + mu). Forming it via log1p(-p) suffers
  # catastrophic cancellation once a draw drives mu ~ 0: p rounds to exactly 1,
  # so 1 - p collapses to 0 and log1p(-p) is -Inf -- giving -Inf for y > 0 (the
  # true value is large but finite) and 0 * -Inf = NaN for y == 0, either of
  # which poisons the simulated likelihood. Use the cancellation-free form and
  # drop the analytically-zero y == 0 term.
  y_term <- ifelse(y == 0, 0, y * (log(mu) - log(r + mu)))
  lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r * log(p) + y_term
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
