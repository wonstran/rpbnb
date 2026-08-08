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

#' Famoye admissible lambda interval over the random coefficients' support
#'
#' Shared by both engines. The admissible set is a property of the model, not of
#' whatever rule approximates its random-coefficient integral: reducing bounds
#' over a finite draw grid makes the "parameter space" shrink as draws are added
#' (measured, a two-normal-margin model runs [-2.550, 3.131] at 5 draws and
#' [-1.914, 2.261] at 50,000, converging on [-1, 1]) and admits lambda values
#' whose conditional pmf is negative on latent neighbourhoods of positive
#' probability that the draws happened to miss.
#'
#' Construction. For margin t and observation i, eta_ti = xb_ti + offset_ti +
#' sum_j X_tij * dev_tj with the dev_tj independent, so the attainable eta
#' interval is the fixed part plus the sum of the per-column deviation
#' intervals. mu = exp(eta) and c = (1 + d*m*mu)^(-1/m) is decreasing in mu, so
#' that maps to a c-interval. Both quantities bounded by lambda_bounds_vec() are
#' pointwise maxima of bilinear functions of (c1, c2), and a bilinear function on
#' a box attains its extremum at a vertex, so evaluating the four corners of the
#' attainable c-rectangle is exact rather than a sample.
#'
#' The eta range is deliberately NOT clipped to any numerical window. Clipping
#' caps the attainable mean, which stops c reaching 0 at large dispersion and
#' WIDENS the interval -- at log_m = 20 a clipped range gives an upper bound of
#' 8.97e6 instead of 1, admitting lambda = 2. The unclipped limits are exact in
#' ordinary R arithmetic: c_val(Inf, m) = 0 and c_val(0, m) = 1, on the Poisson
#' branch too. A narrower theoretical interval is also safe for a clamped
#' objective, whose means are a subset of the model's.
#'
#' Scales must already be on the natural scale and strictly positive; callers
#' pass exp(clamp(log_scale, -20, 20)) so that a scale which underflows in R
#' still counts as varying, matching the template.
#'
#' @keywords internal
#' @noRd
.dev_support_range <- function(dist, sgn, b, s) {
  n <- length(dist)
  lo <- numeric(n); hi <- numeric(n)
  for (j in seq_len(n)) {
    if (dist[j] %in% c("uniform", "triangular")) {
      # coef = b + s*base with base bounded, so the deviation is bounded too.
      # Treating these as spanning (0,1) in c would be OVER-strict, not merely
      # conservative: it would reject admissible fits.
      lo[j] <- -s[j]; hi[j] <- s[j]
    } else if (identical(dist[j], "lognormal")) {
      # dev = sign*exp(b + s*base) - b, one-sided.
      if (sgn[j] >= 0) { lo[j] <- -b[j]; hi[j] <- Inf }
      else             { lo[j] <- -Inf;  hi[j] <- -b[j] }
    } else {
      lo[j] <- -Inf; hi[j] <- Inf          # normal
    }
  }
  cbind(lo, hi)
}

#' @keywords internal
#' @noRd
.c_support_range <- function(X, beta, off, rand_idx, dist, sgn, s, m_v) {
  xb <- as.vector(X %*% beta)
  if (!is.null(off) && length(off)) xb <- xb + off
  add_lo <- rep(0, length(xb)); add_hi <- rep(0, length(xb))
  if (length(rand_idx)) {
    # The Rcpp engine's legacy path leaves dist/sign NULL, meaning all-normal.
    if (is.null(dist)) dist <- rep("normal", length(rand_idx))
    if (is.null(sgn))  sgn  <- rep(1, length(rand_idx))
    lim <- .dev_support_range(dist, sgn, beta[rand_idx], s)
    for (j in seq_along(rand_idx)) {
      xj <- X[, rand_idx[j]]
      a <- xj * lim[j, 1]; b2 <- xj * lim[j, 2]
      # A zero loading contributes nothing; without this, 0 * Inf is NaN.
      a[xj == 0] <- 0; b2[xj == 0] <- 0
      add_lo <- add_lo + pmin(a, b2)
      add_hi <- add_hi + pmax(a, b2)
    }
  }
  cbind(lower = c_val(exp(xb + add_hi), m_v),
        upper = c_val(exp(xb + add_lo), m_v))
}

#' @keywords internal
#' @noRd
famoye_support_bounds <- function(X1, X2, off1, off2,
                                  rand_idx1, rand_idx2,
                                  dist1, dist2, sign1, sign2,
                                  beta1, beta2, s1, s2, m1, m2) {
  r1 <- .c_support_range(X1, beta1, off1, rand_idx1, dist1, sign1, s1, m1)
  r2 <- .c_support_range(X2, beta2, off2, rand_idx2, dist2, sign2, s2, m2)
  b <- lambda_bounds_vec(
    c(r1[, "lower"], r1[, "lower"], r1[, "upper"], r1[, "upper"]),
    c(r2[, "lower"], r2[, "upper"], r2[, "lower"], r2[, "upper"])
  )
  c(lower = b[1], upper = b[2])
}
