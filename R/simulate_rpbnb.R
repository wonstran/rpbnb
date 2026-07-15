#' Simulate data from a random-parameter bivariate NB process
#'
#' @param n Number of observations.
#' @param beta1,beta2 Named numeric vectors of fixed coefficient means per
#'   equation; must include "(Intercept)".
#' @param random_1,random_2 Named lists giving random coefficients. Each value
#'   is a list with `dist` (one of "normal", "lognormal", "uniform",
#'   "triangular"; default "normal"), `scale` (or `sd`) for the dispersion, and
#'   `sign` (-1/1, lognormal only). Means come from `beta1`/`beta2`; for a
#'   lognormal coefficient the `beta` entry is the log-location and the realized
#'   coefficient is `sign * exp(log_location + scale * z)`.
#' @param dispersion Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersions.
#' @param lambda Famoye dependence parameter (0 = independent margins).
#' @param covariates Optional data frame of covariates; if NULL, standard-normal
#'   columns are generated for every non-intercept name.
#' @param seed Optional random seed. If `NULL` (default) the RNG is left
#'   untouched and draws continue from the caller's current stream, so repeated
#'   calls yield distinct datasets; supply an integer for reproducible output.
#' @return A list with `data` (y1, y2, covariates), `coef_realized`
#'   (per-obs coefficients per equation), `mu` (conditional means), `true`
#'   (parameters), `settings`, and `meta` (R/seed/timestamp passed by caller).
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 500,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' head(sim$data)
simulate_rpbnb <- function(n, beta1, beta2,
                           random_1 = NULL, random_2 = NULL,
                           dispersion = c(m1 = 0.5, m2 = 0.5),
                           lambda = 0, covariates = NULL, seed = NULL) {
  stopifnot("(Intercept)" %in% names(beta1), "(Intercept)" %in% names(beta2))
  chk_dispersion(dispersion)
  spec1 <- parse_rand_spec(random_1)
  spec2 <- parse_rand_spec(random_2)
  chk_rand_spec(spec1, beta1, "beta1"); chk_rand_spec(spec2, beta2, "beta2")
  if (lambda != 0) {
    stop("Phase 1 simulate_rpbnb supports lambda = 0 (independent margins) only.",
         call. = FALSE)
  }

  # Only touch the RNG when a seed is explicitly supplied; with seed = NULL the
  # draws continue from the caller's current RNG stream (idiomatic R), so
  # repeated calls in a Monte Carlo loop produce distinct datasets rather than
  # silently identical ones.
  if (!is.null(seed)) set.seed(seed)
  if (is.null(covariates)) covariates <- .sim_default_covariates(beta1, beta2, n)
  .check_sim_covariates(covariates, beta1, beta2, n)

  d1 <- .build_sim_X(beta1, covariates, n); X1 <- d1$X; b1 <- d1$beta
  d2 <- .build_sim_X(beta2, covariates, n); X2 <- d2$X; b2 <- d2$beta

  realize <- function(bv, spec, X) {
    B <- matrix(rep(bv, each = n), nrow = n, dimnames = list(NULL, names(bv)))
    for (i in seq_along(spec$names)) {
      nm  <- spec$names[i]
      reg <- rand_dist_registry[[spec$dist[i]]]
      base <- if (reg$base == "normal") stats::rnorm(n)
              else reg$u_to_base(stats::runif(n))
      B[, nm] <- reg$coef(bv[[nm]], spec$scale[i], base, spec$sign[i])
    }
    B
  }
  # realize()/rowSums() below pair X's and B's columns by position, so both
  # must be built from the same (Intercept)-first-reordered coefficient
  # vector — using the caller's original beta1/beta2 here would silently
  # mismatch columns whenever "(Intercept)" isn't listed first.
  B1 <- realize(b1, spec1, X1); B2 <- realize(b2, spec2, X2)
  mu1 <- exp(rowSums(X1 * B1)); mu2 <- exp(rowSums(X2 * B2))
  m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]
  y1 <- rnbinom(n, size = 1 / m1, mu = mu1)
  y2 <- rnbinom(n, size = 1 / m2, mu = mu2)

  data <- data.frame(y1 = y1, y2 = y2, covariates)
  list(
    data = data,
    coef_realized = list(eq1 = B1, eq2 = B2),
    mu = data.frame(mu1 = mu1, mu2 = mu2),
    true = list(beta1 = beta1, beta2 = beta2, random_1 = spec1,
                random_2 = spec2, dispersion = dispersion, lambda = lambda),
    settings = list(n = n, seed = seed),
    meta = list(seed = seed, r_version = R.version.string)
  )
}
