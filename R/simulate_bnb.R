#' Simulate data from the Famoye/Sarmanov bivariate NB distribution
#'
#' Generates paired count outcomes `(y1, y2)` from the fixed-parameter
#' Famoye/Sarmanov bivariate NB2 joint PMF:
#'
#' \deqn{P(Y_1=y_1, Y_2=y_2) = p_1(y_1)\,p_2(y_2)\,
#'   [1 + \lambda(e^{-y_1}-c_1)(e^{-y_2}-c_2)]}
#'
#' where \eqn{c_k = E[e^{-Y_k}]} under NB2(\eqn{\mu_k, m_k}).
#'
#' @param n Number of observations.
#' @param beta1,beta2 Named numeric vectors of coefficients for each equation;
#'   must include `"(Intercept)"`.
#' @param dispersion Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersion
#'   parameters (variance = mu + m * mu^2).
#' @param lambda Famoye/Sarmanov dependence parameter. Must lie within the
#'   valid bounds implied by the marginal means and dispersions.
#' @param covariates Optional data frame of covariates. If `NULL`, standard-
#'   normal columns are generated for every non-intercept coefficient name.
#' @param seed Optional random seed. If `NULL` (default) the RNG is left
#'   untouched and draws continue from the caller's current stream, so repeated
#'   calls yield distinct datasets; supply an integer for reproducible output.
#' @return A list with:
#'   \describe{
#'     \item{`data`}{data frame with `y1`, `y2`, and covariate columns}
#'     \item{`mu`}{data frame with `mu1`, `mu2` (per-obs conditional means)}
#'     \item{`true`}{list of true parameters: `beta1`, `beta2`, `dispersion`, `lambda`}
#'     \item{`settings`}{list with `n` and `seed`}
#'     \item{`meta`}{list with `seed` and `r_version`}
#'   }
#' @export
#' @examples
#' sim <- simulate_bnb(n = 500,
#'   beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
#'   beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 1)
#' head(sim$data)
simulate_bnb <- function(n, beta1, beta2,
                         dispersion = c(m1 = 0.5, m2 = 0.5),
                         lambda = 0,
                         covariates = NULL,
                         seed = NULL) {
  if (!"(Intercept)" %in% names(beta1))
    stop("`beta1` must include an '(Intercept)' element.", call. = FALSE)
  if (!"(Intercept)" %in% names(beta2))
    stop("`beta2` must include an '(Intercept)' element.", call. = FALSE)
  # Shared validator (also used by simulate_rpbnb / simulate_rpbnb_copula):
  # requires named c(m1 = ., m2 = .) with finite, strictly positive values, so a
  # negative/zero/non-finite dispersion errors here rather than silently
  # producing NaN counts downstream via rnbinom(size = 1/m).
  chk_dispersion(dispersion)

  # Only touch the RNG when a seed is explicitly supplied; with seed = NULL the
  # draws continue from the caller's current RNG stream (idiomatic R), so
  # repeated calls in a Monte Carlo loop produce distinct datasets rather than
  # silently identical ones.
  if (!is.null(seed)) set.seed(seed)

  if (is.null(covariates)) covariates <- .sim_default_covariates(beta1, beta2, n)
  .check_sim_covariates(covariates, beta1, beta2, n)

  d1 <- .build_sim_X(beta1, covariates, n)
  d2 <- .build_sim_X(beta2, covariates, n)
  mu1 <- as.vector(exp(d1$X %*% d1$beta))
  mu2 <- as.vector(exp(d2$X %*% d2$beta))

  m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]
  c1 <- c_val(mu1, m1)
  c2 <- c_val(mu2, m2)

  if (lambda != 0) {
    bnds <- lambda_bounds_vec(c1, c2)
    if (lambda < bnds[1] || lambda > bnds[2])
      stop("lambda (", lambda, ") is outside the valid bounds [",
           round(bnds[1], 4), ", ", round(bnds[2], 4), "].", call. = FALSE)
  }

  # Draw Y1 from its marginal NB2
  y1 <- rnbinom(n, size = 1 / m1, mu = mu1)

  if (lambda == 0) {
    # The Sarmanov weight W = 1 + lambda*(...) is identically 1 when
    # lambda == 0, so the conditional collapses exactly to the NB2 marginal —
    # draw it directly instead of building the truncated conditional grid.
    y2 <- rnbinom(n, size = 1 / m2, mu = mu2)
  } else {
    # Draw Y2 from conditional distribution P(Y2=y2|Y1=y1) via vectorized matrix.
    # NOTE: this is an approximate sampler. The conditional support is truncated to
    # 0:ymax, where ymax is the 0.9999 NB quantile of the largest-mean margin, so
    # the extreme upper tail (prob < 1e-4 per obs) is dropped. This is formula-
    # consistent with the Famoye/Sarmanov PMF but not exact. For exact generation
    # over the full NB tail, use rejection sampling (cf. legacy rBNBR()).
    ymax <- ceiling(max(qnbinom(0.9999, size = 1 / m2, mu = mu2)))
    y2_grid <- 0:ymax

    # P2[i, j] = dnbinom(y2_grid[j], size=1/m2, mu=mu2[i])
    P2 <- outer(mu2, y2_grid, function(mu, y) dnbinom(y, size = 1 / m2, mu = mu))

    # W[i, j] = 1 + lambda * (exp(-y1[i]) - c1[i]) * (exp(-y2_grid[j]) - c2[i])
    # c_per_row recycles down each column of the n x (ymax+1) matrix since its
    # length (n) equals the row count.
    row_factor <- lambda * (exp(-y1) - c1)   # length n
    c_per_row  <- row_factor * c2            # per-obs constant subtracted from each row
    W <- 1 + outer(row_factor, exp(-y2_grid)) - c_per_row
    Q <- pmax(P2 * W, 0)
    # Normalize rows (should already sum to ~1, but guard against float drift)
    Q <- Q / rowSums(Q)

    # Vectorized inverse-CDF sampling: one runif() draw per row and a single
    # max.col() lookup, instead of n interpreted sample() calls. Cumulative
    # sums are built with a loop over the (typically small) grid columns
    # rather than apply()+t() over rows, which is both faster and preserves
    # Q's matrix shape when the grid has exactly one column (ymax == 0) —
    # apply() silently simplifies a single-column result to a plain vector.
    cum <- Q
    for (j in seq_len(ncol(cum))[-1]) cum[, j] <- cum[, j - 1L] + cum[, j]
    u  <- runif(n) * cum[, ncol(cum)]
    y2 <- y2_grid[max.col(cum >= u, ties.method = "first")]
  }

  data <- data.frame(y1 = y1, y2 = y2, covariates)
  list(
    data     = data,
    mu       = data.frame(mu1 = mu1, mu2 = mu2),
    true     = list(beta1 = beta1, beta2 = beta2,
                    dispersion = dispersion, lambda = lambda),
    settings = list(n = n, seed = seed),
    meta     = list(seed = seed, r_version = R.version.string)
  )
}
