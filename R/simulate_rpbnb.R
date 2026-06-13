#' Simulate data from a random-parameter bivariate NB process
#'
#' @param n Number of observations.
#' @param beta1,beta2 Named numeric vectors of fixed coefficient means per
#'   equation; must include "(Intercept)".
#' @param random_1,random_2 Named lists giving random coefficients, e.g.
#'   `list(x1 = list(sd = 0.5))`. Means come from `beta1`/`beta2`.
#' @param dispersion Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersions.
#' @param lambda Famoye dependence parameter (0 = independent margins).
#' @param covariates Optional data frame of covariates; if NULL, standard-normal
#'   columns are generated for every non-intercept name.
#' @param seed Random seed.
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
                           lambda = 0, covariates = NULL, seed = 1234) {
  stopifnot("(Intercept)" %in% names(beta1), "(Intercept)" %in% names(beta2))
  if (!all(c("m1", "m2") %in% names(dispersion))) {
    stop("`dispersion` must be a named vector c(m1 = ., m2 = .).", call. = FALSE)
  }
  chk_rand <- function(rl, bv, lbl) {
    for (nm in names(rl)) if (!nm %in% names(bv)) {
      stop("random name '", nm, "' not in ", lbl, ".", call. = FALSE)
    }
  }
  chk_rand(random_1, beta1, "beta1"); chk_rand(random_2, beta2, "beta2")
  if (lambda != 0) {
    stop("Phase 1 simulate_rpbnb supports lambda = 0 (independent margins) only.",
         call. = FALSE)
  }

  set.seed(seed)
  vars <- setdiff(union(names(beta1), names(beta2)), "(Intercept)")
  if (is.null(covariates)) {
    covariates <- as.data.frame(stats::setNames(
      lapply(vars, function(v) rnorm(n)), vars))
  }
  missing_cov <- setdiff(vars, names(covariates))
  if (length(missing_cov)) {
    stop("covariates is missing required column(s): ",
         paste(missing_cov, collapse = ", "), ".", call. = FALSE)
  }
  build_X <- function(bv) {
    X <- cbind(`(Intercept)` = rep(1, n))
    for (nm in setdiff(names(bv), "(Intercept)")) X <- cbind(X, covariates[[nm]])
    colnames(X) <- names(bv)
    X
  }
  X1 <- build_X(beta1); X2 <- build_X(beta2)

  realize <- function(bv, rl, X) {
    B <- matrix(rep(bv, each = n), nrow = n, dimnames = list(NULL, names(bv)))
    for (nm in names(rl)) B[, nm] <- bv[[nm]] + rl[[nm]]$sd * rnorm(n)
    B
  }
  B1 <- realize(beta1, random_1, X1); B2 <- realize(beta2, random_2, X2)
  mu1 <- exp(rowSums(X1 * B1)); mu2 <- exp(rowSums(X2 * B2))
  m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]
  y1 <- rnbinom(n, size = 1 / m1, mu = mu1)
  y2 <- rnbinom(n, size = 1 / m2, mu = mu2)

  data <- data.frame(y1 = y1, y2 = y2, covariates)
  list(
    data = data,
    coef_realized = list(eq1 = B1, eq2 = B2),
    mu = data.frame(mu1 = mu1, mu2 = mu2),
    true = list(beta1 = beta1, beta2 = beta2, random_1 = random_1,
                random_2 = random_2, dispersion = dispersion, lambda = lambda),
    settings = list(n = n, seed = seed),
    meta = list(seed = seed, r_version = R.version.string)
  )
}
