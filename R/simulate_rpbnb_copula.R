#' Sample copula uniforms (u, v) for a given family and native parameter.
#'
#' Conditional-inversion sampling: draw independent u, w ~ U(0,1); set
#' v = C_{2|1}^{-1}(w | u). Gaussian uses a bivariate-normal draw directly.
#' @keywords internal
#' @noRd
.rcopula_uv <- function(n, family, theta) {
  u <- stats::runif(n)
  if (family == "normal") {
    rho <- theta
    z1 <- stats::qnorm(u)
    z2 <- rho * z1 + sqrt(1 - rho^2) * stats::rnorm(n)
    return(list(u = u, v = stats::pnorm(z2)))
  }
  w <- stats::runif(n)
  if (family == "frank") {
    if (abs(theta) < 1e-8) return(list(u = u, v = w))
    et  <- exp(-theta)
    etu <- exp(-theta * u)
    v <- -log(1 + w * (et - 1) / (1 + (1 - w) * (etu - 1))) / theta
    v <- pmin(pmax(v, 1e-12), 1 - 1e-12)
    return(list(u = u, v = v))
  }
  if (family == "kimeldorf") {          # Clayton, theta > 0
    if (theta < 1e-8) return(list(u = u, v = w))
    v <- (u^(-theta) * (w^(-theta / (theta + 1)) - 1) + 1)^(-1 / theta)
    v <- pmin(pmax(v, 1e-12), 1 - 1e-12)
    return(list(u = u, v = v))
  }
  stop("unknown copula family: ", family, call. = FALSE)
}

#' Simulate data from a copula RP-BNB process
#'
#' @param n Number of observations.
#' @param beta1,beta2 Named coefficient means; must include "(Intercept)".
#' @param random_1,random_2 Random-coefficient specs (see [simulate_rpbnb()]).
#' @param dispersion Named `c(m1=, m2=)` NB2 dispersions.
#' @param copula An [copula()] object giving the family and native parameter `par`.
#' @param covariates Optional covariate data frame; NULL -> standard-normal columns.
#' @param seed Optional RNG seed.
#' @return list(data, mu, true, settings).
#' @export
simulate_rpbnb_copula <- function(n, beta1, beta2,
                                  random_1 = NULL, random_2 = NULL,
                                  dispersion = c(m1 = 0.5, m2 = 0.5),
                                  copula, covariates = NULL, seed = NULL) {
  stopifnot(inherits(copula, "rpbnb_copula"), !is.null(copula$par),
            "(Intercept)" %in% names(beta1), "(Intercept)" %in% names(beta2))
  spec1 <- parse_rand_spec(random_1); spec2 <- parse_rand_spec(random_2)
  if (!is.null(seed)) set.seed(seed)
  if (is.null(covariates)) covariates <- .sim_default_covariates(beta1, beta2, n)
  .check_sim_covariates(covariates, beta1, beta2, n)

  d1 <- .build_sim_X(beta1, covariates, n); X1 <- d1$X; b1 <- d1$beta
  d2 <- .build_sim_X(beta2, covariates, n); X2 <- d2$X; b2 <- d2$beta
  realize <- function(bv, spec, X) {
    B <- matrix(rep(bv, each = n), n, dimnames = list(NULL, names(bv)))
    for (i in seq_along(spec$names)) {
      reg  <- rand_dist_registry[[spec$dist[i]]]
      base <- if (reg$base == "normal") stats::rnorm(n) else reg$u_to_base(stats::runif(n))
      B[, spec$names[i]] <- reg$coef(bv[[spec$names[i]]], spec$scale[i], base, spec$sign[i])
    }
    B
  }
  B1 <- realize(b1, spec1, X1); B2 <- realize(b2, spec2, X2)
  mu1 <- exp(rowSums(X1 * B1)); mu2 <- exp(rowSums(X2 * B2))
  m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]

  uv <- .rcopula_uv(n, copula$family, copula$par)
  y1 <- stats::qnbinom(uv$u, size = 1 / m1, mu = mu1)
  y2 <- stats::qnbinom(uv$v, size = 1 / m2, mu = mu2)

  td <- copula_tau_and_deriv(copula$family, switch(copula$family,
        frank = copula$par, normal = atanh(copula$par), kimeldorf = log(copula$par)))
  list(
    data = data.frame(y1 = y1, y2 = y2, covariates),
    mu   = data.frame(mu1 = mu1, mu2 = mu2),
    true = list(beta1 = beta1, beta2 = beta2, random_1 = spec1, random_2 = spec2,
                dispersion = dispersion, copula = copula$family,
                theta = copula$par, tau = td$tau),
    settings = list(n = n, seed = seed)
  )
}
