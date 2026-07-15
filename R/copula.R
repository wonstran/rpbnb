#' Specify a copula dependence structure
#'
#' Pass the result as the `dependence` argument to [fit_bnb()] or
#' [fit_rpbnb()] to estimate a discrete-copula bivariate NB model instead of
#' the default Famoye/Sarmanov dependence, or as the `copula` argument to
#' [simulate_rpbnb_copula()] to simulate from one. The joint pmf is built from
#' the two NB2 marginal CDFs and the chosen copula CDF (rectangle
#' differencing); see the package vignette for the "Copula dependence"
#' section.
#'
#' @param family One of `"frank"`, `"normal"` (Gaussian), or `"kimeldorf"`
#'   (Clayton).
#' @param par Optional native dependence parameter (Frank's theta, the
#'   Gaussian rho, or Clayton's theta) used by [simulate_rpbnb_copula()] to
#'   generate data. Ignored by [fit_bnb()] and [fit_rpbnb()], which estimate
#'   the parameter from the data.
#' @return An object of class `rpbnb_copula`.
#' @export
#' @examples
#' copula("frank")
#' copula("normal", par = 0.3)
#' copula("kimeldorf")
copula <- function(family = c("frank", "normal", "kimeldorf"), par = NULL) {
  family <- match.arg(family)
  structure(list(family = family, par = par), class = "rpbnb_copula")
}
