#' Specify a copula dependence structure for fit_bnb()
#'
#' @param family One of `"frank"`, `"normal"`, or `"kimeldorf"` (Clayton).
#' @param par Optional dependence parameter value for simulation (not used
#'   by [fit_bnb()], which estimates it).
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
