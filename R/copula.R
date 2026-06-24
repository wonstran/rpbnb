#' Specify a copula dependence structure for fit_bnb()
#'
#' @param family One of `"frank"`, `"normal"`, or `"kimeldorf"` (Clayton).
#' @return An object of class `rpbnb_copula`.
#' @export
#' @examples
#' copula("frank")
#' copula("normal")
#' copula("kimeldorf")
copula <- function(family = c("frank", "normal", "kimeldorf")) {
  family <- match.arg(family)
  structure(list(family = family), class = "rpbnb_copula")
}
