#' Numerically stable per-row log-sum-exp
#' @keywords internal
#' @noRd
row_log_sum_exp <- function(M) {
  m <- apply(M, 1, max)
  m + log(rowSums(exp(M - m)))
}

#' Significance stars for a vector of p-values
#' @keywords internal
#' @noRd
signif_stars <- function(p) {
  as.character(stats::symnum(
    p, corr = FALSE, na = FALSE,
    cutpoints = c(0, .001, .01, .05, .1, 1),
    symbols   = c("***", "**", "*", ".", " ")
  ))
}

#' Conditional mean bounded away from both overflow (exp() -> Inf feeding
#' downstream NB functions) and underflow (mu -> 0, a 0/0 NaN source in
#' several analytic gradients). Used by every likelihood family so the
#' objective, gradient, and Hessian for a given fit always see the exact
#' same (implicitly capped) function of the parameters.
#' @keywords internal
#' @noRd
.bound_mu <- function(X, beta) {
  pmin(pmax(as.vector(exp(X %*% beta)), 1e-300), 1e15)
}
