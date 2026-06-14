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
