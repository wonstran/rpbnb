#' Numerically stable per-row log-sum-exp
#' @keywords internal
#' @noRd
row_log_sum_exp <- function(M) {
  m <- apply(M, 1, max)
  m + log(rowSums(exp(M - m)))
}

#' Trace of strictly-improving log-likelihood values
#' @keywords internal
#' @noRd
accepted_ll_trace <- function(ll_vals, tol = 1e-8) {
  ll_vals <- ll_vals[is.finite(ll_vals)]
  if (!length(ll_vals)) {
    return(data.frame(iter = integer(0), logLik = numeric(0)))
  }
  best <- -Inf; iter <- integer(0); val <- numeric(0)
  for (i in seq_along(ll_vals)) {
    if (ll_vals[i] > best + tol) {
      best <- ll_vals[i]
      iter <- c(iter, length(iter) + 1L)
      val  <- c(val, ll_vals[i])
    }
  }
  data.frame(iter = iter, logLik = val)
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
