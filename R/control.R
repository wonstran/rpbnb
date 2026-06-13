#' Control parameters for rpbnb estimators
#'
#' @param method Optimizer passed to [maxLik::maxLik()]. One of "BFGS", "NR",
#'   "BHHH", "NM".
#' @param iterlim Maximum optimizer iterations.
#' @param reltol Relative convergence tolerance.
#' @param print_level Verbosity passed to the optimizer (0 = silent).
#' @param draws_hessian Number of simulation draws used for the random-parameter
#'   Hessian (smaller than the optimization draws for speed). Ignored by [fit_bnb()].
#' @param halton_burn,halton_skip Halton sequence burn-in and skip.
#' @param n_cores Worker processes for the optional cluster path (1 = sequential).
#' @param compute_se If FALSE, skip the Hessian and standard errors.
#' @param hess_eps,hess_r Step and Richardson order for [numDeriv::hessian()].
#'
#' @return An object of class `rpbnb_control` (a named list).
#' @export
#' @examples
#' rpbnb_control(method = "BFGS", iterlim = 200)
rpbnb_control <- function(method = c("BFGS", "NR", "BHHH", "NM"),
                          iterlim = 300L,
                          reltol = 1e-8,
                          print_level = 0L,
                          draws_hessian = 100L,
                          halton_burn = 300L,
                          halton_skip = 100L,
                          n_cores = 1L,
                          compute_se = TRUE,
                          hess_eps = 1e-5,
                          hess_r = 4L) {
  if (length(method) > 1) method <- method[1]
  allowed <- c("BFGS", "NR", "BHHH", "NM")
  if (!method %in% allowed) {
    stop("`method` must be one of: ", paste(allowed, collapse = ", "), call. = FALSE)
  }
  if (!is.numeric(n_cores) || n_cores < 1) {
    stop("`n_cores` must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(draws_hessian) || draws_hessian < 1) {
    stop("`draws_hessian` must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(iterlim) || iterlim < 1) {
    stop("`iterlim` must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(reltol) || reltol <= 0) {
    stop("`reltol` must be a positive number.", call. = FALSE)
  }
  structure(
    list(method = method, iterlim = as.integer(iterlim), reltol = reltol,
         print_level = as.integer(print_level),
         draws_hessian = as.integer(draws_hessian),
         halton_burn = as.integer(halton_burn),
         halton_skip = as.integer(halton_skip),
         n_cores = as.integer(n_cores), compute_se = isTRUE(compute_se),
         hess_eps = hess_eps, hess_r = as.integer(hess_r)),
    class = "rpbnb_control"
  )
}
