# Estimator for the copula random-parameter BNB model. Optimizes the copula
# simulated log-likelihood (R/rpbnb_copula_likelihood.R) with BFGS + the
# analytic gradient (bnbr_rp_copula_ll_grad); standard errors from OPG
# (default recommendation) or the numeric Hessian, per control$se_method.
# Internal.

#' @keywords internal
#' @noRd
.fit_rpbnb_copula <- function(formula_1, formula_2, data,
                              random_1, random_2, draws, draw_type,
                              seed, start, control, family) {
  draw_type <- match.arg(draw_type, "halton")
  spec1 <- parse_rand_spec(random_1); spec2 <- parse_rand_spec(random_2)

  prep <- .prepare_bnb_data(formula_1, formula_2, data)
  Y1 <- prep$Y1; Y2 <- prep$Y2; X1 <- prep$X1; X2 <- prep$X2
  k1 <- ncol(X1); k2 <- ncol(X2)

  idx_from_names <- function(who, X) {
    if (!length(who)) return(integer(0))
    miss <- who[!who %in% colnames(X)]
    if (length(miss)) stop("random name(s) not found: ",
                           paste(miss, collapse = ", "), call. = FALSE)
    as.integer(match(who, colnames(X)))
  }
  rand_idx1 <- idx_from_names(spec1$names, X1)
  rand_idx2 <- idx_from_names(spec2$names, X2)
  dist1 <- spec1$dist; sign1 <- spec1$sign
  dist2 <- spec2$dist; sign2 <- spec2$sign
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  XR1 <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2 <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else NULL

  set.seed(seed)
  if ((q1 + q2) > 0) {
    Z  <- halton_uniform(draws, q1 + q2, burn = control$halton_burn)
    Z1 <- if (q1 > 0) Z[, 1:q1, drop = FALSE] else matrix(0, draws, 0)
    Z2 <- if (q2 > 0) Z[, (q1 + 1):(q1 + q2), drop = FALSE] else matrix(0, draws, 0)
  } else {
    Z1 <- matrix(0, 1, 0); Z2 <- matrix(0, 1, 0)
  }

  scale_lab <- function(dist, cols)
    vapply(seq_along(dist),
           function(j) paste0(rand_dist_registry[[dist[j]]]$scale_label, cols[j]),
           character(1))
  par_names <- c(paste0("b1:", colnames(X1)), paste0("b2:", colnames(X2)),
                 if (q1 > 0) scale_lab(dist1, paste0("1:", colnames(X1)[rand_idx1])),
                 if (q2 > 0) scale_lab(dist2, paste0("2:", colnames(X2)[rand_idx2])),
                 "log_m1", "log_m2", "z_theta")
  if (is.null(start))
    start <- c(rep(0, k1 + k2),
               if (q1 > 0) rep(log(0.2), q1), if (q2 > 0) rep(log(0.2), q2),
               log(0.5), log(0.5), 0)
  names(start) <- par_names

  se_method <- if (is.null(control$se_method)) "numeric" else control$se_method

  # Preferred fast path: multithreaded (OpenMP) C++ likelihood, mirroring the
  # Famoye path in fit_rpbnb.R -- n_cores is interpreted as the OpenMP thread
  # count for the C++ core (there is no process-cluster fallback here).
  use_cpp <- rpbnb_copula_cpp_available()
  cpp_threads <- max(1L, as.integer(control$n_cores))

  ll_trace <- numeric(0)
  ll_fun <- function(p) {
    v <- if (use_cpp)
      bnbr_rp_copula_ll_grad_cpp(p, Y1, Y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                                 Z1, Z2, family, dist1, dist2, sign1, sign2,
                                 n_threads = cpp_threads)
    else
      bnbr_rp_copula_ll_grad(p, Y1, Y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                             Z1, Z2, family, dist1, dist2, sign1, sign2)
    ll_trace[[length(ll_trace) + 1L]] <<- as.numeric(v)
    v
  }
  fit <- maxLik::maxLik(logLik = ll_fun, start = start, method = "BFGS",
                        control = list(iterlim = control$iterlim,
                                       reltol = control$reltol,
                                       printLevel = control$print_level))
  par_hat <- stats::coef(fit); names(par_hat) <- par_names
  npar <- length(par_hat)

  if (isTRUE(control$compute_se)) {
    if (identical(se_method, "analytic")) {
      stop("se_method = 'analytic' is not available for copula dependence; use ",
           "'opg' (recommended) or 'numeric'.", call. = FALSE)
    } else if (identical(se_method, "opg")) {
      res <- if (use_cpp)
        bnbr_rp_copula_ll_grad_cpp(par_hat, Y1, Y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2,
                                   Z1, Z2, family, dist1, dist2, sign1, sign2,
                                   want_scores = TRUE, n_threads = cpp_threads)
      else
        bnbr_rp_copula_ll_grad(par_hat, Y1, Y2, X1, X2, XR1, XR2,
                              rand_idx1, rand_idx2, Z1, Z2, family,
                              dist1, dist2, sign1, sign2, want_scores = TRUE)
      vc <- opg_vcov(attr(res, "scores"), par_names)
      se <- sqrt(pmax(diag(vc), 0)); names(se) <- par_names
    } else {  # "numeric"
      H <- numDeriv::hessian(function(p) bnbr_rp_copula_ll(p, Y1, Y2, X1, X2, XR1, XR2,
                             rand_idx1, rand_idx2, Z1, Z2, family, dist1, dist2, sign1, sign2),
                             par_hat,
                             method.args = list(r = control$hess_r, eps = control$hess_eps))
      info <- -(H + t(H)) / 2
      vc <- try(solve(info), silent = TRUE)
      if (inherits(vc, "try-error")) vc <- MASS::ginv(info)
      se <- sqrt(pmax(diag(vc), 0))
    }
  } else {
    vc <- matrix(NA_real_, npar, npar); se <- rep(NA_real_, npar)
  }
  dimnames(vc) <- list(par_names, par_names); names(se) <- par_names

  idx_end <- k1 + k2 + q1 + q2
  m1_hat <- unname(exp(par_hat[idx_end + 1]))
  m2_hat <- unname(exp(par_hat[idx_end + 2]))

  ll_hat <- as.numeric(stats::logLik(fit))
  convergence <- list(converged = fit$code <= 2L, code = fit$code,
                      message = fit$message, iterations = fit$iterations)

  new_rpbnb_fit(
    coef = par_hat, vcov = vc, se = se, logLik = ll_hat,
    nobs = length(Y1), npar = npar, m1 = m1_hat, m2 = m2_hat,
    lambda = NULL, bounds = NULL, mu1 = NULL, mu2 = NULL,
    X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
    rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
    formula_1 = formula_1, formula_2 = formula_2,
    draws = draws, draw_type = draw_type, seed = seed,
    ll_trace = ll_trace, convergence = convergence,
    cop_family = family, call = match.call()
  )
}
