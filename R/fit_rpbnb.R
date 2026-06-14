# Public fit_rpbnb() plus the rpbnb_fit constructor and minimal S3 methods.
# Maximum simulated likelihood estimation of the bivariate random-parameter
# negative binomial model (Famoye/Sarmanov dependence, normal random
# coefficients). Ported from Rcodes/rpbnbr_faymore.R (rpbnbr_bfgs).

#' Construct an rpbnb_fit object
#' @keywords internal
#' @noRd
new_rpbnb_fit <- function(coef, vcov, se, logLik, nobs, npar,
                          m1, m2, lambda, bounds, mu1, mu2, X1, X2, Y1, Y2,
                          rand_idx1, rand_idx2, formula_1, formula_2,
                          draws, draw_type, seed, ll_trace, convergence, call) {
  structure(
    list(coef = coef, vcov = vcov, se = se, logLik = logLik,
         nobs = nobs, npar = npar, m1 = m1, m2 = m2, lambda = lambda,
         bounds = bounds, mu1 = mu1, mu2 = mu2,
         X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
         rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
         formula_1 = formula_1, formula_2 = formula_2,
         draws = draws, draw_type = draw_type, seed = seed,
         ll_trace = ll_trace, convergence = convergence,
         AIC = -2 * logLik + 2 * npar, BIC = -2 * logLik + log(nobs) * npar,
         call = call),
    class = "rpbnb_fit"
  )
}

#' Fit a bivariate random-parameter negative binomial model
#'
#' Maximum simulated likelihood estimation with normal random coefficients and
#' Famoye/Sarmanov dependence. Random coefficients are selected per equation by
#' name via `random_1` / `random_2`.
#'
#' @param formula_1,formula_2 Model formulas for the two count outcomes.
#' @param data A data frame.
#' @param random_1,random_2 Character vectors of coefficient names (matching
#'   `model.matrix` columns) to treat as normal random coefficients in each
#'   equation. NULL means all-fixed for that equation.
#' @param draws Number of simulation draws for the optimization.
#' @param draw_type Quasi-random draw type. Only "halton" is supported in this version.
#' @param seed Random seed for the simulation draws.
#' @param start Optional starting parameter vector.
#' @param control An [rpbnb_control()] object. Estimation uses BFGS;
#'   `control$method` is not currently honored by `fit_rpbnb` (the
#'   simulated-likelihood objective returns an aggregate gradient).
#' @return An object of class `rpbnb_fit`.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 600,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 100, control = rpbnb_control(compute_se = FALSE))
#' coef(fit)
fit_rpbnb <- function(formula_1, formula_2, data,
                      random_1 = NULL, random_2 = NULL,
                      draws = 400, draw_type = "halton",
                      seed = 1234, start = NULL,
                      control = rpbnb_control()) {
  stopifnot(is.data.frame(data))
  draw_type <- match.arg(draw_type, "halton")

  rand_names1 <- random_1
  rand_names2 <- random_2
  n_draws      <- draws
  n_draws_hess <- control$draws_hessian
  halton_burn  <- control$halton_burn
  n_cores      <- control$n_cores
  compute_se   <- control$compute_se
  method       <- "BFGS"
  ml_control   <- list(iterlim = control$iterlim,
                       reltol = control$reltol,
                       printLevel = control$print_level)

  prep <- .prepare_bnb_data(formula_1, formula_2, data)
  Y1 <- prep$Y1; Y2 <- prep$Y2
  X1 <- prep$X1; X2 <- prep$X2

  # random names -> indices; unknown names are an error (not silently dropped)
  idx_from_names <- function(who, X) {
    if (!length(who)) return(integer(0))
    miss <- who[!who %in% colnames(X)]
    if (length(miss)) {
      stop("random name(s) not found: ", paste(miss, collapse = ", "),
           call. = FALSE)
    }
    as.integer(match(who, colnames(X)))
  }
  rand_idx1 <- idx_from_names(rand_names1, X1)
  rand_idx2 <- idx_from_names(rand_names2, X2)

  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  XR1 <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2 <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else NULL

  # Halton normals for the optimization phase. set.seed(seed) drives the
  # Cranley-Patterson rotation in halton_normal, so `seed` selects a reproducible
  # randomized-QMC draw set; the same draws are reused across all optimizer
  # evaluations for a smooth simulated likelihood.
  set.seed(seed)
  if ((q1 + q2) > 0) {
    Z_opt  <- halton_normal(n_draws, q1 + q2, burn = halton_burn)
    Z1_opt <- if (q1 > 0) Z_opt[, 1:q1, drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
    Z2_opt <- if (q2 > 0) Z_opt[, (q1+1):(q1+q2), drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
  } else {
    Z1_opt <- matrix(0, nrow = n_draws, ncol = 0)
    Z2_opt <- matrix(0, nrow = n_draws, ncol = 0)
  }

  par_names <- c(paste0("b1:", colnames(X1)),
                 paste0("b2:", colnames(X2)),
                 if (q1 > 0) paste0("log_sd1:", colnames(X1)[rand_idx1]) else NULL,
                 if (q2 > 0) paste0("log_sd2:", colnames(X2)[rand_idx2]) else NULL,
                 "log_m1", "log_m2", "z_lambda")

  if (is.null(start)) {
    start <- c(rep(0, k1 + k2),
               if (q1 > 0) rep(log(0.2), q1) else NULL,
               if (q2 > 0) rep(log(0.2), q2) else NULL,
               log(0.5), log(0.5), 0)
  }
  names(start) <- par_names

  # Optional cluster (for optimization draws); fall back to sequential.
  use_parallel <- n_cores > 1 && requireNamespace("parallel", quietly = TRUE)
  if (n_cores > 1 && !use_parallel) {
    warning("Package 'parallel' not available; running sequentially.", call. = FALSE)
  }
  cl <- NULL
  if (use_parallel) {
    cl <- parallel::makeCluster(max(1L, as.integer(n_cores)))
    on.exit({ try(parallel::stopCluster(cl), silent = TRUE) }, add = TRUE)
    y1 <- Y1; y2 <- Y2
    # clusterExport must list every transitive callee: workers run in a fresh
    # globalenv without the rpbnb namespace, so all helpers (and their callees)
    # must be exported by name. Adding a new internal call inside any pass
    # function requires adding it here too.
    parallel::clusterExport(cl,
      c("X1", "X2", "XR1", "XR2", "y1", "y2", "Z1_opt", "Z2_opt",
        "c_val", "lambda_bounds_vec", "nb_logpmf_y_mu_r", "dct_dm",
        "dc_dbeta_mat", "d_const"),
      envir = environment())
  }

  # LL trace (one total logLik per function evaluation)
  ll_trace <- numeric(0)
  ll_fun <- function(p) {
    v <- bnbr_rp_ll_and_grad(p, Y1, Y2, X1, X2, XR1, XR2,
                             rand_idx1, rand_idx2, Z1_opt, Z2_opt, cl = cl)
    ll_trace <<- c(ll_trace, as.numeric(v))
    v
  }

  fit <- maxLik::maxLik(logLik = ll_fun, start = start,
                        method = method, control = ml_control)
  par_hat <- stats::coef(fit); names(par_hat) <- par_names

  # --- Frozen lambda-bounds at the optimum (over the optimization draws) ---
  rebuild_bounds <- function(p) {
    beta1 <- p[1:k1]; beta2 <- p[(k1+1):(k1+k2)]
    lg1 <- if (q1 > 0) (k1+k2+1):(k1+k2+q1) else integer(0)
    lg2 <- if (q2 > 0) (k1+k2+q1+1):(k1+k2+q1+q2) else integer(0)
    log_sd1 <- if (q1 > 0) p[lg1] else numeric(0)
    log_sd2 <- if (q2 > 0) p[lg2] else numeric(0)
    idx_end <- k1 + k2 + q1 + q2
    m1 <- exp(p[idx_end+1]); m2 <- exp(p[idx_end+2])
    sd1 <- if (q1 > 0) exp(log_sd1) else numeric(0)
    sd2 <- if (q2 > 0) exp(log_sd2) else numeric(0)
    xb1 <- as.vector(X1 %*% beta1); xb2 <- as.vector(X2 %*% beta2)
    Z1sd <- if (q1 > 0) sweep(Z1_opt, 2, sd1, `*`) else matrix(0, nrow = n_draws, ncol = 0)
    Z2sd <- if (q2 > 0) sweep(Z2_opt, 2, sd2, `*`) else matrix(0, nrow = n_draws, ncol = 0)
    lamLo <- -Inf; lamHi <- Inf
    Rloc <- if (q1 + q2 > 0) nrow(Z1sd) else 1L
    for (r in 1:Rloc) {
      mu1_r <- if (q1 > 0) exp(xb1 + as.vector(XR1 %*% Z1sd[r,])) else exp(xb1)
      mu2_r <- if (q2 > 0) exp(xb2 + as.vector(XR2 %*% Z2sd[r,])) else exp(xb2)
      b <- lambda_bounds_vec(c_val(mu1_r, m1), c_val(mu2_r, m2))
      lamLo <- max(lamLo, b[1]); lamHi <- min(lamHi, b[2])
    }
    c(lamLo, lamHi)
  }
  lam_b   <- rebuild_bounds(par_hat)
  lamLo_h <- as.numeric(lam_b[1]); lamHi_h <- as.numeric(lam_b[2])
  if (!(lamLo_h < lamHi_h)) {
    warning("Frozen bounds invalid at optimum; SEs may be unstable.", call. = FALSE)
  }

  # --- Standard errors via small-draw Hessian with frozen bounds ---
  npar <- length(par_hat)
  if (isTRUE(compute_se)) {
    set.seed(seed + 1L)
    if ((q1 + q2) > 0) {
      Z_h  <- halton_normal(n_draws_hess, q1 + q2,
                            burn = max(50, floor(halton_burn / 3)))
      Z1_h <- if (q1 > 0) Z_h[, 1:q1, drop = FALSE] else matrix(0, nrow = n_draws_hess, ncol = 0)
      Z2_h <- if (q2 > 0) Z_h[, (q1+1):(q1+q2), drop = FALSE] else matrix(0, nrow = n_draws_hess, ncol = 0)
    } else {
      Z1_h <- matrix(0, nrow = n_draws_hess, ncol = 0)
      Z2_h <- matrix(0, nrow = n_draws_hess, ncol = 0)
    }

    cl_h <- NULL
    if (use_parallel) {
      cl_h <- parallel::makeCluster(max(1L, as.integer(min(n_cores, 4))))
      on.exit({ try(parallel::stopCluster(cl_h), silent = TRUE) }, add = TRUE)
      y1 <- Y1; y2 <- Y2
      parallel::clusterExport(cl_h,
        c("X1", "X2", "XR1", "XR2", "y1", "y2", "Z1_h", "Z2_h",
          "c_val", "lambda_bounds_vec", "nb_logpmf_y_mu_r", "dct_dm",
          "dc_dbeta_mat", "d_const"),
        envir = environment())
    }

    ll_fb <- function(p) bnbr_rp_ll_fixed_bounds(p, Y1, Y2, X1, X2, XR1, XR2,
                                                 rand_idx1, rand_idx2, Z1_h, Z2_h,
                                                 lamLo_h, lamHi_h, cl = cl_h)
    H <- numDeriv::hessian(ll_fb, par_hat,
                           method.args = list(r = control$hess_r, eps = control$hess_eps))
    info <- -H; info <- (info + t(info)) / 2
    if (any(!is.finite(info))) {
      warning("Non-finite information; retrying Hessian with larger step.",
              call. = FALSE)
      H <- numDeriv::hessian(ll_fb, par_hat,
                             method.args = list(r = max(6, control$hess_r + 2),
                                                eps = max(1e-4, 5 * control$hess_eps)))
      info <- -H; info <- (info + t(info)) / 2
    }
    ok <- try(eigen(info, symmetric = TRUE, only.values = TRUE), silent = TRUE)
    if (inherits(ok, "try-error") || any(!is.finite(ok$values)) || min(ok$values) <= 0) {
      ridge <- if (inherits(ok, "try-error") || any(!is.finite(ok$values))) {
        1e-2
      } else {
        1e-8 - min(ok$values)
      }
      info <- info + diag(ridge, nrow(info))
    }
    vc <- try(solve(info), silent = TRUE)
    if (inherits(vc, "try-error")) vc <- MASS::ginv(info)
    se <- sqrt(pmax(diag(vc), 0))
    dimnames(vc) <- list(par_names, par_names)
    names(se) <- par_names
  } else {
    # Keep vcov() type-consistent with bnb_fit (an NA-filled matrix, not NULL)
    # when SEs are skipped, so downstream code can rely on a matrix shape.
    vc <- matrix(NA_real_, npar, npar, dimnames = list(par_names, par_names))
    se <- rep(NA_real_, npar); names(se) <- par_names
  }

  # --- Natural-scale fields ---
  idx_end    <- k1 + k2 + q1 + q2
  m1_hat     <- unname(exp(par_hat[idx_end + 1]))
  m2_hat     <- unname(exp(par_hat[idx_end + 2]))
  z_hat      <- unname(par_hat[idx_end + 3])
  eps        <- 1e-6; sig <- plogis(z_hat)
  lambda_hat <- lamLo_h + (lamHi_h - lamLo_h) * (eps + (1 - 2*eps) * sig)

  # --- Fitted draw-averaged unconditional means ---
  beta1_hat <- par_hat[1:k1]; beta2_hat <- par_hat[(k1+1):(k1+k2)]
  xb1 <- as.vector(X1 %*% beta1_hat); xb2 <- as.vector(X2 %*% beta2_hat)
  if (q1 > 0) {
    sd1  <- exp(par_hat[(k1+k2+1):(k1+k2+q1)])
    Z1sd <- sweep(Z1_opt, 2, sd1, `*`)
    mu1_mat <- vapply(seq_len(n_draws),
                      function(r) exp(xb1 + as.vector(XR1 %*% Z1sd[r, ])),
                      numeric(length(Y1)))
    mu1_hat <- rowMeans(mu1_mat)
  } else {
    mu1_hat <- exp(xb1)
  }
  if (q2 > 0) {
    sd2  <- exp(par_hat[(k1+k2+q1+1):(k1+k2+q1+q2)])
    Z2sd <- sweep(Z2_opt, 2, sd2, `*`)
    mu2_mat <- vapply(seq_len(n_draws),
                      function(r) exp(xb2 + as.vector(XR2 %*% Z2sd[r, ])),
                      numeric(length(Y2)))
    mu2_hat <- rowMeans(mu2_mat)
  } else {
    mu2_hat <- exp(xb2)
  }

  ll_hat <- as.numeric(stats::logLik(fit))
  convergence <- list(converged = fit$code <= 2L, code = fit$code,
                      message = fit$message, iterations = fit$iterations)

  new_rpbnb_fit(
    coef = par_hat, vcov = vc, se = se, logLik = ll_hat,
    nobs = length(Y1), npar = npar,
    m1 = m1_hat, m2 = m2_hat, lambda = lambda_hat,
    bounds = c(lamLo_h, lamHi_h), mu1 = mu1_hat, mu2 = mu2_hat,
    X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
    rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
    formula_1 = formula_1, formula_2 = formula_2,
    draws = n_draws, draw_type = draw_type, seed = seed,
    ll_trace = ll_trace, convergence = convergence, call = match.call())
}
