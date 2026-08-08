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
                          draws, draw_type, seed, ll_trace, convergence,
                          cop_family = NULL, call, hessian_diag = NULL,
                          rp_meta = NULL, predict_meta = NULL,
                          poisson_1 = FALSE, poisson_2 = FALSE) {
  structure(
    list(coef = coef, vcov = vcov, se = se, logLik = logLik,
         nobs = nobs, npar = npar, m1 = m1, m2 = m2, lambda = lambda,
         bounds = bounds, mu1 = mu1, mu2 = mu2,
         X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
         rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
         formula_1 = formula_1, formula_2 = formula_2,
         draws = draws, draw_type = draw_type, seed = seed,
         ll_trace = ll_trace, convergence = convergence,
         cop_family = cop_family,
         AIC = -2 * logLik + 2 * npar, BIC = -2 * logLik + log(nobs) * npar,
         call = call, hessian_diag = hessian_diag,
         # Random-coefficient distributions, signs, and the optimization draws:
         # everything predict() needs to reproduce the integrated (population)
         # mean E[exp(x'beta)] for any supported distribution on new data.
         rp_meta = rp_meta,
         # terms/xlevels/contrasts/offsets for column-stable predict() designs.
         predict_meta = predict_meta,
         # Poisson-limit (m = 0) restriction flags per margin, so residuals /
         # diagnostics use the exact Poisson CDF/variance for a restricted margin.
         poisson_1 = poisson_1, poisson_2 = poisson_2),
    class = "rpbnb_fit"
  )
}

#' Fit a bivariate random-parameter negative binomial model
#'
#' Maximum simulated likelihood estimation with normal random coefficients and
#' Famoye/Sarmanov dependence. Random coefficients are selected per equation by
#' name via `random_1` / `random_2`.
#'
#' @param formula_1,formula_2 Model formulas for the two count outcomes. An
#'   equation-specific `offset()` term (e.g. `y ~ x + offset(log(exposure))`) is
#'   supported: the offset enters that margin's linear predictor additively
#'   (integrated mean `E[exp(x'beta + offset)]`) during estimation and is carried
#'   through the stored fitted means and `predict()`.
#' @param data A data frame.
#' @param random_1,random_2 Random coefficients per equation. Either a character
#'   vector of `model.matrix` column names (all Normal), or a named list whose
#'   values are a distribution name (`"normal"`, `"lognormal"`, `"uniform"`,
#'   `"triangular"`) or a list `list(dist = ..., sign = ...)` (`sign` is -1/1 and
#'   lognormal-only). NULL means all-fixed for that equation.
#' @param draws Number of simulation draws for the optimization.
#' @param draw_type Quasi-random draw type. Only "halton" is supported in this version.
#' @param seed Random seed for the simulation draws.
#' @param start Optional starting parameter vector.
#' @param control An [rpbnb_control()] object. Estimation uses BFGS, the only
#'   optimizer `control$method` accepts.
#' @param dependence Dependence structure: "famoye" (default; Famoye/Sarmanov)
#'   or an [copula()] object for copula dependence (Frank / Gaussian /
#'   Clayton). Both paths use the multithreaded (OpenMP) C++ simulated
#'   likelihood; the copula path is more numerically expensive per evaluation
#'   (discrete-copula pmf + per-draw NB CDF corners), so fits typically take
#'   noticeably longer than the Famoye path at comparable `draws`/`n`. Random
#'   coefficients on 0/1 dummy regressors are weakly identified under the
#'   copula path (NB dispersion trades off against the random-coefficient
#'   scale); prefer random coefficients on continuous regressors.
#' @param poisson_1,poisson_2 Fit the corresponding margin at its exact Poisson
#'   limit (NB2 dispersion `m = 0`): the margin's `log_m` is held fixed, so it is
#'   not a free parameter and the fit is a properly nested restriction of the NB
#'   model. Pair with [lr_test()] (`boundary = TRUE`) to test a margin for
#'   overdispersion (`H0: m = 0`). The simulated likelihood uses the exact `m = 0`
#'   branch -- the per-draw margin log-pmf is `dpois` and its Famoye dependence
#'   constant is `exp(-d*mu)` -- so it is accurate at any fitted mean, not a
#'   fixed-dispersion approximation. Supported with both Famoye/Sarmanov and
#'   [copula()] dependence (the copula path uses the same exact m = 0 branch).
#' @param .fixed Internal. A named numeric vector of parameters (in the
#'   optimization/log-scale parameterization) to pin at the supplied values and
#'   hold fixed during estimation. Used by [rpbnb_boundary_tests()] to construct
#'   scale-zero (SD boundary) restricted fits; not intended for direct use.
#' @param .opt_draws Internal. A list `list(Z1, Z2)` of uniform Halton draw
#'   matrices to use verbatim instead of generating them, so a restricted refit
#'   reuses a full fit's draws (common random numbers). Used by
#'   [rpbnb_boundary_tests()]; not intended for direct use.
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
#'
#' \donttest{
#' # Copula dependence instead of Famoye/Sarmanov (slower; fewer draws here
#' # for a quick example -- use more in practice)
#' sim_cop <- simulate_rpbnb_copula(n = 600,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.3)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5),
#'   copula = copula("normal", par = 0.5), seed = 1)
#' fit_cop <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim_cop$data, random_1 = "x1",
#'                      dependence = copula("normal"), draws = 100,
#'                      control = rpbnb_control(compute_se = FALSE))
#' tanh(coef(fit_cop)[["z_theta"]])  # estimated copula rho
#' }
fit_rpbnb <- function(formula_1, formula_2, data,
                      random_1 = NULL, random_2 = NULL,
                      draws = 400, draw_type = "halton",
                      seed = 1234, start = NULL,
                      control = rpbnb_control(),
                      dependence = "famoye",
                      poisson_1 = FALSE, poisson_2 = FALSE,
                      .fixed = NULL, .opt_draws = NULL) {
  stopifnot(is.data.frame(data))
  .chk_poisson_flag(poisson_1, "poisson_1")
  .chk_poisson_flag(poisson_2, "poisson_2")

  if (inherits(dependence, "rpbnb_copula")) {
    return(.fit_rpbnb_copula(formula_1, formula_2, data, random_1, random_2,
                             draws, draw_type, seed, start, control,
                             family = dependence$family,
                             poisson_1 = poisson_1, poisson_2 = poisson_2,
                             .fixed = .fixed, .opt_draws = .opt_draws))
  }
  if (!identical(dependence, "famoye")) {
    stop("`dependence` must be \"famoye\" or a copula() object; got ",
         if (is.character(dependence)) dQuote(dependence) else class(dependence)[1],
         ".", call. = FALSE)
  }

  draw_type <- match.arg(draw_type, "halton")

  spec1 <- parse_rand_spec(random_1)
  spec2 <- parse_rand_spec(random_2)
  n_draws      <- draws
  halton_burn  <- control$halton_burn
  n_cores      <- control$n_cores
  compute_se   <- control$compute_se
  se_method    <- if (is.null(control$se_method)) "numeric" else control$se_method
  method       <- "BFGS"
  ml_control   <- list(iterlim = control$iterlim,
                       reltol = control$reltol,
                       printLevel = control$print_level)

  prep <- .prepare_bnb_data(formula_1, formula_2, data)
  Y1 <- prep$Y1; Y2 <- prep$Y2
  X1 <- prep$X1; X2 <- prep$X2
  off1 <- prep$off1; off2 <- prep$off2   # equation-specific offsets (0 if none)

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
  rand_idx1 <- idx_from_names(spec1$names, X1)
  rand_idx2 <- idx_from_names(spec2$names, X2)
  dist1 <- spec1$dist; sign1 <- spec1$sign
  dist2 <- spec2$dist; sign2 <- spec2$sign

  k1 <- ncol(X1); k2 <- ncol(X2)
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  XR1 <- if (q1 > 0) X1[, rand_idx1, drop = FALSE] else NULL
  XR2 <- if (q2 > 0) X2[, rand_idx2, drop = FALSE] else NULL

  # Halton uniform draws for the optimization phase. set.seed(seed) drives the
  # Cranley-Patterson rotation in halton_uniform, so `seed` selects a reproducible
  # randomized-QMC draw set; the same draws are reused across all optimizer
  # evaluations for a smooth simulated likelihood. The likelihood applies each
  # column's u_to_base transform (via rand_realize), so we pass raw uniforms here.
  if (!is.null(.opt_draws)) {
    # Internal common-random-number path (rpbnb_boundary_tests): reuse a full
    # fit's stored uniform draw matrices verbatim, so a restricted refit is
    # compared to the full fit on identical draws. Validate the shapes against
    # the parsed spec so a caller cannot silently misalign columns.
    Z1_opt <- .opt_draws$Z1; Z2_opt <- .opt_draws$Z2
    if (!is.matrix(Z1_opt) || !is.matrix(Z2_opt) ||
        nrow(Z1_opt) != n_draws || nrow(Z2_opt) != n_draws ||
        ncol(Z1_opt) != q1 || ncol(Z2_opt) != q2) {
      stop("`.opt_draws` must supply Z1 (", n_draws, "x", q1, ") and Z2 (",
           n_draws, "x", q2, ") uniform draw matrices.", call. = FALSE)
    }
  } else {
    set.seed(seed)
    if ((q1 + q2) > 0) {
      Z_opt  <- halton_uniform(n_draws, q1 + q2, burn = halton_burn)
      Z1_opt <- if (q1 > 0) Z_opt[, 1:q1, drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
      Z2_opt <- if (q2 > 0) Z_opt[, (q1+1):(q1+q2), drop = FALSE] else matrix(0, nrow = n_draws, ncol = 0)
    } else {
      Z1_opt <- matrix(0, nrow = n_draws, ncol = 0)
      Z2_opt <- matrix(0, nrow = n_draws, ncol = 0)
    }
  }

  scale_lab <- function(dist, cols) {
    vapply(seq_along(dist),
           function(j) paste0(rand_dist_registry[[dist[j]]]$scale_label, cols[j]),
           character(1))
  }
  par_names <- c(paste0("b1:", colnames(X1)),
                 paste0("b2:", colnames(X2)),
                 if (q1 > 0) paste0(scale_lab(dist1, paste0("1:", colnames(X1)[rand_idx1]))) else NULL,
                 if (q2 > 0) paste0(scale_lab(dist2, paste0("2:", colnames(X2)[rand_idx2]))) else NULL,
                 "log_m1", "log_m2", "z_lambda")

  # Zero mean-coefficient starts (see fit_bnb_famoye: marginal glm.nb starts
  # converged worse under the frozen-bounds famoye gradient). User starts are
  # resolved (positional or named, reordered) against par_names.
  start <- .resolve_start(
    start,
    c(rep(0, k1 + k2),
      if (q1 > 0) rep(log(0.2), q1) else NULL,
      if (q2 > 0) rep(log(0.2), q2) else NULL,
      log(0.5), log(0.5), 0),
    par_names, "start")

  # Poisson-limit margins: pin log_m at log(POISSON_M) and hold it fixed during
  # optimization (maxLik `fixed=`), so it is not an estimated parameter and the
  # fit nests inside the NB model for an overdispersion LR test.
  fixed_names <- c(if (isTRUE(poisson_1)) "log_m1", if (isTRUE(poisson_2)) "log_m2")
  if (length(fixed_names)) start[fixed_names] <- log(POISSON_M)

  # Internal boundary-test path: pin additional parameters (e.g. a random-slope
  # log-scale for an SD boundary LR test, whose draw column rpbnb_boundary_tests()
  # zeroes so the scale is inert) at supplied values and hold them fixed, so the
  # restricted fit nests inside the full model on a named-parameter restriction.
  # `.fixed` is a named numeric vector in the optimization (log-scale)
  # parameterization.
  if (!is.null(.fixed)) {
    if (is.null(names(.fixed)) || any(!nzchar(names(.fixed)))) {
      stop("`.fixed` must be a fully named numeric vector.", call. = FALSE)
    }
    unknown <- setdiff(names(.fixed), par_names)
    if (length(unknown)) {
      stop("`.fixed` names not in the model: ",
           paste(unknown, collapse = ", "), ".", call. = FALSE)
    }
    start[names(.fixed)] <- .fixed
    fixed_names <- union(fixed_names, names(.fixed))
  }
  free <- !(par_names %in% fixed_names)

  # Preferred fast path: multithreaded (OpenMP) C++ likelihood. When available
  # it supersedes the process-based cluster entirely -- the draw loop is
  # parallelised across threads inside C++ with shared memory (no per-call
  # serialization), so n_cores is interpreted as the OpenMP thread count.
  use_cpp <- rpbnb_cpp_available()
  cpp_threads <- max(1L, as.integer(n_cores))

  # Optional cluster (R fallback for the optimization draws) -- only when the
  # C++ core is not compiled. Falls back to sequential if 'parallel' is missing.
  use_parallel <- !use_cpp && n_cores > 1 && requireNamespace("parallel", quietly = TRUE)
  if (!use_cpp && n_cores > 1 && !requireNamespace("parallel", quietly = TRUE)) {
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
  # The admissible interval is FROZEN at the starting values for the whole
  # optimization, and the objective is handed that fixed interval.
  #
  # This is what makes the analytic gradient the actual derivative of the
  # function being optimized. The kernels differentiate lam through lamLo/lamHi
  # treating them as constants; recomputing the interval from the current
  # parameters on every call would silently add d(lamLo)/d(par) and
  # d(lamHi)/d(par) terms to the objective that the gradient omits. Those terms
  # are zero only when the bound happens not to depend on the parameters (normal
  # or lognormal coefficients loaded in both margins, where it is the constant
  # [-1, 1]); with a single varying margin or a uniform/triangular coefficient
  # they are not, and the discrepancy was measured at 2.05 on a one-margin
  # uniform fixture -- every coordinate except z_lambda, which is the only one
  # the bounds do not involve.
  #
  # Freezing matches what the TMB engine already does, so the two engines now
  # agree on semantics as well as on the interval, and the post-fit
  # admissibility check below is the same guard TMB applies.
  lam_frozen <- .rp_support_bounds(start, X1, X2, rand_idx1, rand_idx2,
                                   dist1, dist2, sign1, sign2,
                                   pois1 = poisson_1, pois2 = poisson_2,
                                   off1 = off1, off2 = off2)
  if (!(lam_frozen[["lower"]] < lam_frozen[["upper"]] &&
        all(is.finite(lam_frozen)))) {
    stop("Invalid Famoye lambda bounds at the starting values: [",
         signif(lam_frozen[["lower"]], 6), ", ",
         signif(lam_frozen[["upper"]], 6), "].", call. = FALSE)
  }
  # A poisson_* margin uses the exact m = 0 (Poisson) branch; the pinned log_m is
  # only a display placeholder, so the fit nests inside the NB model for the
  # overdispersion LR test while the likelihood/CDF stay exactly Poisson.
  ll_fun <- function(p) {
    v <- if (use_cpp)
      bnbr_rp_ll_and_grad_cpp(p, Y1, Y2, X1, X2, XR1, XR2,
                              rand_idx1, rand_idx2, Z1_opt, Z2_opt,
                              dist1, dist2, sign1, sign2, n_threads = cpp_threads,
                              pois1 = poisson_1, pois2 = poisson_2,
                              off1 = off1, off2 = off2,
                              lam_bounds = lam_frozen)
    else
      bnbr_rp_ll_and_grad(p, Y1, Y2, X1, X2, XR1, XR2,
                          rand_idx1, rand_idx2, Z1_opt, Z2_opt,
                          dist1, dist2, sign1, sign2, cl = cl,
                          pois1 = poisson_1, pois2 = poisson_2,
                          off1 = off1, off2 = off2,
                          lam_bounds = lam_frozen)
    ll_trace <<- c(ll_trace, as.numeric(v))
    v
  }

  fit <- maxLik::maxLik(logLik = ll_fun, start = start,
                        method = method, control = ml_control,
                        fixed = if (length(fixed_names)) fixed_names else NULL)
  par_hat <- stats::coef(fit)
  # maxLik returns a NULL estimate when the objective is non-finite at the start
  # (code 100, "Initial value out of range"): a random slope on a large-scale
  # covariate can make the simulated likelihood overflow/underflow before the
  # optimizer takes a step. Fail with an actionable message instead of the
  # opaque "attempt to set an attribute on NULL" from naming a NULL vector.
  if (is.null(par_hat)) {
    stop("RP-BNB optimization failed before it could start (maxLik code ",
         fit$code, ": ", fit$message, ").\n",
         "The simulated log-likelihood was non-finite at the starting values. ",
         "This usually means a random coefficient sits on a large-scale ",
         "covariate; try centering/scaling it or supplying `start`.",
         call. = FALSE)
  }
  names(par_hat) <- par_names

  # --- Frozen lambda-bounds at the optimum (over the optimization draws) ---
  rebuild_bounds <- function(p) {
    beta1 <- p[1:k1]; beta2 <- p[(k1+1):(k1+k2)]
    idx_end <- k1 + k2 + q1 + q2
    # A Poisson-restricted margin uses m = 0 (c = exp(-d*mu)) so the frozen bounds
    # match the objective's exact Poisson margin.
    m1 <- if (isTRUE(poisson_1)) 0 else exp(p[idx_end+1])
    m2 <- if (isTRUE(poisson_2)) 0 else exp(p[idx_end+2])
    sd1 <- if (q1 > 0) exp(p[(k1+k2+1):(k1+k2+q1)]) else numeric(0)
    sd2 <- if (q2 > 0) exp(p[(k1+k2+q1+1):(k1+k2+q1+q2)]) else numeric(0)
    # The support bound, matching the objective. This used to reduce
    # lambda_bounds_vec() over the optimization draws, which made the
    # reconstructed interval -- and so the frozen-bounds Hessian built from it --
    # a function of `draws` rather than of the model.
    sb <- famoye_support_bounds(
      X1, X2, off1, off2, rand_idx1, rand_idx2,
      dist1, dist2, sign1, sign2, beta1, beta2,
      if (q1 > 0) exp(pmax(p[(k1+k2+1):(k1+k2+q1)], -20)) else numeric(0),
      if (q2 > 0) exp(pmax(p[(k1+k2+q1+1):(k1+k2+q1+q2)], -20)) else numeric(0),
      m1, m2
    )
    c(sb[["lower"]], sb[["upper"]])
  }
  lam_b   <- rebuild_bounds(par_hat)
  lamLo_h <- as.numeric(lam_b[1]); lamHi_h <- as.numeric(lam_b[2])
  if (!(lamLo_h < lamHi_h)) {
    warning("Frozen bounds invalid at optimum; SEs may be unstable.", call. = FALSE)
  }

  # Admissibility at the OPTIMUM, the same guard the TMB engine applies.
  #
  # The interval handed to the objective was frozen at the starting values so
  # that the analytic gradient is genuinely the derivative of the optimized
  # function. The price is the one TMB already pays: where the bound depends on
  # the parameters (a single varying margin, or uniform/triangular
  # coefficients), the fit can move somewhere the frozen box is wider than the
  # admissible one, leaving the joint pmf negative in the count tails while the
  # objective stays finite at the observed cells. Detect and report; the
  # optimized objective cannot be repaired after the fact.
  lambda_admissible <- NA
  lam_hat_chk <- lamLo_h + (lamHi_h - lamLo_h) *
    (1e-6 + (1 - 2e-6) * stats::plogis(par_hat[[length(par_hat)]]))
  if (is.finite(lam_hat_chk) && lamLo_h < lamHi_h) {
    lambda_admissible <- isTRUE(lam_hat_chk >= lamLo_h && lam_hat_chk <= lamHi_h)
  }
  if (isTRUE(!lambda_admissible)) {
    warning(
      "The fitted Famoye lambda (", signif(lam_hat_chk, 6), ") lies outside ",
      "the admissible interval recomputed at the fitted parameters [",
      signif(lamLo_h, 6), ", ", signif(lamHi_h, 6), "]. The interval used by ",
      "the likelihood was frozen at the starting values ([",
      signif(lam_frozen[["lower"]], 6), ", ",
      signif(lam_frozen[["upper"]], 6), "]), so the optimizer was free to ",
      "leave the valid region: the joint pmf is negative somewhere in the ",
      "count tails and this fit should not be interpreted. Refit from ",
      "starting values closer to the optimum.", call. = FALSE
    )
  }

  # --- Standard errors ---
  # npar (df) counts only free parameters; a pinned Poisson dispersion is fixed.
  # The vcov/se objects stay full-length (NA for the pinned coordinate) so they
  # align with the full coefficient vector downstream.
  npar       <- length(par_hat) - length(fixed_names)
  npar_total <- length(par_hat)
  use_opg      <- isTRUE(compute_se) && use_cpp && identical(se_method, "opg")
  use_analytic <- isTRUE(compute_se) && identical(se_method, "analytic")
  if (use_opg) {
    # Analytic BHHH / outer-product-of-gradients covariance from the
    # per-observation scores, evaluated at the optimum with the SAME simulation
    # draws used for estimation. One score pass -- no numeric Hessian, so no
    # finite-difference digamma singularities.
    S <- bnbr_rp_scores_cpp(par_hat, Y1, Y2, X1, X2, XR1, XR2,
                            rand_idx1, rand_idx2, Z1_opt, Z2_opt,
                            dist1, dist2, sign1, sign2, n_threads = cpp_threads,
                            pois1 = poisson_1, pois2 = poisson_2,
                            off1 = off1, off2 = off2)
    inv <- .free_index_vcov(crossprod(S), par_names, free, label = "OPG (BHHH)")
    vc <- inv$vcov; se <- inv$se; hdiag <- inv$diag
  } else if (use_analytic) {
    # Closed-form observed-information Hessian (Famoye 2010 per-draw second
    # derivatives + Louis mixture formula), at the optimum with the SAME draws
    # and the frozen lambda-bounds used for the objective. Exact, and far faster
    # than the numeric Hessian for larger models.
    H <- bnbr_rp_hessian(par_hat, Y1, Y2, X1, X2, XR1, XR2,
                         rand_idx1, rand_idx2, Z1_opt, Z2_opt,
                         dist1, dist2, sign1, sign2,
                         lamLo = lamLo_h, lamHi = lamHi_h,
                         pois1 = poisson_1, pois2 = poisson_2,
                         off1 = off1, off2 = off2)
    info <- -H
    inv <- .free_index_vcov(info, par_names, free, label = "RP-BNB (analytic Hessian)")
    vc <- inv$vcov; se <- inv$se; hdiag <- inv$diag
  } else if (isTRUE(compute_se)) {
    # Same-draw curvature: differentiate the frozen-bounds objective on the SAME
    # optimization draws (Z1_opt/Z2_opt) that produced the estimate, so the
    # numeric Hessian is the curvature of the actual optimized simulated
    # likelihood -- matching the analytic and OPG paths. (Previously this path
    # resimulated a smaller seed+1 Halton set, so `draws_hessian` no longer
    # affects the Famoye numeric SEs.)
    cl_h <- NULL
    if (use_parallel) {
      cl_h <- parallel::makeCluster(max(1L, as.integer(min(n_cores, 4))))
      on.exit({ try(parallel::stopCluster(cl_h), silent = TRUE) }, add = TRUE)
      y1 <- Y1; y2 <- Y2
      parallel::clusterExport(cl_h,
        c("X1", "X2", "XR1", "XR2", "y1", "y2", "Z1_opt", "Z2_opt",
          "c_val", "lambda_bounds_vec", "nb_logpmf_y_mu_r", "dct_dm",
          "dc_dbeta_mat", "d_const"),
        envir = environment())
    }

    ll_fb <- if (use_cpp)
      function(p) bnbr_rp_ll_fixed_bounds_cpp(p, Y1, Y2, X1, X2, XR1, XR2,
                                              rand_idx1, rand_idx2, Z1_opt, Z2_opt,
                                              lamLo_h, lamHi_h,
                                              dist1, dist2, sign1, sign2,
                                              n_threads = cpp_threads,
                                              pois1 = poisson_1, pois2 = poisson_2,
                                              off1 = off1, off2 = off2)
    else
      function(p) bnbr_rp_ll_fixed_bounds(p, Y1, Y2, X1, X2, XR1, XR2,
                                          rand_idx1, rand_idx2, Z1_opt, Z2_opt,
                                          lamLo_h, lamHi_h,
                                          dist1, dist2, sign1, sign2,
                                          cl = cl_h,
                                          pois1 = poisson_1, pois2 = poisson_2,
                                          off1 = off1, off2 = off2)
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
    inv <- .free_index_vcov(info, par_names, free, label = "RP-BNB (numeric Hessian)")
    vc <- inv$vcov; se <- inv$se; hdiag <- inv$diag
  } else {
    # Keep vcov() type-consistent with bnb_fit (an NA-filled matrix, not NULL)
    # when SEs are skipped, so downstream code can rely on a matrix shape.
    vc <- matrix(NA_real_, npar_total, npar_total, dimnames = list(par_names, par_names))
    se <- rep(NA_real_, npar_total); names(se) <- par_names
    hdiag <- NULL
  }

  # --- Natural-scale fields ---
  # A Poisson-restricted margin is exactly m = 0; its pinned log_m is only the
  # POISSON_M display placeholder, so store 0 rather than exp(log(1e-6)) = 1e-6
  # (matches fit_bnb() and keeps the natural-scale contract consistent).
  idx_end    <- k1 + k2 + q1 + q2
  m1_hat     <- if (isTRUE(poisson_1)) 0 else unname(exp(par_hat[idx_end + 1]))
  m2_hat     <- if (isTRUE(poisson_2)) 0 else unname(exp(par_hat[idx_end + 2]))
  z_hat      <- unname(par_hat[idx_end + 3])
  eps        <- 1e-6; sig <- plogis(z_hat)
  lambda_hat <- lamLo_h + (lamHi_h - lamLo_h) * (eps + (1 - 2*eps) * sig)

  # --- Fitted draw-averaged unconditional means ---
  beta1_hat <- par_hat[1:k1]; beta2_hat <- par_hat[(k1+1):(k1+k2)]
  xb1 <- as.vector(X1 %*% beta1_hat) + off1; xb2 <- as.vector(X2 %*% beta2_hat) + off2
  if (q1 > 0) {
    sd1  <- exp(par_hat[(k1+k2+1):(k1+k2+q1)])
    dev1 <- rand_realize(Z1_opt, dist1, sign1, beta1_hat[rand_idx1], sd1)$dev
    mu1_mat <- vapply(seq_len(n_draws),
                      function(r) pmin(exp(xb1 + as.vector(XR1 %*% dev1[r, ])), 1e15),
                      numeric(length(Y1)))
    mu1_hat <- rowMeans(mu1_mat)
  } else {
    mu1_hat <- exp(xb1)
  }
  if (q2 > 0) {
    sd2  <- exp(par_hat[(k1+k2+q1+1):(k1+k2+q1+q2)])
    dev2 <- rand_realize(Z2_opt, dist2, sign2, beta2_hat[rand_idx2], sd2)$dev
    mu2_mat <- vapply(seq_len(n_draws),
                      function(r) pmin(exp(xb2 + as.vector(XR2 %*% dev2[r, ])), 1e15),
                      numeric(length(Y2)))
    mu2_hat <- rowMeans(mu2_mat)
  } else {
    mu2_hat <- exp(xb2)
  }

  ll_hat <- as.numeric(stats::logLik(fit))
  # maxLik BFGS (optim-based) returns code 0 on success; code 1 is
  # "iteration limit exceeded" -- NOT convergence. Only 0 counts as converged.
  convergence <- list(converged = isTRUE(fit$code == 0L), code = fit$code,
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
    ll_trace = ll_trace, convergence = convergence, call = match.call(),
    hessian_diag = hdiag,
    rp_meta = list(dist1 = dist1, dist2 = dist2, sign1 = sign1, sign2 = sign2,
                   Z1 = Z1_opt, Z2 = Z2_opt, halton_burn = halton_burn),
    predict_meta = .prep_predict_meta(prep),
    poisson_1 = poisson_1, poisson_2 = poisson_2)
}
