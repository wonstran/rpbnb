# Public fit_bnb() plus the bnb_fit constructor and the two internal fitters
# (Famoye/Sarmanov bivariate NB and independence = two univariate NB2 margins).

#' Construct a bnb_fit object
#' @keywords internal
#' @noRd
new_bnb_fit <- function(coef, vcov, se, logLik, nobs, npar, dependence,
                        lambda, bounds, mu1, mu2, X1, X2, Y1, Y2,
                        formula_1, formula_2, ll_trace, convergence, call,
                        cop_family = NULL, cop_par = NULL, cop_tau = NULL,
                        hessian_diag = NULL, predict_meta = NULL,
                        poisson_1 = FALSE, poisson_2 = FALSE) {
  structure(
    list(coef = coef, vcov = vcov, se = se, logLik = logLik,
         nobs = nobs, npar = npar, dependence = dependence,
         lambda = lambda, bounds = bounds, mu1 = mu1, mu2 = mu2,
         X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
         formula_1 = formula_1, formula_2 = formula_2,
         ll_trace = ll_trace, convergence = convergence,
         AIC = -2 * logLik + 2 * npar, BIC = -2 * logLik + log(nobs) * npar,
         call = call,
         cop_family = cop_family, cop_par = cop_par, cop_tau = cop_tau,
         hessian_diag = hessian_diag,
         # terms/xlevels/contrasts/offsets for column-stable predict() designs.
         predict_meta = predict_meta,
         # Poisson-limit restriction flags per margin, so bnb_gof() can rebuild a
         # same-family (Poisson-restricted) null instead of an unrestricted NB one.
         poisson_1 = poisson_1, poisson_2 = poisson_2),
    class = "bnb_fit"
  )
}

#' Internal estimator: Famoye/Sarmanov BNB via maxLik BFGS + analytic gradient
#'
#' Receives pre-built response vectors and design matrices. Returns a plain
#' list with fields consumed by [fit_bnb()].
#' @keywords internal
#' @noRd
fit_bnb_famoye <- function(Y1, Y2, X1, X2, cn1, cn2, start, control,
                           poisson_1 = FALSE, poisson_2 = FALSE,
                           off1 = NULL, off2 = NULL) {
  p1 <- NCOL(X1); p2 <- NCOL(X2)
  off1 <- .as_offset(off1, nrow(X1)); off2 <- .as_offset(off2, nrow(X2))

  par_names <- c(paste0("b1:", cn1), paste0("b2:", cn2),
                 "log_m1", "log_m2", "z_lambda")
  zero_start <- c(rep(0, p1 + p2), log(0.5), log(0.5), 0)

  # Poisson-limit margins: pin log_m at log(POISSON_M) and hold it fixed during
  # optimization (maxLik `fixed=`), so it is not an estimated parameter.
  fixed_names <- c(if (isTRUE(poisson_1)) "log_m1", if (isTRUE(poisson_2)) "log_m2")
  free <- !(par_names %in% fixed_names)

  # Multi-start policy. The famoye analytic gradient freezes the lambda-bounds,
  # so the BFGS objective is start-sensitive and no single start dominates
  # (inst/validation/start_sensitivity_famoye.R: a zero start wins on rwm1984 and
  # low/mid-mean data; marginal glm.nb starts win on high-mean data). With no
  # user start, optimize from BOTH candidates and keep the best converged
  # objective. A user-supplied start (positional or named) is honored as given.
  if (is.null(start)) {
    nb   <- .marginal_nb_starts(Y1, X1, Y2, X2)
    cand <- list(zero  = zero_start,
                 glmnb = c(nb$b1, nb$b2, nb$log_m1, nb$log_m2, 0))
  } else {
    cand <- list(user = .resolve_start(start, zero_start, par_names, "start"))
  }
  cand <- lapply(cand, function(s) { names(s) <- par_names; s })
  # Pin the Poisson-limit dispersions in every start so maxLik holds them there.
  if (length(fixed_names)) {
    cand <- lapply(cand, function(s) { s[fixed_names] <- log(POISSON_M); s })
  }

  # --- capture logLik at every evaluation (reset per candidate) ---
  # A poisson_* margin uses the exact m = 0 (Poisson) branch; the pinned log_m in
  # the parameter vector is only a display placeholder, ignored by the math.
  .ll_eval <- numeric(0)
  ll_fun <- function(p) {
    v <- bnb_loglik_vec(p, Y1, Y2, X1, X2, pois1 = poisson_1, pois2 = poisson_2,
                        off1 = off1, off2 = off2)
    .ll_eval <<- c(.ll_eval, sum(v))         # record total logLik
    v
  }
  grad_fun <- function(p) bnb_grad_vec(p, Y1, Y2, X1, X2, pois1 = poisson_1,
                                       pois2 = poisson_2, off1 = off1, off2 = off2)

  ml_control <- list(iterlim = control$iterlim,
                     reltol = control$reltol,
                     printLevel = control$print_level)

  run_one <- function(s0) {
    .ll_eval <<- numeric(0)
    f <- tryCatch(
      maxLik::maxLik(logLik = ll_fun, grad = grad_fun, start = s0,
                     method = "BFGS", control = ml_control,
                     fixed = if (length(fixed_names)) fixed_names else NULL),
      error = function(e) NULL)
    if (is.null(f)) NULL else list(fit = f, trace = .ll_eval)
  }
  runs <- Filter(Negate(is.null), lapply(cand, run_one))
  if (!length(runs)) {
    stop("famoye optimization failed from all candidate starts.", call. = FALSE)
  }
  conv <- vapply(runs, function(r) isTRUE(r$fit$code == 0L), logical(1))
  pool <- if (any(conv)) runs[conv] else runs      # prefer converged fits (code 0)
  lls  <- vapply(pool, function(r) as.numeric(stats::logLik(r$fit)), numeric(1))
  best <- pool[[which.max(lls)]]
  fit  <- best$fit
  .ll_eval <- best$trace                            # trace of the winning run
  par_hat <- stats::coef(fit)

  # Back-transform and lambda at fitted bounds
  beta1_hat <- par_hat[paste0("b1:", cn1)]
  beta2_hat <- par_hat[paste0("b2:", cn2)]
  m1_hat <- unname(exp(par_hat["log_m1"]))
  m2_hat <- unname(exp(par_hat["log_m2"]))
  z_hat  <- unname(par_hat["z_lambda"])

  mu1_hat <- as.vector(exp(X1 %*% beta1_hat + off1))
  mu2_hat <- as.vector(exp(X2 %*% beta2_hat + off2))
  # A Poisson-restricted margin uses the exact m = 0 limit everywhere, so its
  # dependence constant is c = exp(-d*mu) (m -> 0). Force m = 0 here too, so the
  # frozen lambda-bounds used for the SEs match the objective's Poisson margin.
  if (isTRUE(poisson_1)) m1_hat <- 0
  if (isTRUE(poisson_2)) m2_hat <- 0
  c1_hat  <- c_val(mu1_hat, m1_hat)
  c2_hat  <- c_val(mu2_hat, m2_hat)
  bnds_hat <- lambda_bounds_vec(c1_hat, c2_hat)
  eps <- 1e-6
  lambda_hat <- unname(bnds_hat[1] + (bnds_hat[2] - bnds_hat[1]) *
                         (eps + (1 - 2*eps) * plogis(z_hat)))

  ll_hat <- as.numeric(stats::logLik(fit))

  # Standard errors from the Hessian, with lambda-bounds frozen at optimum. The
  # analytic and numeric paths differentiate the same frozen-bounds objective.
  if (isTRUE(control$compute_se)) {
    lamLo_h <- bnds_hat[1]; lamHi_h <- bnds_hat[2]
    ll_fb <- function(p) bnbr_loglik_fixed_bounds(p, Y1, Y2, X1, X2, lamLo_h, lamHi_h,
                                                  pois1 = poisson_1, pois2 = poisson_2,
                                                  off1 = off1, off2 = off2)

    H <- if (identical(control$hessian, "analytic")) {
      bnb_hessian_fixed_bounds(par_hat, Y1, Y2, X1, X2, lamLo_h, lamHi_h,
                               pois1 = poisson_1, pois2 = poisson_2,
                               off1 = off1, off2 = off2)
    } else {
      numDeriv::hessian(ll_fb, par_hat,
                        method.args = list(r = control$hess_r, eps = control$hess_eps))
    }
    info <- -H; info <- (info + t(info)) / 2
    if (any(!is.finite(info))) {
      warning("Non-finite information; retrying Hessian with larger step.")
      H <- numDeriv::hessian(ll_fb, par_hat,
                             method.args = list(r = max(6, control$hess_r + 2),
                                                eps = max(1e-4, 5 * control$hess_eps)))
      info <- -H; info <- (info + t(info)) / 2
    }
    inv <- .free_index_vcov(info, names(par_hat), free, label = "famoye BNB")
    vc <- inv$vcov; se <- inv$se; hdiag <- inv$diag
  } else {
    vc <- matrix(NA_real_, length(par_hat), length(par_hat))
    se <- rep(NA_real_, length(par_hat))
    hdiag <- NULL
  }
  dimnames(vc) <- list(names(par_hat), names(par_hat))
  names(se) <- names(par_hat)

  # maxLik BFGS (optim-based) returns code 0 on success; code 1 is
  # "iteration limit exceeded" -- NOT convergence. Only 0 counts as converged.
  convergence <- list(converged = isTRUE(fit$code == 0L), code = fit$code,
                      message = fit$message, iterations = fit$iterations)

  list(coef = par_hat, vcov = vc, se = se, logLik = ll_hat,
       npar = length(par_hat) - length(fixed_names),   # pinned dispersions not free
       lambda = lambda_hat, bounds = c(bnds_hat[1], bnds_hat[2]),
       mu1 = mu1_hat, mu2 = mu2_hat,
       ll_trace = .ll_eval, convergence = convergence, hessian_diag = hdiag)
}

#' Internal estimator: copula with NB2 margins via maxLik BFGS + analytic gradient
#' @keywords internal
#' @noRd
fit_bnb_copula <- function(Y1, Y2, X1, X2, cn1, cn2, family, start, control,
                           off1 = NULL, off2 = NULL) {
  p1 <- NCOL(X1); p2 <- NCOL(X2)
  off1 <- .as_offset(off1, nrow(X1)); off2 <- .as_offset(off2, nrow(X2))

  nb <- .marginal_nb_starts(Y1, X1, Y2, X2)             # z_theta = 0 (independence)
  par_names <- c(paste0("b1:", cn1), paste0("b2:", cn2),
                 "log_m1", "log_m2", "z_theta")
  start <- .resolve_start(start, c(nb$b1, nb$b2, nb$log_m1, nb$log_m2, 0),
                          par_names, "start")

  .ll_eval <- numeric(0)
  ll_fun <- function(p) {
    v <- copula_loglik_vec(p, Y1, Y2, X1, X2, family, off1 = off1, off2 = off2)
    .ll_eval <<- c(.ll_eval, sum(v))
    v
  }
  grad_fun <- function(p) copula_grad_vec(p, Y1, Y2, X1, X2, family,
                                          off1 = off1, off2 = off2)

  ml_control <- list(iterlim = control$iterlim, reltol = control$reltol,
                     printLevel = control$print_level)

  fit <- maxLik::maxLik(logLik = ll_fun, grad = grad_fun, start = start,
                        method = "BFGS", control = ml_control)
  par_hat <- stats::coef(fit)

  beta1_hat <- par_hat[paste0("b1:", cn1)]
  beta2_hat <- par_hat[paste0("b2:", cn2)]
  z_hat     <- unname(par_hat["z_theta"])
  mu1_hat   <- as.vector(exp(X1 %*% beta1_hat + off1))
  mu2_hat   <- as.vector(exp(X2 %*% beta2_hat + off2))

  native_hat <- z_to_native(family, z_hat)
  td         <- copula_tau_and_deriv(family, z_hat)
  ll_hat     <- as.numeric(stats::logLik(fit))

  if (isTRUE(control$compute_se)) {
    ll_total <- function(p) sum(copula_loglik_vec(p, Y1, Y2, X1, X2, family,
                                                  off1 = off1, off2 = off2))
    H <- numDeriv::hessian(ll_total, par_hat,
                           method.args = list(r = control$hess_r, eps = control$hess_eps))
    info <- -H; info <- (info + t(info)) / 2
    if (any(!is.finite(info))) {
      H    <- numDeriv::hessian(ll_total, par_hat,
                                method.args = list(r = max(6L, control$hess_r + 2L),
                                                   eps = max(1e-4, 5 * control$hess_eps)))
      info <- -H; info <- (info + t(info)) / 2
    }
    inv <- .observed_info_vcov(info, names(par_hat),
                               label = paste0(family, " copula BNB"))
    vc <- inv$vcov; se <- inv$se; hdiag <- inv$diag
  } else {
    vc <- matrix(NA_real_, length(par_hat), length(par_hat))
    se <- rep(NA_real_, length(par_hat))
    hdiag <- NULL
  }
  dimnames(vc) <- list(names(par_hat), names(par_hat))
  names(se) <- names(par_hat)

  # maxLik BFGS (optim-based) returns code 0 on success; code 1 is
  # "iteration limit exceeded" -- NOT convergence. Only 0 counts as converged.
  convergence <- list(converged = isTRUE(fit$code == 0L), code = fit$code,
                      message = fit$message, iterations = fit$iterations)

  list(coef = par_hat, vcov = vc, se = se, logLik = ll_hat,
       npar = length(par_hat),
       mu1 = mu1_hat, mu2 = mu2_hat,
       ll_trace = .ll_eval, convergence = convergence,
       cop_family = family, cop_par = native_hat, cop_tau = td$tau,
       hessian_diag = hdiag)
}

# Fit one independence margin: NB2 (MASS::glm.nb) or, when Poisson, an exact
# Poisson GLM. Returns the fit, its beta vcov, log_m (log(POISSON_M) pinned for
# Poisson), var(log_m) (NA for Poisson -- fixed, not estimated), and whether the
# dispersion is a free parameter. `which` labels errors ("Margin 1"/"Margin 2").
.fit_independence_margin <- function(formula, data, poisson, which) {
  if (isTRUE(poisson)) {
    g <- tryCatch(stats::glm(formula, family = stats::poisson, data = data),
                  error = function(e) stop(which, " (", deparse(formula),
                    ") Poisson glm failed: ", conditionMessage(e), call. = FALSE))
    list(g = g, vbeta = stats::vcov(g), log_m = log(POISSON_M),
         var_log_m = NA_real_, m_free = FALSE)
  } else {
    g <- tryCatch(MASS::glm.nb(formula, data = data),
                  error = function(e) stop(which, " (", deparse(formula),
                    ") glm.nb failed: ", conditionMessage(e), call. = FALSE))
    list(g = g, vbeta = stats::vcov(g), log_m = log(1 / g$theta),
         var_log_m = (g$SE.theta / g$theta)^2, m_free = TRUE)   # delta-method
  }
}

#' Internal estimator: independence = two univariate margins (NB2 or Poisson)
#' @keywords internal
#' @noRd
fit_bnb_independence <- function(formula_1, formula_2, data, cn1, cn2,
                                 X1, X2, Y1, Y2,
                                 poisson_1 = FALSE, poisson_2 = FALSE) {
  f1 <- .fit_independence_margin(formula_1, data, poisson_1, "Margin 1")
  f2 <- .fit_independence_margin(formula_2, data, poisson_2, "Margin 2")
  g1 <- f1$g; g2 <- f2$g
  p1 <- length(stats::coef(g1)); p2 <- length(stats::coef(g2))

  # Independence has no dependence parameter: the coefficient vector holds only
  # the two betas and the two dispersions (no fabricated z_lambda). A Poisson
  # margin pins log_m at the Poisson limit and does not count it as free.
  par <- c(stats::coef(g1), stats::coef(g2), f1$log_m, f2$log_m)
  names(par) <- c(paste0("b1:", cn1), paste0("b2:", cn2),
                  "log_m1", "log_m2")

  ntot <- length(par)
  vc <- matrix(0, ntot, ntot, dimnames = list(names(par), names(par)))
  vc[1:p1, 1:p1] <- f1$vbeta
  vc[p1 + (1:p2), p1 + (1:p2)] <- f2$vbeta
  # log_m = log(1/theta); delta-method var(log_m) = (SE.theta / theta)^2. A
  # pinned Poisson dispersion gets NA (fixed, not estimated).
  vc[p1 + p2 + 1, p1 + p2 + 1] <- f1$var_log_m
  vc[p1 + p2 + 2, p1 + p2 + 2] <- f2$var_log_m

  se <- sqrt(pmax(diag(vc), 0)); names(se) <- names(par)
  n_free_m <- f1$m_free + f2$m_free

  ll <- as.numeric(stats::logLik(g1)) + as.numeric(stats::logLik(g2))

  convergence <- list(converged = isTRUE(g1$converged) && isTRUE(g2$converged),
                      code = NA_integer_,
                      message = if (n_free_m == 2L) "glm.nb" else "glm.nb/poisson",
                      iterations = NA_integer_)

  list(coef = par, vcov = vc, se = se, logLik = ll,
       npar = p1 + p2 + n_free_m,        # betas + free dispersions (no lambda)
       lambda = 0, bounds = c(NA_real_, NA_real_),
       mu1 = unname(stats::fitted(g1)), mu2 = unname(stats::fitted(g2)),
       ll_trace = numeric(0), convergence = convergence)
}

#' Fit a bivariate negative binomial regression model
#'
#' @param formula_1,formula_2 Model formulas for the two count outcomes. An
#'   equation-specific `offset()` term (e.g. `y ~ x + offset(log(exposure))`) is
#'   supported on every dependence path: the offset enters that margin's linear
#'   predictor additively (mean `exp(x'beta + offset)`) during estimation, and is
#'   carried through the stored fitted means and both `predict()` methods.
#' @param data A data frame.
#' @param dependence Dependence structure: "independence" (two univariate NB2
#'   margins), "famoye" (Famoye/Sarmanov bivariate NB), or a [copula()] object
#'   (Frank / Gaussian / Clayton discrete-copula bivariate NB; the dependence
#'   parameter is estimated).
#' @param start Optional starting parameter vector. May be positional (length
#'   equal to the number of parameters) or named; a named vector is reordered to
#'   the canonical parameter order and a named partial vector is merged into the
#'   defaults (unknown or duplicate names are rejected). When `start` is `NULL`,
#'   the famoye path uses a multi-start policy: it optimizes from both an all-zero
#'   mean-coefficient start and marginal `glm.nb` starts and keeps the better
#'   converged objective (the frozen-bounds gradient makes the objective
#'   start-sensitive and neither start dominates).
#' @param control An [rpbnb_control()] object. The famoye and copula estimators
#'   both use BFGS, the only optimizer `control$method` accepts.
#' @param poisson_1,poisson_2 Fit the corresponding margin at its exact Poisson
#'   limit (NB2 dispersion `m = 0`) instead of estimating the dispersion. The
#'   margin's `log_m` is held fixed, so it is not a free parameter and the fit is
#'   a properly nested restriction of the NB model -- pair it with [lr_test()]
#'   (`boundary = TRUE`) to test for overdispersion. This is the exact `m = 0`
#'   restriction at any fitted mean: the margin's log-pmf is `dpois` and its
#'   Famoye dependence constant is `exp(-d*mu)` (the `m -> 0` limit), not an NB2
#'   at a tiny pinned dispersion. The famoye and independence paths are both
#'   exact (the independence path fits a Poisson GLM margin). Not supported with
#'   a [copula()] dependence.
#' @return An object of class `bnb_fit`.
#' @export
#' @examples
#' d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
#' fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
#'                dependence = "famoye")
#' summary(fit)
#'
#' # Overdispersion test for margin 1 (H0: m1 = 0, Poisson)
#' fit_p1 <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
#'                   dependence = "famoye", poisson_1 = TRUE)
#' lr_test(fit_p1, fit, boundary = TRUE)
#'
#' # Gaussian copula dependence instead of Famoye/Sarmanov
#' fit_cop <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
#'                    dependence = copula("normal"))
#' fit_cop$cop_tau  # estimated Kendall's tau
fit_bnb <- function(formula_1, formula_2, data,
                    dependence = c("independence", "famoye"),
                    start = NULL, control = rpbnb_control(),
                    poisson_1 = FALSE, poisson_2 = FALSE) {

  .chk_poisson_flag(poisson_1, "poisson_1")
  .chk_poisson_flag(poisson_2, "poisson_2")

  prep <- .prepare_bnb_data(formula_1, formula_2, data)
  Y1 <- prep$Y1; Y2 <- prep$Y2
  X1 <- prep$X1; X2 <- prep$X2
  cn1 <- prep$cn1; cn2 <- prep$cn2

  if (inherits(dependence, "rpbnb_copula")) {
    if (isTRUE(poisson_1) || isTRUE(poisson_2)) {
      stop("poisson_1 / poisson_2 (Poisson-limit margins) are not supported ",
           "with a copula() dependence.", call. = FALSE)
    }
    res <- fit_bnb_copula(Y1, Y2, X1, X2, cn1, cn2,
                          family = dependence$family, start = start, control = control,
                          off1 = prep$off1, off2 = prep$off2)
    return(new_bnb_fit(
      coef = res$coef, vcov = res$vcov, se = res$se,
      logLik = res$logLik, nobs = length(Y1), npar = res$npar,
      dependence = res$cop_family,
      lambda = NULL, bounds = c(NA_real_, NA_real_),
      mu1 = res$mu1, mu2 = res$mu2, X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
      formula_1 = formula_1, formula_2 = formula_2,
      ll_trace = res$ll_trace, convergence = res$convergence, call = match.call(),
      cop_family = res$cop_family, cop_par = res$cop_par, cop_tau = res$cop_tau,
      hessian_diag = res$hessian_diag, predict_meta = .prep_predict_meta(prep)
    ))
  }

  dependence <- match.arg(dependence)

  res <- if (dependence == "famoye") {
    fit_bnb_famoye(Y1, Y2, X1, X2, cn1, cn2, start, control,
                   poisson_1 = poisson_1, poisson_2 = poisson_2,
                   off1 = prep$off1, off2 = prep$off2)
  } else {
    fit_bnb_independence(formula_1, formula_2, prep$data, cn1, cn2, X1, X2, Y1, Y2,
                         poisson_1 = poisson_1, poisson_2 = poisson_2)
  }

  new_bnb_fit(coef = res$coef, vcov = res$vcov, se = res$se,
              logLik = res$logLik, nobs = length(Y1), npar = res$npar,
              dependence = dependence, lambda = res$lambda, bounds = res$bounds,
              mu1 = res$mu1, mu2 = res$mu2, X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
              formula_1 = formula_1, formula_2 = formula_2,
              ll_trace = res$ll_trace, convergence = res$convergence,
              call = match.call(), hessian_diag = res$hessian_diag,
              predict_meta = .prep_predict_meta(prep),
              poisson_1 = poisson_1, poisson_2 = poisson_2)
}
